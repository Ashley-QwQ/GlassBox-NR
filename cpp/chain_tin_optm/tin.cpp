#include "../geometry.hpp"
#include "primitives.inc"
#include <map>
#include <cfenv>
#include <chrono>
#include <condition_variable>
#include <sstream>
#include <iomanip>
namespace {
std::mutex abi_mutex;
struct AbiLease {std::unique_lock<std::mutex> lock;AbiLease():lock(abi_mutex,std::try_to_lock){if(!lock.owns_lock())throw std::runtime_error("CONFIGURE_RUN_BUSY");}};
struct Matrix { int K=0,N=0; std::vector<uint8_t> v; uint8_t at(int k,int n)const{return v.at(size_t(k)*N+n);} };
uint16_t read16(const uint8_t* p){return uint16_t(p[0])|uint16_t(p[1])<<8;}
uint32_t read32(const uint8_t* p){uint32_t v=0;for(int i=0;i<4;++i)v|=uint32_t(p[i])<<(8*i);return v;}
uint64_t read64(const uint8_t* p){uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t(p[i])<<(8*i);return v;}
// All launch-owned immutable panels are built before GIL release and workers.
struct Panel {
 int K=0,N=0;bool decoded=true;const E4Table* table=nullptr;
 std::vector<uint8_t> bytes;std::vector<E4Decoded> values;
 E4Decoded at(int k,int j)const{return decoded?values[size_t(j)*K+k]:(*table)[bytes[size_t(j)*K+k]];}
 size_t storage()const{return bytes.capacity()+values.capacity()*sizeof(E4Decoded);}
};
struct HeadPanels {Panel ex,br,out;};

struct Wt { Wt()=default;Wt(const Wt&)=delete;Wt& operator=(const Wt&)=delete;Wt(Wt&&)=delete;Wt& operator=(Wt&&)=delete;int C,H; Matrix ex,branch,out,qkv,proj,extra;std::vector<uint16_t> cs,ca,sin,bias,scale; E4Table table;Panel pex,pbranch,pout,pqkv,pproj,pextra;std::vector<HeadPanels> heads; };
Matrix unpack(const uint8_t* p,int K,int N,bool n128=false){
 Matrix m{K,N,std::vector<uint8_t>(size_t(K)*N)}; const int rowperm[16]={0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15};
 for(int k=0;k<K;++k)for(int n=0;n<N;++n){int rk=(k/16)*16+rowperm[k%16],kt=rk/32,nt=n/32,ki=rk%32,ni=n%32;
 int lane=(ni%8)*4+(ki%16)/4,elem=(ki%4)+(ki/16)*4,frag=(ni/8)%2,pair=ni/16;
 int tile=n128?((nt/4)*(K/32)+kt)*4+nt%4:kt*(N/32)+nt;
 m.v[size_t(k)*N+n]=p[size_t(tile)*1024+pair*512+lane*16+frag*8+elem];}return m;
}
int tokeninternal(int t){return ((t/8)/4)*32+((t%8)/4)*16+((t/8)%4)*4+t%4;}
void decode_weights(Wt& w,const uint8_t* p,size_t n,int C,bool up,bool down){
 w.C=C;w.H=C/32;size_t o=0;
 auto mat=[&](int K,int N,bool major=false){size_t len=size_t(K)*N;if(o+len>n)throw std::out_of_range("weight matrix");Matrix m=unpack(p+o,K,N,major);o+=len;return m;};
 auto halves=[&](int count){if(o+2*count>n)throw std::out_of_range("weight half");std::vector<uint16_t> v(count);for(int i=0;i<count;++i)v[i]=read16(p+o+2*i);o+=2*count;return v;};
 if(C==32){w.ex=mat(32,128);w.branch=mat(128,32);}else{w.ex=mat(C,4*C+128,true);w.out=mat(C,C);}
 if(up)w.extra=mat(2*C,C);
 if(!up||C==32)o+=16;
 w.cs=halves(C);
 if(up){if(C==32)o+=16;w.sin=halves(C);}else o+=16;
 w.qkv=mat(C,3*C);
 if(C>=64){Matrix q=w.qkv;for(int k=0;k<C;++k)for(int j=0;j<3*C;++j){int type=j/C,h=(j%C)/32,c=j%32;w.qkv.v[size_t(k)*3*C+j]=q.at(k,h*96+type*32+c);}}
 auto b=halves(w.H*4096);w.bias.resize(b.size());
 for(int h=0;h<w.H;++h)for(int i=0;i<64;++i)for(int j=0;j<64;++j){int r=tokeninternal(i),c=tokeninternal(j);
 int tile=(r/16)*4+c/16,ri=r%16,ci=c%16;
 int lane=(ri%8)*4+(ci%8)/2,el=(ci/8)*4+(ri/8)*2+ci%2;
 w.bias[size_t(h)*4096+i*64+j]=b[size_t(h)*4096+tile*256+lane*8+el];}
 w.scale.resize(w.H);for(int h=0;h<w.H;++h){if(o+4>n)throw std::out_of_range("weight scale");uint32_t u=read32(p+o);w.scale[h]=half_round(optm_bit_cast<float>(u));o+=4;}
 o=(o+15)&~size_t(15);w.proj=mat(C,C);w.ca=halves(C);if(down)w.extra=mat(C,2*C);
}
Panel panel(const Matrix& m,const E4Table& table,bool decoded){
 Panel p;p.K=m.K;p.N=m.N;p.table=&table;p.decoded=decoded;
 if(decoded)p.values.resize(m.v.size());else p.bytes.resize(m.v.size());
 for(int j=0;j<m.N;++j)for(int k=0;k<m.K;++k){auto c=m.at(k,j);size_t i=size_t(j)*m.K+k;if(decoded)p.values[i]=table[c];else p.bytes[i]=c;}
 return p;
}
void prepare_panels(Wt& w,bool decoded){
 w.table=make_e4_table();
 w.pex=panel(w.ex,w.table,decoded);w.pbranch=panel(w.branch,w.table,decoded);w.pout=panel(w.out,w.table,decoded);
 w.pqkv=panel(w.qkv,w.table,decoded);w.pproj=panel(w.proj,w.table,decoded);w.pextra=panel(w.extra,w.table,decoded);
 if(w.C>32){w.heads.resize(w.H);for(int h=0;h<w.H;++h){int C=w.C;
  Matrix ex{C,128,std::vector<uint8_t>(size_t(C)*128)},br{128,32,std::vector<uint8_t>(128*32)},out{32,C,std::vector<uint8_t>(size_t(32)*C)};
  for(int k=0;k<C;++k)for(int j=0;j<128;++j)ex.v[size_t(k)*128+j]=w.ex.at(k,h*128+j);
  for(int k=0;k<128;++k)for(int j=0;j<32;++j)br.v[k*32+j]=w.ex.at(h*32+k%32,4*C+(k/32)*32+j);
  for(int k=0;k<32;++k)for(int j=0;j<C;++j)out.v[size_t(k)*C+j]=w.out.at(h*32+k,j);
  w.heads[h]={panel(ex,w.table,decoded),panel(br,w.table,decoded),panel(out,w.table,decoded)};
 }}
}

// Layouts follow sm120_b1.py (C32), wo_spec.py (C64/128/256), planar16_wh.
size_t logical_of(size_t p,int W,int H,int C,int layout){
 const int perm[16]={0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15};
 if(layout!=0){size_t j=p%16,x=(p/16)%W,y=(p/(16*W))%H,plane=p/(16*size_t(W)*H);return (y*W+x)*C+plane*16+perm[j];}
 if(C==32){size_t tile=p/512,l=(p%512)/16,b=p%16;int x=(l>>2)&3,y=(l>>4)+((b>>2)&1)*2,c=(l&3)*2+(b&1)+((b>>1)&1)*8+(b>>3)*16;return (((tile/(W/4))*4+y)*W+(tile%(W/4))*4+x)*32+c;}
 size_t off=p%4096,blk=p/4096;int cb=optm_countr_zero(unsigned(C)),xs=1024/C;
 int c=0,x=0,y=0;const char kind[9]={'c','c','y','c','c','c','x','x','y'};const int bit[9]={0,3,1,4,1,2,0,1,0};
 for(int i=0;i<9;++i){int v=int((off>>i)&1)<<bit[i];if(kind[i]=='c')c|=v;else if(kind[i]=='x')x|=v;else y|=v;}
 int pos=9;for(int b=5;b<cb;++b)c|=int((off>>pos++)&1)<<b;
 for(int b=2;(1<<b)<xs;++b)x|=int((off>>pos++)&1)<<b;
 return (((blk/(W/xs))*4+y)*W+(blk%(W/xs))*xs+x)*C+c;
}
std::vector<uint8_t> decode_layout(const uint8_t* p,int W,int H,int C,int layout){std::vector<uint8_t> v(size_t(W)*H*C);for(size_t i=0;i<v.size();++i)v[logical_of(i,W,H,C,layout)]=p[i];return v;}
std::vector<uint8_t> encode_layout(const std::vector<uint8_t>& v,int W,int H,int C,int layout){std::vector<uint8_t> r(v.size());for(size_t i=0;i<r.size();++i)r[i]=v.at(logical_of(i,W,H,C,layout));return r;}
std::vector<uint16_t> RSQ,RCP;
uint16_t lookup(const std::vector<uint16_t>& t,uint16_t b){if(b<0x410||size_t(b-0x410)>=t.size())throw std::domain_error("MUFU outside table");return t[b-0x410];}
std::vector<uint16_t> multiply(const std::vector<uint8_t>& a,int rows,const Panel& b,const std::vector<uint16_t>* seed=nullptr,int start=0,int K=-1){
 if(K<0)K=b.K;std::vector<uint16_t> r(size_t(rows)*b.N);
 int ak=int(a.size()/rows);
 // Decoding a row only records flags; unused codes never throw.
 std::vector<E4Decoded> av(K);
 for(int i=0;i<rows;++i){for(int k=0;k<K;++k)av[k]=(*b.table)[a[size_t(i)*ak+start+k]];
  for(int j=0;j<b.N;++j){uint16_t acc=seed?seed->at(size_t(i)*b.N+j):0;
   for(int k=0;k<K;k+=32){
    if(b.decoded){const auto* bv=b.values.data()+size_t(j)*b.K+start+k;acc=qmma_decoded(av.data()+k,[bv](int l){return bv[l];},acc);}
    else{const auto* bv=b.bytes.data()+size_t(j)*b.K+start+k;const auto* table=b.table;acc=qmma_decoded(av.data()+k,[bv,table](int l){return (*table)[bv[l]];},acc);}
   }r[size_t(i)*b.N+j]=acc;
  }
 }return r;
}


std::vector<uint8_t> publish(const std::vector<uint16_t>& a){std::vector<uint8_t> r(a.size());for(size_t i=0;i<a.size();++i)r[i]=sat_e4(a[i]);return r;}
uint8_t gate_reference(uint16_t e){double x=half_value(e),c=std::clamp(x,-4.,4.);uint16_t l=half_round(std::abs(c)*-0.055908203125+0.447265625);uint16_t f=half_round(c*half_value(l)+0.89453125);return sat_e4(half_round(x*half_value(f)));}
// Exact extraction of the original TIN score inline expression, preserving
// float intermediate statements and the original operator/guard order.
uint16_t score_reference(uint16_t score){
 float mul=float(half_value(score))*0.044921875f;float affine=mul+1.30078125f;uint16_t clipped=h_min(h_max(half_round(affine),half_round(1.03125)),half_round(1.5693359375));return uint16_t((uint32_t(clipped)<<5)+0x8000);
}
#include "unary_tables_optb2.inc"
uint8_t gate(uint16_t e){return optb2_gate_enabled?optb2_gate_table.lookup(e):gate_reference(e);}
uint16_t score_unary(uint16_t e){return optb2_score_enabled?optb2_score_table.lookup(e):score_reference(e);}
void norm32(const uint16_t* x,uint8_t* out,uint16_t scale){
 uint16_t t[8];for(int i=0;i<8;++i)t[i]=h_add(h_fma(x[i],x[i],h_mul(x[16+i],x[16+i])),h_fma(x[8+i],x[8+i],h_mul(x[24+i],x[24+i])));
 uint16_t a[8],b[8];for(int i=0;i<8;++i)a[i]=h_add(t[i],t[i^4]);for(int i=0;i<8;++i)b[i]=h_add(a[i],a[i^2]);uint16_t n=h_max(h_add(b[0],b[1]),0x410),r=lookup(RSQ,n);
 for(int i=0;i<32;++i)out[i]=sat_e4(h_mul(h_mul(x[i],r),scale));
}
void attn(const Wt& w,const std::vector<uint16_t>& qkv,int C,std::vector<uint8_t>& av){
 av.resize(64*C);std::vector<uint8_t> q(64*32),k(q.size()),v(q.size()),p(4096);uint8_t ac[32],bc[32];
 for(int h=0;h<w.H;++h){for(int i=0;i<64;++i){norm32(&qkv[size_t(i)*3*C+h*32],&q[i*32],w.scale[h]);norm32(&qkv[size_t(i)*3*C+C+h*32],&k[i*32],0x3c00);for(int l=0;l<32;++l)v[i*32+l]=sat_e4(qkv[size_t(i)*3*C+2*C+h*32+l]);}
 for(int i=0;i<64;++i){uint16_t ww[64];for(int j=0;j<64;++j){uint16_t score=qmma_dot(&q[i*32],&k[j*32],w.bias[size_t(h)*4096+i*64+j]);
 ww[j]=score_unary(score);}
 auto leg=[&](int b){uint16_t x=h_add(h_add(ww[b],ww[b+16]),h_add(ww[b+4],ww[b+20]));x=h_add(x,h_add(ww[b+32],ww[b+48]));return h_add(x,h_add(ww[b+36],ww[b+52]));};
 uint16_t d=h_add(h_add(h_add(h_add(leg(0),leg(2)),leg(8)),leg(10)),h_add(h_add(h_add(leg(1),leg(3)),leg(9)),leg(11)));
 if(d<0x1c00||d>0x60e0)throw std::domain_error("attention denominator");uint16_t r=lookup(RCP,d);for(int j=0;j<64;++j)p[i*64+j]=sat_e4(h_mul(ww[j],r));}
 for(int i=0;i<64;++i)for(int j=0;j<32;++j){uint16_t acc=0;for(int s=0;s<2;++s){for(int l=0;l<32;++l){ac[l]=p[i*64+s*32+l];bc[l]=v[(s*32+l)*32+j];}acc=qmma_dot(ac,bc,acc);}av[size_t(i)*C+h*32+j]=sat_e4(acc);}
 }
}
std::vector<uint16_t> window(const Wt& w,const std::vector<uint8_t>& x,const std::vector<uint16_t>* sx=nullptr){
 int C=w.C;std::vector<uint16_t> z(64*C);for(size_t i=0;i<z.size();++i)z[i]=h_mul(sx?sx->at(i):e4_half(x[i]),w.cs[i%C]);
 if(C==32){auto e=multiply(x,64,w.pex);std::vector<uint8_t> g(e.size());for(size_t i=0;i<g.size();++i)g[i]=gate(e[i]);z=multiply(g,64,w.pbranch,&z);}
 else for(int h=0;h<w.H;++h){const auto& p=w.heads[h];
 auto e=multiply(x,64,p.ex);std::vector<uint8_t> g(e.size());for(size_t i=0;i<g.size();++i)g[i]=gate(e[i]);auto u=publish(multiply(g,64,p.br));z=multiply(u,64,p.out,&z);}
 auto qz=publish(z);auto qkv=multiply(qz,64,w.pqkv);std::vector<uint8_t> av;attn(w,qkv,C,av);std::vector<uint16_t> seed(z.size());for(size_t i=0;i<seed.size();++i)seed[i]=h_mul(C==32?z[i]:e4_half(qz[i]),w.ca[i%C]);return multiply(av,64,w.pproj,&seed);
}
bool panel_decoded=true;
struct LaunchSpec{int C=0,extent=4,shift=5,inputlayout=0,outputlayout=0,pooledword=-1,skipword=-1,inW=0,inH=0,releaseword=-1;size_t weightbytes=0;bool up=false;};
std::map<int,LaunchSpec> specs;uint64_t wsbase=0,weightbase=0;
py::dict run_launch(int L,py::bytearray ws,py::bytes par,py::buffer weights,py::dict textures,int threads){
 AbiLease lease;py::buffer workspace_owner=py::reinterpret_borrow<py::buffer>(ws);py::buffer_info workspace_pin=workspace_owner.request();auto it=specs.find(L);if(it==specs.end())throw std::invalid_argument("launch not configured");auto s=it->second;
 std::string p=par;geo2::parameters(L,reinterpret_cast<const uint8_t*>(p.data()),p.size());auto q=[&](int i){if(i<0||size_t(i+1)*8>p.size())throw std::out_of_range("params qword");return read64(reinterpret_cast<const uint8_t*>(p.data())+i*8);};
 int W=int(q(s.extent)>>32),H=int(uint32_t(q(s.extent))),ox=int32_t(uint32_t(q(s.shift))),oy=int32_t(q(s.shift)>>32);
 if(W<=0||H<=0||W%4||H%4||W>geo2::g().nw||H>geo2::g().nh||!(ox==0||ox==-4)||!(oy==0||oy==-4))throw std::invalid_argument("geometry/origin");
 size_t wslen=PyByteArray_Size(ws.ptr());auto* dst=reinterpret_cast<uint8_t*>(PyByteArray_AsString(ws.ptr()));
 auto span=[&](uint64_t ptr,size_t n){if(ptr<wsbase||ptr-wsbase>wslen||n>wslen-size_t(ptr-wsbase))throw std::out_of_range("workspace span");return dst+size_t(ptr-wsbase);};
 auto wb=weights.request();if(wb.itemsize!=1||wb.ndim!=1||wb.strides[0]!=1)throw std::invalid_argument("weights require contiguous byte buffer");size_t wn=wb.size;
 uint64_t wp=q(2);if(wp<weightbase||wp-weightbase>wn||s.weightbytes>wn-size_t(wp-weightbase))throw std::out_of_range("weight span");
 Wt w;decode_weights(w,static_cast<const uint8_t*>(wb.ptr)+size_t(wp-weightbase),s.weightbytes,s.C,s.up,s.pooledword>=0);
 prepare_panels(w,panel_decoded);
 size_t n=geo2::product(geo2::product(size_t(W),size_t(H)),size_t(s.C));auto* op=span(q(1),n);auto input=decode_layout(span(q(0),s.up?size_t(s.inW)*s.inH*2*s.C:n),s.up?s.inW:W,s.up?s.inH:H,s.up?2*s.C:s.C,s.inputlayout);
 std::vector<uint16_t> merged;
 if(s.up){auto sk=decode_layout(span(q(s.skipword),n),W,H,s.C,0);auto proj=multiply(input,s.inW*s.inH,w.pextra);input.resize(n);merged.resize(n);for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int c=0;c<s.C;++c){size_t i=(size_t(y)*W+x)*s.C+c;uint16_t a=proj[(size_t(y/2)*s.inW+x/2)*s.C+c];merged[i]=h_fma(e4_half(sk[i]),w.sin[c],a);input[i]=sat_e4(merged[i]);}}
 int CW=s.C==32?W-2*ox:((W-ox+7)/8)*8,CH=s.C==32?H-2*oy:((H-oy+7)/8)*8;
 if(CW%8||CH%8)throw std::invalid_argument("window canvas");int GX=CW/8,GY=CH/8;std::vector<uint16_t> yhalf(n);
 threads=std::clamp(threads,1,12);optb2_last_launch=L;optb2_last_threads=threads;auto task_order=optb2_task_order(GX*GY);std::atomic<int> completed{0};std::atomic<int> next{0};std::exception_ptr ep;std::mutex err;std::atomic<bool> failed{false};
 {py::gil_scoped_release release;auto worker=[&](){try{while(!failed){int ticket=next.fetch_add(1);if(ticket>=GX*GY)break;int task=task_order[size_t(ticket)];std::vector<uint8_t> tile(64*s.C,0);std::vector<uint16_t> seed(64*s.C,0);int bx=task%GX,by=task/GX;
 for(int t=0;t<64;++t){int x=bx*8+t%8+ox,y=by*8+t/8+oy;if(x>=0&&x<W&&y>=0&&y<H)for(int c=0;c<s.C;++c){size_t a=(size_t(y)*W+x)*s.C+c,b=size_t(t)*s.C+c;tile[b]=input[a];if(s.up&&s.C==32)seed[b]=merged[a];}}
 auto out=window(w,tile,s.up&&s.C==32?&seed:nullptr);for(int t=0;t<64;++t){int x=bx*8+t%8+ox,y=by*8+t/8+oy;if(x>=0&&x<W&&y>=0&&y<H)for(int c=0;c<s.C;++c)yhalf[(size_t(y)*W+x)*s.C+c]=out[size_t(t)*s.C+c];}
 ++completed;}}catch(...){failed=true;std::lock_guard<std::mutex> l(err);if(!ep)ep=std::current_exception();}};std::vector<std::thread> pool;for(int t=1;t<threads;++t)pool.emplace_back(worker);worker();for(auto& t:pool)t.join();}
 optb2_last_claimed=std::min(next.load(),GX*GY);optb2_last_completed=completed.load();if(ep)std::rethrow_exception(ep);auto output=encode_layout(publish(yhalf),W,H,s.C,s.outputlayout);std::memcpy(op,output.data(),n);py::list ranges;ranges.append(py::make_tuple(q(1)-wsbase,n));
 if(s.pooledword>=0){int PW=W/2,PH=H/2;if(s.C==256){PW=int(q(10)>>32);PH=int(uint32_t(q(10)));}if(PW*2<W||PH*2<H)throw std::invalid_argument("negative pooling pad");std::vector<uint8_t> pin(size_t(PW)*PH*s.C);
 auto at=[&](int x,int y,int c){return x<W&&y<H?yhalf[(size_t(y)*W+x)*s.C+c]:uint16_t(0);};for(int y=0;y<PH;++y)for(int x=0;x<PW;++x)for(int c=0;c<s.C;++c){auto a=h_add(at(x*2,y*2,c),at(x*2+1,y*2,c)),b=h_add(at(x*2,y*2+1,c),at(x*2+1,y*2+1,c));pin[(size_t(y)*PW+x)*s.C+c]=sat_e4(h_mul(h_add(a,b),0x3400));}
 auto po=encode_layout(publish(multiply(pin,PW*PH,w.pextra)),PW,PH,2*s.C,1);auto* pd=span(q(s.pooledword),po.size());std::memcpy(pd,po.data(),po.size());ranges.append(py::make_tuple(q(s.pooledword)-wsbase,po.size()));
 // The ds_wait padding stores use 64 planar16 planes (literal zero),
 // including the upper 32 planes beyond the nominal 512-channel output.
 // Preserve this physical write footprint; do not clear valid/aliased data.
 if(s.C==256){const int validW=(W+1)/2,validH=(H+1)/2;
  for(int plane=0;plane<64;++plane)for(int y=0;y<PH;++y){
   const int begin=y<validH?validW:0;if(begin>=PW)continue;
   const size_t off=geo2::product((size_t(plane)*PH+y)*PW+begin,16);
   const size_t len=geo2::product(size_t(PW-begin),16);
   if(q(s.pooledword)>UINT64_MAX-off)throw std::out_of_range("pool padding pointer");
   auto* pad=span(q(s.pooledword)+off,len);std::memset(pad,0,len);
   ranges.append(py::make_tuple(q(s.pooledword)-wsbase+off,len));
  }
 }
}
 // Release store: one zero int32 per CTA, index=ctaid.y*grid.x+ctaid.x.
 if(s.releaseword>=0){size_t len=size_t(GX)*GY*4;auto* rp=span(q(s.releaseword),len);std::memset(rp,0,len);ranges.append(py::make_tuple(q(s.releaseword)-wsbase,len));}
 py::dict ret;ret["launch"]=L;ret["written_ranges"]=ranges;ret["textures_out"]=py::dict();return ret;
}
}
PYBIND11_MODULE(dlssnr_chain_tin_optm,m){
m.def("configure_geometry",&geo2::configure);m.def("geometry_probe",&geo2::probe);
 optb2_gate_table.generate(gate_reference);optb2_score_table.generate(score_reference);
#include "bindings_optb2.inc"
 m.def("set_panel_mode",[](bool decoded){AbiLease lease;if(!decoded)throw std::invalid_argument("OPTM_DECODED_PANEL_REQUIRED");panel_decoded=true;});
 m.def("primitive_batch",[](std::string kind,uint64_t start,uint64_t count){
  const auto table=make_e4_table();std::map<std::string,uint64_t> outcomes;
  auto rng=[](uint64_t& s){s+=0x9e3779b97f4a7c15ULL;uint64_t z=s;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);};
  struct Outcome{uint16_t bits=0;std::string error;bool operator==(const Outcome& o)const{return bits==o.bits&&error==o.error;}};
  auto observe=[](auto f){Outcome o;try{o.bits=f();}catch(const std::domain_error& e){o.error=std::string("domain_error:")+e.what();}return o;};
  auto qcheck=[&](const uint8_t* a,const uint8_t* b,uint16_t seed,const E4Table& tab,bool reverse=false){
   E4Decoded av[32],bv[32];for(int k=0;k<32;++k){av[k]=tab[a[reverse?31-k:k]];bv[k]=tab[b[reverse?31-k:k]];}
   auto ref=observe([&](){return qmma_dot(a,b,seed);});auto got=observe([&](){return qmma_decoded(av,[&](int k){return bv[k];},seed);});
   if(!(ref==got))throw std::runtime_error("QMMA_DIFF:"+ref.error+":"+got.error);
   auto byte_got=observe([&](){return qmma_decoded(av,[&](int k){return tab[b[reverse?31-k:k]];},seed);});
   if(!(ref==byte_got))throw std::runtime_error("QMMA_BYTE_LOOKUP_DIFF:"+ref.error+":"+byte_got.error);
   outcomes[ref.error.empty()?"FINITE_OUTPUT":ref.error]++;
  };
  auto hcheck=[&](const uint16_t* a,const uint16_t* b){double av[16],bv[16];for(int k=0;k<16;++k){av[k]=half_value(a[k]);bv[k]=half_value(b[k]);}
   auto ref=observe([&](){return hmma_dot(a,b);});auto got=observe([&](){return hmma_cached(a,b,av,bv);});if(!(ref==got))throw std::runtime_error("HMMA_DIFF:"+ref.error+":"+got.error);outcomes[ref.error.empty()?"FINITE_OUTPUT":ref.error]++;
  };
  {py::gil_scoped_release release;
   if(kind=="table"){
    for(int c=0;c<256;++c){try{int v=e4_units(uint8_t(c));if(table[c].nan||table[c].value!=v)throw std::runtime_error("TABLE_DIFF");}catch(const std::domain_error& e){if(!table[c].nan||std::string(e.what())!="E4M3 NaN code")throw;}}
    for(int c=0;c<256;++c)if(table[c].nan!=(c==127||c==255))throw std::runtime_error("TABLE_NAN_FLAGS_DIFF");
    uint8_t a[32]{},b[32]{};a[0]=b[0]=0x38;auto flipped=table;flipped[0x38].value^=1;bool caught=false;try{qcheck(a,b,0,flipped);}catch(const std::runtime_error&){caught=true;}if(!caught)throw std::runtime_error("TABLE_FLIP_MISSED");
    a[0]=0x7f;auto missing=table;missing[127].nan=false;caught=false;try{qcheck(a,b,0,missing);}catch(const std::runtime_error&){caught=true;}if(!caught)throw std::runtime_error("NAN_FLAG_MISSED");outcomes["table_flip_DIFFERENT"]++;outcomes["nan_flag_DIFFERENT"]++;
   }else for(uint64_t i=start;i<start+count;++i){uint64_t s=i^0x46414c4cULL;
    if(kind=="qm_pairs"){uint8_t a[32]{},b[32]{};int k=int(i/65536);if(k>=32)throw std::invalid_argument("PAIR_RANGE");a[k]=uint8_t(i/256);b[k]=uint8_t(i);qcheck(a,b,0,table);}
    else if(kind=="qm_random"||kind=="qm_reverse"){
     uint8_t a[32],b[32];for(int k=0;k<32;++k){a[k]=uint8_t(rng(s));b[k]=uint8_t(rng(s));if(i%16){if((a[k]&127)==127)a[k]^=1;if((b[k]&127)==127)b[k]^=1;}}
     const uint16_t boundary[]={0,0x8000,1,0x8001,0x7bff,0xfbff,0x7c00,0x7e00};uint16_t seed=i%16==0?boundary[(i/16)%8]:uint16_t(rng(s)&0xfbff);
     qcheck(a,b,seed,table,kind=="qm_reverse");
    }else if(kind=="hm_half"){uint16_t a[16]{},b[16]{};int side=int(i/65536);if(side>1)throw std::invalid_argument("HALF_RANGE");a[0]=side?0x3c00:uint16_t(i);b[0]=side?uint16_t(i):0x3c00;hcheck(a,b);}
    else if(kind=="hm_random"){uint16_t a[16],b[16];for(int k=0;k<16;++k){a[k]=uint16_t(rng(s));b[k]=uint16_t(rng(s));if(i%16){if((a[k]&0x7c00)==0x7c00)a[k]^=0x400;if((b[k]&0x7c00)==0x7c00)b[k]^=0x400;}}hcheck(a,b);}
    else if(kind=="hm_crafted"){
     uint16_t a[16]{},b[16]{};switch(i){
      case 0:a[0]=0x8000;b[0]=0x3c00;break;
      case 1:a[0]=b[0]=1;break;
      case 2:for(int k=0;k<16;++k)a[k]=b[k]=0x7bff;break;
      case 3:a[0]=0x7c00;b[0]=0;break;
      case 4:a[15]=0x7e00;b[15]=0x3c00;break;
      case 5:a[0]=0xfc00;b[0]=0x7e00;a[15]=b[15]=0x7bff;break;
      case 6:a[0]=a[1]=0x7bff;b[0]=0x7bff;b[1]=0xfbff;a[2]=b[2]=1;break;
      default:throw std::invalid_argument("HM_CRAFTED_RANGE");}hcheck(a,b);
    }
    else if(kind=="crafted"){
     uint8_t a[32]{},b[32]{};uint16_t seed=0;
     switch(i){case 0:a[0]=0x7f;b[1]=0xff;seed=0x7c00;break;case 1:a[0]=0x7f;b[1]=0xff;break;
     case 2:a[0]=b[0]=0x7e;break;
     case 3:a[0]=a[1]=0x7e;b[0]=0x7e;b[1]=0xfe;a[2]=b[2]=1;break;
     case 4:a[0]=a[1]=0x7e;b[0]=0x7e;b[1]=0xfe;a[2]=b[2]=1;a[31]=0x7f;break;
     case 5:seed=0x8000;break;
     default:throw std::invalid_argument("CRAFTED_RANGE");}qcheck(a,b,seed,table);
    }else throw std::invalid_argument("UNKNOWN_PRIMITIVE_KIND");
   }
  }
  py::dict d;d["status"]="MATCH";d["kind"]=kind;d["start"]=start;d["count"]=count;d["outcomes"]=outcomes;return d;
 },py::arg("kind"),py::arg("start"),py::arg("count"));

 m.def("check_panels",[](py::bytes raw,int C,bool up,bool down,bool decoded){
  if(!decoded)throw std::invalid_argument("OPTM_DECODED_PANEL_REQUIRED");std::string r=raw;Wt w;decode_weights(w,reinterpret_cast<const uint8_t*>(r.data()),r.size(),C,up,down);prepare_panels(w,decoded);size_t checked=0,storage=sizeof(w.table),transient=0;
  auto check=[&](const Matrix& a,const Panel& p){storage+=p.storage();for(int k=0;k<a.K;++k)for(int j=0;j<a.N;++j){auto want=w.table[a.at(k,j)],got=p.at(k,j);if(want.value!=got.value||want.nan!=got.nan)throw std::runtime_error("PANEL_DIFF");if(!decoded&&p.bytes[size_t(j)*a.K+k]!=a.at(k,j))throw std::runtime_error("PANEL_BYTE_DIFF");++checked;}};
  check(w.ex,w.pex);check(w.branch,w.pbranch);check(w.out,w.pout);check(w.qkv,w.pqkv);check(w.proj,w.pproj);check(w.extra,w.pextra);
  for(int h=0;h<int(w.heads.size());++h){auto& p=w.heads[h];storage+=p.ex.storage()+p.br.storage()+p.out.storage();transient=size_t(C)*128+4096+size_t(32)*C;
   for(int k=0;k<C;++k)for(int j=0;j<128;++j){auto want=w.table[w.ex.at(k,h*128+j)],got=p.ex.at(k,j);if(want.value!=got.value||want.nan!=got.nan)throw std::runtime_error("HEAD_EX_DIFF");++checked;}
   for(int k=0;k<128;++k)for(int j=0;j<32;++j){auto want=w.table[w.ex.at(h*32+k%32,4*C+(k/32)*32+j)],got=p.br.at(k,j);if(want.value!=got.value||want.nan!=got.nan)throw std::runtime_error("HEAD_BR_DIFF");++checked;}
   for(int k=0;k<32;++k)for(int j=0;j<C;++j){auto want=w.table[w.out.at(h*32+k,j)],got=p.out.at(k,j);if(want.value!=got.value||want.nan!=got.nan)throw std::runtime_error("HEAD_OUT_DIFF");++checked;}
  }
  // A one-byte wrong column entry must be detected by precisely the same checker.
  bool caught=false;Panel bad=panel(w.qkv,w.table,false);bad.bytes[0]^=1;try{check(w.qkv,bad);}catch(const std::runtime_error&){caught=true;}
  if(!caught)throw std::runtime_error("PANEL_NEGATIVE_CONTROL_MISSED");
  py::dict d;d["status"]="PANEL_MATCH";d["elements"]=checked;d["mode"]=decoded?"decoded":"byte";
  d["panel_storage_bytes"]=storage-bad.storage();d["construction_scratch_bytes"]=transient;
  d["launch_metadata_bytes"]=6*sizeof(Panel)+sizeof(std::vector<HeadPanels>)+w.heads.capacity()*sizeof(HeadPanels);
  d["row_metadata_bytes"]=sizeof(std::vector<E4Decoded>);d["head_scratch_metadata_bytes"]=3*sizeof(Matrix);
  d["decoded_element_bytes"]=sizeof(E4Decoded);d["negative_control"]="DIFF_AS_REQUIRED";return d;
 });

 m.def("optb_sat_reference",&sat_e4_reference);
 m.def("optb_sat_lookup",&sat_e4);
 m.def("optb_table_bytes",[](){return py::bytes(reinterpret_cast<const char*>(sat_e4_table.data()),sat_e4_table.size());});
 m.def("optb_validate_table",[](py::bytes raw){std::string b=raw;
   if(b.size()!=sat_e4_table.size() || std::memcmp(b.data(),sat_e4_table.data(),b.size()))
     throw std::invalid_argument("OPTB_TABLE_IDENTITY_MISMATCH");
 });

 m.def("configure",[](uint64_t base,uint64_t wb,py::dict rows,py::array_t<uint16_t> rsq,py::array_t<uint16_t> rcp){AbiLease lease;wsbase=base;weightbase=wb;RSQ.assign(rsq.data(),rsq.data()+rsq.size());RCP.assign(rcp.data(),rcp.data()+rcp.size());specs.clear();for(auto kv:rows){int L=py::cast<int>(kv.first);py::dict d=py::cast<py::dict>(kv.second);LaunchSpec s;s.C=py::cast<int>(d["C"]);s.extent=py::cast<int>(d["extent"]);s.shift=py::cast<int>(d["shift"]);s.inputlayout=py::cast<int>(d["inputlayout"]);s.outputlayout=py::cast<int>(d["outputlayout"]);s.pooledword=py::cast<int>(d["pooledword"]);s.skipword=py::cast<int>(d["skipword"]);s.inW=py::cast<int>(d["inW"]);s.inH=py::cast<int>(d["inH"]);s.weightbytes=py::cast<size_t>(d["weightbytes"]);s.up=py::cast<bool>(d["up"]);s.releaseword=py::cast<int>(d["releaseword"]);specs[L]=s;}});
 m.def("run_launch",&run_launch,py::arg("launch"),py::arg("workspace"),py::arg("params"),py::arg("weights"),py::arg("textures"),py::arg("threads")=1);
 m.def("decode_matrix",[](py::bytes raw,int K,int N,bool major){std::string r=raw;if(r.size()!=size_t(K)*N)throw std::invalid_argument("matrix size");auto m=unpack(reinterpret_cast<const uint8_t*>(r.data()),K,N,major);return py::bytes(reinterpret_cast<const char*>(m.v.data()),m.v.size());});
 m.def("inspect_weights",[](py::bytes raw,int C,bool up,bool down){std::string r=raw;Wt w;decode_weights(w,reinterpret_cast<const uint8_t*>(r.data()),r.size(),C,up,down);py::dict d;auto mat=[&](const char* name,const Matrix& a){if(!a.v.empty())d[name]=py::bytes(reinterpret_cast<const char*>(a.v.data()),a.v.size());};auto half=[&](const char* name,const std::vector<uint16_t>& a){d[name]=a;};mat("ex",w.ex);mat("branch",w.branch);mat("out",w.out);mat("qkv",w.qkv);mat("proj",w.proj);mat("extra",w.extra);half("cs",w.cs);half("ca",w.ca);half("sin",w.sin);half("bias",w.bias);half("scale",w.scale);return d;});
 m.def("layout",[](py::bytes raw,int W,int H,int C,int kind,bool encode){std::string r=raw;if(r.size()!=size_t(W)*H*C)throw std::invalid_argument("layout size");std::vector<uint8_t> v(r.begin(),r.end());auto out=encode?encode_layout(v,W,H,C,kind):decode_layout(v.data(),W,H,C,kind);return py::bytes(reinterpret_cast<const char*>(out.data()),out.size());});
}

