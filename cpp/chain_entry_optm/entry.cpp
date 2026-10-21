#include "../geometry.hpp"
#include "path_policy.hpp"
#include "mapexec_base.hpp"
#include <functional>
#include "noise_online_optd.hpp"
namespace {
// C2: py::buffer_info exports pin the original byte owner across the GIL release.
struct BorrowedBytes {
 py::buffer owner; py::buffer_info view;
 explicit BorrowedBytes(py::handle h):owner(py::reinterpret_borrow<py::buffer>(h)),view(owner.request()) {
  if(view.ndim!=1||view.itemsize!=1||view.strides[0]!=1)throw std::invalid_argument("texture requires contiguous byte buffer");
  py::object root=py::reinterpret_borrow<py::object>(h);
  while(PyMemoryView_Check(root.ptr()))root=root.attr("obj");
  if(!view.readonly||!PyBytes_Check(root.ptr()))throw std::invalid_argument("texture borrow requires immutable bytes owner; mutable owners need snapshot");
 }
 size_t size()const{return size_t(view.size);}
 const char* data()const{return static_cast<const char*>(view.ptr);}
};


std::mutex abi_mutex;
struct AbiLease {std::unique_lock<std::mutex> lock;AbiLease():lock(abi_mutex,std::try_to_lock){if(!lock.owns_lock())throw std::runtime_error("CONFIGURE_RUN_BUSY");}};
py::dict* assets=nullptr;
std::unique_ptr<dlssnr_optd::NoiseAssets> noise_owner;
uint64_t wsbase=0;
uint64_t native_weight_base=0xc400000ULL;
uint64_t asset_epoch=0,asset_releases=0;
bool last_probe_joined=false;
void clear_assets(){noise_owner.reset();if(assets){*assets=py::dict();++asset_releases;}}
struct AssetCleanup{~AssetCleanup(){clear_assets();}};
bool reverse_red=false;
uint32_t le32(const uint8_t* p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
uint64_t le64(const uint8_t* p){return uint64_t(le32(p))|(uint64_t(le32(p+4))<<32);}
uint16_t le16(const uint8_t* p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}
void put16(uint8_t* p,uint16_t v){p[0]=uint8_t(v);p[1]=uint8_t(v>>8);}
void par(size_t n,int threads,const std::function<void(size_t)>& fn){
 threads=std::clamp(threads,1,12);std::atomic<size_t> next{0};std::atomic<bool> fail{false};std::exception_ptr ep;std::mutex mu;
 auto f=[&](){try{while(!fail){size_t i=next.fetch_add(1);if(i>=n)break;fn(i);}}catch(...){fail=true;std::lock_guard<std::mutex> l(mu);if(!ep)ep=std::current_exception();}};
 std::vector<std::thread> pool;for(int i=1;i<threads;++i)pool.emplace_back(f);f();for(auto& t:pool)t.join();if(ep)std::rethrow_exception(ep);
}
struct Ws {
 uint8_t* data;size_t n;std::vector<uint64_t> q;py::list written;
 Ws(py::bytearray b,const std::string& p):data(reinterpret_cast<uint8_t*>(PyByteArray_AsString(b.ptr()))),n(size_t(PyByteArray_Size(b.ptr()))){
  if(p.size()%8)throw std::invalid_argument("params qwords");for(size_t i=0;i<p.size();i+=8)q.push_back(le64(reinterpret_cast<const uint8_t*>(p.data()+i)));
 }
 size_t off(int word,size_t len)const{if(size_t(word)>=q.size()||q[word]<wsbase||q[word]-wsbase>n||len>n-(q[word]-wsbase))throw std::out_of_range("workspace span");return size_t(q[word]-wsbase);}
 void store(int word,const std::string& b,size_t delta=0){size_t o=off(word,b.size()+delta)+delta;std::memcpy(data+o,b.data(),b.size());written.append(py::make_tuple(o,b.size()));}
 void fill(int word,size_t len,uint8_t value){size_t o=off(word,len);std::memset(data+o,value,len);written.append(py::make_tuple(o,len));}
};
const uint8_t* weight_span(const Ws& w,int word,size_t size,py::buffer_info& b){
 if(b.ndim!=1||b.itemsize!=1||b.strides[0]!=1)throw std::invalid_argument("weights require flat byte memoryview");
 uint64_t o=w.q.at(word)-native_weight_base;if(w.q[word]<native_weight_base||o>uint64_t(b.size)||size>uint64_t(b.size)-o)throw std::out_of_range("weight pointer");return static_cast<const uint8_t*>(b.ptr)+o;
}
void decoder(Ws& w,const uint8_t* wt,int threads){
 if(w.q.size()!=10)throw std::invalid_argument("L100 captured geometry");
 auto mp=(*assets)["l100_maps"].cast<py::dict>();auto mj=(*assets)["l100_json"].cast<py::dict>();
 py::array_t<int32_t,py::array::c_style> A=mp["A_src"].cast<py::array_t<int32_t,py::array::c_style>>(),B=mp["B_off"].cast<py::array_t<int32_t,py::array::c_style>>(),O=mp["OUTMAP"].cast<py::array_t<int32_t,py::array::c_style>>(),R=mp["REDMAP"].cast<py::array_t<int32_t,py::array::c_style>>(),RO=mp["RED_OFF"].cast<py::array_t<int32_t,py::array::c_style>>();
 if(A.ndim()!=7||geo2::product(geo2::product(size_t(A.shape(0)),size_t(A.shape(1))),geo2::product(size_t(A.shape(2)),size_t(A.shape(3))))!=geo2::product(uint32_t(w.q[8]),uint32_t(w.q[8]>>32))||A.shape(4)!=16||A.shape(5)!=8||A.shape(6)!=32||A.size()!=B.size()||O.ndim()!=2||O.shape(1)!=12||size_t(O.shape(0))!=geo2::product(geo2::product(uint32_t(w.q[9]),uint32_t(w.q[9]>>32)),512)||R.shape(0)!=RO.size()||R.shape(1)!=3||R.shape(2)!=4)throw std::invalid_argument("L100 map geometry");
 const size_t nt=geo2::product(uint32_t(w.q[8]),uint32_t(w.q[8]>>32)),mainbytes=geo2::product(nt,1024),nout=geo2::product(geo2::product(uint32_t(w.q[9]),uint32_t(w.q[9]>>32)),512),nchain=16,ns=8,nred=size_t(RO.size());const auto *a=A.data(),*b=B.data(),*om=O.data(),*rm=R.data(),*ro=RO.data();
 size_t mo=w.off(0,mainbytes),so=w.off(1,nout);std::vector<uint16_t> D(nt*nchain*128);std::vector<uint16_t> red(nred);std::string output(nout,'\0');
 for(py::ssize_t i=0;i<A.size();++i)if(a[i]<0||a[i]+16>mainbytes||b[i]<0||b[i]+8>525312)throw std::out_of_range("L100 operand bounds");
 auto di=[&](int t,int ch,int lane,int slot)->size_t{if(t<0||t>=int(nt)||ch<0||ch>=16||lane<0||lane>=32||slot<0||slot>=4)throw std::out_of_range("L100 D index");return ((size_t(t)*16+ch)*32+lane)*4+slot;};
 int rlo=*std::min_element(ro,ro+nred),rhi=*std::max_element(ro,ro+nred)+2;std::string redbytes(size_t(rhi-rlo),'\0');
 for(size_t i=0;i<nred;++i)if(rlo<0||int64_t(ro[i])!=int64_t(rlo)+2*int64_t(i))throw std::invalid_argument("L100 REDG output ownership not exactly once");
 {py::gil_scoped_release release;
  par(nt*nchain,threads,[&](size_t tile){uint16_t c[128]{};for(size_t s=0;s<ns;++s){size_t base=(tile*ns+s)*32;
   for(int row=0;row<16;++row)for(int col=0;col<8;++col){uint8_t av[32],bv[32];for(int k=0;k<32;++k){int lane=(row%8)*4+(k%16)/4,reg=row/8+2*(k/16);av[k]=w.data[mo+size_t(a[base+lane])+4*reg+k%4];lane=col*4+(k%16)/4;bv[k]=wt[size_t(b[base+lane])+4*(k/16)+k%4];}c[row*8+col]=qmma_dot(av,bv,c[row*8+col]);}}
   for(int row=0;row<16;++row)for(int col=0;col<8;++col){int lane=(row%8)*4+col/2,slot=2*(row/8)+col%2;D[tile*128+lane*4+slot]=c[row*8+col];}
  });
  par(nred,threads,[&](size_t i){uint16_t v[3];for(int z=0;z<3;++z){const int32_t* x=rm+i*12+z*4;v[z]=D[di(x[0],x[1],x[2],x[3])];}red[i]=reverse_red?h_add(h_add(v[2],v[1]),v[0]):h_add(h_add(v[0],v[1]),v[2]);put16(reinterpret_cast<uint8_t*>(redbytes.data())+ro[i]-rlo,red[i]);});
  par(size_t(O.shape(0)),threads,[&](size_t i){const int32_t* x=om+i*12;auto it=std::lower_bound(ro,ro+nred,x[0]);if(it==ro+nred||*it!=x[0]||x[5]<0||x[5]>=nout||x[6]<0||x[6]+2>525312)throw std::out_of_range("L100 output map");auto d3=D[di(x[1],x[2],x[3],x[4])];auto merged=h_fma(e4_half(w.data[so+x[5]]),le16(wt+x[6]),h_add(red[size_t(it-ro)],d3));output[i]=static_cast<char>(sat_e4(merged));});
 }
 w.store(2,output);w.store(6,redbytes,size_t(rlo));
 auto cw=mj["coverage"].cast<py::dict>()["counter_writes"].cast<py::dict>();
 for(auto it:cw){std::string k=py::str(it.first);auto pos=k.find('+');if(pos==std::string::npos)throw std::invalid_argument("counter key");std::string role=k.substr(0,pos);if(role!="ctr"&&role!="ctr2")throw std::invalid_argument("counter role");size_t o=std::stoull(k.substr(pos+1));int word=role=="ctr"?4:5;auto values=it.second.cast<std::vector<int>>();std::sort(values.begin(),values.end());if(values!=(word==4?std::vector<int>{0,1,2,3}:std::vector<int>{0}))throw std::invalid_argument("counter release sequence");std::string val(4,'\0');if(word==4)val[0]=3;w.store(word,val,o);}
}
void b00(Ws& w,const uint8_t* wt,py::dict textures,int threads);
py::dict run_launch(int launch,py::bytearray workspace,py::bytes params,py::buffer weights,py::dict textures,int threads){
 AbiLease lease;AssetCleanup cleanup;py::buffer workspace_owner=py::reinterpret_borrow<py::buffer>(workspace);py::buffer_info workspace_pin=workspace_owner.request();
 std::string p=params;geo2::parameters(launch,reinterpret_cast<const uint8_t*>(p.data()),p.size());Ws w(workspace,p);if(wsbase==0)throw std::runtime_error("configure required");py::dict tx;
 if(launch==1){if(w.q.size()!=2)throw std::invalid_argument("L1 params");uint32_t count=uint32_t(w.q[1]);w.fill(0,size_t(count)*4,255);}
 else if(launch==0){
  // The captured TSV maps merged input handle to uploaded colour, output UAV to id10.
  // Texture coordinates land on texel centres for the captured identity transform.
  const auto& geom=geo2::g();
  if(w.q.size()!=9||!geom.l0_identity||uint32_t(w.q[8])!=uint32_t(geom.cw)||uint32_t(w.q[8]>>32)!=uint32_t(geom.ch))throw std::invalid_argument("L0 captured frame0 identity contract");
  BorrowedBytes raw(textures[py::int_(1)]);std::string b(raw.data(),raw.size());
  if(b.size()!=geo2::product(geo2::product(size_t(geom.cw),size_t(geom.ch)),8))throw std::invalid_argument("L0 upload RGBA16F bytes");
  for(size_t i=0;i<b.size();i+=2)if((le16(reinterpret_cast<const uint8_t*>(b.data()+i))&0x7c00)==0x7c00)throw std::domain_error("L0 nonfinite conversion not proven");
  tx[py::int_(10)]=py::bytes(b);textures[py::int_(10)]=py::bytes(b);
 }
 else {auto wb=weights.request();
  if(launch==132){if(w.q.size()!=9||w.q[5]||w.q[6]||w.q[8]!=0xfffffffc00000000ULL)throw std::invalid_argument("L132 captured geometry and signed origin");const uint8_t* wt=weight_span(w,3,263168,wb);auto cfg=(*assets)["l132"].cast<py::dict>();size_t n=geo2::product(geo2::product(uint32_t(w.q[4]),uint32_t(w.q[4]>>32)),512);size_t a=w.off(0,n),b=w.off(1,n);std::string input(reinterpret_cast<char*>(w.data+a),n);input.append(reinterpret_cast<char*>(w.data+b),n);
   std::string raw(reinterpret_cast<const char*>(wt),263168);auto result=execute(cfg["wiring"].cast<py::dict>(),cfg["maps"].cast<py::dict>(),py::bytes(input),py::bytes(raw),py::dict(),(*assets)["rsq"].cast<py::array_t<float>>(),(*assets)["rcp"].cast<py::array_t<float>>(),n,threads,0);w.store(2,result.cast<std::string>());
  }else if(launch==100)decoder(w,weight_span(w,7,525312,wb),threads);
  else if(launch==2)b00(w,weight_span(w,28,21696,wb),textures,threads);
  else throw std::invalid_argument("launch outside entry CARD");
 }
 py::dict out;out["launch"]=launch;out["written_ranges"]=w.written;out["textures_out"]=tx;return out;
}
}
#include "native_temporal.hpp"
#include "b00.hpp"
PYBIND11_MODULE(dlssnr_chain_entry_optm,m){
m.def("configure_geometry",&geo2::configure);m.def("geometry_probe",&geo2::probe);
#include "noise_exports_optd.inc"
 m.def("native_history",[](py::bytes params,py::bytes history,py::bytes motion,int nw,int nh,int threads){
 std::string p=params,hist=history,mv=motion;py::bytearray dummy("x",1);Ws ws(dummy,p);
 if(p.size()!=264||nw!=1920||nh!=1152||hist.size()!=1920*1080*8||mv.size()!=1920*1080*4)throw std::invalid_argument("native_history 1080 scope");
 std::string out(size_t(nw)*nh*6,'\0');{py::gil_scoped_release release;par(size_t(nh),threads,[&](size_t row){int y=int(row);y=y<1080?y:2158-y;for(int x=0;x<nw;++x){uint16_t v[3];native_history_pixel(reinterpret_cast<const uint8_t*>(hist.data()),reinterpret_cast<const uint8_t*>(mv.data()),ws,x,y,1920,1080,v);for(int c=0;c<3;++c)put16(reinterpret_cast<uint8_t*>(out.data())+(row*nw+x)*6+c*2,v[c]);}});}return py::bytes(out);
 },py::arg("params"),py::arg("history"),py::arg("motion"),py::arg("net_w")=1920,py::arg("net_h")=1152,py::arg("threads")=12);
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

 m.def("optb_sat_reference",&sat_e4_reference);
 m.def("optb_sat_lookup",&sat_e4);
 m.def("optb_table_bytes",[](){return py::bytes(reinterpret_cast<const char*>(sat_e4_table.data()),sat_e4_table.size());});
 m.def("optb_validate_table",[](py::bytes raw){std::string b=raw;
   if(b.size()!=sat_e4_table.size() || std::memcmp(b.data(),sat_e4_table.data(),b.size()))
     throw std::invalid_argument("OPTB_TABLE_IDENTITY_MISMATCH");
 });

 m.def("release_assets",[](){AbiLease lease;clear_assets();});
 m.def("asset_state",[](){py::dict d;d["epoch"]=asset_epoch;d["releases"]=asset_releases;d["live_keys"]=assets?py::len(*assets):0;d["last_probe_joined"]=last_probe_joined;return d;});
 m.def("abi_borrow_probe",[](py::object h){BorrowedBytes b(h);return py::bytes(b.data(),b.size());});
 m.def("abi_lifetime_probe",[](py::object hook){
  AbiLease lease;AssetCleanup cleanup;last_probe_joined=false;hook();
  std::atomic<int> active{0};
  try {py::gil_scoped_release release;par(2,2,[&](size_t){
   struct Active {std::atomic<int>& a;Active(std::atomic<int>& x):a(x){++a;}~Active(){--a;}} worker(active);
   throw std::runtime_error("ABI_PROBE_WORKER_FAILURE");
  });}catch(const std::runtime_error& e){
   last_probe_joined=(active.load()==0);
   if(!last_probe_joined)throw std::runtime_error("ABI_PROBE_ACTIVE_WORKER_AFTER_JOIN");
   throw;
  }
 });
 m.def("configure",[](uint64_t base,py::dict a){AbiLease lease;if(base!=geo2::g().workspace_base)throw std::invalid_argument("GEO2_ENTRY_BASE_DIFF");wsbase=base;native_weight_base=a.contains("native_device_weight_base")?a["native_device_weight_base"].cast<uint64_t>():0xc400000ULL;if(native_weight_base!=geo2::g().weight_base)throw std::invalid_argument("device weight base outside frozen allocations");++asset_epoch;if(assets) *assets=a;else assets=new py::dict(a);});
 m.def("set_reverse_red_control",[](bool x){AbiLease lease;reverse_red=x;});
 m.def("run_launch",&run_launch,py::arg("launch"),py::arg("workspace"),py::arg("params"),py::arg("weights"),py::arg("textures"),py::arg("threads")=1);
 m.def("debug_b00_window",[](py::bytes params,py::buffer weights,py::bytes colour,int wx,int wy){
  AbiLease lease;AssetCleanup cleanup;std::string p=params,upload=colour;py::bytearray dummy("x",1);Ws w(dummy,p);auto wb=weights.request();auto wt=weight_span(w,28,21696,wb);B0Weight W(wt);
  if(!noise_owner)throw std::runtime_error("D noise assets require L2 preparation");auto owner=std::move(noise_owner);const auto& noise=*owner;
  auto rsq=(*assets)["rsq_half"].cast<py::array_t<uint16_t>>(),rcp=(*assets)["rcp_half"].cast<py::array_t<uint16_t>>();uint16_t y[2048];
  if(upload.size()!=geo2::product(geo2::product(size_t(geo2::g().cw),size_t(geo2::g().ch)),8)||wx<0||wx>=geo2::g().nw/8||wy<0||wy>=geo2::g().nh/8)throw std::invalid_argument("debug B00 window scope");
  {py::gil_scoped_release release;b0window(wx,wy,geo2::g().cw,geo2::g().ch,reinterpret_cast<const uint8_t*>(upload.data()),w,W,noise,rsq.data(),rcp.data(),y);}
  std::string out(4096,'\0');for(int i=0;i<2048;++i)put16(reinterpret_cast<uint8_t*>(out.data())+2*i,y[i]);return py::bytes(out);
 });
}

