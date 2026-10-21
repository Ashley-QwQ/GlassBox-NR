// Formula transcription: d20_backend_wh.py:121-158; p1_front_0_60.py:177-224;
// sm120_b1.py:342-421, noise_reference.py:33-200; unpack_dlssnr_weights.py:51-144.
namespace {
uint32_t pcg(uint32_t v){return ((v>>((v>>28)+4))^v)*0x108ef2d9U;}
float mul32(float a,float b){volatile float x=a*b;return x;}
struct B0Weight {
 std::vector<uint8_t> w1,w2,qkv,proj;uint16_t adapter[512],ffskip[32],attskip[32],bias[4096],scale;
 double adapter_value[512];uint16_t adapter_column[512];
 static std::vector<uint8_t> matrix(const uint8_t* raw,int K,int N){
  static const int order[16]={0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15};std::vector<uint8_t> fragment(size_t(K)*N),out(size_t(K)*N);
  for(int kt=0;kt<K/32;++kt)for(int nt=0;nt<N/32;++nt)for(int i=0;i<1024;++i){int e=i%8,f=(i/8)%2,l=(i/16)%32,g=i/512;int k=(l%4)*4+(e&3)+(e>=4?16:0),n=(g*2+f)*8+l/4;fragment[(kt*32+k)*N+nt*32+n]=raw[(kt*(N/32)+nt)*1024+i];}
  for(int k=0;k<K;++k)for(int n=0;n<N;++n)out[k*N+n]=fragment[(k-k%16+order[k%16])*N+n];return out;
 }
 B0Weight(const uint8_t* w):w1(matrix(w,32,128)),w2(matrix(w+4096,128,32)),qkv(matrix(w+9312,32,96)),proj(matrix(w+20592,32,32)){
  for(int nt=0;nt<2;++nt)for(int i=0;i<256;++i){int e=i%8,l=i/8,fe=e%4,k=(l%4)*2+(fe&1)+(fe>=2?8:0),n=l/4+(e>=4?8:0);adapter[k*32+nt*16+n]=le16(w+8208+2*(nt*256+i));}
  for(int c=0;c<32;++c)for(int j=0;j<16;++j){adapter_column[c*16+j]=adapter[j*32+c];adapter_value[c*16+j]=half_value(adapter[j*32+c]);}
  for(int c=0;c<32;++c){ffskip[c]=le16(w+9232+2*c);attskip[c]=le16(w+21616+2*c);}
  scale=half_round(double(optm_bit_cast<float>(le32(w+20576))));
  auto bit=[](int x,int n){return (x>>n)&1;};
  for(int q=0;q<64;++q)for(int k=0;k<64;++k){int qy=q/8,qx=q%8,ky=k/8,kx=k%8;int ix=(bit(qy,2)<<11)|(bit(qx,2)<<10)|(bit(ky,2)<<9)|(bit(kx,2)<<8)|(bit(qy,0)<<7)|(bit(qx,1)<<6)|(bit(qx,0)<<5)|(bit(ky,0)<<4)|(bit(kx,1)<<3)|(bit(ky,1)<<2)|(bit(qy,1)<<1)|bit(kx,0);bias[q*64+k]=le16(w+12384+2*ix);}
 }
};
uint16_t b0dot(const uint8_t* x,const std::vector<uint8_t>& mat,int N,int col,int step,uint16_t seed){uint8_t b[32];for(int k=0;k<32;++k)b[k]=mat[(step*32+k)*N+col];return qmma_dot(x,b,seed);}
uint8_t gate(uint16_t v){uint16_t cl=h_max(h_min(v,0x4400),0xc400);auto linear=h_fma(cl&0x7fff,half_round(-0.055908203125),half_round(0.447265625));auto factor=h_fma(cl,linear,half_round(0.89453125));return sat_e4(h_mul(v,factor));}
uint16_t norm32(const uint16_t* x){uint16_t t[8],u[8],v[8];for(int i=0;i<8;++i){auto a=h_fma(x[i],x[i],h_mul(x[16+i],x[16+i]));auto b=h_fma(x[8+i],x[8+i],h_mul(x[24+i],x[24+i]));t[i]=h_add(a,b);}for(int g=0;g<4;++g)for(int j=0;j<2;++j)u[g*2+j]=h_add(t[g*2+j],t[(g^2)*2+j]);for(int g=0;g<4;++g)for(int j=0;j<2;++j)v[g*2+j]=h_add(u[g*2+j],u[(g^1)*2+j]);return h_max(h_add(v[0],v[1]),0x410);}
uint16_t denom64(const uint16_t* w){auto leg=[&](int b){auto x=h_add(h_add(w[b],w[b+16]),h_add(w[b+4],w[b+20]));x=h_add(x,h_add(w[b+32],w[b+48]));return h_add(x,h_add(w[b+36],w[b+52]));};return h_add(h_add(h_add(h_add(leg(0),leg(2)),leg(8)),leg(10)),h_add(h_add(h_add(leg(1),leg(3)),leg(9)),leg(11)));}
void b0window(int wx,int wy,int cw,int ch,const uint8_t* upload,const Ws& ws,const B0Weight& W,const dlssnr_optd::NoiseAssets& noise,const uint16_t* rsq,const uint16_t* rcp,uint16_t* y,const uint8_t* history=nullptr,const uint8_t* mv=nullptr){
 uint16_t z[2048];uint8_t q[2048],k[2048],v[2048],pw[4096];
 uint32_t frame=uint32_t(ws.q[25]);uint16_t constants[6];auto paramf=[&](int word,int hi){return optm_bit_cast<float>(uint32_t(ws.q[word]>>(hi*32)));};
 bool p5=std::min(paramf(23,0),paramf(23,1))>=0;constants[0]=half_round(paramf(22,1));constants[1]=half_round(paramf(21,1));constants[2]=half_round(p5?1.0:paramf(22,0));constants[3]=half_round(p5?paramf(23,0):-1.0);constants[4]=half_round(p5?paramf(23,1):-1.0);constants[5]=0;
 // In this frame's reference p5 is false; refuse the unresolved branch for future inputs.
 if(p5)throw std::domain_error("B00 p5 branch outside measured domain");
 bool temporal=uint32_t(ws.q[24])!=0&&ws.q[1]!=0&&ws.q[2]!=0;
 if(temporal&&(!history||!mv))throw std::invalid_argument("B00 temporal inputs missing");
 uint16_t cs=half_round(double(paramf(24,1)+paramf(24,1)));
 const uint32_t mul[4]={0xcaa5b80dU,0x2c9277b5U,0x83232c31U,0xfa6dc5f9U},add[4]={0x21dd796bU,0xac564b05U,0x3463e0acU,0x4712a88eU};
 for(int t=0;t<64;++t){int x=wx*8+t%8,yy=wy*8+t/8;int tx=x<cw?x:2*cw-2-x,ty=yy<ch?yy:2*ch-2-yy;if(tx<0||ty<0||tx>=cw||ty>=ch)throw std::out_of_range("B00 mirror");
  const auto nt=noise.trace_xy(uint32_t(x),uint32_t(yy),frame);
  uint16_t feat[16];for(int j=0;j<3;++j)feat[j]=half_round(optm_bit_cast<float>(nt.products[j]));feat[3]=0x3c00;
  for(int c=0;c<3;++c)feat[4+c]=feat[7+c]=h_mul(h_add(le16(upload+(size_t(ty)*cw+tx)*8+c*2),0xb800),cs);if(temporal)native_history_pixel(history,mv,ws,tx,ty,cw,ch,feat+7);for(int c=0;c<6;++c)feat[10+c]=constants[c];
  double feat_value[16];for(int j=0;j<16;++j)feat_value[j]=half_value(feat[j]);
  uint16_t adapter[32];uint8_t xc[32];for(int c=0;c<32;++c){adapter[c]=hmma_cached(feat,W.adapter_column+c*16,feat_value,W.adapter_value+c*16);xc[c]=sat_e4(adapter[c]);}
  uint8_t hidden[128];for(int c=0;c<128;++c)hidden[c]=gate(b0dot(xc,W.w1,128,c,0,0));
  for(int c=0;c<32;++c){auto acc=h_mul(adapter[c],W.ffskip[c]);for(int s=0;s<4;++s)acc=b0dot(hidden+s*32,W.w2,32,c,s,acc);z[t*32+c]=acc;xc[c]=sat_e4(acc);}
  uint16_t qh[32],kh[32];for(int c=0;c<32;++c){qh[c]=b0dot(xc,W.qkv,96,c,0,0);kh[c]=b0dot(xc,W.qkv,96,c+32,0,0);v[t*32+c]=sat_e4(b0dot(xc,W.qkv,96,c+64,0,0));}
  auto nq=norm32(qh),nk=norm32(kh);if(nq<0x410||nq>0x7c00||nk<0x410||nk>0x7c00)throw std::domain_error("B00 rsq domain");for(int c=0;c<32;++c){q[t*32+c]=sat_e4(h_mul(h_mul(qh[c],rsq[nq-0x410]),W.scale));k[t*32+c]=sat_e4(h_mul(kh[c],rsq[nk-0x410]));}
 }
 for(int row=0;row<64;++row){uint16_t un[64];for(int key=0;key<64;++key){auto score=qmma_dot(q+row*32,k+key*32,W.bias[row*64+key]);volatile float prod=mul32(float(half_value(score)),0.044921875f);volatile float affine32=prod+1.30078125f;auto affine=half_round(double(affine32));auto cl=h_min(h_max(affine,half_round(1.03125)),half_round(1.5693359375));un[key]=uint16_t((uint32_t(cl)<<5)+0x8000);}
  auto den=denom64(un);if(den<0x1c00||den>0x60e0)throw std::domain_error("B00 reciprocal domain");for(int key=0;key<64;++key)pw[row*64+key]=sat_e4(h_mul(un[key],rcp[den-0x410]));
 }
 for(int row=0;row<64;++row){uint8_t av[32];for(int c=0;c<32;++c){uint16_t acc=0;for(int s=0;s<2;++s){uint8_t vv[32];for(int j=0;j<32;++j)vv[j]=v[(s*32+j)*32+c];acc=qmma_dot(pw+row*64+s*32,vv,acc);}av[c]=sat_e4(acc);}for(int c=0;c<32;++c)y[row*32+c]=b0dot(av,W.proj,32,c,0,h_mul(z[row*32+c],W.attskip[c]));}
}
void b00(Ws& ws,const uint8_t* wt,py::dict textures,int threads){
 if(ws.q.size()!=33)throw std::invalid_argument("B00 params");
 const auto& geom=geo2::g();uint32_t frame=uint32_t(ws.q[25]);
 if(frame>0&&(uint32_t(ws.q[24])!=1||ws.q[1]!=geom.textures.at(6)[3]||ws.q[2]!=geom.textures.at(3)[3]))throw std::invalid_argument("B00 temporal frame handles contract");int cw=int(uint32_t(ws.q[26]>>32)),ch=int(uint32_t(ws.q[26])),nw=int(uint32_t(ws.q[30]>>32)),nh=int(uint32_t(ws.q[30]));if(cw!=geo2::g().cw||ch!=geo2::g().ch||nw!=geo2::g().nw||nh!=geo2::g().nh)throw std::invalid_argument("B00 1080 frame0 scope");
 BorrowedBytes upload(textures[py::int_(1)]);if(upload.size()!=size_t(cw)*ch*8)throw std::invalid_argument("B00 upload length");
 bool temporal=uint32_t(ws.q[24])!=0&&ws.q[1]!=0&&ws.q[2]!=0;
 std::unique_ptr<BorrowedBytes> history,mv;
 if(temporal){history=std::make_unique<BorrowedBytes>(textures[py::int_(6)]);mv=std::make_unique<BorrowedBytes>(textures[py::int_(3)]);if(history->size()!=size_t(cw)*ch*8||mv->size()!=size_t(cw)*ch*4)throw std::invalid_argument("B00 history/MV length");}
 if(!noise_owner)throw std::runtime_error("D noise assets require L2 preparation");auto owner=std::move(noise_owner);const auto& noise=*owner;
 auto rsq=(*assets)["rsq_half"].cast<py::array_t<uint16_t,py::array::c_style>>(),rcp=(*assets)["rcp_half"].cast<py::array_t<uint16_t,py::array::c_style>>();if(rsq.size()!=30705||rcp.size()!=30704)throw std::invalid_argument("B00 half table size");B0Weight W(wt);
 std::string full(size_t(nw)*nh*32,'\0'),pool(size_t(nw/2)*(nh/2)*32,'\0');
 {py::gil_scoped_release release;par(size_t(nw/8)*(nh/8),threads,[&](size_t win){int wx=int(win%size_t(nw/8)),wy=int(win/size_t(nw/8));uint16_t y[2048];b0window(wx,wy,cw,ch,reinterpret_cast<const uint8_t*>(upload.data()),ws,W,noise,rsq.data(),rcp.data(),y,temporal?reinterpret_cast<const uint8_t*>(history->data()):nullptr,temporal?reinterpret_cast<const uint8_t*>(mv->data()):nullptr);
  for(int ly=0;ly<8;++ly)for(int lx=0;lx<8;++lx)for(int c=0;c<32;++c){int lane=(lx%4)*4+(ly%2)*16+(c%8)/2;int b=((ly/2)%2)*4+(c%2)+((c/8)%2)*2+(c/16)*8;size_t tile=size_t((wy*8+ly)/4)*(nw/4)+(wx*8+lx)/4;full[tile*512+lane*16+b]=static_cast<char>(sat_e4(y[(ly*8+lx)*32+c]));}
  for(int ly=0;ly<4;++ly)for(int lx=0;lx<4;++lx)for(int physical=0;physical<32;++physical){int c=(physical/16)*16+((physical%16)/4)*2+(physical%2)+((physical%4)/2)*8;auto even=h_add(y[((ly*2)*8+lx*2)*32+c],y[((ly*2)*8+lx*2+1)*32+c]);auto odd=h_add(y[((ly*2+1)*8+lx*2)*32+c],y[((ly*2+1)*8+lx*2+1)*32+c]);auto v=h_mul(h_add(even,odd),0x3400);size_t o=size_t(physical/16)*(nw/2)*(nh/2)*16+(size_t(wy*4+ly)*(nw/2)+wx*4+lx)*16+physical%16;pool[o]=static_cast<char>(sat_e4(v));}
 });}
 ws.store(27,full);ws.store(31,pool);
}
}
