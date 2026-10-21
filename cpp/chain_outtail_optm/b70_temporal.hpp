// Temporal-tail numeric path: independent rewrite, checked byte-exact against GPU measurements.
// No use of B00 normalized feature transform; return weighted raw history.
struct B70ReadTexture {
 py::dict desc;py::object owner;const uint8_t* p;size_t n;int w,h,pitch,channels;uint64_t tex;
 B70ReadTexture(py::dict all,int id,int format,int nc):desc(py::cast<py::dict>(all[py::int_(id)])),owner(desc["data"]),channels(nc){
  require(PyByteArray_Check(owner.ptr()),"temporal read data must be pinned bytearray");p=reinterpret_cast<const uint8_t*>(PyByteArray_AsString(owner.ptr()));n=size_t(PyByteArray_Size(owner.ptr()));
  w=py::cast<int>(desc["width"]);h=py::cast<int>(desc["height"]);pitch=py::cast<int>(desc["pitch"]);tex=py::cast<uint64_t>(desc["texture_handle"]);geo2::texture(id,w,h,pitch,tex,py::cast<uint64_t>(desc["surface_handle"]));
  require(py::cast<int>(desc["format"])==format&&w==geo2::g().cw&&h==geo2::g().ch&&pitch==w*nc*2&&n==size_t(h)*pitch,"temporal texture captured 1080 domain");
 }
 float sample(float nx,float ny,int c)const{
  // Native TEX pixel-center/q8 joint weights, half-input alignment and half RTA.
  float x=std::clamp(fm(nx,float(w)),.5f,float(w)-.5f),y=std::clamp(fm(ny,float(h)),.5f,float(h)-.5f);
  int xq=int(std::floor(double(x)*256+.5))-128,yq=int(std::floor(double(y)*256+.5))-128;
  int ix=xq/256,iy=yq/256,jx=std::min(ix+1,w-1),jy=std::min(iy+1,h-1),a=xq%256,b=yq%256,br=(a*b+128)/256;
  int wt[4]={256-a-b+br,a-br,b-br,br};size_t off[4]={size_t(iy)*pitch+ix*channels*2,size_t(iy)*pitch+jx*channels*2,size_t(jy)*pitch+ix*channels*2,size_t(jy)*pitch+jx*channels*2};
  double col[4],largest=0;int active=0,single=-1;
  for(int k=0;k<4;++k){auto bits=load<uint16_t>(p,off[k]+c*2);require((bits&0x7c00)!=0x7c00,"temporal TEX nonfinite outside measured domain");col[k]=half_value(bits);if(wt[k]){largest=std::max(largest,std::abs(col[k]));++active;single=k;}}
  int exponent=0;std::frexp(largest,&exponent);double step=std::ldexp(1.,exponent-15),total=0;
  for(int k=0;k<4;++k)total+=std::trunc(col[k]/step)*step*(double(wt[k])/256.);
  uint16_t bits=half_round(total);double near=half_value(bits);
  if(near!=total){uint16_t other=bits;if(total>near)other=(bits&0x8000)?uint16_t(bits-1):uint16_t(bits+1);else other=(bits&0x8000)?uint16_t(bits+1):uint16_t(bits-1);double adj=half_value(other);if(total==(near+adj)/2&&std::abs(adj)>std::abs(near))bits=other;}
  if(active==1&&total==0&&std::signbit(col[single]))bits=0x8000;
  return float(half_value(bits));
 }
};
void b70_temporal_rgb(const uint8_t* p,int x,int y,const uint16_t* head,const float* encoded,float confidence,const B70ReadTexture& history,const B70ReadTexture& motion,float* out){
 const float w=float(geo2::g().cw),h=float(geo2::g().ch),rw=rcp(w),rh=rcp(h);auto pf=[&](size_t o){return load<float>(p,o);};
 float ux=fm(fa(float(x),.5f),rw),uy=fm(fa(float(y),.5f),rh);
 float mxcoord=fm(ff(pf(148),ux,pf(140)),pf(156)),mycoord=fm(ff(pf(152),uy,pf(144)),pf(160));
 float mx=motion.sample(mxcoord,mycoord,0),my=motion.sample(mxcoord,mycoord,1);
 float ax=load<uint32_t>(p,112)==0?ux:ff(pf(164),mx,ux),ay=load<uint32_t>(p,112)==0?uy:ff(pf(168),my,uy);
 // Sample position: fused multiply-add width*ax-.5, floor, then fused multiply-add width*ax-x0.
 float x0=fa(std::floor(ff(w,ax,-.5f)),.5f),y0=fa(std::floor(ff(h,ay,-.5f)),.5f);
 float t=sat(ff(w,ax,-x0)),v=sat(ff(h,ay,-y0)),t2=fm(t,t),v2=fm(v,v),t3=fm(t,t2),v3=fm(v,v2);
 float left=ff(fa(t,t3),-.5f,t2),top=ff(fa(v,v3),-.5f,v2);
 float m0x=fa(ff(t3,1.5f,-fm(t2,2.5f)),1.f),m0y=fa(ff(v3,1.5f,-fm(v2,2.5f)),1.f);
 float right=fm(fa(t3,-t2),.5f),bottom=fm(fa(v3,-v2),.5f);
 float m1x=fa(fa(fa(1.f,-left),-m0x),-right),m1y=fa(fa(fa(1.f,-top),-m0y),-bottom);
 float cx=fa(m0x,m1x),cy=fa(m0y,m1y);
 float xc=std::clamp(ff(m1x,rcp(cx),x0),.5f,w-.5f),yc=std::clamp(ff(m1y,rcp(cy),y0),.5f,h-.5f);
 float xl=std::clamp(fa(x0,-1.f),.5f,w-.5f),xr=std::clamp(fa(x0,2.f),.5f,w-.5f),yt=std::clamp(fa(y0,-1.f),.5f,h-.5f),yb=std::clamp(fa(y0,2.f),.5f,h-.5f);
 auto sample=[&](float sx,float sy,int c){float hx=fm(ff(pf(124),fm(rw,sx),pf(116)),pf(132)),hy=fm(ff(pf(128),fm(rh,sy),pf(120)),pf(136));return history.sample(hx,hy,c);};
 float wl=fm(left,cy),wt=fm(top,cx),wc=fm(cx,cy),wb=fm(bottom,cx),wr=fm(right,cy);
 float den=fa(wr,fa(wb,fa(wc,fa(wl,wt)))),inv=rcp(den);
 float argument=fm(float(half_value(head[3])),fb(0xbfb8aa3b));
 float sigmoid=rcp(fa(ex(argument),1.f));float factor=sat(ff(sigmoid,confidence,0.f));
 for(int c=0;c<3;++c){float weighted=fm(sample(xc,yt,c),wt);weighted=ff(sample(xl,yc,c),wl,weighted);weighted=ff(sample(xc,yc,c),wc,weighted);weighted=ff(sample(xc,yb,c),wb,weighted);weighted=ff(sample(xr,yc,c),wr,weighted);
  // Normalization and current subtraction share one fused multiply-add.
  float delta=ff(inv,weighted,-encoded[c]);out[c]=ff(factor,delta,encoded[c]);}
}
