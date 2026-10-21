// Isolated transcription of fix_prototype/native_temporal.py native_fma.
// Explicit fmaf only where the native fused operator was established. Other
// operations compile under -ffp-contract=off; top,left,center,bottom,right stay.
namespace {
float tn_add(float a,float b){volatile float z=a+b;return z;}
float tn_mul(float a,float b){volatile float z=a*b;return z;}
float tn_div(float a,float b){volatile float z=a/b;return z;}
float tn_fma(float a,float b,float c){return std::fma(a,b,c);}
float tn_clip(float x,float lo,float hi){return std::clamp(x,lo,hi);}
float tn_sample(const uint8_t* tex,int w,int h,float x,float y,int c){
 x=tn_clip(x,.5f,float(w)-.5f);y=tn_clip(y,.5f,float(h)-.5f);
 int xq=int(std::floor(double(x)*256+.5))-128,yq=int(std::floor(double(y)*256+.5))-128;
 int ix=xq/256,iy=yq/256,jx=std::min(ix+1,w-1),jy=std::min(iy+1,h-1),a=xq%256,b=yq%256;
 int br=(a*b+128)/256,wt[4]={256-a-b+br,a-br,b-br,br};
 size_t off[4]={(size_t(iy)*w+ix)*8,(size_t(iy)*w+jx)*8,(size_t(jy)*w+ix)*8,(size_t(jy)*w+jx)*8};
 double col[4],largest=0;int active=0,single=-1;
 for(int k=0;k<4;++k){col[k]=half_value(le16(tex+off[k]+c*2));if(wt[k]){largest=std::max(largest,std::abs(col[k]));++active;single=k;}}
 int exponent=0;std::frexp(largest,&exponent);double step=std::ldexp(1.,exponent-15),total=0;
 for(int k=0;k<4;++k)total+=std::trunc(col[k]/step)*step*(double(wt[k])/256.);
 uint16_t bits=half_round(total);double near=half_value(bits);
 // Finite binary16 nearest with ties away from zero (measured TEX half RTA).
 if(near!=total){uint16_t other=bits;if(total>near)other=(bits&0x8000)?uint16_t(bits-1):uint16_t(bits+1);else other=(bits&0x8000)?uint16_t(bits+1):uint16_t(bits-1);
  double adj=half_value(other);if(total==(near+adj)/2&&std::abs(adj)>std::abs(near))bits=other;}
 if(active==1&&total==0&&std::signbit(col[single]))bits=0x8000;
 return float(half_value(bits));
}
void native_history_pixel(const uint8_t* hist,const uint8_t* mv,const Ws& ws,int x,int y,int w,int h,uint16_t* out){
 auto p=[&](int i){return optm_bit_cast<float>(uint32_t(ws.q[i/2]>>((i%2)*32)));};
 float rw=tn_div(1.f,float(w)),rh=tn_div(1.f,float(h));size_t o=(size_t(y)*w+x)*4;
 float ax=tn_fma(float(half_value(le16(mv+o))),p(40),tn_mul(float(x)+.5f,rw));
 float ay=tn_fma(float(half_value(le16(mv+o+2))),p(41),tn_mul(float(y)+.5f,rh));
 float x0=tn_add(std::floor(tn_fma(float(w),ax,-.5f)),.5f),y0=tn_add(std::floor(tn_fma(float(h),ay,-.5f)),.5f);
 float t=tn_clip(tn_fma(float(w),ax,-x0),0,1),v=tn_clip(tn_fma(float(h),ay,-y0),0,1);
 float t2=tn_mul(t,t),v2=tn_mul(v,v),t3=tn_mul(t,t2),v3=tn_mul(v,v2);
 float left=tn_fma(tn_add(t,t3),-.5f,t2),top=tn_fma(tn_add(v,v3),-.5f,v2);
 float m0x=tn_add(tn_fma(t3,1.5f,-tn_mul(t2,2.5f)),1.f),m0y=tn_add(tn_fma(v3,1.5f,-tn_mul(v2,2.5f)),1.f);
 float right=tn_mul(tn_add(t3,-t2),.5f),bottom=tn_mul(tn_add(v3,-v2),.5f);
 float m1x=tn_add(tn_add(tn_add(1.f,-left),-m0x),-right),m1y=tn_add(tn_add(tn_add(1.f,-top),-m0y),-bottom);
 float cx=tn_add(m0x,m1x),cy=tn_add(m0y,m1y);
 float xc=tn_clip(tn_fma(m1x,tn_div(1.f,cx),x0),.5f,float(w)-.5f),yc=tn_clip(tn_fma(m1y,tn_div(1.f,cy),y0),.5f,float(h)-.5f);
 float xl=tn_clip(tn_add(x0,-1.f),.5f,float(w)-.5f),xr=tn_clip(tn_add(x0,2.f),.5f,float(w)-.5f);
 float yt=tn_clip(tn_add(y0,-1.f),.5f,float(h)-.5f),yb=tn_clip(tn_add(y0,2.f),.5f,float(h)-.5f);
 float wl=tn_mul(left,cy),wt=tn_mul(top,cx),wc=tn_mul(cx,cy),wb=tn_mul(bottom,cx),wr=tn_mul(right,cy);
 float den=tn_add(tn_add(tn_add(tn_add(wl,wt),wc),wb),wr);uint16_t gain=half_round(tn_mul(p(49),2.f));
 for(int c=0;c<3;++c){float value=tn_mul(tn_sample(hist,w,h,xc,yt,c),wt);
  value=tn_fma(tn_sample(hist,w,h,xl,yc,c),wl,value);value=tn_fma(tn_sample(hist,w,h,xc,yc,c),wc,value);
  value=tn_fma(tn_sample(hist,w,h,xc,yb,c),wb,value);value=tn_fma(tn_sample(hist,w,h,xr,yc,c),wr,value);
  value=tn_mul(value,tn_div(1.f,den));out[c]=h_mul(h_add(half_round(value),0xb800),gain);}
}
}
