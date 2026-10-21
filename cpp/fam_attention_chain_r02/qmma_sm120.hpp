#if defined(_MSC_VER)
#include <intrin.h>
#endif
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

// Explicit native instruction target; independent of the host CPU/GPU.
// QMMA_TARGET_SM120=1 is selected only by the isolated sm120 build.
namespace qmma_sm120_r01 {
inline uint64_t mag(int64_t x) { return x<0?uint64_t(-x):uint64_t(x); }
inline int log2u(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
 return x ? 63-__builtin_clzll(x) : -1;
#elif defined(_MSC_VER)
 if(!x)return -1;unsigned long index=0;_BitScanReverse64(&index,x);return int(index);
#else
 int e=-1;while(x){x>>=1;++e;}return e;
#endif
}
inline int32_t decode(uint8_t c) {
 if((c&127)==127)throw std::domain_error("E4M3 NaN code");
 int e=(c>>3)&15,m=c&7,v=e?((8+m)<<(e-1)):m;
 return (c&128)?-v:v;
}
inline int64_t seed_units(uint16_t h) {
 int e=(h>>10)&31,m=h&1023;
 if(e==31)throw std::domain_error("nonfinite QMMA seed");
 int64_t v=e?(int64_t(1024+m)<<(e-1)):m;
 return (h&0x8000)?-v:v;
}
inline int field(int32_t u) { return std::max(log2u(mag(u))-9,-6); }
inline int64_t place(int64_t x,int shift) {
 uint64_t p=(mag(x)>>shift)<<shift;return x<0?-int64_t(p):int64_t(p);
}
inline uint16_t publish(int64_t units) {
 if(!units)return 0; // Frozen model canonicalizes zero, including input -0.
 uint16_t sign=units<0?0x8000:0;uint64_t a=mag(units);
 int top=log2u(a),sh=std::max(top-10,0);uint64_t q=a>>sh;
 if(sh){uint64_t r=a- (q<<sh),tie=uint64_t(1)<<(sh-1);
  if(r>tie||(r==tie&&(q&1)))++q;}
 if(q>=2048){q>>=1;++top;}
 if(top<10)return sign|uint16_t(q);
 int E=top-9;if(E>=31)return sign|0x7c00;
 return sign|uint16_t(E<<10)|uint16_t(q-1024);
}
// G getters return exact signed 2^-9 units, after NaN flag checking.
// model=false preserves the old production guard contract and messages.
template<class A,class B> inline uint16_t dot(A a,B b,uint16_t seed,bool model) {
 int64_t c=seed_units(seed),terms[32];
 int E=c?log2u(mag(c))-24:-99;
 uint64_t largest=mag(c),least=UINT64_MAX;int64_t exact=c;
 for(int k=0;k<32;++k){int32_t av=a(k),bv=b(k);
  int64_t t=int64_t(av)*int64_t(bv)*64;terms[k]=t;exact+=t;
  uint64_t m=mag(t);largest=std::max(largest,m);
  if(m){least=std::min(least,m);E=std::max(E,field(av)+field(bv));}}
 int sh=std::max(E-1,0);int64_t sum=place(c,sh);
 for(auto t:terms)sum+=place(t,sh); // whole K32, no intermediate half.
 if(mag(sum)>=(uint64_t(1)<<52))throw std::domain_error("QMMA accumulator outside exact range");
 if(!model && largest && least!=UINT64_MAX){
  int top=log2u(largest)-24,range=log2u(largest)-log2u(least);
  int cancel=exact?log2u(largest)-log2u(mag(exact)):99;
  if(top>=12 && range>=20 && cancel>=2)throw std::domain_error("D27 QMMA counterexample domain");}
 uint16_t out=publish(sum);
 if(!model && (out&0x7c00)==0x7c00)throw std::domain_error("QMMA half overflow");
 return out;
}
inline uint16_t raw(const uint8_t* a,const uint8_t* b,uint16_t seed,bool model=false){
 return dot([a](int k){return decode(a[k]);},[b](int k){return decode(b[k]);},seed,model);
}
template<class D,class B> inline uint16_t decoded(const D* a,B b,uint16_t seed,bool model=false){
 return dot([a](int k){auto v=a[k];if(v.nan)throw std::domain_error("E4M3 NaN code");return v.value;},
  [b](int k){auto v=b(k);if(v.nan)throw std::domain_error("E4M3 NaN code");return v.value;},seed,model);
}
}
