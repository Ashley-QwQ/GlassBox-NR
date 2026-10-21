#pragma once
#include "../decoder/compact_noise_optd.hpp"

#include <array>
#include <memory>
namespace dlssnr_optd {
struct NoiseTrace {
 std::array<uint32_t,4> k;
 std::array<uint32_t,2> radius;
 std::array<uint32_t,3> trig,products;
};
inline float noise_mul32(float a,float b) {volatile float p=a*b;return p;}
inline uint32_t noise_pcg(uint32_t v) {return ((v>>((v>>28)+4))^v)*0x108ef2d9U;}
inline std::array<uint32_t,4> noise_seed_k(uint32_t x,uint32_t y,uint32_t frame) {
 const uint32_t mul[4]={0xcaa5b80dU,0x2c9277b5U,0x83232c31U,0xfa6dc5f9U};
 const uint32_t add[4]={0x21dd796bU,0xac564b05U,0x3463e0acU,0x4712a88eU};
 uint32_t h=noise_pcg((y*0xd8163841U)^(x*0x8da6b343U)^(frame*0x9e3779b9U)^0x243f6a88U);
 h=(h>>22)^h;std::array<uint32_t,4> k{};
 for(int j=0;j<4;++j){uint32_t v=noise_pcg(h*mul[j]+add[j]);k[j]=((v>>30)^(v>>8))+1;}
 return k;
}
struct NoiseAssets {
 CompactTable radius,cosine,sine;
 NoiseTrace trace(const std::array<uint32_t,4>& k) const {
  NoiseTrace t;t.k=k;
  t.radius={radius.at(k[0]),radius.at(k[1])};
  t.trig={cosine.at(k[3]),sine.at(k[3]),cosine.at(k[2])};
  for(int i=0;i<3;++i)t.products[i]=optm_bit_cast<uint32_t>(noise_mul32(optm_bit_cast<float>(t.radius[i==2?0:1]),optm_bit_cast<float>(t.trig[i])));
  return t;
 }
 NoiseTrace trace_xy(uint32_t x,uint32_t y,uint32_t frame)const{return trace(noise_seed_k(x,y,frame));}
};
}
