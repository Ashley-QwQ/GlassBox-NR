#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace opte1 {
// Integer only: base is zero or a positive normal <1, E=111..125.
// q-base = ((q<<shift)-significand) * 2^-shift. Discard low bits.
inline unsigned bit_length(uint64_t v) {
    unsigned n=0;
    if(v>>32){v>>=32;n+=32;} if(v>>16){v>>=16;n+=16;}
    if(v>>8){v>>=8;n+=8;} if(v>>4){v>>=4;n+=4;}
    if(v>>2){v>>=2;n+=2;} if(v>>1){v>>=1;n+=1;}
    return n+1;
}
inline uint32_t trunc_negative_sum(int E,uint32_t base) {
    unsigned be=(base>>23)&255;
    unsigned shift=base?150-be:23;
    uint64_t sig=base?uint64_t((base&0x7fffff)|0x800000):0;
    uint64_t num=(uint64_t(127-E)<<shift)-sig;
    unsigned k=bit_length(num)-1;
    uint32_t mant=uint32_t(num>>(k-23)); // k>=23 over this domain.
    return 0x80000000u|((k-shift+127)<<23)|(mant&0x7fffff);
}
struct Decoder {
    std::vector<uint32_t> base,e126;
    std::vector<uint8_t> levels;
    uint32_t entry(int E,uint32_t m) const {
        if(E==127)return base[m]; if(E==126)return e126[m];
        size_t at=size_t(m>>3)*3;
        uint32_t word=levels[at]|(uint32_t(levels[at+1])<<8)|(uint32_t(levels[at+2])<<16);
        unsigned level=(word>>(3*(m&7)))&7;
        unsigned group=E<=118?0:(E<=122?1:(E<=124?2:3));
        return trunc_negative_sum(E,base[m])-uint32_t(level<=group);
    }
    size_t bytes()const{return (base.capacity()+e126.capacity())*4+levels.capacity();}
};
}
