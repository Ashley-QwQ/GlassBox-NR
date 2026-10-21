#pragma once
#include <cstring>
#include <type_traits>
namespace optm {
template<class To, class From> To bit_cast(const From& src) noexcept {
    static_assert(sizeof(To)==sizeof(From), "bit_cast size");
    static_assert(std::is_trivially_copyable<To>::value && std::is_trivially_copyable<From>::value, "bit_cast type");
    To dst;
    std::memcpy(&dst,&src,sizeof(dst));
    return dst;
}
inline bool little_endian() noexcept { unsigned x=1; return *reinterpret_cast<const unsigned char*>(&x)==1; }
inline int countl_zero(unsigned long long x) noexcept {
    if(!x)return 64;
    int n=0;
    for(unsigned long long mask=1ULL<<63; !(x&mask); mask>>=1)++n;
    return n;
}
}
