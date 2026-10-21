#pragma once
#include <cstring>
#include <type_traits>
#include <cstdint>
template<class To,class From> To optm_bit_cast(const From& x) noexcept {
 static_assert(sizeof(To)==sizeof(From),"bit cast size");
 static_assert(std::is_trivially_copyable<To>::value&&std::is_trivially_copyable<From>::value,"bit cast type");
 To out;std::memcpy(&out,&x,sizeof(out));return out;
}
inline int optm_countr_zero(unsigned x) noexcept {
 if(!x)return int(sizeof(x)*8);int n=0;while((x&1u)==0u){++n;x>>=1;}return n;
}
inline int optm_countl_zero(uint64_t x) noexcept {
 if(!x)return 64;int n=0;uint64_t mask=uint64_t(1)<<63;while((x&mask)==0){++n;mask>>=1;}return n;
}
