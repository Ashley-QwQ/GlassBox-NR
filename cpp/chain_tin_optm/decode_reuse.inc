// CARD F: generated only by the original decoder; decoding never publishes errors.
struct E4Decoded { int32_t value=0; bool nan=false; };
using E4Table=std::array<E4Decoded,256>;
E4Table make_e4_table() {
 E4Table t{};for(int c=0;c<256;++c){try{t[c].value=e4_units(uint8_t(c));}
 catch(const std::domain_error& e){if(std::string(e.what())!="E4M3 NaN code")throw;t[c].nan=true;}}
 return t;
}
// NaN flags are consumed only at the original dot position, after f16_units(seed).
// Correspondence: cpp/opt/B/tin/primitives.inc:125
template<class B> uint16_t qmma_decoded(const E4Decoded* a, B b, uint16_t seed) {
#if QMMA_TARGET_SM120 == 1
    return qmma_sm120_r01::decoded(a,b,seed,false);
#endif
// Correspondence: cpp/opt/B/tin/primitives.inc:126
    int64_t terms[32]; int64_t seed_units=f16_units(seed); uint64_t largest = magnitude(seed_units);
// Correspondence: cpp/opt/B/tin/primitives.inc:127
    uint64_t min_nonzero = UINT64_MAX;
// Correspondence: cpp/opt/B/tin/primitives.inc:128
    for (int k = 0; k < 32; ++k) {
// Correspondence: cpp/opt/B/tin/primitives.inc:129
        const auto av=a[k]; if(av.nan)throw std::domain_error("E4M3 NaN code");
        const auto bv=b(k); if(bv.nan)throw std::domain_error("E4M3 NaN code");
        terms[k] = int64_t(av.value) * int64_t(bv.value) * 64;
// Correspondence: cpp/opt/B/tin/primitives.inc:130
        auto mag = magnitude(terms[k]);
// Correspondence: cpp/opt/B/tin/primitives.inc:131
        largest = std::max(largest, mag);
// Correspondence: cpp/opt/B/tin/primitives.inc:132
        if (mag) min_nonzero = std::min(min_nonzero, mag);
// Correspondence: cpp/opt/B/tin/primitives.inc:133
    }
// Correspondence: cpp/opt/B/tin/primitives.inc:134
    int shift = std::max(lg2_u64(largest) + 1 - 27, 0);
// Correspondence: cpp/opt/B/tin/primitives.inc:135
    auto place = [shift](int64_t v) {
// Correspondence: cpp/opt/B/tin/primitives.inc:136
        uint64_t a = magnitude(v); uint64_t p = (a >> shift) << shift;
// Correspondence: cpp/opt/B/tin/primitives.inc:137
        return v < 0 ? -int64_t(p) : int64_t(p);
// Correspondence: cpp/opt/B/tin/primitives.inc:138
    };
// Correspondence: cpp/opt/B/tin/primitives.inc:139
    int64_t sum = place(seed_units), exact_sum=seed_units;
// Correspondence: cpp/opt/B/tin/primitives.inc:140
    for (auto t : terms) { sum += place(t); exact_sum += t; }
// Correspondence: cpp/opt/B/tin/primitives.inc:141
    if (magnitude(sum) >= (uint64_t(1) << 52)) throw std::domain_error("QMMA accumulator outside exact range");
// Correspondence: cpp/opt/B/tin/primitives.inc:142
    if (largest && min_nonzero != UINT64_MAX) {
// Correspondence: cpp/opt/B/tin/primitives.inc:143
        int top = lg2_u64(largest) - 24;
// Correspondence: cpp/opt/B/tin/primitives.inc:144
        int range = lg2_u64(largest) - lg2_u64(min_nonzero);
// Correspondence: cpp/opt/B/tin/primitives.inc:145
        int cancel = exact_sum ? lg2_u64(largest) - lg2_u64(magnitude(exact_sum)) : 99;
// Correspondence: cpp/opt/B/tin/primitives.inc:146
        if (top >= 12 && range >= 20 && cancel >= 2) throw std::domain_error("D27 QMMA counterexample domain");
// Correspondence: cpp/opt/B/tin/primitives.inc:147
    }
// Correspondence: cpp/opt/B/tin/primitives.inc:148
    uint16_t result = half_round(std::ldexp(double(sum), -24));
// Correspondence: cpp/opt/B/tin/primitives.inc:149
    if ((result & 0x7c00) == 0x7c00) throw std::domain_error("QMMA half overflow");
// Correspondence: cpp/opt/B/tin/primitives.inc:150
    return result;
// Correspondence: cpp/opt/B/tin/primitives.inc:151
}

// Correspondence: cpp/opt/B/tin/primitives.inc:152
uint16_t hmma_cached(const uint16_t* a, const uint16_t* b, const double* av, const double* bv) {
// Correspondence: cpp/opt/B/tin/primitives.inc:153
    int e = -21; bool any = false;
// Correspondence: cpp/opt/B/tin/primitives.inc:154
    for (int k=0;k<16;++k) {
// Correspondence: cpp/opt/B/tin/primitives.inc:155
        double x=av[k],y=bv[k];
// Correspondence: cpp/opt/B/tin/primitives.inc:156
        if (!std::isfinite(x)||!std::isfinite(y)) throw std::domain_error("LOONG HMMA nonfinite operand");
// Correspondence: cpp/opt/B/tin/primitives.inc:157
        if(x==0||y==0)continue;
// Correspondence: cpp/opt/B/tin/primitives.inc:158
        int ae=(a[k]>>10)&31,be=(b[k]>>10)&31;
// Correspondence: cpp/opt/B/tin/primitives.inc:159
        e=std::max(e,(ae?ae-15:-14)+(be?be-15:-14));any=true;
// Correspondence: cpp/opt/B/tin/primitives.inc:160
    }
// Correspondence: cpp/opt/B/tin/primitives.inc:161
    if(!any)return 0;
// Correspondence: cpp/opt/B/tin/primitives.inc:162
    int grid=e-25;int64_t total=0;
// Correspondence: cpp/opt/B/tin/primitives.inc:163
    for(int k=0;k<16;++k){
// Correspondence: cpp/opt/B/tin/primitives.inc:164
        double p=av[k]*bv[k];
// Correspondence: cpp/opt/B/tin/primitives.inc:165
        double scaled=std::ldexp(p,-grid);
// Correspondence: cpp/opt/B/tin/primitives.inc:166
        if(!std::isfinite(scaled)||std::abs(scaled)>=double(uint64_t(1)<<52))throw std::domain_error("LOONG HMMA accumulator range");
// Correspondence: cpp/opt/B/tin/primitives.inc:167
        total+=int64_t(std::trunc(scaled));
// Correspondence: cpp/opt/B/tin/primitives.inc:168
    }
// Correspondence: cpp/opt/B/tin/primitives.inc:169
    const uint16_t h=half_round(std::ldexp(double(total),grid));
    // sm_120 proof sweep A: canonical +0 at half-zero publication (as in seal 49a98a0b).
    return h==0x8000u?uint16_t(0):h;
// Correspondence: cpp/opt/B/tin/primitives.inc:170
}

