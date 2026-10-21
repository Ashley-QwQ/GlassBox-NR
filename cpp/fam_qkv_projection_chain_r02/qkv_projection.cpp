#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "mem_vectors.hpp"
#include <memory>
#include <unordered_map>

namespace py = pybind11;
namespace {
using V = std::array<uint32_t, 32>;
template<class T> T bits_cast(uint32_t x) { return std::bit_cast<T>(x); }
uint32_t f32bits(float x) { return std::bit_cast<uint32_t>(x); }

double half_value(uint16_t x) {
    int sign = (x & 0x8000) ? -1 : 1;
    int e = (x >> 10) & 31, m = x & 1023;
    if (e == 31) return m ? std::numeric_limits<double>::quiet_NaN() : sign * std::numeric_limits<double>::infinity();
    if (e == 0) return sign * std::ldexp(double(m), -24);
    return sign * std::ldexp(double(1024 + m), e - 25);
}

// Exact binary64 -> binary16 RN-even. Every finite operation below is made in
// binary64 before this one publication point; integer products stay exact.
uint16_t half_round(double x) {
    uint64_t raw = std::bit_cast<uint64_t>(x);
    uint16_t sign = (raw >> 48) & 0x8000;
    if (std::isnan(x)) throw std::domain_error("NaN half publication outside measured domain");
    if (std::isinf(x)) return sign | 0x7c00;
    double a = std::abs(x);
    if (a == 0) return sign;
    if (a > 65520.0 || a == 65520.0) return sign | 0x7c00;
    int e;
    std::frexp(a, &e);
    int step_exp = std::max(e - 11, -24);
    double q = std::ldexp(a, -step_exp);
    double floor_q = std::floor(q);
    uint32_t n = uint32_t(floor_q);
    double rem = q - floor_q;
    if (rem > .5 || (rem == .5 && (n & 1))) ++n;
    double rounded = std::ldexp(double(n), step_exp);
    if (rounded >= 65536.0) return sign | 0x7c00;
    if (rounded < std::ldexp(1.0, -14)) return sign | uint16_t(n);
    int out_e;
    std::frexp(rounded, &out_e);
    uint32_t mant = uint32_t(std::ldexp(rounded, 11 - out_e));
    if (mant == 2048) { ++out_e; mant = 1024; }
    return sign | uint16_t((out_e + 14) << 10) | uint16_t(mant - 1024);
}

int e4_units(uint8_t c) {
    if ((c & 0x7f) == 0x7f) throw std::domain_error("E4M3 NaN code");
    int e = (c >> 3) & 15, m = c & 7;
    int v = e ? ((8 + m) << (e - 1)) : m;
    return (c & 0x80) ? -v : v;
}
uint16_t e4_half(uint8_t c) { return (c & 0x7f) == 0 ? ((c & 0x80) ? 0x8000 : 0) : half_round(std::ldexp(double(e4_units(c)), -9)); }
uint8_t sat_e4(uint16_t h) {
    double x = half_value(h);
    if (std::isnan(x)) throw std::domain_error("NaN into E4M3 quantizer");
    uint8_t sign = (h & 0x8000) ? 0x80 : 0;
    if (x == 0) return sign;
    double a = std::abs(x);
    if (a >= 448) return sign | 0x7e;
    uint8_t best = 0;
    double best_d = a;
    for (int c = 1; c <= 0x7e; ++c) {
        double d = std::abs(a - std::ldexp(double(e4_units(uint8_t(c))), -9));
        if (d < best_d || (d == best_d && !(c & 1) && (best & 1))) { best = uint8_t(c); best_d = d; }
    }
    return sign | best;
}
uint16_t h_add(uint16_t a, uint16_t b) { return half_round(half_value(a) + half_value(b)); }
uint16_t h_mul(uint16_t a, uint16_t b) { return half_round(half_value(a) * half_value(b)); }
uint16_t h_fma(uint16_t a, uint16_t b, uint16_t c) {
    return half_round(std::fma(half_value(a), half_value(b), half_value(c)));
}
uint16_t h_min(uint16_t a, uint16_t b) {
    double x = half_value(a), y = half_value(b);
    if (std::isnan(x) || std::isnan(y)) throw std::domain_error("NaN min");
    return x < y ? a : b;
}
uint16_t h_max(uint16_t a, uint16_t b) {
    double x = half_value(a), y = half_value(b);
    if (std::isnan(x) || std::isnan(y)) throw std::domain_error("NaN max");
    return x > y ? a : b;
}
int64_t f16_units(uint16_t h) {
    double x = half_value(h);
    if (!std::isfinite(x)) throw std::domain_error("nonfinite QMMA seed");
    return int64_t(std::ldexp(x, 24));
}
int lg2_u64(uint64_t x) { return x ? 63 - int(std::countl_zero(x)) : -99; }
uint64_t magnitude(int64_t x) { return x < 0 ? uint64_t(-x) : uint64_t(x); }
uint16_t qmma_dot(const uint8_t* a, const uint8_t* b, uint16_t seed) {
#if QMMA_TARGET_SM120 == 1
    return qmma_sm120_r01::raw(a,b,seed,false);
#endif
    int64_t terms[32]; int64_t seed_units=f16_units(seed); uint64_t largest = magnitude(seed_units);
    uint64_t min_nonzero = UINT64_MAX;
    for (int k = 0; k < 32; ++k) {
        terms[k] = int64_t(e4_units(a[k])) * int64_t(e4_units(b[k])) * 64;
        auto mag = magnitude(terms[k]);
        largest = std::max(largest, mag);
        if (mag) min_nonzero = std::min(min_nonzero, mag);
    }
    int shift = std::max(lg2_u64(largest) + 1 - 27, 0);
    auto place = [shift](int64_t v) {
        uint64_t a = magnitude(v); uint64_t p = (a >> shift) << shift;
        return v < 0 ? -int64_t(p) : int64_t(p);
    };
    int64_t sum = place(seed_units), exact_sum=seed_units;
    for (auto t : terms) { sum += place(t); exact_sum += t; }
    if (magnitude(sum) >= (uint64_t(1) << 52)) throw std::domain_error("QMMA accumulator outside exact range");
    if (largest && min_nonzero != UINT64_MAX) {
        int top = lg2_u64(largest) - 24;
        int range = lg2_u64(largest) - lg2_u64(min_nonzero);
        int cancel = exact_sum ? lg2_u64(largest) - lg2_u64(magnitude(exact_sum)) : 99;
        if (top >= 12 && range >= 20 && cancel >= 2) throw std::domain_error("D27 QMMA counterexample domain");
    }
    uint16_t result = half_round(std::ldexp(double(sum), -24));
    if ((result & 0x7c00) == 0x7c00) throw std::domain_error("QMMA half overflow");
    return result;
}

#include "qkv_projection_impl.hpp"

