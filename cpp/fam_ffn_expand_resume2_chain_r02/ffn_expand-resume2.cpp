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
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "mem_vectors.hpp"

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

// The binary16/E4M3/QMMA scalar primitives above are copied from
// cpp/src/step2a.cpp:1-131, sha256
// 91a28b28a9ca9771d70973fec37d772f60f0b9713a1d3c79267b309913d1110b.
// No AVX target flag applies.

struct Step { int epoch, a_slot, b_slot, b_reg; };
struct Chain { int final_epoch, final_slot; ChainVector<Step> steps; };
struct Source { int epoch, slot, d_reg, half; };
struct Store { int slot; std::array<Source,16> source; };

uint8_t gate(uint16_t expansion) {
    if ((expansion & 0x7c00) == 0x7c00) throw std::domain_error("nonfinite gate input");
    const uint16_t clamped = h_min(h_max(expansion, half_round(-4.0)), half_round(4.0));
    const uint16_t linear = half_round(std::fma(std::abs(half_value(clamped)), -0.055908203125, 0.447265625));
    const uint16_t factor = half_round(std::fma(half_value(clamped), half_value(linear), 0.89453125));
    return sat_e4(h_mul(expansion, factor));
}

template<class T> py::array_t<T> array_input(const py::handle& x, const char* name, int rank) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(x);
    if (!a || a.ndim() != rank) throw std::invalid_argument(std::string("bad ") + name);
    return a;
}

ChainVector<Chain> read_chains(py::dict wiring) {
    auto root = py::cast<py::dict>(wiring["chains"]);
    auto raw = py::cast<py::list>(root["chains"]);
    ChainVector<Chain> result;
    for (auto item : raw) {
        auto d = py::cast<py::dict>(item);
        auto f = py::cast<py::list>(d["final"]);
        Chain chain{py::cast<int>(f[0]), py::cast<int>(f[1]), {}};
        for (auto raw_step : py::cast<py::list>(d["steps"])) {
            auto s = py::cast<py::dict>(raw_step);
            auto b = py::cast<py::list>(s["b_reg_index"]);
            int r = py::cast<int>(b[0]);
            if (py::cast<int>(b[1]) != r + 1) throw std::invalid_argument("nonadjacent B registers");
            chain.steps.push_back({py::cast<int>(s["epoch"]), py::cast<int>(s["a_read_slot"]),
                                   py::cast<int>(s["b_load_slot"]), r});
        }
        if (chain.steps.size() != 32) throw std::invalid_argument("chain step count");
        result.push_back(std::move(chain));
    }
    if (result.size() != 32) throw std::invalid_argument("chain count");
    return result;
}

ChainVector<Store> read_stores(py::dict wiring) {
    ChainVector<Store> result;
    for (auto item : py::cast<py::list>(wiring["store_table"])) {
        auto d = py::cast<py::dict>(item);
        Store store{};
        store.slot = py::cast<int>(d["slot"]);
        auto bytes = py::cast<py::list>(d["bytes"]);
        if (bytes.size() != 16) throw std::invalid_argument("store width");
        for (int i=0;i<16;++i) {
            auto s = py::cast<py::dict>(bytes[i]);
            store.source[i] = {py::cast<int>(s["mma_epoch"]), py::cast<int>(s["mma_slot"]),
                               py::cast<int>(s["d_reg"]), py::cast<int>(s["half"])};
        }
        result.push_back(store);
    }
    if (result.size() != 8) throw std::invalid_argument("store rows");
    return result;
}

py::dict execute_original(py::bytes input_bytes, py::bytes weights_bytes,
                 py::handle py_src, py::handle py_wo, py::handle py_so,
                 py::dict wiring, int threads, bool shuffled) {
    std::string input = input_bytes, weights = weights_bytes;
    auto src = array_input<int32_t>(py_src,"sread_src",5);
    auto wo = array_input<int64_t>(py_wo,"gload_weight_offset",5);
    auto so = array_input<int64_t>(py_so,"store_output_offset",4);
    const size_t nc = size_t(src.shape(0));
    if (src.shape(1)!=4 || src.shape(2)!=16 || src.shape(3)!=8 || src.shape(4)!=32 ||
        wo.shape(0)!=src.shape(0) || wo.shape(1)!=4 || wo.shape(2)!=16 ||
        wo.shape(3)!=8 || wo.shape(4)!=32 || so.shape(0)!=src.shape(0) ||
        so.shape(1)!=4 || so.shape(2)!=8 || so.shape(3)!=32)
        throw std::invalid_argument("map shape mismatch");
    if (weights.size() < 4194304) throw std::invalid_argument("short weight tensor");
    auto chains = read_chains(wiring);
    auto stores = read_stores(wiring);
    const auto* sr = src.data(); const auto* wr = wo.data(); const auto* dst = so.data();
    auto source_at = [&](size_t c,int y,int ep,int slot,int lane)->int32_t {
        return sr[((((c*4+y)*16+ep)*8+slot)*32+lane)];
    };
    auto weight_at = [&](size_t c,int y,int ep,int slot,int lane)->int64_t {
        return wr[((((c*4+y)*16+ep)*8+slot)*32+lane)];
    };
    auto dest_at = [&](size_t c,int y,int slot,int lane)->int64_t {
        return dst[(((c*4+y)*8+slot)*32+lane)];
    };
    const auto cov = py::cast<py::dict>(wiring["coverage"]);
    const size_t output_len = py::cast<size_t>(cov["output_bytes"]);
    const size_t release_bytes = py::cast<size_t>(cov["release_bytes"]);
    const size_t release_ctas = py::cast<size_t>(cov["release_ctas"]);
    std::string output(output_len,'\0');
    ChainVector<uint8_t> cover(output_len,0);
    ChainVector<std::pair<size_t,int>> tiles;
    for(size_t c=0;c<nc;++c) for(int y=0;y<4;++y) {
        bool live = false;
        for(int slot=0;slot<8;++slot) for(int lane=0;lane<32;++lane)
            if(dest_at(c,y,slot,lane)>=0) live=true;
        if (!live) continue;
        for(const auto& store: stores) for(int lane=0;lane<32;++lane) {
            int64_t base=dest_at(c,y,store.slot,lane);
            if(base<0)continue;
            if(uint64_t(base)+16>output_len) throw std::out_of_range("store offset");
            for(int b=0;b<16;++b) {
                auto& n=cover[size_t(base)+b];
                if(n++) throw std::invalid_argument("output has multiple owners");
            }
        }
        tiles.emplace_back(c,y);
    }
    if(std::find(cover.begin(),cover.end(),0)!=cover.end()) throw std::invalid_argument("output coverage gap");
    if (tiles.empty()) throw std::invalid_argument("no live tiles");
    // One zero s32 release store per CTA x. CTA x maps bijectively to
    // 32*(x mod Q)+floor(x/Q), with grid.x=32*Q, so the write set is dense.
    if (release_bytes != 4*release_ctas || release_ctas != nc || tiles.size() > 4*nc)
        throw std::invalid_argument("release CTA extent mismatch");
    ChainVector<int> final_to_chain(16*64,-1);
    for(int i=0;i<int(chains.size());++i) {
        auto& ch=chains[i];
        if(ch.final_epoch<0||ch.final_epoch>=16||ch.final_slot<0||ch.final_slot>=64)
            throw std::invalid_argument("final index");
        final_to_chain[ch.final_epoch*64+ch.final_slot]=i;
    }
    auto worker = [&](size_t ordinal) {
        auto [c,y]=tiles[shuffled ? tiles.size()-1-ordinal : ordinal];
        std::array<std::array<uint8_t,128>,32> codes{};
        for(int ci=0;ci<32;++ci) {
            uint16_t acc[16][8]{};
            for(const auto& step:chains[ci].steps) {
                uint8_t a[16][32]{},b[32][8]{};
                for(int lane=0;lane<32;++lane) {
                    int g=lane/4,t=lane%4;
                    int32_t ab=source_at(c,y,step.epoch,step.a_slot,lane);
                    int64_t bb=weight_at(c,y,step.epoch,step.b_slot,lane)+4*step.b_reg;
                    if((ab>=0&&uint64_t(ab)+16>input.size())||bb<0||uint64_t(bb)+8>weights.size())
                        throw std::out_of_range("A/B map bounds");
                    for(int r=0;r<4;++r)for(int sb=0;sb<4;++sb)
                        a[g+8*(r%2)][16*(r/2)+4*t+sb]=ab<0?0:uint8_t(input[size_t(ab)+4*r+sb]);
                    for(int r=0;r<2;++r)for(int sb=0;sb<4;++sb)
                        b[16*r+4*t+sb][g]=uint8_t(weights[size_t(bb)+4*r+sb]);
                }
                for(int row=0;row<16;++row)for(int col=0;col<8;++col) {
                    uint8_t bv[32];for(int k=0;k<32;++k)bv[k]=b[k][col];
                    acc[row][col]=qmma_dot(a[row],bv,acc[row][col]);
                }
            }
            for(int lane=0;lane<32;++lane)for(int reg=0;reg<2;++reg)for(int half=0;half<2;++half) {
                int row=lane/4+8*reg,col=2*(lane%4)+half;
                codes[ci][lane*4+2*reg+half]=gate(acc[row][col]);
            }
        }
        for(const auto& store:stores)for(int lane=0;lane<32;++lane) {
            int64_t raw_base=dest_at(c,y,store.slot,lane);
            if(raw_base<0)continue;
            size_t base=size_t(raw_base);
            for(int b=0;b<16;++b) {
                const auto& s=store.source[b];
                if(s.epoch<0||s.epoch>=16||s.slot<0||s.slot>=64||s.d_reg<0||s.d_reg>=2||s.half<0||s.half>=2)
                    throw std::invalid_argument("store source index");
                int ci=final_to_chain[s.epoch*64+s.slot];
                if(ci<0) throw std::invalid_argument("unknown final source");
                output[base+b]=char(codes[ci][lane*4+2*s.d_reg+s.half]);
            }
        }
    };
    if(threads<1||threads>12) throw std::invalid_argument("threads must be 1..12");
    // Each tile owns disjoint byte addresses, proved above before workers start.
    std::atomic<size_t> next{0};std::exception_ptr failure;std::mutex failure_mutex;
    ChainVector<std::thread> pool;
    { py::gil_scoped_release release;
      for(int tid=0;tid<threads;++tid)pool.emplace_back([&] {
          try { for(;;){size_t i=next.fetch_add(1);if(i>=tiles.size())break;worker(i);} }
          catch(...){std::lock_guard<std::mutex> guard(failure_mutex);if(!failure)failure=std::current_exception();}
      });
      for(auto& t:pool)t.join();
    }
    if(failure)std::rethrow_exception(failure);
    py::dict result;
    result["output"]=py::bytes(output);
    auto releases=py::cast<py::list>(wiring["releases"]);
    if(releases.size()!=4)throw std::invalid_argument("release write count");
    for(auto item:releases) {
        auto r=py::cast<py::dict>(item);
        if(py::cast<int>(r["offset_from_rd4"])!=0||py::cast<int>(r["value"])!=0)
            throw std::invalid_argument("unsupported release write");
    }
    result["release"]=py::bytes(std::string(release_bytes,'\0'));
    result["release_bytes"]=release_bytes;
    result["tiles"]=tiles.size();
    result["output_bytes"]=output_len;
    return result;
}
} // namespace


py::dict execute(py::bytes input_bytes, py::bytes weights_bytes,
                 py::handle py_src, py::handle py_wo, py::handle py_so,
                 py::dict wiring, int threads, bool shuffled){chain_mem::begin();try{auto result=execute_original(input_bytes,weights_bytes,py_src,py_wo,py_so,wiring,threads,shuffled);result["mem"]=chain_mem::finish();return result;}catch(...){chain_mem::active=false;throw;}}
PYBIND11_MODULE(dlssnr_fam_ffn_expand_resume2_chain_r02,m) {
    m.def("memory_counter_selftest",[](){chain_mem::begin();{ChainVector<uint8_t> a(4096);ChainVector<uint16_t> b(2048);std::string text(4096,'x');}return chain_mem::finish();});
    m.def("execute",&execute,py::arg("input"),py::arg("weights"),py::arg("src"),
          py::arg("weight_offsets"),py::arg("store_offsets"),py::arg("wiring"),
          py::arg("threads")=1,py::arg("shuffled")=false);
    m.def("gate",&gate);
    m.def("qmma_dot",[](py::bytes a,py::bytes b,uint16_t seed){
        std::string x=a,y=b;
        if(x.size()!=32||y.size()!=32)throw std::invalid_argument("vectors must have 32 codes");
        return qmma_dot(reinterpret_cast<const uint8_t*>(x.data()),
                        reinterpret_cast<const uint8_t*>(y.data()),seed);
    });
}
