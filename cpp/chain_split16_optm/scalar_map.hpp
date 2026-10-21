#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <type_traits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
namespace {
using V = std::array<uint32_t, 32>;
template<class To, class From> To portable_bit_cast(const From& x) { static_assert(sizeof(To)==sizeof(From), "bit cast size"); static_assert(std::is_trivially_copyable<To>::value && std::is_trivially_copyable<From>::value, "bit cast type"); To out; std::memcpy(&out,&x,sizeof(out)); return out; }
template<class T> T bits_cast(uint32_t x) { return portable_bit_cast<T>(x); }
uint32_t f32bits(float x) { return portable_bit_cast<uint32_t>(x); }

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
    uint64_t raw = portable_bit_cast<uint64_t>(x);
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
int lg2_u64(uint64_t x) {
#if defined(_MSC_VER)
 if(!x)return -99;unsigned long index=0;_BitScanReverse64(&index,x);return int(index);
#else
 if(!x)return -99; int out=0; while(x>>=1)++out; return out;
#endif
}
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
uint16_t hmma_dot(const uint16_t* a, const uint16_t* b) {
    int e = -21; bool any = false;
    for (int k=0;k<16;++k) {
        double x=half_value(a[k]),y=half_value(b[k]);
        if (!std::isfinite(x)||!std::isfinite(y)) throw std::domain_error("LOONG HMMA nonfinite operand");
        if(x==0||y==0)continue;
        int ae=(a[k]>>10)&31,be=(b[k]>>10)&31;
        e=std::max(e,(ae?ae-15:-14)+(be?be-15:-14));any=true;
    }
    if(!any)return 0;
    int grid=e-25;int64_t total=0;
    for(int k=0;k<16;++k){
        double p=half_value(a[k])*half_value(b[k]);
        double scaled=std::ldexp(p,-grid);
        if(!std::isfinite(scaled)||std::abs(scaled)>=double(uint64_t(1)<<52))throw std::domain_error("LOONG HMMA accumulator range");
        total+=int64_t(std::trunc(scaled));
    }
    return half_round(std::ldexp(double(total),grid));
}

struct Array {
    py::array owner;
    const char* data = nullptr;
    std::vector<py::ssize_t> shape, stride;
    py::ssize_t item = 0;
    explicit Array(py::handle o) : owner(py::reinterpret_borrow<py::array>(o)) {
        if(o.is_none())return;
        auto b = owner.request();
        if (!py::cast<bool>(owner.dtype().attr("isnative")) || py::cast<std::string>(owner.dtype().attr("kind")) != "i" || (b.itemsize!=1 && b.itemsize!=2 && b.itemsize!=4 && b.itemsize!=8)) throw std::invalid_argument("map array requires native signed integer dtype");
        if (!(owner.flags() & py::array::c_style)) throw std::invalid_argument("map array must be C contiguous");
        data = static_cast<const char*>(b.ptr); item = b.itemsize;
        shape.assign(b.shape.begin(), b.shape.end()); stride.assign(b.strides.begin(), b.strides.end());
    }
    int64_t at(int c, int y, int l, int s, int t = -1) const {
        if(!data)throw std::invalid_argument("node requires absent map array");
        if(shape.size()<4 || c<0 || c>=shape[0] || y<0 || y>=shape[1] || l<0 || l>=shape[2] || s<0 || s>=shape[3] || (t>=0 && (shape.size()<5 || t>=shape[4])))throw std::out_of_range("map index");
        py::ssize_t p = c*stride[0] + y*stride[1] + l*stride[2] + s*stride[3] + (t < 0 ? 0 : t*stride[4]);
        const char* q = data + p;
        if (item == 8) { int64_t v; std::memcpy(&v,q,8); return v; }
        if (item == 4) { int32_t v; std::memcpy(&v,q,4); return v; }
        if (item == 2) { int16_t v; std::memcpy(&v,q,2); return v; }
        if (item == 1) return portable_bit_cast<int8_t>(uint8_t(*q));
        throw std::invalid_argument("unsupported map array dtype");
    }
};
struct Node { std::string kind, op; int a=0,b=0,c=0,d=0,k=0; std::vector<int> args; };
struct Mma { std::array<int,16> a; std::array<int,8> b; std::array<int,4> c; };
struct Spec {
    std::vector<Node> nodes; std::vector<Mma> mmas;
    std::vector<int> pcs; std::vector<std::array<int,16>> stores;
    std::vector<int> contract;
    struct ClassStore {int cls,pc; std::array<int,16> atoms;};
    std::vector<ClassStore> class_stores;
};
Spec parse_spec(py::dict w, py::dict contract) {
    Spec s;
    py::list ns = w["nodes"].cast<py::list>(); s.nodes.reserve(ns.size());
    for (auto it : ns) {
        py::list n = py::reinterpret_borrow<py::list>(it); Node x; x.kind=n[0].cast<std::string>();
        if (x.kind == "F") { x.op=n[1].cast<std::string>(); x.args=n[2].cast<std::vector<int>>(); }
        else if (x.kind == "WI") { x.op=n[1].cast<std::string>(); x.a=n[2].cast<int>(); x.b=n[3].cast<int>(); x.c=n[4].cast<int>(); x.k=n[5].cast<int>(); }
        else { if(n.size()>1)x.a=n[1].cast<int>(); if(n.size()>2)x.b=n[2].cast<int>(); if(n.size()>3)x.c=n[3].cast<int>(); }
        s.nodes.push_back(std::move(x));
    }
    for (auto it : w["mmas"].cast<py::list>()) {
        py::dict m=py::reinterpret_borrow<py::dict>(it); Mma x;
        auto a=m["A"].cast<std::vector<int>>(),b=m["B"].cast<std::vector<int>>(),c=m["C"].cast<std::vector<int>>();
        if(a.size()!=16||b.size()!=8||c.size()!=4)throw std::invalid_argument("bad MMA operand length");
        std::copy(a.begin(),a.end(),x.a.begin());std::copy(b.begin(),b.end(),x.b.begin());std::copy(c.begin(),c.end(),x.c.begin());s.mmas.push_back(x);
    }
    s.contract.assign(s.nodes.size(),-1);
    for(auto it:contract) { int i=std::stoi(py::str(it.first)); int k=py::cast<int>(it.second); if(i<0||i>=int(s.nodes.size())||k<0||k>1)throw std::invalid_argument("bad contraction");s.contract[i]=k; }
    for(auto it:w["gstore_pcs"].cast<py::list>()) {
        int pc=py::cast<int>(it);s.pcs.push_back(pc);
        auto v=w.contains("classes") ? std::vector<int>(16,0) : w["gstore_atoms"].cast<py::dict>()[py::str(std::to_string(pc))].cast<std::vector<int>>();
        if(v.size()!=16)throw std::invalid_argument("store not 16 bytes");std::array<int,16> a;std::copy(v.begin(),v.end(),a.begin());s.stores.push_back(a);
    }
    if(w.contains("classes"))for(auto obj:w["classes"].cast<py::list>()) {
        py::dict cl=py::reinterpret_borrow<py::dict>(obj);
        for(auto pcobj:cl["gstore_pcs"].cast<py::list>()) {
            int pc=py::cast<int>(pcobj);auto atoms=cl["gstore_atoms"].cast<py::dict>()[py::str(std::to_string(pc))].cast<std::vector<int>>();
            if(atoms.size()!=16)throw std::invalid_argument("class store width");
            Spec::ClassStore cs;cs.cls=cl["cls"].cast<int>();cs.pc=pc;std::copy(atoms.begin(),atoms.end(),cs.atoms.begin());s.class_stores.push_back(cs);
        }
    }
    return s;
}
struct Worker {
    const Spec& s; const Array &wl,&src,&go,&xs,&xw,&sh,&ps;
    const uint8_t *inp,*weight; size_t ni,nw;
    const float *rsq,*rcp; size_t nt;
    std::vector<V> mem; std::vector<uint8_t> seen;
    int c=0,y=0;
    const Array* tc=nullptr;
    std::vector<V> invalid;
    Worker(const Spec& sp,const Array& w,const Array& sr,const Array& g,const Array& x,const Array& xw0,const Array& sh0,const Array& p,
           const uint8_t* i,size_t il,const uint8_t* wt,size_t wn,const float* r,const float* rc,size_t tn)
      :s(sp),wl(w),src(sr),go(g),xs(x),xw(xw0),sh(sh0),ps(p),inp(i),weight(wt),ni(il),nw(wn),rsq(r),rcp(rc),nt(tn),mem(sp.nodes.size()),seen(sp.nodes.size()){}
    const V& val(int i) {
        if(i<0||i>=int(s.nodes.size()))throw std::out_of_range("node id");
        if(seen[i]==2)return mem[i];
        if(seen[i]==1)throw std::runtime_error("cycle in wiring DAG");
        Node const& n=s.nodes[i]; if(n.kind=="M")throw std::runtime_error("MMA value before its stage");
        seen[i]=1; V &v=mem[i]; V* bad=tc ? &invalid[i] : nullptr;if(tc)bad->fill(0);
        if(n.kind=="W")for(int l=0;l<32;++l){auto off=wl.at(c,y,l,n.a)+n.b;if(off<0||uint64_t(off)>=nw)throw std::out_of_range("weight load");v[l]=weight[off];}
        else if(n.kind=="S")for(int l=0;l<32;++l){auto off=src.at(c,y,l,n.a,n.b);if(off< -1||off>=int64_t(ni))throw std::out_of_range("shared read");v[l]=off<0?0:inp[off];}
        else if(n.kind=="X"||n.kind=="XH")for(int l=0;l<32;++l){uint8_t q[4];for(int j=0;j<4;++j){auto off=xs.at(c,y,l,n.a,j);auto wo=xw.at(c,y,l,n.a,j);if(off>=0&&off<int64_t(ni))q[j]=inp[off];else if(off==-3&&wo>=0&&wo<int64_t(nw))q[j]=weight[wo];else if(off==-1)q[j]=0;else if(tc&&off==-7){q[j]=0;(*bad)[l]=1;}else throw std::out_of_range("global word source");}v[l]=n.kind=="X"?q[n.b]:uint32_t(q[2*n.b])|(uint32_t(q[2*n.b+1])<<8);}
        else if(n.kind=="K")v.fill(n.a);
        else if(n.kind=="E"){auto &a=val(n.a);for(int l=0;l<32;++l)v[l]=e4_half(uint8_t(a[l]));}
        else if(n.kind=="Q"){auto &a=val(n.a);for(int l=0;l<32;++l)v[l]=sat_e4(uint16_t(a[l]));}
        else if(n.kind=="HB"){V a=val(n.a),b=val(n.b);for(int l=0;l<32;++l)v[l]=(a[l]&255)|((b[l]&255)<<8);}
        else if(n.kind=="B"){V a=val(n.b);for(int l=0;l<32;++l)v[l]=a[l^n.a];}
        else if(n.kind=="SI"){V a=val(n.b);for(int l=0;l<32;++l){auto ix=sh.at(c,y,l,n.a);if(tc&&ix==-2){(*bad)[l]=1;ix=0;}if(ix<0||ix>=32)throw std::out_of_range("shuffle index");v[l]=a[ix];if(tc)(*bad)[l]|=invalid[n.b][ix];}}
        else if(n.kind=="PS"){V a=val(n.b),b=val(n.c);for(int l=0;l<32;++l){auto p=ps.at(c,y,l,n.a);if(tc&&p==-2){(*bad)[l]=1;p=0;}if(p!=0&&p!=1)throw std::domain_error("missing select predicate");v[l]=p?a[l]:b[l];if(tc)(*bad)[l]|=invalid[p?n.b:n.c][l];}}
        else if(n.kind=="MV"){V a=val(n.b),b=val(n.c);for(int l=0;l<32;++l){int cell=2*l+n.a, z=8*(cell%8)+cell/8;v[l]=(z%2?b:a)[z/2];}}
        else if(n.kind=="WI"){V a=val(n.b),b=val(n.c);for(int l=0;l<32;++l){uint32_t q=(a[l]&65535)|((b[l]&65535)<<16);if(n.op=="shl"){if(n.a<0||n.a>=32)throw std::domain_error("word shift");q<<=n.a;}else if(n.op=="add")q+=uint32_t(n.a);else throw std::domain_error("word op");v[l]=(q>>(16*n.k))&65535;}}
        else if(n.kind=="F") {
            if(n.op=="add"&&s.contract[i]>=0){auto &m0=s.nodes[n.args[0]],&m1=s.nodes[n.args[1]];if(m0.kind!="F"||m1.kind!="F"||m0.op!="mul"||m1.op!="mul"||m0.args[0]!=m0.args[1]||m1.args[0]!=m1.args[1])throw std::domain_error("bad contraction DAG");int k=s.contract[i];V a=val(s.nodes[n.args[k]].args[0]), b=val(n.args[1-k]);for(int l=0;l<32;++l)v[l]=h_fma(a[l],a[l],b[l]);}
            else {std::vector<V> a;for(int id:n.args)a.push_back(val(id));
                for(int l=0;l<32;++l){
                    if(n.op=="add")v[l]=h_add(a[0][l],a[1][l]);
                    else if(n.op=="mul")v[l]=h_mul(a[0][l],a[1][l]);
                    else if(n.op=="fma")v[l]=h_fma(a[0][l],a[1][l],a[2][l]);
                    else if(n.op=="min")v[l]=h_min(a[0][l],a[1][l]);
                    else if(n.op=="max")v[l]=h_max(a[0][l],a[1][l]);
                    else if(n.op=="abs")v[l]=a[0][l]&0x7fff;
                    else if(n.op=="rsqrt"||n.op=="rcp"){uint32_t ix=a[0][l];if(tc&&invalid[n.args[0]][l])ix=0x3c00;if(ix<0x410||ix>=0x7c00||ix-0x410>=nt)throw std::domain_error("MUFU table domain");v[l]=f32bits((n.op=="rsqrt"?rsq:rcp)[ix-0x410]);}
                    else if(n.op=="rne16")v[l]=half_round(double(bits_cast<float>(a[0][l])));
                    else if(n.op=="rne16w"){uint32_t raw=(a[0][l]&255)|((a[1][l]&255)<<8)|((a[2][l]&255)<<16)|((a[3][l]&255)<<24);v[l]=half_round(double(bits_cast<float>(raw)));}
                    else throw std::domain_error("unknown F operation");
                }
            }
        }else throw std::domain_error("unknown node kind");
        if(tc){
            auto merge=[&](int id){for(int l=0;l<32;++l)(*bad)[l]|=invalid[id][l];};
            if(n.kind=="E"||n.kind=="Q")merge(n.a);
            else if(n.kind=="HB"){merge(n.a);merge(n.b);}
            else if(n.kind=="B")for(int l=0;l<32;++l)(*bad)[l]|=invalid[n.b][l^n.a];
            else if(n.kind=="MV")for(int l=0;l<32;++l){int cell=2*l+n.a,z=8*(cell%8)+cell/8;(*bad)[l]|=invalid[z%2?n.c:n.b][z/2];}
            else if(n.kind=="WI"){merge(n.b);merge(n.c);}
            else if(n.kind=="F"){
                if(n.op=="add"&&s.contract[i]>=0){int k=s.contract[i];merge(s.nodes[n.args[k]].args[0]);merge(n.args[1-k]);}
                else for(int id:n.args)merge(id);
            }
        }
        seen[i]=2;return v;
    }
    void mma(int o) {
        auto const& m=s.mmas[o];
        std::array<V,16> aa;std::array<V,8> bb;std::array<V,4> cc;
        for(int j=0;j<16;++j)aa[j]=val(m.a[j]);for(int j=0;j<8;++j)bb[j]=val(m.b[j]);for(int j=0;j<4;++j)cc[j]=val(m.c[j]);
        if(tc)for(auto group:{std::vector<int>(m.a.begin(),m.a.end()),std::vector<int>(m.b.begin(),m.b.end()),std::vector<int>(m.c.begin(),m.c.end())})for(int id:group)for(auto b:invalid[id])if(b)throw std::domain_error("MMA depends on absent class site");
        V out[4]{};
        for(int row=0;row<16;++row)for(int col=0;col<8;++col){
            uint8_t ac[32],bc[32];
            for(int k=0;k<32;++k){int l=(row%8)*4+((k%16)/4),r=row/8+2*(k/16);ac[k]=uint8_t(aa[4*r+k%4][l]);l=col*4+((k%16)/4);r=k/16;bc[k]=uint8_t(bb[4*r+k%4][l]);}
            int l=(row%8)*4+col/2,ix=2*(row/8)+col%2;
            out[ix][l]=qmma_dot(ac,bc,uint16_t(cc[ix][l]));
        }
        // Four M atoms are assigned in wiring order by the stage number.
        for(int k=0;k<4;++k){int id=m_atoms[4*o+k];mem[id]=out[k];seen[id]=2;}
    }
    std::vector<int> m_atoms;
    void setup_atoms(){m_atoms.assign(s.mmas.size()*4,-1);for(size_t i=0;i<s.nodes.size();++i)if(s.nodes[i].kind=="M"){auto &n=s.nodes[i];if(n.a<0||n.a>=int(s.mmas.size())||n.b<0||n.b>=4)throw std::invalid_argument("bad M atom");m_atoms[4*n.a+n.b]=int(i);}for(int x:m_atoms)if(x<0)throw std::invalid_argument("missing M atom");}
    void run(int ci,int yi,uint8_t* output){
        c=ci;y=yi;std::fill(seen.begin(),seen.end(),0);if(tc)for(auto& b:invalid)b.fill(0);
        for(int o=0;o<int(s.mmas.size());++o)mma(o);
        if(!tc){for(size_t j=0;j<s.stores.size();++j){std::array<V,16> vals;for(int b=0;b<16;++b)vals[b]=val(s.stores[j][b]);for(int l=0;l<32;++l){auto off=go.at(c,y,l,int(j));if(off<0)continue;for(int b=0;b<16;++b)output[off+b]=uint8_t(vals[b][l]);}}}
        else for(auto& st:s.class_stores){int j=int(std::find(s.pcs.begin(),s.pcs.end(),st.pc)-s.pcs.begin());std::array<V,16> vals;for(int b=0;b<16;++b)vals[b]=val(st.atoms[b]);for(int l=0;l<32;++l){if(tc->at(c,y,l,0)!=st.cls)continue;auto off=go.at(c,y,l,j);if(off<0)continue;for(int b=0;b<16;++b){if(invalid[st.atoms[b]][l])throw std::domain_error("class store depends on absent site");output[off+b]=uint8_t(vals[b][l]);}}}
    }
};
py::bytes execute_prepared(const Spec* prepared,py::dict wiring,py::dict maps,py::bytes input,py::bytes weights,py::dict contract,py::array_t<float> rsq,py::array_t<float> rcp,size_t output_len,int threads,uint32_t queue_seed){
    Spec s=prepared ? *prepared : parse_spec(wiring,contract);
    Array wl(maps["wload_off"]),sr(maps["sread_src"]),go(maps["gstore_off"]),xs((maps.contains("xsite_src") ? py::object(maps["xsite_src"]) : py::none())),xw((maps.contains("xsite_wsrc") ? py::object(maps["xsite_wsrc"]) : py::none())),sh((maps.contains("shfl_idx") ? py::object(maps["shfl_idx"]) : py::none())),ps((maps.contains("psel") ? py::object(maps["psel"]) : py::none()));
    std::unique_ptr<Array> tc;if(maps.contains("thread_class"))tc=std::make_unique<Array>(maps["thread_class"]);
    if(wl.shape.size()!=4||wl.shape[2]!=32||go.shape.size()!=4||go.shape[3]!=py::ssize_t(s.stores.size()))throw std::invalid_argument("map geometry");
    if(rsq.size()!=rcp.size()||rsq.size()<0x7c00-0x410)throw std::invalid_argument("MUFU table size");
    std::string in=input,wt=weights;
    std::vector<uint8_t> cover(output_len,0),out(output_len,0);
    std::vector<int> tasks; tasks.reserve(wl.shape[0]*wl.shape[1]);
    for(int c=0;c<wl.shape[0];++c)for(int y=0;y<wl.shape[1];++y){tasks.push_back(c*int(wl.shape[1])+y);for(int l=0;l<32;++l)for(int j=0;j<int(s.stores.size());++j){auto off=go.at(c,y,l,j);if(off<0)continue;if(tc){int cls=int(tc->at(c,y,l,0));bool found=false;for(auto& st:s.class_stores)if(st.cls==cls&&st.pc==s.pcs[j])found=true;if(!found)throw std::domain_error("store outside thread class");}if(uint64_t(off)+16>output_len)throw std::out_of_range("store outside output");for(int b=0;b<16;++b)if(++cover[off+b]!=1)throw std::runtime_error("output ownership collision");}}
    if(std::find(cover.begin(),cover.end(),0)!=cover.end())throw std::runtime_error("output coverage incomplete");
    if(queue_seed){uint32_t z=queue_seed;for(int i=int(tasks.size())-1;i>0;--i){z=1664525*z+1013904223;std::swap(tasks[i],tasks[z%uint32_t(i+1)]);}}
    threads=std::max(1,std::min(threads,12));std::atomic<size_t> next{0};std::atomic<bool> fail{false};std::exception_ptr ep;std::mutex error_lock;
    {
      py::gil_scoped_release release;
      auto work=[&](){try{Worker w(s,wl,sr,go,xs,xw,sh,ps,reinterpret_cast<const uint8_t*>(in.data()),in.size(),reinterpret_cast<const uint8_t*>(wt.data()),wt.size(),rsq.data(),rcp.data(),rsq.size());w.tc=tc.get();if(w.tc)w.invalid.resize(s.nodes.size());w.setup_atoms();while(!fail){size_t ix=next.fetch_add(1);if(ix>=tasks.size())break;int t=tasks[ix];w.run(t/int(wl.shape[1]),t%int(wl.shape[1]),out.data());}}catch(...){fail=true;std::lock_guard<std::mutex> guard(error_lock);if(!ep)ep=std::current_exception();}};
      std::vector<std::thread> pool;for(int i=1;i<threads;++i)pool.emplace_back(work);work();for(auto &t:pool)t.join();
    }
    if(ep)std::rethrow_exception(ep);
    return py::bytes(reinterpret_cast<const char*>(out.data()),out.size());
}

// The ViT lane evaluator publishes one 16x8 tile per QMMA. Inputs here are
// already E4M3 codes and binary16 seed bits; no float conversion or BLAS is
// allowed between the recorded input codes and the W27 accumulation.
py::array_t<uint16_t> qmma_tiles(py::array_t<uint8_t,py::array::c_style> a,
                                  py::array_t<uint8_t,py::array::c_style> b,
                                  py::array_t<uint16_t,py::array::c_style> seed,
                                  int threads) {
    if (a.ndim()!=3 || b.ndim()!=3 || seed.ndim()!=3 ||
        a.shape(0)!=b.shape(0) || a.shape(0)!=seed.shape(0) ||
        a.shape(1)!=16 || a.shape(2)!=32 || b.shape(1)!=32 || b.shape(2)!=8 ||
        seed.shape(1)!=16 || seed.shape(2)!=8)
        throw std::invalid_argument("QMMA tile geometry must be [batch,16,32], [batch,32,8], [batch,16,8]");
    const size_t batch=size_t(a.shape(0));
    py::array_t<uint16_t> out({a.shape(0),py::ssize_t(16),py::ssize_t(8)});
    auto* aa=a.data(); auto* bb=b.data(); auto* cc=seed.data(); auto* dst=out.mutable_data();
    threads=std::max(1,std::min(threads,12));
    std::atomic<size_t> next{0}; std::atomic<bool> fail{false};
    std::exception_ptr ep; std::mutex error_lock;
    {
        py::gil_scoped_release release;
        auto work=[&]() {
            try {
                while (!fail) {
                    size_t tile=next.fetch_add(1); if(tile>=batch) break;
                    for(int row=0;row<16;++row) for(int col=0;col<8;++col) {
                        uint8_t av[32],bv[32];
                        for(int k=0;k<32;++k) {
                            av[k]=aa[tile*512+row*32+k];
                            bv[k]=bb[tile*256+k*8+col];
                        }
                        dst[tile*128+row*8+col]=qmma_dot(av,bv,cc[tile*128+row*8+col]);
                    }
                }
            } catch (...) {
                fail=true; std::lock_guard<std::mutex> guard(error_lock);
                if(!ep) ep=std::current_exception();
            }
        };
        std::vector<std::thread> pool;
        for(int i=1;i<threads;++i) pool.emplace_back(work);
        work(); for(auto& t:pool)t.join();
    }
    if(ep)std::rethrow_exception(ep);
    return out;
}
} // namespace
py::bytes execute(py::dict wiring,py::dict maps,py::bytes input,py::bytes weights,py::dict contract,py::array_t<float> rsq,py::array_t<float> rcp,size_t output_len,int threads,uint32_t queue_seed){return execute_prepared(nullptr,wiring,maps,input,weights,contract,rsq,rcp,output_len,threads,queue_seed);}
