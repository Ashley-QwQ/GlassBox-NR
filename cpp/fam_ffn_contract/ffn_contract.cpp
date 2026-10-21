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

struct Map {
    std::string name;
    py::array holder;
    const int32_t* data;
    size_t nc;
    int32_t at(size_t c, int z, int warp, int lane) const {
        return data[(((c * 4 + z) * 4 + warp) * 32 + lane)];
    }
};

enum class Op { Read, Mma, Cat, C16, Part, Dq, Sat, Add, Mul };
struct Node {
    Op op;
    int map=-1, ep=0, slot=0, part=0;
    uint16_t constant=0;
    std::vector<Node> child;
};
struct Step {
    int ep, slot;
    std::vector<Node> a,b,c;
};
struct Store {
    std::string kind;
    int map;
    Node value;
};
struct Release {int z;std::string region;size_t offset;int32_t value;int count;};
struct Stage { std::vector<Step> wiring; std::vector<Store> stores; };
struct Bytes { std::array<uint8_t,16> data{}; int n=0; };

struct Program {
    std::vector<Map> maps;
    std::unordered_map<std::string,int> names;
    std::array<std::array<Stage,3>,4> stages; // native live / upper live (dead) / masked edge
    std::vector<Release> releases;
    size_t nc=0;
    int ny=0; size_t tokens=0, bands=0;
    size_t input_len=0, residual_len=0, weights_len=0, buffer_len=0, output_len=0;
    size_t release60_len=0,release61_len=0;
};

int get_map(const Program& p, const std::string& key) {
    auto it=p.names.find(key);
    if (it==p.names.end()) throw std::invalid_argument("missing address map: "+key);
    return it->second;
}
std::string map_key(const std::string& kind,int ep,int slot) {
    return kind+"|"+std::to_string(ep)+"|"+std::to_string(slot);
}

Node parse_node(const py::handle& obj, const Program& p) {
    py::list x=py::cast<py::list>(obj);
    std::string kind=py::cast<std::string>(x[0]);
    Node n{};
    if (kind=="sread" || kind=="gload" || kind=="g32" || kind=="cg" || kind=="bufload") {
        n.op=Op::Read;
        n.ep=py::cast<int>(x[1]); n.slot=py::cast<int>(x[2]);
        n.part=kind=="g32" ? -1 : py::cast<int>(x[3]);
        n.map=get_map(p,map_key(kind=="sread" ? "src" : kind,n.ep,n.slot));
    } else if (kind=="mma") {
        n.op=Op::Mma; n.ep=py::cast<int>(x[1]); n.slot=py::cast<int>(x[2]); n.part=py::cast<int>(x[3]);
        if(n.ep<0||n.ep>=16||n.slot<0||n.slot>=64||n.part<0||n.part>=2) throw std::invalid_argument("invalid MMA node");
    } else if (kind=="cat") {
        n.op=Op::Cat;
        for(size_t i=1;i<x.size();++i)n.child.push_back(parse_node(x[i],p));
    } else if (kind=="c16") {
        n.op=Op::C16;
        n.constant=half_round(py::isinstance<py::str>(x[1])
            ? std::stod(py::cast<std::string>(x[1])) : py::cast<double>(x[1]));
    } else if (kind=="part") {
        n.op=Op::Part; n.child.push_back(parse_node(x[1],p)); n.part=py::cast<int>(x[2]);
    } else if (kind=="dq"||kind=="sat") {
        n.op=kind=="dq"?Op::Dq:Op::Sat; n.child.push_back(parse_node(x[1],p));
    } else if (kind=="fop") {
        std::string sub=py::cast<std::string>(x[1]);
        n.op=sub=="add"?Op::Add:sub=="mul"?Op::Mul:throw std::invalid_argument("unknown fop");
        n.child.push_back(parse_node(x[2],p));n.child.push_back(parse_node(x[3],p));
    } else throw std::invalid_argument("unknown template node: "+kind);
    return n;
}

// Native geometry and copy/store predicates, independent of supplied map signs.
// The descriptor stores whole 32-token rows; this round admits aligned T only.
int native_template_class(const Program& p,size_t c,int warp) {
    if(warp<2)return 0;
    return 128*(c%p.bands)+64>=p.tokens ? 2 : 1;
}
bool native_zero_source(const Program& p,size_t c,int warp,int slot) {
    return 8*(c%p.bands)+4*(warp/2)+slot/2 >= 2*((p.tokens+31)/32);
}
void validate_native_maps(const Program& p) {
    if(p.maps.size()!=512)throw std::invalid_argument("native event set size");
    const int qoff[8]={0,512,1024,1536,32768,33280,33792,34304};
    for(const auto& m:p.maps){
        auto a=m.name.find('|'),b=m.name.find('|',a+1);
        if(a==std::string::npos||b==std::string::npos)throw std::invalid_argument("native event key");
        std::string kind=m.name.substr(0,a);
        int ep=std::stoi(m.name.substr(a+1,b-a-1)),j=std::stoi(m.name.substr(b+1));
        if(m.name!=map_key(kind,ep,j))throw std::invalid_argument("noncanonical event key");
        bool valid=(kind=="gload"&&ep>=-1&&ep<15&&j>=0&&j<8)||
          ((kind=="sread"||kind=="src")&&ep>=0&&ep<16&&j>=0&&j<8)||
          (kind=="cg"&&ep==-1&&j>=0&&j<8)||(kind=="g32"&&ep==-1&&j>=0&&j<64)||
          ((kind=="bufstore"||kind=="red"||kind=="bufload")&&ep==15&&j>=0&&j<16)||
          (kind=="outstore"&&ep==15&&j>=0&&j<8);
        if(!valid)throw std::invalid_argument("unknown native event: "+m.name);
        for(size_t c=0;c<p.nc;++c)for(int z=0;z<4;++z)for(int y=0;y<4;++y)for(int l=0;l<32;++l){
            int64_t g=c/p.bands,k=c%p.bands,e=-1;
            bool live=128*k+64*(y/2)<int64_t(p.tokens);
            if(kind=="gload")e=1048576*z+65536*(ep+1)+4096*g+2048*(y%2)+qoff[j]+16*l;
            else if(kind=="sread")e=8192*(ep%2)+4096*(y/2)+1024*(j/2)+512*(j%2)+16*l;
            else if(kind=="src"){
                if(!native_zero_source(p,c,y,j))e=524288*k+262144*(y/2)+65536*(j/2)+16384*z+1024*ep+512*(j%2)+16*l;
            }else if(kind=="g32"){
                if(z==0)e=4194304+4*(64*g+32*(y%2)+4*((j%16)/2)+l%4);
            }else if(kind=="cg"||kind=="outstore"){
                if(live&&z==(kind=="cg"?0:3))e=131072*k+65536*(y/2)+2048*g+1024*(y%2)+16384*(j/2)+512*(j%2)+16*l;
            }else{
                bool stage=kind=="bufstore"?z==0:kind=="red"?(z==1||z==2):z==3;
                if(live&&stage)e=262144*k+131072*(y/2)+4096*g+2048*(y%2)+32768*(j/4)+512*(j%4)+16*l;
            }
            if(m.at(c,z,y,l)!=e)throw std::invalid_argument("native map predicate/address mismatch: "+m.name);
        }
    }
}
py::object native_zero_residual_graph(const py::handle& obj){
    if(!py::isinstance<py::list>(obj))return py::reinterpret_borrow<py::object>(obj);
    py::list x=py::cast<py::list>(obj);
    if(x.size()==2&&py::isinstance<py::str>(x[0])&&py::cast<std::string>(x[0])=="dq"&&py::isinstance<py::list>(x[1])){
        py::list a=py::cast<py::list>(x[1]);
        if(a.size()==3&&py::isinstance<py::str>(a[0])&&py::cast<std::string>(a[0])=="part"&&py::isinstance<py::list>(a[1])){
            py::list b=py::cast<py::list>(a[1]);
            if(b.size()==4&&py::cast<std::string>(b[0])=="cg"){
                py::list c1,c2,cat,sat,dq;
                c1.append("c16");c1.append("0.0");c2.append("c16");c2.append("0.0");
                cat.append("cat");cat.append(c1);cat.append(c2);sat.append("sat");sat.append(cat);dq.append("dq");dq.append(sat);return dq;
            }
        }
    }
    py::list out;for(auto v:x)out.append(native_zero_residual_graph(v));return out;
}

Program parse_program(const py::dict& templates,const py::dict& arrays,
                      size_t input_len,size_t residual_len,size_t weights_len,
                      size_t buffer_len,size_t output_len) {
    Program p;
    p.input_len=input_len;p.residual_len=residual_len;p.weights_len=weights_len;
    p.buffer_len=buffer_len;p.output_len=output_len;
    for(auto item: arrays) {
        std::string key=py::cast<std::string>(item.first);
        py::array a=py::cast<py::array>(item.second);
        if(a.ndim()!=4||a.shape(1)!=4||a.shape(2)!=4||a.shape(3)!=32||a.dtype().kind()!='i'||a.itemsize()!=4 || !(a.flags()&py::array::c_style))
            throw std::invalid_argument("map must be contiguous int32 [NC,4,4,32]: "+key);
        if(p.nc==0)p.nc=size_t(a.shape(0));
        if(p.nc!=size_t(a.shape(0)))throw std::invalid_argument("inconsistent CTA count");
        p.names.emplace(key,int(p.maps.size()));
        p.maps.push_back(Map{key,a,static_cast<const int32_t*>(a.data()),p.nc});
    }
    if(!p.nc||p.nc>10000)throw std::invalid_argument("invalid CTA count");
    py::dict regions=py::cast<py::dict>(templates["regions"]);
    p.tokens=py::cast<size_t>(py::cast<py::list>(regions["input"])[1])/4096;
    p.bands=(p.tokens+127)/128;
    if(!p.tokens||p.tokens%32||p.nc!=8*p.bands||
       py::cast<size_t>(py::cast<py::list>(regions["input"])[1])!=4096*p.tokens||
       py::cast<size_t>(py::cast<py::list>(regions["residual"])[1])!=1024*p.tokens||
       py::cast<size_t>(py::cast<py::list>(regions["output"])[1])!=1024*p.tokens||
       py::cast<size_t>(py::cast<py::list>(regions["buffer"])[1])!=2048*p.tokens)
        throw std::invalid_argument("native descriptor geometry not admitted");
    if((input_len&&input_len!=4096*p.tokens)||(residual_len&&residual_len!=1024*p.tokens)||
       (weights_len&&weights_len!=4196352)||(buffer_len&&buffer_len!=2048*p.tokens)||
       (output_len&&output_len!=1024*p.tokens))throw std::invalid_argument("native role extent mismatch");
    validate_native_maps(p);
    py::dict ts=py::cast<py::dict>(templates["templates"]);
    for(int z=0;z<4;++z)for(int cls=0;cls<3;++cls) {
        const bool has_edge=128*p.bands-p.tokens>=64;
        std::string suffix=cls==0?"live":cls==1?"dead":has_edge?"dead_edge":"live";
        std::string key=std::to_string(z)+"|"+suffix;
        py::dict t=py::cast<py::dict>(ts[py::str(key)]);
        py::dict live=py::cast<py::dict>(ts[py::str(std::to_string(z)+"|live")]);
        py::object expected=(cls==2&&has_edge&&z==0)?native_zero_residual_graph(live["wiring"]):py::reinterpret_borrow<py::object>(live["wiring"]);
        if(!py::cast<bool>(expected.attr("__eq__")(t["wiring"])))throw std::invalid_argument("native masked template wiring mismatch");
        if(cls==2&&has_edge){if(py::len(t["stores"])!=0)throw std::invalid_argument("native masked edge emits stores");}
        else if(!py::cast<bool>(live["stores"].attr("__eq__")(t["stores"])))throw std::invalid_argument("native live stores mismatch");
        Stage& stage=p.stages[z][cls];
        for(auto entry:py::cast<py::list>(t["wiring"])) {
            py::list w=py::cast<py::list>(entry);
            Step s{};s.ep=py::cast<int>(w[0]);s.slot=py::cast<int>(w[1]);
            if(s.ep<0||s.ep>=16||s.slot<0||s.slot>=64)throw std::invalid_argument("invalid wiring slot");
            for(auto n:py::cast<py::list>(w[3]))s.a.push_back(parse_node(n,p));
            for(auto n:py::cast<py::list>(w[4]))s.b.push_back(parse_node(n,p));
            for(auto n:py::cast<py::list>(w[5]))s.c.push_back(parse_node(n,p));
            stage.wiring.push_back(std::move(s));
        }
        for(auto entry:py::cast<py::list>(t["stores"])) {
            py::list w=py::cast<py::list>(entry);
            std::string kind=py::cast<std::string>(w[0]); int slot=py::cast<int>(w[1]);
            std::string suffix="|"+std::to_string(slot);
            int found=-1;
            for(size_t i=0;i<p.maps.size();++i)if(p.maps[i].name.starts_with(kind+"|")&&p.maps[i].name.ends_with(suffix)) {
                if(found>=0)throw std::invalid_argument("ambiguous store map: "+kind+suffix);
                found=int(i);
            }
            if(found<0)throw std::invalid_argument("missing store map: "+kind+suffix);
            stage.stores.push_back(Store{kind,found,parse_node(w[3],p)});
        }
    }
    p.release60_len=py::cast<size_t>(py::cast<py::list>(regions["release60"])[1]);
    p.release61_len=py::cast<size_t>(py::cast<py::list>(regions["release61"])[1]);
    for(auto obj:py::cast<py::list>(templates["releases"])){
        py::dict e=py::cast<py::dict>(obj);
        Release r{py::cast<int>(e["z"]),py::cast<std::string>(e["region"]),
                  py::cast<size_t>(e["offset"]),py::cast<int32_t>(e["value"]),py::cast<int>(e["count"])};
        if(r.z<0||r.z>3||r.count!=32||r.offset%4||
           (r.region=="release60"?r.offset+4>p.release60_len:
            r.region=="release61"?r.offset+4>p.release61_len:true))
            throw std::invalid_argument("release record outside expected role");
        p.releases.push_back(r);
    }
    p.ny=4; // All four physical warps execute; native template selects masked stores.
    return p;
}

uint16_t get16(const uint8_t* b){return uint16_t(b[0])|(uint16_t(b[1])<<8);}
void put16(uint8_t* b,uint16_t h){b[0]=uint8_t(h);b[1]=uint8_t(h>>8);}

struct Context {
    const Program& p;
    const uint8_t *input,*residual,*weights;
    uint8_t *buffer,*output,*written,*cover;
    size_t cta;
    int z,warp,lane;
    std::vector<std::array<uint8_t,32*8>>& mma;
};

Bytes evaluate(const Node& n,Context& c){
    Bytes out{};
    switch(n.op){
    case Op::Read: {
        const Map& m=c.p.maps[n.map];
        int32_t off=m.at(c.cta,c.z,c.warp,c.lane);
        size_t width=n.part==-1?4:16;
        const uint8_t* base=nullptr;size_t len=0;
        if(m.name.starts_with("src|")){base=c.input;len=c.p.input_len;}
        else if(m.name.starts_with("cg|")){base=c.residual;len=c.p.residual_len;}
        else if(m.name.starts_with("bufload|")){base=c.buffer;len=c.p.buffer_len;}
        else {base=c.weights;len=c.p.weights_len;}
        if(off==-1&&m.name.starts_with("src|")&&native_zero_source(c.p,c.cta,c.warp,n.slot)){
            out.n=4; // Native masked copy writes positive zero into shared memory.
            break;
        }
        if(off<0||uint64_t(off)+width>len)throw std::out_of_range("mapped load outside role: "+m.name);
        if(m.name.starts_with("bufload|"))for(size_t i=0;i<width;++i)if(!c.written[off+i])throw std::runtime_error("buffer read before write");
        out.n=4;std::memcpy(out.data.data(),base+off+(n.part==-1?0:4*n.part),4);
        break;
    }
    case Op::Mma: {
        out.n=4;
        const auto& tile=c.mma[size_t(n.ep*64+n.slot)];
        std::memcpy(out.data.data(),tile.data()+c.lane*8+n.part*4,4);
        break;
    }
    case Op::Cat: {
        for(const auto& child:n.child){Bytes part=evaluate(child,c);if(out.n+part.n>16)throw std::runtime_error("cat longer than 16 bytes");
            std::memcpy(out.data.data()+out.n,part.data.data(),part.n);out.n+=part.n;}
        break;
    }
    case Op::C16:out.n=2;put16(out.data.data(),n.constant);break;
    case Op::Part: {
        Bytes x=evaluate(n.child[0],c);if(2*n.part+2>x.n)throw std::runtime_error("part outside node");
        out.n=2;std::memcpy(out.data.data(),x.data.data()+2*n.part,2);break;
    }
    case Op::Dq: {
        Bytes x=evaluate(n.child[0],c);out.n=x.n*2;
        if(out.n>16)throw std::runtime_error("decode too wide");
        for(int i=0;i<x.n;++i)put16(out.data.data()+2*i,e4_half(x.data[i]));break;
    }
    case Op::Sat: {
        Bytes x=evaluate(n.child[0],c);if(x.n&1)throw std::runtime_error("odd half width");
        out.n=x.n/2;for(int i=0;i<out.n;++i)out.data[i]=sat_e4(get16(x.data.data()+2*i));break;
    }
    case Op::Add:case Op::Mul: {
        Bytes a=evaluate(n.child[0],c),b=evaluate(n.child[1],c);
        if(a.n!=b.n||(a.n&1))throw std::runtime_error("fop shape mismatch");
        out.n=a.n;
        for(int i=0;i<a.n;i+=2){uint16_t x=get16(a.data.data()+i),y=get16(b.data.data()+i);
            uint16_t result=n.op==Op::Add?h_add(x,y):h_mul(x,y);
            if((result&0x7c00)==0x7c00)throw std::domain_error("half fop overflow");
            put16(out.data.data()+i,result);}
        break;
    }
    }
    return out;
}

void calculate_cta(const Program& p,size_t cta,const uint8_t* input,const uint8_t* residual,const uint8_t* weights,
                   uint8_t* buffer,uint8_t* output,uint8_t* written,uint8_t* cover,bool reverse_red){
    std::array<std::vector<std::array<uint8_t,32*8>>,4> tiles;
    for(auto& v:tiles)v.resize(16*64);
    const int stage_order[4]={0,reverse_red?2:1,reverse_red?1:2,3};
    for(int stage_index=0;stage_index<4;++stage_index){
        int z=stage_order[stage_index];
        for(int warp=0;warp<p.ny;++warp){
            const Stage& stage=p.stages[z][native_template_class(p,cta,warp)];
            Context c{p,input,residual,weights,buffer,output,written,cover,cta,z,warp,0,tiles[warp]};
            for(const auto& step:stage.wiring){
                std::array<uint8_t,16*32> aa{};std::array<uint8_t,32*8> bb{};std::array<uint16_t,16*8> cc{};
                for(int lane=0;lane<32;++lane){
                    c.lane=lane;Bytes a{},b{},seed{};
                    for(const auto& n:step.a){Bytes v=evaluate(n,c);if(a.n+v.n>16)throw std::runtime_error("A lane too wide");std::memcpy(a.data.data()+a.n,v.data.data(),v.n);a.n+=v.n;}
                    for(const auto& n:step.b){Bytes v=evaluate(n,c);if(b.n+v.n>8)throw std::runtime_error("B lane too wide");std::memcpy(b.data.data()+b.n,v.data.data(),v.n);b.n+=v.n;}
                    for(const auto& n:step.c){Bytes v=evaluate(n,c);if(seed.n+v.n>8)throw std::runtime_error("C lane too wide");std::memcpy(seed.data.data()+seed.n,v.data.data(),v.n);seed.n+=v.n;}
                    if(a.n!=16||b.n!=8||seed.n!=8)throw std::runtime_error("QMMA lane shape");
                    int g=lane/4,t=lane%4;
                    for(int r=0;r<4;++r)for(int sb=0;sb<4;++sb)aa[(g+8*(r%2))*32+16*(r/2)+4*t+sb]=a.data[4*r+sb];
                    for(int r=0;r<2;++r)for(int sb=0;sb<4;++sb)bb[(16*r+4*t+sb)*8+g]=b.data[4*r+sb];
                    for(int r=0;r<2;++r)for(int h=0;h<2;++h)cc[(g+8*r)*8+2*t+h]=get16(seed.data.data()+2*(2*r+h));
                }
                std::array<uint16_t,16*8> dd{};
                for(int row=0;row<16;++row)for(int col=0;col<8;++col){
                    uint8_t av[32],bv[32];for(int k=0;k<32;++k){av[k]=aa[row*32+k];bv[k]=bb[k*8+col];}
                    dd[row*8+col]=qmma_dot(av,bv,cc[row*8+col]);
                }
                auto& tile=c.mma[size_t(step.ep*64+step.slot)];
                for(int lane=0;lane<32;++lane){int g=lane/4,t=lane%4;
                    for(int r=0;r<2;++r)for(int h=0;h<2;++h)put16(tile.data()+lane*8+2*(2*r+h),dd[(g+8*r)*8+2*t+h]);}
            }
            for(const auto& s:stage.stores){
                const Map& m=p.maps[s.map];
                for(int lane=0;lane<32;++lane){
                    c.lane=lane;int32_t off=m.at(cta,z,warp,lane);Bytes val=evaluate(s.value,c);
                    if(off<0||val.n!=16)throw std::runtime_error("store map or width invalid");
                    if(s.kind=="outstore"){
                        if(z!=3||uint64_t(off)+16>p.output_len)throw std::out_of_range("output store");
                        for(int i=0;i<16;++i)if(cover[off+i]++)throw std::runtime_error("duplicate output owner");
                        std::memcpy(output+off,val.data.data(),16);
                    }else if(s.kind=="bufstore"){
                        if(z!=0||uint64_t(off)+16>p.buffer_len)throw std::out_of_range("buffer store");
                        for(int i=0;i<16;++i)if(written[off+i]++)throw std::runtime_error("duplicate buffer initialization");
                        std::memcpy(buffer+off,val.data.data(),16);
                    }else if(s.kind=="red"){
                        if(z!=1&&z!=2)throw std::runtime_error("REDG stage invalid");
                        if(uint64_t(off)+16>p.buffer_len)throw std::out_of_range("REDG store");
                        for(int i=0;i<16;i+=2){if(!written[off+i]||!written[off+i+1])throw std::runtime_error("REDG on unwritten buffer");
                            uint16_t lhs=get16(buffer+off+i),rhs=get16(val.data.data()+i);
                            uint16_t sum=h_add(lhs,rhs);
                            if((sum&0x7c00)==0x7c00)throw std::domain_error("REDG half overflow");
                            put16(buffer+off+i,sum);}
                    }else throw std::runtime_error("unknown store kind");
                }
            }
        }
    }
}

py::dict execute_contract(py::dict templates,py::dict maps,py::bytes py_input,py::bytes py_residual,
                          py::bytes py_weights,size_t buffer_len,size_t output_len,int threads,uint32_t queue_seed,
                          bool reverse_red){
    std::string input=py_input,residual=py_residual,weights=py_weights;
    Program p=parse_program(templates,maps,input.size(),residual.size(),weights.size(),buffer_len,output_len);
    std::vector<uint8_t> buffer(buffer_len),output(output_len),written(buffer_len),cover(output_len);
    std::vector<size_t> order(p.nc);std::iota(order.begin(),order.end(),0);
    if(queue_seed){uint32_t x=queue_seed;for(size_t i=order.size();i>1;--i){x=x*1664525u+1013904223u;std::swap(order[i-1],order[x%i]);}}
    threads=std::clamp(threads,1,12);
    std::atomic<size_t> next{0};std::atomic<bool> stop{false};std::exception_ptr error;std::mutex lock;
    {
        py::gil_scoped_release release;
        auto work=[&](){try{while(!stop){size_t i=next.fetch_add(1);if(i>=order.size())break;
                    calculate_cta(p,order[i],reinterpret_cast<const uint8_t*>(input.data()),
                        reinterpret_cast<const uint8_t*>(residual.data()),reinterpret_cast<const uint8_t*>(weights.data()),
                        buffer.data(),output.data(),written.data(),cover.data(),reverse_red);
                }}catch(...){stop=true;std::lock_guard<std::mutex> g(lock);if(!error)error=std::current_exception();}};
        std::vector<std::thread> pool;for(int i=1;i<threads;++i)pool.emplace_back(work);
        work();for(auto& t:pool)t.join();
    }
    if(error)std::rethrow_exception(error);
    if(std::any_of(cover.begin(),cover.end(),[](uint8_t x){return x!=1;}))throw std::runtime_error("output coverage not exactly once");
    std::vector<uint8_t> release60(p.release60_len),release61(p.release61_len);
    std::vector<uint8_t> release60_seen(p.release60_len/4),release61_seen(p.release61_len/4);
    for(int z=0;z<4;++z)for(const auto& r:p.releases)if(r.z==z){
        auto& target=r.region=="release60"?release60:release61;
        auto& seen=r.region=="release60"?release60_seen:release61_seen;
        ++seen[r.offset/4];
        uint32_t u=uint32_t(r.value);
        for(int i=0;i<4;++i)target[r.offset+i]=uint8_t(u>>(8*i));
    }
    if(std::any_of(release60_seen.begin(),release60_seen.end(),[](uint8_t x){return x!=4;})||
       std::any_of(release61_seen.begin(),release61_seen.end(),[](uint8_t x){return x!=1;}))
        throw std::runtime_error("release publication coverage mismatch");
    py::dict roles;
    roles["output"]=py::bytes(reinterpret_cast<const char*>(output.data()),output.size());
    roles["buffer"]=py::bytes(reinterpret_cast<const char*>(buffer.data()),buffer.size());
    roles["release60"]=py::bytes(reinterpret_cast<const char*>(release60.data()),release60.size());
    roles["release61"]=py::bytes(reinterpret_cast<const char*>(release61.data()),release61.size());
    roles["cta_count"]=p.nc;roles["live_warps"]=p.ny;
    return roles;
}
} // namespace

PYBIND11_MODULE(dlssnr_fam_ffn_contract,m){
    m.doc()="Portable scalar FFN contract map executor with fixed-order REDG";
    m.def("inspect_program",[](py::dict templates,py::dict maps){
        Program p=parse_program(templates,maps,0,0,0,0,0);
        py::dict info;
        info["cta_count"]=p.nc;info["live_warps"]=p.ny;info["map_count"]=p.maps.size();
        py::list steps,stores;
        for(const auto& group:p.stages){steps.append(group[0].wiring.size());stores.append(group[0].stores.size());}
        info["wiring_counts"]=steps;info["store_counts"]=stores;
        py::list routes;size_t masked=0;
        for(size_t c=0;c<p.nc;++c)for(int y=0;y<4;++y){int cls=native_template_class(p,c,y);routes.append(py::make_tuple(c,y,cls));if(cls==2)++masked;}
        info["native_template_routes"]=routes;info["masked_edge_warps"]=masked;
        info["tokens"]=p.tokens;info["source_policy"]="native predicate only; illegal negative map rejected";
        return info;
    });
    m.def("execute",&execute_contract,py::arg("templates"),py::arg("maps"),py::arg("input"),
          py::arg("residual"),py::arg("weights"),py::arg("buffer_len"),py::arg("output_len"),
          py::arg("threads")=1,py::arg("queue_seed")=0,py::arg("reverse_red")=false);
    m.def("qmma_dot",[](py::bytes a,py::bytes b,uint16_t seed){std::string x=a,y=b;
        if(x.size()!=32||y.size()!=32)throw std::invalid_argument("QMMA vectors must have 32 codes");
        return qmma_dot(reinterpret_cast<const uint8_t*>(x.data()),reinterpret_cast<const uint8_t*>(y.data()),seed);});
}
