#include "../geometry.hpp"
// Scalar primitives and map evaluator copied and extended from step2a.cpp.
// Exact source ranges and SHA are emitted by prepare.py into s01-bindings.json.
#include "scalar_map.hpp"
namespace {
py::dict* launch_specs=nullptr;
py::array_t<float>* table_rsq=nullptr;
py::array_t<float>* table_rcp=nullptr;
uint64_t workspace_base=0;
uint32_t optg_queue_seed=0;
uint64_t qword(const std::string& p,int i) {
    if(i<0 || size_t(i)*8+8>p.size())throw std::invalid_argument("params qword outside bytes");
    uint64_t v=0; for(int k=0;k<8;++k)v|=uint64_t(uint8_t(p[size_t(i)*8+k]))<<(8*k); return v;
}
constexpr size_t MAX_METADATA=8ULL<<20,MAX_MAPS=512ULL<<20;
struct Role {int word;size_t bytes;};
struct Plan {int launch;std::string identity,params;Spec spec;py::dict maps;std::vector<Role> inputs,outputs,releases;int weight_word;size_t weight_bytes,output_bytes,map_bytes=0;};
std::unique_ptr<Plan> current;
struct Reader {
 const std::string& b;size_t p=0;
 void need(size_t n){if(n>b.size()-p)throw std::invalid_argument("truncated precompiled metadata");}
 std::string bytes(size_t n){need(n);auto v=b.substr(p,n);p+=n;return v;}
 uint32_t u32(){need(4);uint32_t v=0;for(int k=0;k<4;++k)v|=uint32_t(uint8_t(b[p++]))<<(8*k);return v;}
 int i32(){return portable_bit_cast<int32_t>(u32());}
 uint64_t u64(){uint64_t v=u32();return v|(uint64_t(u32())<<32);}
};
std::unique_ptr<Plan> decode(int launch,const std::string&metadata,py::dict maps,const std::string&params,const std::string&identity){
 if(metadata.size()<96||metadata.size()>MAX_METADATA||identity.size()!=32)throw std::invalid_argument("precompiled metadata/key bound");
 Reader r{metadata};if(r.bytes(8)!="IG1SPEC1"||r.u32()!=1)throw std::invalid_argument("precompiled metadata version");
 if(r.i32()!=launch)throw std::invalid_argument("precompiled launch mismatch");if(r.bytes(32)!=identity)throw std::invalid_argument("precompiled identity mismatch");
 std::array<uint32_t,7> counts;for(auto&x:counts)x=r.u32();auto nn=counts[0],nm=counts[1],ns=counts[2],nc=counts[3];
 uint64_t expected=96ULL+uint64_t(nn)*52+uint64_t(nm)*112+uint64_t(ns)*68+uint64_t(nc)*72+uint64_t(counts[4]+uint64_t(counts[5])+counts[6])*12;
 if(expected!=metadata.size()||nn>65536||nm>16384||ns>4096||nc>65536||counts[4]>1024||counts[5]>1024||counts[6]>1024)throw std::invalid_argument("precompiled count/length bound");
 auto p=std::make_unique<Plan>();p->launch=launch;p->identity=identity;p->params=params;p->maps=maps;p->weight_word=r.i32();p->weight_bytes=size_t(r.u64());p->output_bytes=size_t(r.u64());
 if(p->weight_bytes>MAX_MAPS||p->output_bytes>MAX_MAPS)throw std::invalid_argument("precompiled role byte bound");qword(params,p->weight_word);
 static const std::array<const char*,15> kinds{{"W","S","X","XH","K","E","Q","HB","B","SI","PS","MV","WI","F","M"}};
 static const std::array<const char*,12> ops{{"","add","mul","fma","min","max","abs","rsqrt","rcp","rne16","rne16w","shl"}};
 p->spec.nodes.reserve(nn);
 for(uint32_t i=0;i<nn;++i){auto kind=r.u32(),op=r.u32();Node n;n.a=r.i32();n.b=r.i32();n.c=r.i32();int d=r.i32();n.k=r.i32();auto arity=r.u32();std::array<int,4> args;for(auto&x:args)x=r.i32();
  if(kind>=kinds.size()||op>=ops.size()||d!=0||arity>4)throw std::invalid_argument("precompiled node opcode/arity");n.kind=kinds[kind];n.op=ops[op];
  for(size_t j=arity;j<4;++j)if(args[j]!=0)throw std::invalid_argument("precompiled node padding");
  if(n.kind=="F"){int need=op==3?3:op==10?4:(op==1||op==2||op==4||op==5)?2:(op==6||op==7||op==8||op==9)?1:0;if(!need||arity!=uint32_t(need))throw std::invalid_argument("precompiled F arity");n.args.assign(args.begin(),args.begin()+arity);}
  else if(arity|| (n.kind=="WI"?(op!=1&&op!=11):op!=0))throw std::invalid_argument("precompiled non-F opcode/arity");
  p->spec.nodes.push_back(std::move(n));
 }
 p->spec.mmas.resize(nm);for(auto&m:p->spec.mmas){for(auto&x:m.a)x=r.i32();for(auto&x:m.b)x=r.i32();for(auto&x:m.c)x=r.i32();}
 p->spec.contract.resize(nn);for(auto&x:p->spec.contract){x=r.i32();if(x< -1||x>1)throw std::invalid_argument("bad contraction");}
 p->spec.pcs.resize(ns);p->spec.stores.resize(ns);for(size_t j=0;j<ns;++j){p->spec.pcs[j]=r.i32();for(auto&x:p->spec.stores[j])x=r.i32();}
 p->spec.class_stores.resize(nc);for(auto&st:p->spec.class_stores){st.cls=r.i32();st.pc=r.i32();for(auto&x:st.atoms)x=r.i32();}
 auto roles=[&](std::vector<Role>&v,uint32_t n){v.reserve(n);for(uint32_t i=0;i<n;++i){int word=r.i32();uint64_t size=r.u64();if(size>MAX_MAPS)throw std::invalid_argument("precompiled role bound");qword(params,word);v.push_back({word,size_t(size)});}};
 roles(p->inputs,counts[4]);roles(p->outputs,counts[5]);roles(p->releases,counts[6]);
 if(r.p!=metadata.size())throw std::invalid_argument("precompiled trailing metadata");size_t total=0;for(auto&v:p->outputs){if(v.bytes>MAX_MAPS-total)throw std::invalid_argument("precompiled output sum bound");total+=v.bytes;}if(total!=p->output_bytes)throw std::invalid_argument("precompiled output role sum");
 auto node=[&](int i){if(i<0||size_t(i)>=nn)throw std::invalid_argument("precompiled node id");};
 for(auto&n:p->spec.nodes){if(n.kind=="E"||n.kind=="Q")node(n.a);else if(n.kind=="HB"){node(n.a);node(n.b);}else if(n.kind=="B"||n.kind=="SI")node(n.b);else if(n.kind=="PS"||n.kind=="MV"||n.kind=="WI"){node(n.b);node(n.c);}else if(n.kind=="F")for(int a:n.args)node(a);}
 for(auto&m:p->spec.mmas){for(int a:m.a)node(a);for(int a:m.b)node(a);for(int a:m.c)node(a);}for(auto&st:p->spec.stores)for(int a:st)node(a);for(auto&st:p->spec.class_stores)for(int a:st.atoms)node(a);
 static const std::array<const char*,8> fields{{"wload_off","sread_src","gstore_off","xsite_src","xsite_wsrc","shfl_idx","psel","thread_class"}};
 for(auto entry:maps){std::string name=py::str(entry.first);if(std::find(fields.begin(),fields.end(),name)==fields.end())throw std::invalid_argument("unknown precompiled map field");if(!py::isinstance<py::array>(entry.second))throw std::invalid_argument("precompiled map must be array");auto a=py::reinterpret_borrow<py::array>(entry.second);auto dtype=a.dtype();if(dtype.kind()!='i'||(a.itemsize()!=1&&a.itemsize()!=2&&a.itemsize()!=4&&a.itemsize()!=8)||!(a.flags()&py::array::c_style)||a.writeable())throw std::invalid_argument("precompiled map must be readonly contiguous signed i8/i16/i32/i64");if(size_t(a.nbytes())>MAX_MAPS-p->map_bytes)throw std::invalid_argument("precompiled map byte bound");p->map_bytes+=size_t(a.nbytes());}
 for(auto field:{"wload_off","sread_src","gstore_off"})if(!maps.contains(field))throw std::invalid_argument("missing precompiled map field");
 return p;
}
py::dict run_launch(int launch,py::bytearray workspace,py::bytes params,py::buffer weights,py::dict textures,int threads) {
    auto key=py::int_(launch);
    if(!launch_specs || !launch_specs->contains(key))throw std::invalid_argument("launch not configured for split16");
    py::dict sp=(*launch_specs)[key].cast<py::dict>();
    std::string p=params;geo2::parameters(launch,reinterpret_cast<const uint8_t*>(p.data()),p.size());std::string expected=sp["params"].cast<py::bytes>();
    if(p!=expected)throw std::invalid_argument("params differs from frozen 1080 frame0 binding");
    auto* ws=reinterpret_cast<uint8_t*>(PyByteArray_AsString(workspace.ptr()));
    size_t wsz=size_t(PyByteArray_Size(workspace.ptr()));
    if(workspace_base!=geo2::g().workspace_base||wsz!=geo2::g().workspace_bytes)throw std::invalid_argument("GEO2_SPLIT_WORKSPACE_DESCRIPTOR_DIFF");
    auto wb=weights.request(); if(wb.ndim!=1 || wb.itemsize!=1 || wb.strides[0]!=1)throw std::invalid_argument("weights must be flat contiguous bytes");
    auto check=[&](uint64_t ptr,size_t len){if(ptr<workspace_base||ptr-workspace_base>wsz||len>wsz-size_t(ptr-workspace_base))throw std::out_of_range("workspace role outside buffer");return size_t(ptr-workspace_base);};
    std::string input;
    for(auto x:sp["inputs"].cast<py::list>()) {auto r=py::cast<std::vector<int64_t>>(x);size_t n=size_t(r[1]),off=check(qword(p,int(r[0])),n);input.append(reinterpret_cast<const char*>(ws+off),n);}
    size_t weight_len=sp["weight_bytes"].cast<size_t>(); uint64_t ptr=qword(p,sp["weight_word"].cast<int>());
    if(ptr<geo2::g().weight_base||ptr-geo2::g().weight_base>size_t(wb.size)||weight_len>size_t(wb.size)-size_t(ptr-geo2::g().weight_base))throw std::out_of_range("weight slice outside WEIGHTS_HT device layout");
    py::bytes wt(static_cast<const char*>(wb.ptr)+size_t(ptr-geo2::g().weight_base),weight_len);
    py::bytes output=execute_prepared(current && current->launch==launch ? &current->spec : nullptr,sp["wiring"].cast<py::dict>(),sp["maps"].cast<py::dict>(),py::bytes(input),wt,sp["contract"].cast<py::dict>(),*table_rsq,*table_rcp,sp["output_bytes"].cast<size_t>(),threads,optg_queue_seed);
    std::string out=output; size_t cursor=0; py::list ranges;
    for(auto x:sp["outputs"].cast<py::list>()) {auto r=py::cast<std::vector<int64_t>>(x);size_t n=size_t(r[1]),off=check(qword(p,int(r[0])),n);if(n>out.size()-cursor)throw std::out_of_range("output role slicing"); std::memcpy(ws+off,out.data()+cursor,n);cursor+=n;ranges.append(py::make_tuple(off,n));}
    if(cursor!=out.size())throw std::runtime_error("unassigned output bytes");
    for(auto x:sp["releases"].cast<py::list>()) {auto r=py::cast<std::vector<int64_t>>(x);size_t n=size_t(r[1]),off=check(qword(p,int(r[0])),n);std::memset(ws+off,0,n);ranges.append(py::make_tuple(off,n));}
    py::dict result;result["launch"]=launch;result["written_ranges"]=ranges;result["textures_out"]=py::dict();return result;
}
}
PYBIND11_MODULE(dlssnr_chain_split16_optm,m) {
m.def("configure_geometry",&geo2::configure);m.def("geometry_probe",&geo2::probe);
    m.def("_configure_precompiled",[](int launch,py::bytes metadata,py::dict maps,py::bytes params,py::array_t<float> rsq,py::array_t<float> rcp,uint64_t base,py::bytes identity){
        auto next=decode(launch,std::string(metadata),maps,std::string(params),std::string(identity));
        py::dict sp;sp["params"]=params;sp["wiring"]=py::dict();sp["maps"]=maps;sp["contract"]=py::dict();
        auto roles=[](const std::vector<Role>& values){py::list out;for(auto&r:values)out.append(py::make_tuple(r.word,r.bytes));return out;};
        sp["inputs"]=roles(next->inputs);sp["outputs"]=roles(next->outputs);sp["releases"]=roles(next->releases);
        sp["weight_word"]=next->weight_word;sp["weight_bytes"]=next->weight_bytes;sp["output_bytes"]=next->output_bytes;
        py::dict specs;specs[py::int_(launch)]=sp;
        current.reset();delete launch_specs;delete table_rsq;delete table_rcp;
        launch_specs=new py::dict(specs);table_rsq=new py::array_t<float>(rsq);table_rcp=new py::array_t<float>(rcp);workspace_base=base;current=std::move(next);
    },py::arg("launch"),py::arg("metadata"),py::arg("maps"),py::arg("params"),py::arg("rsq"),py::arg("rcp"),py::arg("workspace_base"),py::arg("identity_key"));

    m.def("set_queue_seed",[](uint32_t seed){optg_queue_seed=seed;});
    m.def("audit_execute",&execute);
    m.def("audit_array_at",[](py::array a,int c,int y,int l,int s,int t){return Array(a).at(c,y,l,s,t);});

    // Capsule is released with the module while Python still owns the GIL,
    // never from a DLL static destructor after interpreter finalization.
    m.add_object("_state_cleanup",py::capsule(reinterpret_cast<void*>(1),[](void*){current.reset();delete launch_specs;delete table_rsq;delete table_rcp;launch_specs=nullptr;table_rsq=nullptr;table_rcp=nullptr;}));
    m.def("configure",[](py::dict specs,py::array_t<float> rsq,py::array_t<float> rcp,uint64_t base){current.reset();delete launch_specs;delete table_rsq;delete table_rcp;launch_specs=new py::dict(specs);table_rsq=new py::array_t<float>(rsq);table_rcp=new py::array_t<float>(rcp);workspace_base=base;});
    m.def("release_assets",[](){current.reset();delete launch_specs;delete table_rsq;delete table_rcp;launch_specs=nullptr;table_rsq=nullptr;table_rcp=nullptr;});
    m.def("run_launch",&run_launch,py::arg("launch"),py::arg("workspace"),py::arg("params"),py::arg("weights"),py::arg("textures"),py::arg("threads"));
    m.def("redg",[](uint16_t seed,std::vector<uint16_t> values){for(auto h:values)seed=h_add(seed,h);return seed;});
}
