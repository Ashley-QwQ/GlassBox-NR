// Independent attention executor. Original scalar helpers in attention.cpp:
// cpp/src/step2a.cpp lines 1-131, SHA256
// 91a28b28a9ca9771d70973fec37d772f60f0b9713a1d3c79267b309913d1110b.
struct Bytes32 {
    std::array<uint8_t, 512> b{};
    int width = 0;
    uint8_t at(int lane, int byte) const { return b[size_t(lane)*16+byte]; }
    void put(int lane, int byte, uint8_t x) { b[size_t(lane)*16+byte]=x; }
    uint16_t h(int lane, int part=0) const {
        return uint16_t(at(lane,2*part)) | (uint16_t(at(lane,2*part+1))<<8);
    }
    uint32_t u32(int lane) const {
        uint32_t v=0; for(int j=0;j<4;++j)v|=uint32_t(at(lane,j))<<(8*j); return v;
    }
    void hput(int lane, uint16_t v, int part=0) {
        put(lane,2*part,uint8_t(v)); put(lane,2*part+1,uint8_t(v>>8));
    }
    void uput(int lane, uint32_t v) {
        for(int j=0;j<4;++j)put(lane,j,uint8_t(v>>(8*j)));
    }
};
struct NodeA {
    std::string kind, op;
    ChainVector<int> args;
    int ep=0,slot=0,word=0,pc=0,idx=0,imm=0,bits=0;
    uint32_t raw=0;
};
struct WireA { int ep,slot; std::array<int,4> a; std::array<int,2> b,c; };
struct StoreA { int node; std::string mapkey; };
struct MapA {
    py::array owner;
    const int32_t* data;
    int nc,ny;
    explicit MapA(py::handle h) : owner(py::reinterpret_borrow<py::array>(h)),data(nullptr),nc(0),ny(0) {
        auto info=owner.request();
        if(info.ndim!=3||info.shape[0]!=32||info.shape[2]!=32||info.itemsize!=4||
           owner.dtype().kind()!='i'||!(owner.flags()&py::array::c_style))
            throw std::invalid_argument("attention map must be C-contiguous int32 [32,y,32]");
        data=static_cast<const int32_t*>(info.ptr);nc=int(info.shape[0]);ny=int(info.shape[1]);
    }
    int32_t at(int c,int y,int lane)const {
        if(c<0||c>=nc||y<0||y>=ny||lane<0||lane>=32)throw std::out_of_range("attention map index");
        return data[(c*ny+y)*32+lane];
    }
};
struct EpochA {
    ChainVector<NodeA> nodes;
    ChainVector<ChainVector<WireA>> wiring;
    ChainVector<std::array<ChainVector<StoreA>,32>> stores;
    std::unordered_map<std::string,int> gather;
    std::unordered_map<std::string,MapA> maps;
    int ny=0;
    size_t output_size=0,release63_size=0;
    MapA const& map(std::string const& name)const {
        auto it=maps.find(name);if(it==maps.end())throw std::out_of_range("map key missing: "+name);
        return it->second;
    }
};
int asint(py::handle h){return py::cast<int>(h);}
ChainVector<int> ints(py::handle h){ChainVector<int> out;for(auto x:py::reinterpret_borrow<py::sequence>(h))out.push_back(asint(x));return out;}
std::string key3(std::string const& k,int ep,int slot){return k+"|"+std::to_string(ep)+"|"+std::to_string(slot);}
std::string key4(int y,int pc,int ep,int idx){return std::to_string(y)+"|"+std::to_string(pc)+"|"+std::to_string(ep)+"|"+std::to_string(idx);}
EpochA parse_epoch(py::dict tm,py::dict mp) {
    if(!tm.contains("nodes")||!tm.contains("templates")||!tm.contains("gather_sources"))throw std::invalid_argument("template keys missing");
    EpochA e;
    for(auto v:tm["nodes"].cast<py::list>()) {
        py::list n=py::reinterpret_borrow<py::list>(v);NodeA x;x.kind=py::cast<std::string>(n[0]);auto k=x.kind;
        if(k=="leaf") {x.op=py::cast<std::string>(n[1]);x.ep=asint(n[2]);x.slot=asint(n[3]);x.word=asint(n[4]);}
        else if(k=="c16") {x.raw=uint32_t(half_round(std::stod(py::cast<std::string>(n[1]))));}
        else if(k=="ibits") {x.raw=py::cast<uint32_t>(n[1]);}
        else if(k=="cat") {for(size_t j=1;j<n.size();++j)x.args.push_back(asint(n[j]));}
        else if(k=="part") {x.args={asint(n[1])};x.idx=asint(n[2]);}
        else if(k=="fop") {x.op=py::cast<std::string>(n[1]);for(size_t j=2;j<n.size();++j)x.args.push_back(asint(n[j]));}
        else if(k=="ishl"||k=="iadd") {x.bits=asint(n[1]);x.args={asint(n[2])};x.imm=asint(n[3]);}
        else if(k=="ilow16"||k=="tof32"||k=="rcp"||k=="f16of"||k=="sat") {x.args={asint(n[1])};}
        else if(k=="prmt") {x.imm=asint(n[1]);x.args={asint(n[2]),asint(n[3])};}
        else if(k=="gather") {x.pc=asint(n[1]);x.idx=asint(n[2]);x.ep=asint(n[3]);}
        else if(k=="fmulc") {x.args={asint(n[1])};x.raw=py::cast<uint32_t>(n[2]);}
        else throw std::invalid_argument("unknown attention node: "+k);
        e.nodes.push_back(std::move(x));
    }
    py::dict gs=tm["gather_sources"].cast<py::dict>();e.gather.reserve(gs.size());
    for(auto kv:gs)e.gather.emplace(py::cast<std::string>(kv.first),asint(kv.second));
    for(auto kv:mp)e.maps.emplace(py::cast<std::string>(kv.first),MapA(kv.second));
    if(e.maps.empty())throw std::invalid_argument("empty attention maps");
    py::dict regions=tm["regions"].cast<py::dict>();
    e.output_size=size_t(asint(regions["output"].cast<py::list>()[1]));
    e.release63_size=size_t(asint(regions["release63"].cast<py::list>()[1]));
    if(e.release63_size==0||e.release63_size>e.output_size)throw std::invalid_argument("release63 extent");
    for(auto ri:tm["releases"].cast<py::list>()){
        py::dict r=py::reinterpret_borrow<py::dict>(ri);
        if(py::cast<std::string>(r["region"])!="release63"||asint(r["value"])!=0)
            throw std::invalid_argument("unmodelled attention release");
    }
    e.ny=e.maps.begin()->second.ny;
    if(e.ny<1||e.ny>16)throw std::invalid_argument("unexpected attention y count");
    for(auto const& kv:e.maps)if(kv.second.ny!=e.ny)throw std::invalid_argument("mixed map geometry");
    e.wiring.resize(size_t(e.ny));e.stores.resize(size_t(e.ny));
    py::dict ts=tm["templates"].cast<py::dict>();
    for(int y=0;y<e.ny;++y) {
        int ty=ts.contains(py::str(std::to_string(y)+"|0"))?y:0;
        py::dict t0=ts[py::str(std::to_string(ty)+"|0")].cast<py::dict>();
        for(auto wi:t0["wiring"].cast<py::list>()) {
            py::list w=py::reinterpret_borrow<py::list>(wi);
            auto a=ints(w[3]),b=ints(w[4]),c=ints(w[5]);
            if(a.size()!=4||b.size()!=2||c.size()!=2)throw std::invalid_argument("attention QMMA wiring width");
            e.wiring[y].push_back({asint(w[0]),asint(w[1]),{a[0],a[1],a[2],a[3]},{b[0],b[1]},{c[0],c[1]}});
        }
        for(int lane=0;lane<32;++lane) {
            py::dict t=ts[py::str(std::to_string(ty)+"|"+std::to_string(lane))].cast<py::dict>();
            if(PyObject_RichCompareBool(t["wiring"].ptr(),t0["wiring"].ptr(),Py_EQ)!=1)
                throw std::invalid_argument("lane QMMA wiring differs");
            for(auto si:t["stores"].cast<py::list>()) {
                py::list s=py::reinterpret_borrow<py::list>(si);
                if(py::cast<std::string>(s[0])!="ostore")throw std::invalid_argument("non-output store");
                e.stores[y][lane].push_back({asint(s[4]),key3("ostore",asint(s[1]),asint(s[2]))});
            }
        }
    }
    return e;
}

struct WorkerA {
    EpochA const& e;
    std::array<std::string,3> const& input;
    uint32_t const* rcp;
    size_t rcp_n;
    ChainVector<Bytes32> memo;
    ChainVector<uint8_t> seen;
    std::unordered_map<uint64_t,Bytes32> mma_out;
    int c=0,y=0;
    WorkerA(EpochA const& ep,std::array<std::string,3> const& in,uint32_t const* r,size_t rn)
      :e(ep),input(in),rcp(r),rcp_n(rn),memo(ep.nodes.size()),seen(ep.nodes.size()){}
    Bytes32 const& ev(int id) {
        if(id<0||id>=int(e.nodes.size()))throw std::out_of_range("node id");
        if(seen[id]==2)return memo[id];
        if(seen[id]==1)throw std::runtime_error("attention DAG cycle");
        seen[id]=1;auto const& n=e.nodes[id];auto& out=memo[id];auto const& k=n.kind;
        if(k=="leaf") {
            out.width=4;
            if(n.op=="mma") {
                uint64_t key=(uint64_t(uint32_t(n.ep))<<32)|uint32_t(n.slot);
                auto it=mma_out.find(key);if(it==mma_out.end())throw std::runtime_error("MMA leaf before stage");
                if(n.word<0||n.word>1)throw std::out_of_range("MMA leaf word");
                for(int lane=0;lane<32;++lane)for(int j=0;j<4;++j)out.put(lane,j,it->second.at(lane,4*n.word+j));
            }else{
                int region=n.op=="qload"?0:n.op=="kread"?1:n.op=="vread"?2:-1;
                if(region<0||n.word<0)throw std::invalid_argument("unknown leaf region");
                auto const& m=e.map(key3(n.op=="qload"?n.op:n.op+"_src",n.ep,n.slot));
                for(int lane=0;lane<32;++lane) {
                    int64_t off=m.at(c,y,lane);
                    if(off< -1)throw std::out_of_range("negative leaf map offset");
                    for(int j=0;j<4;++j) {
                        int64_t ix=off+4*n.word+j;
                        if(off>=0&&(ix<0||uint64_t(ix)>=input[region].size()))throw std::out_of_range("leaf read");
                        out.put(lane,j,off<0?0:uint8_t(input[region][size_t(ix)]));
                    }
                }
            }
        }else if(k=="c16"||k=="ibits") {
            out.width=k=="c16"?2:4;
            for(int lane=0;lane<32;++lane)for(int j=0;j<out.width;++j)out.put(lane,j,uint8_t(n.raw>>(8*j)));
        }else if(k=="cat") {
            ChainVector<Bytes32 const*> p;for(int i:n.args)p.push_back(&ev(i));
            out.width=0;for(auto v:p)out.width+=v->width;
            if(out.width>16)throw std::out_of_range("cat width");
            for(int lane=0;lane<32;++lane) {int b=0;for(auto v:p)for(int j=0;j<v->width;++j)out.put(lane,b++,v->at(lane,j));}
        }else if(k=="part"||k=="ilow16") {
            auto const& a=ev(n.args[0]);out.width=2;
            int start=k=="part"?2*n.idx:0;if(start<0||start+2>a.width)throw std::out_of_range("part width");
            for(int lane=0;lane<32;++lane)for(int j=0;j<2;++j)out.put(lane,j,a.at(lane,start+j));
        }else if(k=="fop") {
            ChainVector<Bytes32 const*> p;for(int i:n.args)p.push_back(&ev(i));
            if(p.empty())throw std::invalid_argument("empty fop");out.width=p[0]->width;
            if(out.width<2||out.width>16||(out.width&1))throw std::invalid_argument("fop result width");
            for(auto a:p)if(a->width!=out.width)throw std::invalid_argument("fop operand width node="+std::to_string(id));
            for(int lane=0;lane<32;++lane)for(int part=0;part<out.width/2;++part){
                uint16_t z=0,a=p[0]->h(lane,part),b=p.size()>1?p[1]->h(lane,part):0;
                if(n.op=="add")z=h_add(a,b);
                else if(n.op=="sub")z=half_round(half_value(a)-half_value(b));
                else if(n.op=="mul")z=h_mul(a,b);
                else if(n.op=="fma"&&p.size()==3)z=h_fma(a,b,p[2]->h(lane,part));
                else if(n.op=="min")z=h_min(a,b);
                else if(n.op=="max")z=h_max(a,b);
                else throw std::invalid_argument("unknown fop");
                if((z&0x7c00)==0x7c00)throw std::domain_error("nonfinite fop result");out.hput(lane,z,part);
            }
        }else if(k=="ishl"||k=="iadd") {
            auto const& a=ev(n.args[0]);out.width=n.bits/8;
            if((n.bits!=16&&n.bits!=32)||a.width!=out.width)throw std::invalid_argument("integer width");
            for(int lane=0;lane<32;++lane){
                uint32_t x=out.width==2?a.h(lane):a.u32(lane);
                if(k=="ishl") {if(n.imm<0||n.imm>=32)throw std::domain_error("shift range");x<<=n.imm;}
                else x+=uint32_t(n.imm);
                if(out.width==2)out.hput(lane,uint16_t(x));else out.uput(lane,x);
            }
        }else if(k=="prmt") {
            auto const& a=ev(n.args[0]);auto const& b=ev(n.args[1]);out.width=4;
            if(a.width!=4||b.width!=4)throw std::invalid_argument("prmt width");
            for(int j=0;j<4;++j){int sel=(n.imm>>(4*j))&15;if(sel>=8)throw std::invalid_argument("prmt sign replication");
                for(int lane=0;lane<32;++lane)out.put(lane,j,(sel<4?a:b).at(lane,sel&3));}
        }else if(k=="gather") {
            auto it=e.gather.find(key4(y,n.pc,n.ep,n.idx));if(it==e.gather.end())throw std::out_of_range("gather source");
            auto const& a=ev(it->second);if(n.idx<0||n.idx>=32)throw std::out_of_range("gather lane");
            out.width=a.width;for(int lane=0;lane<32;++lane)for(int j=0;j<out.width;++j)out.put(lane,j,a.at(n.idx,j));
        }else if(k=="tof32") {
            auto const& a=ev(n.args[0]);if(a.width<2||a.width>8||(a.width&1))throw std::invalid_argument("tof32 width");out.width=2*a.width;
            for(int lane=0;lane<32;++lane)for(int part=0;part<a.width/2;++part){
                uint32_t v=f32bits(float(half_value(a.h(lane,part))));
                for(int j=0;j<4;++j)out.put(lane,4*part+j,uint8_t(v>>(8*j)));
            }
        }else if(k=="rcp") {
            auto const& a=ev(n.args[0]);if(a.width<4||a.width>16||(a.width&3))throw std::invalid_argument("rcp width");out.width=a.width;
            for(int lane=0;lane<32;++lane)for(int part=0;part<a.width/4;++part){
                uint32_t bits=0;for(int j=0;j<4;++j)bits|=uint32_t(a.at(lane,4*part+j))<<(8*j);
                float f=bits_cast<float>(bits);uint16_t h=half_round(double(f));
                if(float(half_value(h))!=f)throw std::domain_error("RCP input not half");
                if(h<0x410||h>=0x7c00||size_t(h-0x410)>=rcp_n)throw std::domain_error("RCP table domain");
                uint32_t v=rcp[h-0x410];for(int j=0;j<4;++j)out.put(lane,4*part+j,uint8_t(v>>(8*j)));
            }
        }else if(k=="fmulc") {
            auto const& a=ev(n.args[0]);if(a.width<4||a.width>16||(a.width&3))throw std::invalid_argument("fmulc width");out.width=a.width;
            float cv=bits_cast<float>(n.raw);
            for(int lane=0;lane<32;++lane)for(int part=0;part<a.width/4;++part){
                uint32_t bits=0;for(int j=0;j<4;++j)bits|=uint32_t(a.at(lane,4*part+j))<<(8*j);
                float v=bits_cast<float>(bits)*cv;
                if(!std::isfinite(v)||(v!=0&&std::fpclassify(v)==FP_SUBNORMAL))throw std::domain_error("fmulc domain");
                uint32_t raw=f32bits(v);for(int j=0;j<4;++j)out.put(lane,4*part+j,uint8_t(raw>>(8*j)));
            }
        }else if(k=="f16of"||k=="sat") {
            auto const& a=ev(n.args[0]);out.width=a.width/2;
            if(a.width<2||a.width>16||(a.width&1)||(k=="f16of"&&(a.width&3)))throw std::invalid_argument("float conversion width");
            for(int lane=0;lane<32;++lane)for(int part=0;part<(k=="f16of"?a.width/4:a.width/2);++part){
                if(k=="f16of") {uint32_t bits=0;for(int j=0;j<4;++j)bits|=uint32_t(a.at(lane,4*part+j))<<(8*j);
                    uint16_t h=half_round(double(bits_cast<float>(bits)));
                    if((h&0x7c00)==0x7c00)throw std::domain_error("f16of nonfinite");
                    out.put(lane,2*part,uint8_t(h));out.put(lane,2*part+1,uint8_t(h>>8));}
                else {uint16_t h=a.h(lane,part);if((h&0x7c00)==0x7c00)throw std::domain_error("sat nonfinite");out.put(lane,part,sat_e4(h));}
            }
        }else throw std::invalid_argument("unknown attention node at execution");
        seen[id]=2;return out;
    }
    void mma(WireA const& w) {
        std::array<Bytes32 const*,4> aa;std::array<Bytes32 const*,2> bb,cc;
        for(int j=0;j<4;++j){aa[j]=&ev(w.a[j]);if(aa[j]->width!=4)throw std::invalid_argument("QMMA A width");}
        for(int j=0;j<2;++j){bb[j]=&ev(w.b[j]);cc[j]=&ev(w.c[j]);if(bb[j]->width!=4||cc[j]->width!=4)throw std::invalid_argument("QMMA B/C width");}
        Bytes32 z;z.width=8;
        for(int row=0;row<16;++row)for(int col=0;col<8;++col){
            uint8_t av[32],bv[32];
            for(int k=0;k<32;++k){
                int lane_a=(row%8)*4+((k%16)/4), reg_a=row/8+2*(k/16);
                int lane_b=col*4+((k%16)/4), reg_b=k/16;
                av[k]=aa[reg_a]->at(lane_a,k%4);bv[k]=bb[reg_b]->at(lane_b,k%4);
            }
            int lane=(row%8)*4+col/2, ix=2*(row/8)+col%2;
            uint16_t seed=cc[ix/2]->h(lane,ix%2);
            z.hput(lane,qmma_dot(av,bv,seed),ix);
        }
        uint64_t key=(uint64_t(uint32_t(w.ep))<<32)|uint32_t(w.slot);
        if(!mma_out.emplace(key,z).second)throw std::runtime_error("duplicate QMMA stage");
    }
    void run(int ci,int yi,ChainVector<uint8_t>& output) {
        c=ci;y=yi;std::fill(seen.begin(),seen.end(),0);mma_out.clear();
        for(auto const& w:e.wiring[y])mma(w);
        for(int lane=0;lane<32;++lane)for(auto const& s:e.stores[y][lane]){
            auto const& v=ev(s.node);if(v.width!=16)throw std::invalid_argument("store width");
            int off=e.map(s.mapkey).at(c,y,lane);if(off<0)continue;
            if(size_t(off)+16>output.size())throw std::out_of_range("output store");
            for(int j=0;j<16;++j)output[size_t(off)+j]=v.at(lane,j);
        }
    }
};

py::dict execute_attention_original(py::dict data,py::list epoch_objects,
                           py::array_t<uint32_t,py::array::c_style> rcp_bits,
                           int threads,uint32_t queue_seed) {
    std::array<std::string,3> input;
    for(int i=0;i<3;++i)input[i]=py::cast<std::string>(data[py::str(i==0?"q":i==1?"k":"v")]);
    size_t n=input[0].size();
    if(n==0||n%1024||input[1].size()!=n||input[2].size()!=n)throw std::invalid_argument("q/k/v byte length");
    if(rcp_bits.ndim()!=1||rcp_bits.size()<0x7c00-0x410)throw std::invalid_argument("RCP table length");
    ChainVector<EpochA> epochs;epochs.reserve(epoch_objects.size());
    for(auto obj:epoch_objects){
        py::tuple p=py::reinterpret_borrow<py::tuple>(obj);
        if(p.size()!=2)throw std::invalid_argument("epoch must be template/map pair");
        epochs.push_back(parse_epoch(p[0].cast<py::dict>(),p[1].cast<py::dict>()));
    }
    if(epochs.empty())throw std::invalid_argument("no attention epochs");
    size_t release63_size=epochs[0].release63_size;
    for(auto const& e:epochs)if(e.output_size!=n||e.release63_size!=release63_size)
        throw std::invalid_argument("epoch output/release extents disagree with inputs");
    struct Task {int ep,c,y;};
    ChainVector<Task> tasks;
    ChainVector<uint8_t> cover(n,0),out(n,0);
    for(int ep=0;ep<int(epochs.size());++ep){
        auto const& e=epochs[ep];
        for(int y=0;y<e.ny;++y)for(int c=0;c<32;++c){
            bool active=false;
            for(int lane=0;lane<32;++lane)for(auto const& s:e.stores[y][lane]){
                int64_t off=e.map(s.mapkey).at(c,y,lane);
                if(off< -1)throw std::out_of_range("negative store map offset");
                if(off<0)continue;active=true;
                if(uint64_t(off)+16>n)throw std::out_of_range("store outside output");
                for(int j=0;j<16;++j)if(++cover[size_t(off)+j]!=1)
                    throw std::runtime_error("output ownership collision");
            }
            if(active)tasks.push_back({ep,c,y});
        }
    }
    if(std::find(cover.begin(),cover.end(),0)!=cover.end())throw std::runtime_error("output coverage incomplete");
    if(queue_seed){uint32_t z=queue_seed;for(int i=int(tasks.size())-1;i>0;--i){z=1664525*z+1013904223;std::swap(tasks[i],tasks[z%uint32_t(i+1)]);}}
    threads=std::max(1,std::min(threads,12));
    std::atomic<size_t> next{0};std::atomic<bool> fail{false};std::exception_ptr error;std::mutex error_lock;
    {
        py::gil_scoped_release release;
        auto work=[&](){
            try{
                ChainVector<std::unique_ptr<WorkerA>> worker(epochs.size());
                while(!fail){size_t ix=next.fetch_add(1);if(ix>=tasks.size())break;
                    auto t=tasks[ix];
                    if(!worker[t.ep])worker[t.ep]=std::make_unique<WorkerA>(epochs[t.ep],input,rcp_bits.data(),size_t(rcp_bits.size()));
                    worker[t.ep]->run(t.c,t.y,out);
                }
            }catch(...){fail=true;std::lock_guard<std::mutex> guard(error_lock);if(!error)error=std::current_exception();}
        };
        ChainVector<std::thread> pool;for(int i=1;i<threads;++i)pool.emplace_back(work);
        work();for(auto& t:pool)t.join();
    }
    if(error)std::rethrow_exception(error);
    std::string release63(release63_size,'\0');
    py::dict result;
    result["output"]=py::bytes(reinterpret_cast<const char*>(out.data()),out.size());
    result["release63"]=py::bytes(release63);
    return result;
}
} // namespace


py::dict execute_attention(py::dict data,py::list epoch_objects,
                           py::array_t<uint32_t,py::array::c_style> rcp_bits,
                           int threads,uint32_t queue_seed){chain_mem::begin();try{auto result=execute_attention_original(data,epoch_objects,rcp_bits,threads,queue_seed);result["mem"]=chain_mem::finish();return result;}catch(...){chain_mem::active=false;throw;}}
PYBIND11_MODULE(dlssnr_fam_attention_chain_r02,m){
    m.doc()="Portable scalar attention family executor";
    m.def("memory_counter_selftest",[](){chain_mem::begin();{ChainVector<uint8_t> a(4096);ChainVector<uint16_t> b(2048);std::string text(4096,'x');}return chain_mem::finish();});
    m.def("execute",&execute_attention,py::arg("data"),py::arg("epochs"),py::arg("rcp_bits"),
          py::arg("threads")=1,py::arg("queue_seed")=0);
    m.def("inspect",[](py::list objects){
        py::list out;
        for(auto obj:objects){
            py::tuple p=py::reinterpret_borrow<py::tuple>(obj);
            auto e=parse_epoch(p[0].cast<py::dict>(),p[1].cast<py::dict>());
            py::dict row;row["nodes"]=e.nodes.size();row["maps"]=e.maps.size();
            row["ny"]=e.ny;row["wiring_y0"]=e.wiring[0].size();
            row["stores_y0_lane0"]=e.stores[0][0].size();
            row["output_bytes"]=e.output_size;row["release63_bytes"]=e.release63_size;
            out.append(row);
        }
        return out;
    },py::arg("epochs"));
}
