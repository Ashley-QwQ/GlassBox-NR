// qkv_chained and projection_chained/wait, portable scalar execution.
// The scalar binary16/E4M3/QMMA helpers in qkv_projection.cpp are copied from
// cpp/src/step2a.cpp:1-131, SHA256
// 91a28b28a9ca9771d70973fec37d772f60f0b9713a1d3c79267b309913d1110b.
struct QBytes {
    std::array<uint8_t,512> bytes{};
    int width=0;
    uint8_t at(int lane,int byte)const{return bytes[size_t(lane)*16+byte];}
    void put(int lane,int byte,uint8_t v){bytes[size_t(lane)*16+byte]=v;}
    uint16_t half(int lane,int part)const{return uint16_t(at(lane,2*part))|(uint16_t(at(lane,2*part+1))<<8);}
    void sethalf(int lane,int part,uint16_t v){put(lane,2*part,uint8_t(v));put(lane,2*part+1,uint8_t(v>>8));}
};
struct QNode {
    std::string kind,sub;
    ChainVector<int> args;
    int ep=0,slot=0,word=0,index=0;
    uint16_t half_constant=0;
    uint32_t integer_constant=0;
};
struct QMap {
    py::array owner;
    const int32_t* ptr;
    int nc,nz,ny;
    explicit QMap(py::handle h):owner(py::reinterpret_borrow<py::array>(h)),ptr(nullptr),nc(0),nz(0),ny(0){
        if(owner.ndim()!=4||owner.shape(3)!=32||owner.itemsize()!=4||owner.dtype().kind()!='i'||
           !(owner.flags()&py::array::c_style))throw std::invalid_argument("map requires contiguous int32 [cta,z,warp,32]");
        nc=int(owner.shape(0));nz=int(owner.shape(1));ny=int(owner.shape(2));
        ptr=static_cast<const int32_t*>(owner.data());
    }
    int32_t at(int c,int z,int y,int lane)const{
        if(c<0||c>=nc||z<0||z>=nz||y<0||y>=ny||lane<0||lane>=32)throw std::out_of_range("map index");
        return ptr[((size_t(c)*nz+z)*ny+y)*32+lane];
    }
};
struct QWire{int ep,slot;std::array<int,4>a;std::array<int,2>b,c;};
struct QStore{std::string kind,key;int node;};
struct QTemplate{ChainVector<QWire>wiring;ChainVector<QStore>stores;};
struct QProgram {
    ChainVector<QNode>nodes;
    std::unordered_map<std::string,QMap>maps;
    std::unordered_map<std::string,QTemplate>templates;
    std::unordered_map<std::string,size_t>region_len;
    int nc=0,nz=0,ny=0;
    QMap const& map(std::string const& key)const{
        auto it=maps.find(key);if(it==maps.end())throw std::invalid_argument("missing map "+key);return it->second;
    }
    QTemplate const& tmpl(int z,int y)const{
        auto key=std::to_string(z)+"|"+(y<2?std::to_string(y):"dead");
        auto it=templates.find(key);if(it==templates.end())throw std::invalid_argument("missing template "+key);return it->second;
    }
};
static int qint(py::handle x){return py::cast<int>(x);}
static std::string qkey(std::string const& k,int ep,int slot){return k+"|"+std::to_string(ep)+"|"+std::to_string(slot);}

QProgram parse_qprogram(py::dict tm,py::dict arrays){
    QProgram p;
    for(auto kv:arrays){
        std::string key=py::cast<std::string>(kv.first);
        auto [it,ok]=p.maps.emplace(key,QMap(kv.second));
        if(!ok)throw std::invalid_argument("duplicate map key");
        auto const&m=it->second;
        if(!p.nc){p.nc=m.nc;p.nz=m.nz;p.ny=m.ny;}
        if(m.nc!=p.nc||m.nz!=p.nz||m.ny!=p.ny)throw std::invalid_argument("inconsistent map geometry");
    }
    if(!p.nc||p.nc>10000||p.nz<2||p.nz>4||p.ny<1||p.ny>16)throw std::invalid_argument("invalid map axes");
    for(auto item:tm["nodes"].cast<py::list>()){
        py::list x=py::reinterpret_borrow<py::list>(item);QNode n;
        n.kind=py::cast<std::string>(x[0]);auto k=n.kind;
        if(k=="leaf"){n.sub=py::cast<std::string>(x[1]);n.ep=qint(x[2]);n.slot=qint(x[3]);n.word=qint(x[4]);}
        else if(k=="c16")n.half_constant=half_round(std::stod(py::cast<std::string>(x[1])));
        else if(k=="ibits")n.integer_constant=py::cast<uint32_t>(x[1]);
        else if(k=="fop"){n.sub=py::cast<std::string>(x[1]);for(size_t i=2;i<x.size();++i)n.args.push_back(qint(x[i]));}
        else if(k=="bfly"){n.index=qint(x[1]);n.args.push_back(qint(x[2]));}
        else if(k=="part"){n.args.push_back(qint(x[1]));n.index=qint(x[2]);}
        else if(k=="cat"){for(size_t i=1;i<x.size();++i)n.args.push_back(qint(x[i]));}
        else if(k=="movm"||k=="tof32"||k=="rsq"||k=="f16of"||k=="sat"||k=="dq")n.args.push_back(qint(x[1]));
        else throw std::invalid_argument("unknown qkv/projection node "+k);
        p.nodes.push_back(std::move(n));
    }
    py::dict regions=tm["regions"].cast<py::dict>();
    for(auto kv:regions){auto key=py::cast<std::string>(kv.first);auto x=py::cast<py::list>(kv.second);
        p.region_len[key]=py::cast<size_t>(x[1]);}
    py::dict all=tm["templates"].cast<py::dict>();
    for(auto kv:all){
        auto name=py::cast<std::string>(kv.first);auto t=py::cast<py::dict>(kv.second);QTemplate out;
        for(auto wi:t["wiring"].cast<py::list>()){
            auto w=py::reinterpret_borrow<py::list>(wi);
            auto a=py::cast<ChainVector<int>>(w[3]),b=py::cast<ChainVector<int>>(w[4]),c=py::cast<ChainVector<int>>(w[5]);
            if(a.size()!=4||b.size()!=2||c.size()!=2)throw std::invalid_argument("QMMA wiring shape");
            out.wiring.push_back({qint(w[0]),qint(w[1]),{a[0],a[1],a[2],a[3]},{b[0],b[1]},{c[0],c[1]}});
        }
        for(auto si:t["stores"].cast<py::list>()){
            auto s=py::reinterpret_borrow<py::list>(si);std::string kind=py::cast<std::string>(s[0]);
            out.stores.push_back({kind,qkey(kind,qint(s[1]),qint(s[2])),qint(s[4])});
        }
        p.templates.emplace(name,std::move(out));
    }
    for(auto const&kv:p.templates){for(auto const&s:kv.second.stores)p.map(s.key);}
    return p;
}

struct QWorker {
    QProgram const&p;
    std::unordered_map<std::string,std::string>const&inputs;
    uint32_t const*rsq;
    size_t rsq_n;
    ChainVector<uint8_t>&buffer;
    ChainVector<uint8_t>&written;
    std::unordered_map<std::string,ChainVector<uint8_t>>&outs;
    std::unordered_map<std::string,ChainVector<uint8_t>>&cover;
    ChainVector<uint8_t>&red_count;
    bool reverse_red;
    int c=0,z=0,y=0;
    ChainVector<QBytes>memo;
    ChainVector<uint8_t>seen;
    std::unordered_map<uint64_t,QBytes>mma_out;
    QWorker(QProgram const&pr,std::unordered_map<std::string,std::string>const&in,uint32_t const*r,size_t rn,
            ChainVector<uint8_t>&buf,ChainVector<uint8_t>&wr,
            std::unordered_map<std::string,ChainVector<uint8_t>>&o,
            std::unordered_map<std::string,ChainVector<uint8_t>>&cv,
            ChainVector<uint8_t>&rc,bool rev)
      :p(pr),inputs(in),rsq(r),rsq_n(rn),buffer(buf),written(wr),outs(o),cover(cv),red_count(rc),reverse_red(rev),
       memo(pr.nodes.size()),seen(pr.nodes.size()){}
    QBytes const&ev(int id){
        if(id<0||id>=int(p.nodes.size()))throw std::out_of_range("node id");
        if(seen[id]==2)return memo[id];
        if(seen[id]==1)throw std::runtime_error("template DAG cycle");
        seen[id]=1;auto const&n=p.nodes[id];auto&out=memo[id];auto const&k=n.kind;
        if(k=="leaf"){
            out.width=4;
            if(n.sub=="mma"){
                auto it=mma_out.find((uint64_t(uint32_t(n.ep))<<32)|uint32_t(n.slot));
                if(it==mma_out.end()||n.word<0||n.word>1)throw std::runtime_error("MMA leaf before stage");
                for(int lane=0;lane<32;++lane)for(int j=0;j<4;++j)out.put(lane,j,it->second.at(lane,4*n.word+j));
            }else{
                std::string key=qkey(n.sub=="sread"?"src":n.sub,n.ep,n.slot);
                auto const&m=p.map(key);
                std::string role=n.sub=="sread"?"input":(n.sub=="gload"||n.sub=="g32")?"weights":n.sub=="cg"?"residual":n.sub=="bufload"?"buffer":"";
                if(role.empty())throw std::invalid_argument("unknown leaf role");
                size_t len=role=="buffer"?buffer.size():inputs.at(role).size();
                uint8_t const*base=role=="buffer"?buffer.data():reinterpret_cast<const uint8_t*>(inputs.at(role).data());
                for(int lane=0;lane<32;++lane){
                    int64_t off=m.at(c,z,y,lane);if(off<0&&!(n.sub=="sread"&&off==-1))throw std::out_of_range("missing mapped leaf "+key);
                    for(int j=0;j<4;++j){
                        int64_t ix=off+(n.sub=="g32"?0:4*n.word)+j;
                        if(off>=0&&(ix<0||uint64_t(ix)>=len))throw std::out_of_range("leaf outside "+role);
                        if(off>=0&&role=="buffer"&&!written[size_t(ix)])throw std::runtime_error("buffer read before write");
                        out.put(lane,j,off<0?0:base[size_t(ix)]);
                    }
                }
            }
        }else if(k=="c16"||k=="ibits"){
            out.width=k=="c16"?2:4;uint32_t raw=k=="c16"?n.half_constant:n.integer_constant;
            for(int lane=0;lane<32;++lane)for(int j=0;j<out.width;++j)out.put(lane,j,uint8_t(raw>>(8*j)));
        }else if(k=="cat"){
            ChainVector<QBytes const*>pieces;for(int arg:n.args)pieces.push_back(&ev(arg));
            out.width=0;for(auto a:pieces)out.width+=a->width;if(out.width>16)throw std::out_of_range("cat width");
            for(int lane=0;lane<32;++lane){int pos=0;for(auto a:pieces)for(int j=0;j<a->width;++j)out.put(lane,pos++,a->at(lane,j));}
        }else if(k=="part"){
            auto const&a=ev(n.args[0]);out.width=2;if(n.index<0||2*n.index+2>a.width)throw std::out_of_range("part width");
            for(int lane=0;lane<32;++lane)for(int j=0;j<2;++j)out.put(lane,j,a.at(lane,2*n.index+j));
        }else if(k=="dq"){
            auto const&a=ev(n.args[0]);out.width=2*a.width;if(out.width>16)throw std::out_of_range("decode width");
            for(int lane=0;lane<32;++lane)for(int j=0;j<a.width;++j)out.sethalf(lane,j,e4_half(a.at(lane,j)));
        }else if(k=="fop"){
            ChainVector<QBytes const*>v;for(int arg:n.args)v.push_back(&ev(arg));
            if(v.size()!=2||v[0]->width!=v[1]->width||v[0]->width%2)throw std::invalid_argument("fop operand shape");
            out.width=v[0]->width;
            bool fuse=false;
            if(n.sub=="add"&&p.nodes[n.args[0]].kind=="fop"&&p.nodes[n.args[0]].sub=="mul"&&
               p.nodes[n.args[1]].kind=="fop"&&p.nodes[n.args[1]].sub=="mul")fuse=true;
            for(int lane=0;lane<32;++lane)for(int part=0;part<out.width/2;++part){
                uint16_t a=v[0]->half(lane,part),b=v[1]->half(lane,part),h=0;
                if(fuse){auto const&mul=p.nodes[n.args[0]];auto const&x=ev(mul.args[0]);auto const&yv=ev(mul.args[1]);
                    h=half_round(half_value(x.half(lane,part))*half_value(yv.half(lane,part))+half_value(b));}
                else if(n.sub=="add")h=h_add(a,b);
                else if(n.sub=="mul")h=h_mul(a,b);
                else if(n.sub=="max")h=h_max(a,b);
                else throw std::invalid_argument("unknown fop");
                if((h&0x7c00)==0x7c00)throw std::domain_error("nonfinite fop");out.sethalf(lane,part,h);
            }
        }else if(k=="bfly"){
            auto const&a=ev(n.args[0]);out.width=a.width;
            for(int lane=0;lane<32;++lane)for(int j=0;j<out.width;++j)out.put(lane,j,a.at(lane^n.index,j));
        }else if(k=="movm"){
            auto const&a=ev(n.args[0]);if(a.width!=4)throw std::invalid_argument("movm source width");out.width=4;
            for(int lane=0;lane<32;++lane){int i0=2*lane,i1=i0+1;
                int src0=8*(i0%8)+i0/8,src1=8*(i1%8)+i1/8;
                for(int j=0;j<2;++j){out.put(lane,j,a.at(src0/2,2*(src0%2)+j));
                    out.put(lane,2+j,a.at(src1/2,2*(src1%2)+j));}
            }
        }else if(k=="tof32"){
            auto const&a=ev(n.args[0]);out.width=2*a.width;if(out.width>16)throw std::out_of_range("tof32 width");
            for(int lane=0;lane<32;++lane)for(int j=0;j<a.width/2;++j){uint32_t raw=f32bits(float(half_value(a.half(lane,j))));
                for(int b=0;b<4;++b)out.put(lane,4*j+b,uint8_t(raw>>(8*b)));}
        }else if(k=="rsq"){
            auto const&a=ev(n.args[0]);out.width=a.width;if(out.width%4)throw std::invalid_argument("rsq width");
            for(int lane=0;lane<32;++lane)for(int j=0;j<out.width/4;++j){uint32_t raw=0;for(int b=0;b<4;++b)raw|=uint32_t(a.at(lane,4*j+b))<<(8*b);
                float f=bits_cast<float>(raw);uint16_t h=half_round(double(f));
                if(float(half_value(h))!=f||h<0x410||h>=0x7c00||size_t(h-0x410)>=rsq_n)throw std::domain_error("RSQ table domain");
                raw=rsq[h-0x410];for(int b=0;b<4;++b)out.put(lane,4*j+b,uint8_t(raw>>(8*b)));}
        }else if(k=="f16of"){
            auto const&a=ev(n.args[0]);if(a.width%4)throw std::invalid_argument("f16of width");out.width=a.width/2;
            for(int lane=0;lane<32;++lane)for(int j=0;j<a.width/4;++j){uint32_t raw=0;for(int b=0;b<4;++b)raw|=uint32_t(a.at(lane,4*j+b))<<(8*b);
                uint16_t h=half_round(double(bits_cast<float>(raw)));if((h&0x7c00)==0x7c00)throw std::domain_error("f16of nonfinite");
                out.sethalf(lane,j,h);}
        }else if(k=="sat"){
            auto const&a=ev(n.args[0]);if(a.width%2)throw std::invalid_argument("sat width");out.width=a.width/2;
            for(int lane=0;lane<32;++lane)for(int j=0;j<out.width;++j)out.put(lane,j,sat_e4(a.half(lane,j)));
        }else throw std::invalid_argument("unknown node at execution: "+k);
        seen[id]=2;return out;
    }
    void mma(QWire const&w){
        std::array<QBytes const*,4>a;std::array<QBytes const*,2>b,d;
        for(int j=0;j<4;++j){a[j]=&ev(w.a[j]);if(a[j]->width!=4)throw std::invalid_argument("QMMA A width");}
        for(int j=0;j<2;++j){b[j]=&ev(w.b[j]);d[j]=&ev(w.c[j]);if(b[j]->width!=4||d[j]->width!=4)throw std::invalid_argument("QMMA B/C width");}
        QBytes result;result.width=8;
        for(int row=0;row<16;++row)for(int col=0;col<8;++col){uint8_t aa[32],bb[32];
            for(int k=0;k<32;++k){int al=(row%8)*4+((k%16)/4),ar=row/8+2*(k/16);
                int bl=col*4+((k%16)/4),br=k/16;aa[k]=a[ar]->at(al,k%4);bb[k]=b[br]->at(bl,k%4);}
            int lane=(row%8)*4+col/2,ix=2*(row/8)+col%2;
            result.sethalf(lane,ix,qmma_dot(aa,bb,d[ix/2]->half(lane,ix%2)));
        }
        if(!mma_out.emplace((uint64_t(uint32_t(w.ep))<<32)|uint32_t(w.slot),std::move(result)).second)
            throw std::runtime_error("duplicate QMMA stage");
    }
    void run_class(int ci,int zi,int yi){
        c=ci;z=zi;y=yi;std::fill(seen.begin(),seen.end(),0);mma_out.clear();
        auto const&t=p.tmpl(z,y);for(auto const&w:t.wiring)mma(w);
        for(auto const&s:t.stores){auto const&val=ev(s.node);if(val.width!=16)throw std::invalid_argument("store width");
            auto const&m=p.map(s.key);
            for(int lane=0;lane<32;++lane){int64_t off=m.at(c,z,y,lane);if(off<0)throw std::out_of_range("store map missing");
                if(s.kind=="bufstore"){if(uint64_t(off)+16>buffer.size()||z!=0)throw std::out_of_range("buffer store");
                    for(int j=0;j<16;++j){size_t ix=size_t(off)+j;if(written[ix]++)throw std::runtime_error("buffer duplicate store");buffer[ix]=val.at(lane,j);}}
                else if(s.kind=="red"){if(z!=1&&z!=2||uint64_t(off)+16>buffer.size())throw std::out_of_range("REDG store");
                    for(int j=0;j<16;j+=2){size_t ix=size_t(off)+j;if(!written[ix]||!written[ix+1])throw std::runtime_error("REDG before buffer store");
                        uint16_t old=uint16_t(buffer[ix])|(uint16_t(buffer[ix+1])<<8),sum=h_add(old,val.half(lane,j/2));
                        if((sum&0x7c00)==0x7c00)throw std::domain_error("REDG nonfinite");
                        buffer[ix]=uint8_t(sum);buffer[ix+1]=uint8_t(sum>>8);++red_count[ix];++red_count[ix+1];}}
                else {auto role=s.kind=="outstore"?"output":s.kind=="qstore"?"q":s.kind=="kstore"?"k":s.kind=="vstore"?"v":"";
                    if(!*role)throw std::invalid_argument("unknown store kind");auto&v=outs.at(role);auto&cv=cover.at(role);
                    if(uint64_t(off)+16>v.size())throw std::out_of_range("output store");
                    for(int j=0;j<16;++j){size_t ix=size_t(off)+j;if(cv[ix]++)throw std::runtime_error("duplicate output owner");v[ix]=val.at(lane,j);}}
            }
        }
    }
};

py::dict execute_qkv_projection_original(std::string family,py::dict templates,py::dict maps,py::dict data,
                                py::array_t<uint32_t,py::array::c_style> rsq_bits,
                                int threads,uint32_t queue_seed,bool reverse_red){
    if(family!="qkv"&&family!="projection")throw std::invalid_argument("family must be qkv or projection");
    QProgram p=parse_qprogram(templates,maps);
    if((family=="qkv"&&p.nz!=2)||(family=="projection"&&p.nz!=4))throw std::invalid_argument("family stage count");
    std::unordered_map<std::string,std::string> in;
    for(auto kv:data)in.emplace(py::cast<std::string>(kv.first),py::cast<std::string>(kv.second));
    for(auto role:{"input","weights"})if(!in.contains(role))throw std::invalid_argument("missing input role");
    if(family=="projection"&&!in.contains("residual"))throw std::invalid_argument("missing residual");
    size_t out_len=p.region_len.at(family=="qkv"?"q":"output"),buf_len=p.region_len.at("buffer");
    if(in.at("input").size()!=out_len||buf_len==0||out_len==0||out_len%1024)throw std::invalid_argument("role lengths");
    if(family=="projection"&&in.at("residual").size()!=out_len)throw std::invalid_argument("residual length");
    if(in.at("weights").size()!=p.region_len.at("weights"))throw std::invalid_argument("weight length");
    if(family=="qkv"&&rsq_bits.size()<0x7c00-0x410)throw std::invalid_argument("RSQ table length");
    ChainVector<uint8_t>buffer(buf_len),written(buf_len),red_count(buf_len);
    std::unordered_map<std::string,ChainVector<uint8_t>>outs,cover;
    auto roles=family=="qkv"?ChainVector<std::string>{"q","k","v"}:ChainVector<std::string>{"output"};
    for(auto const&r:roles){if(p.region_len.at(r)!=out_len)throw std::invalid_argument("output extents");outs.emplace(r,ChainVector<uint8_t>(out_len));cover.emplace(r,ChainVector<uint8_t>(out_len));}
    ChainVector<int>order(p.nc);std::iota(order.begin(),order.end(),0);
    if(queue_seed){uint32_t x=queue_seed;for(size_t i=order.size();i>1;--i){x=1664525*x+1013904223;std::swap(order[i-1],order[x%i]);}}
    threads=std::clamp(threads,1,12);
    for(int zi=0;zi<p.nz;++zi){
        int z=reverse_red&&family=="projection"?(zi==1?2:zi==2?1:zi):zi;
        std::atomic<size_t>next{0};std::atomic<bool>stop{false};std::exception_ptr error;std::mutex error_lock;
        {
            py::gil_scoped_release release;
            auto work=[&](){try{
                QWorker worker(p,in,rsq_bits.data(),size_t(rsq_bits.size()),buffer,written,outs,cover,red_count,reverse_red);
                while(!stop){auto ix=next.fetch_add(1);if(ix>=order.size())break;int c=order[ix];
                    for(int y=0;y<p.ny;++y){
                        auto const&t=p.tmpl(z,y);bool active=false;
                        for(auto const&s:t.stores){auto const&m=p.map(s.key);if(m.at(c,z,y,0)>=0){active=true;break;}}
                        if(active)worker.run_class(c,z,y);
                    }
                }
            }catch(...){stop=true;std::lock_guard<std::mutex>guard(error_lock);if(!error)error=std::current_exception();}};
            ChainVector<std::thread>pool;for(int t=1;t<threads;++t)pool.emplace_back(work);work();for(auto&t:pool)t.join();
        }
        if(error)std::rethrow_exception(error);
    }
    if(std::any_of(written.begin(),written.end(),[](uint8_t x){return x!=1;}))throw std::runtime_error("buffer coverage incomplete");
    for(auto const&r:roles)if(std::any_of(cover.at(r).begin(),cover.at(r).end(),[](uint8_t x){return x!=1;}))
        throw std::runtime_error("output coverage incomplete");
    if(family=="projection"&&std::any_of(red_count.begin(),red_count.end(),[](uint8_t x){return x!=2;}))
        throw std::runtime_error("REDG count incomplete");
    py::dict result;
    for(auto const&r:roles){auto const&v=outs.at(r);result[py::str(r)]=py::bytes(reinterpret_cast<const char*>(v.data()),v.size());}
    result["buffer"]=py::bytes(reinterpret_cast<const char*>(buffer.data()),buffer.size());
    return result;
}
} // namespace


py::dict execute_qkv_projection(std::string family,py::dict templates,py::dict maps,py::dict data,
                                py::array_t<uint32_t,py::array::c_style> rsq_bits,
                                int threads,uint32_t queue_seed,bool reverse_red){chain_mem::begin();try{auto result=execute_qkv_projection_original(family,templates,maps,data,rsq_bits,threads,queue_seed,reverse_red);result["mem"]=chain_mem::finish();return result;}catch(...){chain_mem::active=false;throw;}}
PYBIND11_MODULE(dlssnr_fam_qkv_projection_chain_r02,m){
    m.doc()="Portable scalar qkv / projection map executor";
    m.def("memory_counter_selftest",[](){chain_mem::begin();{ChainVector<uint8_t> a(4096);ChainVector<uint16_t> b(2048);std::string text(4096,'x');}return chain_mem::finish();});
    m.def("execute",&execute_qkv_projection,py::arg("family"),py::arg("templates"),py::arg("maps"),
          py::arg("data"),py::arg("rsq_bits"),py::arg("threads")=1,py::arg("queue_seed")=0,py::arg("reverse_red")=false);
    m.def("inspect",[](py::dict tm,py::dict mp){auto p=parse_qprogram(tm,mp);py::dict d;
        d["cta_count"]=p.nc;d["stage_count"]=p.nz;d["warps"]=p.ny;d["node_count"]=p.nodes.size();d["map_count"]=p.maps.size();return d;});
}
