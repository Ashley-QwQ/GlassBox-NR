// C4 stage ownership carries the sealed E1 decoder. Table data is read directly
// into its final owner; only the small manifest crosses Python as bytes.
#include "codec_identity.hpp"
int configured_launch=-1;
bool temporal_tables=false;std::vector<std::string> temporal_param_profiles;
std::vector<std::pair<uint32_t,uint32_t>> ex2_sparse;
bool quant_ready=false;
std::mutex stage_mutex;
struct ReadPin { std::string path,sha; size_t bytes; };
std::vector<ReadPin> read_pins;
std::unique_lock<std::mutex> stage_lock(){std::unique_lock<std::mutex> l(stage_mutex,std::try_to_lock);require(l.owns_lock(),"TAIL_CONFIGURE_RUN_BUSY");return l;}
void clear_tables(){
 mufu_full_r01::clear_cache();
 std::vector<std::pair<uint32_t,uint32_t>>().swap(ex2_sparse);configured_launch=-1;temporal_tables=false;std::vector<std::string>().swap(temporal_param_profiles);ready=false;quant_ready=false;quant.fill(0);
 std::vector<uint32_t>().swap(ex2tab);std::vector<uint32_t>().swap(rcptab);std::vector<uint32_t>().swap(rsqtab);
 std::vector<uint32_t>().swap(lg2codec.base);std::vector<uint32_t>().swap(lg2codec.e126);std::vector<uint8_t>().swap(lg2codec.levels);
 lg2idx.fill(-1);exneg.fill(false);expos.fill(false);
}
struct StageFailureReset{int entering=std::uncaught_exceptions();~StageFailureReset(){if(std::uncaught_exceptions()>entering){clear_tables();mufu_full_r01::clear();}}};
void require_stage(int launch){
 require(configured_launch==launch,"TAIL_CONFIGURED_LAUNCH_MISMATCH");
 if(launch==155){require(ready&&quant_ready,"TAIL_QUANT_NOT_READY");require(rcptab.size()==8388608&&rsqtab.size()==0x7c00-0x0410+1&&(temporal_tables?ex2tab.size()==8388608:ex2tab.empty())&&lg2codec.bytes()==0,"TAIL_155_TABLE_STATE");}
 else if(launch==156){require(!ready&&!quant_ready&&rcptab.empty()&&rsqtab.empty()&&ex2tab.empty()&&lg2codec.bytes()==0,"TAIL_156_TABLE_STATE");}
 else if(launch==157){
  require(ready&&!quant_ready&&rcptab.size()==8388608&&ex2tab.size()==8388608&&rsqtab.empty()&&lg2codec.base.size()==8388608&&lg2codec.e126.size()==8388608&&lg2codec.levels.size()==3145728,"TAIL_157_TABLE_STATE");
  for(int b=0;b<512;++b)require(lg2idx[b]==(b>=111&&b<=127?b-111:-1),"TAIL_LG2_DOMAIN");
  require(std::any_of(exneg.begin(),exneg.end(),[](bool x){return x;})&&std::any_of(expos.begin(),expos.end(),[](bool x){return x;}),"TAIL_TABLE_DOMAINS");
 }else require(false,"tail launch");
}
template<class T> void read_final(std::vector<T>& v,py::dict d){
 auto path=py::cast<std::string>(d["path"]);auto count=py::cast<size_t>(d["bytes"]);
 auto lower=path;std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return char(std::tolower(c));});
 require(!path_is_forbidden(lower),"FORBIDDEN_PATH");require(count>0&&count%sizeof(T)==0,"table length");
 require(v.empty()&&v.capacity()==0,"old table owner must be released");
 std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f)&&f.tellg()==std::streamoff(count),"table file length");f.seekg(0);
 v.resize(count/sizeof(T));f.read(reinterpret_cast<char*>(v.data()),std::streamsize(count));require(bool(f)&&f.gcount()==std::streamsize(count),"table truncated");
 auto view=py::memoryview::from_memory(static_cast<const void*>(v.data()),py::ssize_t(count));
 auto actual=py::module_::import("hashlib").attr("sha256")(view).attr("hexdigest")().cast<std::string>();
 read_pins.push_back({path,actual,count});
 require(actual==py::cast<std::string>(d["sha256"]),"table actual-read SHA mismatch");
}
void load_codec_files(py::dict d){
 auto hash=py::module_::import("hashlib").attr("sha256");
 require(py::cast<std::string>(hash(d["encoded_manifest"]).attr("hexdigest")())==LG2_MANIFEST_SHA,"E1 manifest identity");
 auto manifest=py::module_::import("json").attr("loads")(d["encoded_manifest"]).cast<py::dict>();
 require(py::cast<std::string>(manifest["endian"])=="little","E1 endian");require(optm::little_endian(),"E1 host endian");
 auto files=manifest["files"].cast<py::dict>();
 auto read=[&](auto& owner,const char* filename,const char* key,size_t bytes){
  auto pin=files[filename].cast<py::dict>();auto desc=d[key].cast<py::dict>();
  require(py::cast<size_t>(pin["bytes"])==bytes&&py::cast<size_t>(desc["bytes"])==bytes,"E1 encoded length");
  require(py::cast<std::string>(desc["sha256"])==py::cast<std::string>(pin["sha256"]),"E1 encoded SHA");
  read_final(owner,desc);
 };
 read(lg2codec.base,"base_E127.bin","lg2_base",33554432);
 read(lg2codec.e126,"raw_E126.bin","lg2_e126",33554432);
 read(lg2codec.levels,"level_3bit.bin","lg2_levels",3145728);
}
void init_files(int launch,py::dict d){
 auto exclusion=stage_lock();clear_tables();read_pins.clear();
 try{
  require(optm::little_endian(),"little endian host required");require(launch>=155&&launch<=157,"tail launch");
  if(launch==155){read_final(rcptab,d["rcp"].cast<py::dict>());read_final(rsqtab,d["rsq"].cast<py::dict>());require(rcptab.size()==8388608&&rsqtab.size()==0x7c00-0x0410+1,"MUFU table sizes");}
  if(launch==157){
   read_final(ex2tab,d["ex2"].cast<py::dict>());read_final(rcptab,d["rcp"].cast<py::dict>());load_codec_files(d);
   require(ex2tab.size()==8388608&&rcptab.size()==8388608,"MUFU table sizes");
   auto blocks=py::cast<std::vector<int>>(d["lg2_blocks"]);require(blocks==std::vector<int>({111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127}),"LG2 size");
   for(size_t i=0;i<blocks.size();i++)lg2idx.at(blocks[i])=int(i);
   for(int b:py::cast<std::vector<int>>(d["ex2_negative_blocks"]))exneg.at(b)=true;for(int b:py::cast<std::vector<int>>(d["ex2_positive_blocks"]))expos.at(b)=true;
  }
  if(launch==155&&d.contains("temporal_param_profiles")){
   auto profiles=d["temporal_param_profiles"].cast<py::list>();require(py::len(profiles)>0,"temporal profile set empty");
   for(auto profile:profiles){auto p=py::cast<std::string>(profile);require(p.size()==184,"temporal profile params length");temporal_param_profiles.push_back(p);}
   read_final(ex2tab,d["ex2"].cast<py::dict>());require(ex2tab.size()==8388608,"temporal EX2 table size");
   for(int b:py::cast<std::vector<int>>(d["ex2_negative_blocks"]))exneg.at(b)=true;
   for(int b:py::cast<std::vector<int>>(d["ex2_positive_blocks"]))expos.at(b)=true;
   if(d.contains("ex2_sparse_input")){
    require(d.contains("ex2_sparse_output"),"sparse EX2 output missing");
    std::vector<uint32_t> inputs,outputs;read_final(inputs,d["ex2_sparse_input"].cast<py::dict>());read_final(outputs,d["ex2_sparse_output"].cast<py::dict>());
    require(inputs.size()==outputs.size()&&!inputs.empty()&&inputs.size()<=65536,"sparse EX2 exact lengths");
    ex2_sparse.reserve(inputs.size());for(size_t i=0;i<inputs.size();++i){require((inputs[i]&0x7f800000)!=0x7f800000,"sparse EX2 finite exact input only");ex2_sparse.emplace_back(inputs[i],outputs[i]);}
    std::sort(ex2_sparse.begin(),ex2_sparse.end(),[](const auto& a,const auto& b){return a.first<b.first;});for(size_t i=1;i<ex2_sparse.size();++i)require(ex2_sparse[i-1].first!=ex2_sparse[i].first,"sparse EX2 duplicate exact keys");
   }
   temporal_tables=true;
  }
  if(launch==155){for(int h=0;h<65536;h++)if((h&0x7c00)!=0x7c00)quant[h]=sat_e4(uint16_t(h));quant_ready=true;}
  ready=launch!=156;configured_launch=launch;require_stage(launch);
 }catch(...){clear_tables();mufu_full_r01::clear();throw;}
}
py::dict table_owners(){
 py::dict d;d["ex2_sparse"]=ex2_sparse.capacity()*sizeof(std::pair<uint32_t,uint32_t>);d["ex2"]=ex2tab.capacity()*4;d["rcp"]=rcptab.capacity()*4;d["lg2"]=lg2codec.bytes();d["lg2_base"]=lg2codec.base.capacity()*4;d["lg2_e126"]=lg2codec.e126.capacity()*4;d["lg2_levels"]=lg2codec.levels.capacity();d["rsq"]=rsqtab.capacity()*4;d["ready"]=ready;d["configured_launch"]=configured_launch;d["quant_ready"]=quant_ready;
 py::list logs;for(const auto& p:read_pins){py::dict q;q["path"]=p.path;q["sha256"]=p.sha;q["bytes"]=p.bytes;logs.append(q);}d["actual_reads"]=logs;return d;
}
