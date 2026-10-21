#pragma once
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <mutex>
#include <string>
#include <vector>
#include <stdexcept>
#include "reach_identity.hpp"
namespace mufu_full_r01 {
namespace py=pybind11;
constexpr size_t BLOCK_BYTES=33554432,CACHE_BLOCKS=2;
struct Mapping {int fd=-1;std::string path,sha;unsigned op=0,block=0;size_t bytes=0;struct stat identity{};};
struct Slot {void* p=nullptr;int member=-1;uint64_t age=0;};
inline std::vector<Mapping> owners;
inline std::array<std::array<int,512>,3> member=[](){std::array<std::array<int,512>,3> a{};for(auto& row:a)row.fill(-1);return a;}();
inline std::array<Slot,CACHE_BLOCKS> cache{};
inline std::mutex cache_mutex;
inline bool registered=false,failed=false;
inline uint32_t zero_word=0;
inline uint64_t clock=0,misses=0,registration_read_bytes=0;
inline void drop_cache(){for(auto& s:cache){if(s.p)munmap(s.p,BLOCK_BYTES);s={};}}
inline void clear(){std::lock_guard<std::mutex> lock(cache_mutex);drop_cache();for(auto& m:owners)if(m.fd>=0)close(m.fd);owners.clear();for(auto& row:member)row.fill(-1);registered=false;failed=false;zero_word=0;}
inline void clear_cache(){std::lock_guard<std::mutex> lock(cache_mutex);drop_cache();}
inline bool has(unsigned op,unsigned b){return registered&&!failed&&op<3&&b<512&&member[op][b]>=0;}
inline bool lookup(unsigned op,uint32_t b,uint32_t& value){
 std::lock_guard<std::mutex> lock(cache_mutex);
 if(!has(op,b>>23))return false;
 const int ix=member[op][b>>23];const auto& m=owners.at(ix);
 if(m.bytes==4){if(b!=0)return false;value=zero_word;return true;}
 try{
  for(auto& s:cache)if(s.p&&s.member==ix){s.age=++clock;value=static_cast<const uint32_t*>(s.p)[b&0x7fffff];return true;}
  struct stat current{};if(fstat(m.fd,&current)||current.st_dev!=m.identity.st_dev||current.st_ino!=m.identity.st_ino||current.st_size!=m.identity.st_size||current.st_mtim.tv_sec!=m.identity.st_mtim.tv_sec||current.st_mtim.tv_nsec!=m.identity.st_mtim.tv_nsec||current.st_ctim.tv_sec!=m.identity.st_ctim.tv_sec||current.st_ctim.tv_nsec!=m.identity.st_ctim.tv_nsec)throw std::domain_error("OUT_OF_D PINNED_FD_CHANGED");
  Slot* slot=&cache[0];for(auto& s:cache)if(!s.p){slot=&s;break;}else if(s.age<slot->age)slot=&s;
  if(slot->p){munmap(slot->p,BLOCK_BYTES);*slot={};}
  void* p=mmap(nullptr,BLOCK_BYTES,PROT_READ,MAP_PRIVATE,m.fd,0);if(p==MAP_FAILED)throw std::domain_error("OUT_OF_D CACHE_MMAP");
  slot->p=p;slot->member=ix;slot->age=++clock;misses++;value=static_cast<const uint32_t*>(p)[b&0x7fffff];return true;
 }catch(...){failed=true;drop_cache();throw;}
}
inline void init(py::list descriptors){
 if(registered||failed)throw std::domain_error("PRESERVE_ALREADY_REGISTERED_OR_FAILED_REQUIRES_EXPLICIT_CLEAR");
 std::vector<Mapping> pending;std::array<std::array<int,512>,3> views{};for(auto& row:views)row.fill(-1);
 registration_read_bytes=0;misses=0;clock=0;
 try{
  if(py::len(descriptors)!=reach_identity::COUNT)throw std::domain_error("INVALID_REACH_MEMBER_COUNT");
  for(auto item:descriptors){
   auto d=py::cast<py::dict>(item);unsigned op=py::cast<unsigned>(d["op"]),block=py::cast<unsigned>(d["block"]);auto path=py::cast<std::string>(d["path"]),sha=py::cast<std::string>(d["sha256"]);size_t bytes=py::cast<size_t>(d["bytes"]);std::string low=path;std::transform(low.begin(),low.end(),low.begin(),[](unsigned char c){return std::tolower(c);});
   if(op>=3||block>=512||views[op][block]>=0||path_is_forbidden(low)||low.rfind("f:",0)==0)throw std::domain_error("INVALID_REACH_DESCRIPTOR");
   auto expected=reach_identity::sha(op,block);if(!expected||sha!=expected||bytes!=(op==2&&block==0?4:BLOCK_BYTES))throw std::domain_error("REACH_COMPILED_IDENTITY_DIFF");
   if(bytes==4){if(py::cast<unsigned>(d["word_index"])!=0||py::cast<size_t>(d["parent_bytes"])!=BLOCK_BYTES||py::cast<std::string>(d["parent_sha256"])!=reach_identity::RCP_ZERO_PARENT_SHA||py::cast<std::string>(d["format"])!="uint32-le")throw std::domain_error("RCP_ZERO_MANIFEST_BINDING");}
   int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);if(fd<0)throw std::domain_error("REACH_OPEN_FAILED");struct stat st{};
   pending.push_back({fd,path,sha,op,block,bytes,st});
   if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size!=static_cast<off_t>(bytes))throw std::domain_error("REACH_SIZE");pending.back().identity=st;
   void* p=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);if(p==MAP_FAILED)throw std::domain_error("REACH_VERIFY_MMAP");
   try{auto view=py::memoryview::from_memory(static_cast<const void*>(p),bytes);auto actual=py::module_::import("hashlib").attr("sha256")(view).attr("hexdigest")().cast<std::string>();registration_read_bytes+=bytes;if(actual!=sha)throw std::domain_error("REACH_ACTUAL_READ_SHA");if(bytes==4)std::memcpy(&zero_word,p,4);}catch(...){munmap(p,bytes);throw;}
   munmap(p,bytes);views[op][block]=static_cast<int>(pending.size()-1);
  }
  for(auto& m:pending){struct stat st{};if(fstat(m.fd,&st)||st.st_size!=m.identity.st_size||st.st_mtim.tv_sec!=m.identity.st_mtim.tv_sec||st.st_mtim.tv_nsec!=m.identity.st_mtim.tv_nsec||st.st_ctim.tv_sec!=m.identity.st_ctim.tv_sec||st.st_ctim.tv_nsec!=m.identity.st_ctim.tv_nsec)throw std::domain_error("REACH_REGISTRATION_CHANGED");}
 }catch(...){for(auto& m:pending)close(m.fd);zero_word=0;failed=true;throw;}
 member=views;owners=std::move(pending);registered=true;
}
inline py::list pins(){py::list r;for(auto& m:owners){py::dict d;d["op"]=m.op;d["block"]=m.block;d["path"]=m.path;d["sha256"]=m.sha;d["bytes"]=m.bytes;d["storage"]="verified read-only pinned fd; global two-block mmap cache; explicit frame/failure retirement";r.append(d);}return r;}
inline py::dict stats(){std::lock_guard<std::mutex> lock(cache_mutex);py::dict r;size_t live=0;for(auto& s:cache)if(s.p)live+=BLOCK_BYTES;r["registered"]=registered;r["failed"]=failed;r["mapped_bytes"]=live;r["cache_bound_bytes"]=CACHE_BLOCKS*BLOCK_BYTES;r["registration_verify_bound_bytes"]=BLOCK_BYTES;r["registration_read_bytes"]=registration_read_bytes;r["cache_misses"]=misses;r["fd_count"]=owners.size();return r;}
}
