#pragma once
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <utility>
#include <string>
#include <map>
#include <array>
#include <limits>
#include <stdexcept>
namespace geo2 {
namespace py=pybind11;
struct Geometry {
 int cw=0,ch=0,nw=0,nh=0,windows_x=0,windows_y=0;
 uint64_t workspace_base=0,weight_base=0;
 size_t workspace_bytes=0,weight_bytes=0;
 std::string id;
 std::map<int,std::string> parameters;
 std::map<int,std::array<uint64_t,5>> textures;
 uint64_t b70_feature=0,b70_skip=0,b70_weight=0,b70_blend=0;
 bool l0_identity=false;
 bool configured=false;
};
static Geometry state;
inline size_t product(size_t a,size_t b){
 if(a&&b>std::numeric_limits<size_t>::max()/a)throw std::out_of_range("GEO2_SIZE_OVERFLOW");return a*b;
}
inline const Geometry& g(){if(!state.configured)throw std::invalid_argument("GEO2_DESCRIPTOR_NOT_CONFIGURED");return state;}
inline void configure(py::dict d){
 Geometry n;
 if(py::cast<std::string>(d["schema"])!="GEO2_EXECUTION/1")throw std::invalid_argument("GEO2_SCHEMA");
 n.id=py::cast<std::string>(d["descriptor_sha256"]);
 if(n.id.size()!=64)throw std::invalid_argument("GEO2_DESCRIPTOR_ID");
 n.cw=py::cast<int>(d["colour_width"]);n.ch=py::cast<int>(d["colour_height"]);
 n.nw=py::cast<int>(d["network_width"]);n.nh=py::cast<int>(d["network_height"]);
 n.windows_x=py::cast<int>(d["L155_grid_x"]);n.windows_y=py::cast<int>(d["L155_grid_y"]);
 n.workspace_base=py::cast<uint64_t>(d["workspace_base"]);n.weight_base=py::cast<uint64_t>(d["weight_base"]);
 n.workspace_bytes=py::cast<size_t>(d["workspace_bytes"]);n.weight_bytes=py::cast<size_t>(d["weight_bytes"]);
 n.b70_feature=py::cast<uint64_t>(d["B70_feature_offset"]);n.b70_skip=py::cast<uint64_t>(d["B70_skip_offset"]);
 n.b70_weight=py::cast<uint64_t>(d["B70_weight_offset"]);n.b70_blend=py::cast<uint64_t>(d["B70_blend_offset"]);
 n.l0_identity=py::cast<bool>(d["L0_identity_sampling_proven"]);
 if(n.cw<=0||n.ch<=0||n.nw<n.cw||n.nh<n.ch||n.nw%32||n.nh%32||n.windows_x<=0||n.windows_y<=0||!n.workspace_base||!n.weight_base||!n.workspace_bytes||!n.weight_bytes)throw std::invalid_argument("GEO2_DESCRIPTOR_EXTENTS");
 if(n.workspace_base>UINT64_MAX-n.workspace_bytes||n.weight_base>UINT64_MAX-n.weight_bytes)throw std::out_of_range("GEO2_ALLOCATION_OVERFLOW");
 product(product(size_t(n.nw),size_t(n.nh)),32);product(size_t(n.windows_x),size_t(n.windows_y));
 for(auto item:py::cast<py::dict>(d["native_params"])){
  auto launch=py::cast<int>(item.first);auto raw=py::cast<std::string>(py::cast<py::bytes>(item.second));
  if(launch<0||launch>=158||raw.empty()||raw.size()%8)throw std::invalid_argument("GEO2_CB_DESCRIPTOR");n.parameters.emplace(launch,std::move(raw));
 }
 if(n.parameters.size()!=158)throw std::invalid_argument("GEO2_CB_SET_INCOMPLETE");
 for(auto item:py::cast<py::dict>(d["textures"])){
  int key=py::cast<int>(item.first);auto t=py::cast<py::dict>(item.second);
  n.textures[key]={py::cast<uint64_t>(t["width"]),py::cast<uint64_t>(t["height"]),py::cast<uint64_t>(t["pitch"]),py::cast<uint64_t>(t["texture_handle"]),py::cast<uint64_t>(t["surface_handle"])};
 }
 if(n.textures.size()!=6)throw std::invalid_argument("GEO2_TEXTURE_SET_INCOMPLETE");
 n.configured=true;state=std::move(n);
}
inline void parameters(int launch,const uint8_t* p,size_t n){
 const auto& s=g();auto it=s.parameters.find(launch);
 if(it==s.parameters.end()||it->second.size()!=n||std::memcmp(it->second.data(),p,n))throw std::invalid_argument("GEO2_NATIVE_CB_BINDING_DIFF");
}
inline size_t windows(){return product(size_t(g().windows_x),size_t(g().windows_y));}
inline void texture(int id,int width,int height,int pitch,uint64_t tex,uint64_t surf){
 auto it=g().textures.find(id);std::array<uint64_t,5> a{uint64_t(width),uint64_t(height),uint64_t(pitch),tex,surf};
 if(it==g().textures.end()||it->second!=a)throw std::invalid_argument("GEO2_TEXTURE_BINDING_DIFF");
}
inline py::dict probe(){const auto& s=g();py::dict d;d["descriptor_sha256"]=s.id;d["colour_width"]=s.cw;d["colour_height"]=s.ch;d["network_width"]=s.nw;d["network_height"]=s.nh;d["workspace_base"]=s.workspace_base;d["workspace_bytes"]=s.workspace_bytes;return d;}
}
