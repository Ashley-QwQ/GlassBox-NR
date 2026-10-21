// SHA expectations are compiled from the independently frozen proof manifest.
m.def("prepare_noise_assets",[](py::dict paths){
 AbiLease lease;
 if(noise_owner)throw std::runtime_error("D noise owner already prepared");
 auto owner=std::make_unique<dlssnr_optd::NoiseAssets>();
 auto load=[&](const char* kind,size_t n,const char* sha){
  std::string path=paths[py::str(kind)].cast<std::string>();
  std::string low=path;std::transform(low.begin(),low.end(),low.begin(),[](unsigned char c){return char(std::tolower(c));});
  if(path_is_forbidden(low))throw std::invalid_argument("D forbidden asset path");
  auto t=dlssnr_optd::CompactTable::load(path,n);
  auto view=py::memoryview::from_memory(t.raw_data(),py::ssize_t(t.raw_size()));
  std::string got=py::module_::import("hashlib").attr("sha256")(view).attr("hexdigest")().cast<std::string>();
  if(got!=sha)throw std::invalid_argument("D noise asset SHA mismatch");
  return t;
 };
 owner->radius=load("SQRT",5675352,"2c42bbbabc3ad60efe6f33096539786955e2885c182422cb9c07669f88ec540e");
 owner->cosine=load("COS",8635352,"d95b2ad4375921cc8da901e57ef8b46bef566c14ee84fe6aad3756025b3db5b3");
 owner->sine=load("SIN",8650456,"1ffe7b333e0155adb16ea1b8232c38013a3321bf6978ce07ddacdb6396fef1e3");
 noise_owner=std::move(owner);
});
m.def("noise_assets_live_bytes",[](){AbiLease lease;return noise_owner?noise_owner->radius.raw_size()+noise_owner->cosine.raw_size()+noise_owner->sine.raw_size():size_t(0);});
m.def("release_noise_assets",[](){AbiLease lease;noise_owner.reset();});
m.def("noise_trace_k",[](std::array<uint32_t,4> k){
 AbiLease lease;
 if(!noise_owner)throw std::runtime_error("D noise assets require L2 preparation");
 auto t=noise_owner->trace(k);std::array<uint16_t,3> h;
 for(int i=0;i<3;++i)h[i]=half_round(optm_bit_cast<float>(t.products[i]));
 py::dict d;d["k"]=t.k;d["radius"]=t.radius;d["trig"]=t.trig;d["products"]=t.products;d["half"]=h;return d;
});
m.def("noise_seed_k",&dlssnr_optd::noise_seed_k);
