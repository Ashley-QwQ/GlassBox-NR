m.def("reach_half_exhaustive",[]{
 auto exclusion=stage_lock();require_stage(157);
 py::list rows;float ep=ex(-.1f),ea=ex(0.f),eb=ex(-0.f);
 for(unsigned h=0;h<65536;h++){
  py::dict row;row["raw_half"]=h;py::list calls;std::string step="raw.fa";
  auto call=[&](const char* site,unsigned op,float x){step=site;py::dict c;c["site"]=site;c["op"]=op;c["input_bits"]=optm::bit_cast<uint32_t>(x);c["block"]=optm::bit_cast<uint32_t>(x)>>23;calls.append(c);float v=op==0?lg(x):ex(x);c["output_bits"]=optm::bit_cast<uint32_t>(v);return v;};
  try{
   float rawv=sat(fa(0,float(half_value(uint16_t(h)))));step="exposure.ff";
   float x=sat(ff(rawv,ep,0));step="pre.fm_ff";
   float x2=fm(x,x),twox=fa(x,x),three=fa(-twox,3),smooth=ff(x2,three,-x),pre=ff(smooth,-.25f,x);
   float l1=call("L157.lg1",0,pre);step="lg1.fm";float a=call("L157.ex1",1,fm(l1,ea));
   float l2=call("L157.lg2",0,std::max(0.f,a));step="lg2.fm";float rgb=call("L157.ex2",1,fm(l2,eb));
   row["rgb_bits"]=optm::bit_cast<uint32_t>(rgb);row["status"]="ACCEPTED";row["first_error"]=py::none();
  }catch(const std::exception& e){row["status"]="REJECTED_BY_ORIGINAL";py::dict failure;failure["step"]=step;failure["reason"]=e.what();row["first_error"]=failure;}
  row["calls"]=calls;rows.append(row);
 }
 py::dict result;result["rows"]=rows;result["ep_bits"]=optm::bit_cast<uint32_t>(ep);result["ea_bits"]=optm::bit_cast<uint32_t>(ea);result["eb_bits"]=optm::bit_cast<uint32_t>(eb);return result;
});
