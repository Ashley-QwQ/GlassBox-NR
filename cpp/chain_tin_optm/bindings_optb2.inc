 m.def("set_optb2_tables",[](bool gate,bool score){AbiLease lease;optb2_fp_check();optb2_gate_enabled=gate;optb2_score_enabled=score;},py::arg("gate"),py::arg("score"));
 m.def("set_optb2_task_shuffle",[](uint64_t seed){AbiLease lease;optb2_shuffle_seed=seed;},py::arg("seed"));
 m.def("optb2_task_order_receipt",[](){
  AbiLease lease;py::dict d;d["launch"]=optb2_last_launch;d["threads"]=optb2_last_threads;d["seed"]=optb2_last_seed;d["permutation"]=optb2_last_order;d["permutation_identity_fnv1a64_le_u32"]=optb2_last_permutation;d["task_count"]=optb2_last_order.size();d["claimed_count"]=optb2_last_claimed;d["completed_count"]=optb2_last_completed;
  auto sorted=optb2_last_order;std::sort(sorted.begin(),sorted.end());bool verified=true,changed=false;
  std::string natural,executed;
  for(size_t i=0;i<sorted.size();++i){if(sorted[i]!=int(i))verified=false;if(optb2_last_order[i]!=int(i))changed=true;optb2_put32(natural,uint32_t(i));if(i<size_t(optb2_last_claimed))optb2_put32(executed,uint32_t(optb2_last_order[i]));}
  auto sha=[](const std::string& b){return py::module_::import("hashlib").attr("sha256")(py::bytes(b)).attr("hexdigest")();};
  d["permutation_verified"]=verified;d["order_changed"]=changed;d["natural_order_sha256"]=sha(natural);d["executed_order_sha256"]=sha(executed);
  d["algorithm"]="splitmix64-fisher-yates-v1;seed0=identity";d["scope"]="independent_window_tasks_only;executed-order means actual claimed task IDs in atomic ticket order";return d;
 });
 m.def("optb2_table_export",[](std::string name){AbiLease lease;if(name=="tin_gate")return py::bytes(optb2_export("tin_gate",optb2_gate_table));if(name=="tin_score")return py::bytes(optb2_export("tin_score",optb2_score_table));throw std::invalid_argument("OPTB2_TABLE_NAME");});
 m.def("optb2_table_import",[](std::string name,py::bytes raw){AbiLease lease;std::string b=raw;if(name=="tin_gate")optb2_import("tin_gate",b,optb2_gate_table);else if(name=="tin_score")optb2_import("tin_score",b,optb2_score_table);else throw std::invalid_argument("OPTB2_TABLE_NAME");});
 m.def("optb2_table_metadata",[](){AbiLease lease;py::dict d;d["source_identity"]=optb2_build_identity();d["gate_payload_bytes"]=sizeof(optb2_gate_table.values)+sizeof(optb2_gate_table.outcomes);d["score_payload_bytes"]=sizeof(optb2_score_table.values)+sizeof(optb2_score_table.outcomes);d["gate_object_bytes"]=sizeof(optb2_gate_table);d["score_object_bytes"]=sizeof(optb2_score_table);d["gate_generation_seconds"]=optb2_gate_table.generation_seconds;d["score_generation_seconds"]=optb2_score_table.generation_seconds;d["gate_enabled"]=optb2_gate_enabled;d["score_enabled"]=optb2_score_enabled;d["catalog_scope"]="std::domain_error and exact original what()";return d;});
 auto optb2_evaluate=[](const std::string& name,uint16_t h,bool table){
  py::dict d;uint16_t bits=0;uint8_t outcome=0;std::string cls,message;
  if(name!="tin_gate"&&name!="tin_score")throw std::invalid_argument("OPTB2_TABLE_NAME");
  try{bits=name=="tin_gate"?(table?gate(h):gate_reference(h)):(table?score_unary(h):score_reference(h));}
  catch(const std::domain_error& e){cls="std::domain_error";message=e.what();const auto& errors=name=="tin_gate"?optb2_gate_table.errors:optb2_score_table.errors;for(size_t i=0;i<errors.size();++i)if(errors[i].cls==cls&&errors[i].message==message){outcome=uint8_t(i+1);break;}if(!outcome)throw std::runtime_error("OPTB2_OUTCOME_CATALOG_MISMATCH");}
  d["value"]=bits;d["bits"]=bits;d["outcome"]=outcome;d["error_class"]=cls;d["error_message"]=message;return d;
 };
 m.def("optb2_evaluate",[optb2_evaluate](std::string name,uint16_t h,bool table){AbiLease lease;optb2_fp_check();return optb2_evaluate(name,h,table);},py::arg("name"),py::arg("h"),py::arg("use_table"));
 m.def("optb2_evaluate_batch",[optb2_evaluate](std::string name,const std::vector<uint16_t>& inputs,bool table){AbiLease lease;optb2_fp_check();py::list rows;for(auto h:inputs)rows.append(optb2_evaluate(name,h,table));return rows;},py::arg("name"),py::arg("inputs"),py::arg("use_table"));
 m.def("optb2_throw",[](std::string name,uint16_t h,bool table)->uint16_t{AbiLease lease;optb2_fp_check();if(name=="tin_gate")return table?gate(h):gate_reference(h);if(name=="tin_score")return table?score_unary(h):score_reference(h);throw std::invalid_argument("OPTB2_TABLE_NAME");},py::arg("name"),py::arg("h"),py::arg("use_table"));
 // A call-site ordering fixture. Score roots are intentionally supplied by the
 // fixture AFTER a real QMMA stage; production attn always uses that QMMA result.
 // The qmma fault is a real original E4 NaN refusal, never a fabricated exception.
 m.def("optb2_callsite_probe",[](std::string site,const std::vector<uint16_t>& inputs,int qmma_failure_index,std::string mode){
  AbiLease lease;optb2_fp_check();bool isscore=site=="score";
  if(site!="gate_c32"&&site!="gate_head"&&!isscore)throw std::invalid_argument("OPTB2_CALLSITE_NAME");
  if(mode!="original"&&mode!="lookup"&&mode!="preflight"&&mode!="reverse"&&mode!="nodebatch")throw std::invalid_argument("OPTB2_CALLSITE_MODE");
  py::list events,outputs;py::dict error;bool table=mode!="original";
  std::vector<uint16_t> result(inputs.size());std::vector<int> order(inputs.size());std::iota(order.begin(),order.end(),0);if(mode=="reverse")std::reverse(order.begin(),order.end());
  int active=-1;std::string active_stage;
  auto event=[&](int i,const char* stage,const char* phase){py::dict e;e["index"]=i;e["stage"]=stage;e["phase"]=phase;events.append(e);};
  auto qmma=[&](int i){active=i;active_stage="qmma";event(i,"qmma","begin");uint8_t a[32]{},b[32]{};if(i==qmma_failure_index)a[0]=0x7f;auto v=qmma_dot(a,b,0);(void)v;event(i,"qmma","success");};
  auto unary=[&](int i){active=i;active_stage=isscore?"score":"gate";event(i,active_stage.c_str(),"begin");uint16_t v=isscore?(table?score_unary(inputs[i]):score_reference(inputs[i])):(table?gate(inputs[i]):gate_reference(inputs[i]));event(i,active_stage.c_str(),"success");return v;};
  auto commit=[&](int i,uint16_t v){py::dict o;o["index"]=i;o["bits"]=v;outputs.append(o);event(i,"commit","success");};
  try{
   if(mode=="preflight"){
    // Negative control: actually evaluate the whole sequence before publishing.
    for(int i:order){if(isscore)qmma(i);result[i]=unary(i);}for(int i:order)commit(i,result[i]);
   }else if(mode=="nodebatch"&&isscore){
    // Negative control: all QMMA nodes before the first unary node.
    for(int i:order)qmma(i);for(int i:order){result[i]=unary(i);commit(i,result[i]);}
   }else for(int i:order){if(isscore)qmma(i);result[i]=unary(i);commit(i,result[i]);}
  }catch(const std::domain_error& e){event(active,active_stage.c_str(),"error");error["class"]="std::domain_error";error["message"]=e.what();error["index"]=active;error["stage"]=active_stage;}
  py::dict d;d["events"]=events;d["outputs"]=outputs;d["error"]=error;d["site"]=site;d["mode"]=mode;d["fixture_scope"]="supplied unary roots; real interleaved original qmma stage; not launch evidence";return d;
 },py::arg("site"),py::arg("half_list"),py::arg("qmma_failure_index")=-1,py::arg("mode")="lookup");
 m.def("optb2_microbench",[](std::string name,const std::vector<uint16_t>& inputs,int threads,uint64_t cycles,bool table){
  AbiLease lease;optb2_fp_check();
  if(name!="tin_gate"&&name!="tin_score")throw std::invalid_argument("OPTB2_TABLE_NAME");
  if(threads<1||threads>12||inputs.empty()||!cycles)throw std::invalid_argument("OPTB2_MICROBENCH_GEOMETRY");
  if(cycles>UINT64_MAX/inputs.size())throw std::invalid_argument("OPTB2_MICROBENCH_CALL_OVERFLOW");
  const auto& catalog=name=="tin_gate"?optb2_gate_table.errors:optb2_score_table.errors;
  struct WorkerResult {double seconds=0;uint64_t calls=0,checksum=0;std::vector<uint64_t> errors;};
  std::vector<WorkerResult> results(threads);for(auto& r:results)r.errors.assign(catalog.size(),0);
  double wall=0;int ready=0,done=0;bool go=false;std::exception_ptr ep;std::mutex error_mutex,control_mutex;std::condition_variable control_cv;std::chrono::steady_clock::time_point wall_stop;
  {py::gil_scoped_release release;
   auto worker=[&](int t){auto& r=results[t];{std::unique_lock<std::mutex> l(control_mutex);++ready;control_cv.notify_all();control_cv.wait(l,[&](){return go;});}auto start=std::chrono::steady_clock::now();
    try{for(uint64_t cycle=0;cycle<cycles;++cycle)for(size_t i=size_t(t);i<inputs.size();i+=size_t(threads)){
     uint16_t value=0;uint64_t outcome=0;
     try{value=name=="tin_gate"?(table?gate(inputs[i]):gate_reference(inputs[i])):(table?score_unary(inputs[i]):score_reference(inputs[i]));}
     catch(const std::domain_error& e){size_t k=0;for(;k<catalog.size();++k)if(catalog[k].cls=="std::domain_error"&&catalog[k].message==e.what())break;if(k==catalog.size())throw std::runtime_error("OPTB2_OUTCOME_CATALOG_MISMATCH");++r.errors[k];outcome=k+1;}
     ++r.calls;r.checksum+=uint64_t(value)+(outcome<<32)+uint64_t(inputs[i])*65537ULL+cycle;
    }}catch(...){std::lock_guard<std::mutex> l(error_mutex);if(!ep)ep=std::current_exception();}
    r.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();{std::lock_guard<std::mutex> l(control_mutex);if(++done==threads)wall_stop=std::chrono::steady_clock::now();}control_cv.notify_all();
   };
   // Workers exist and wait before the stopwatch; joins are outside it.
   std::vector<std::thread> workers;for(int t=0;t<threads;++t)workers.emplace_back(worker,t);
   {std::unique_lock<std::mutex> l(control_mutex);control_cv.wait(l,[&](){return ready==threads;});auto start=std::chrono::steady_clock::now();go=true;control_cv.notify_all();control_cv.wait(l,[&](){return done==threads;});wall=std::chrono::duration<double>(wall_stop-start).count();}
   for(auto& w:workers)w.join();
  }
  if(ep)std::rethrow_exception(ep);uint64_t calls=0,checksum=0;double aggregate=0;std::vector<uint64_t> errors(catalog.size(),0);
  for(const auto& r:results){calls+=r.calls;checksum+=r.checksum;aggregate+=r.seconds;for(size_t k=0;k<errors.size();++k)errors[k]+=r.errors[k];}
  py::list errors_out;for(size_t k=0;k<errors.size();++k){py::dict e;e["class"]=catalog[k].cls;e["message"]=catalog[k].message;e["count"]=errors[k];errors_out.append(e);}
  py::dict d;d["wall_sec"]=wall;d["aggregate_worker_sec"]=aggregate;d["calls"]=calls;d["checksum"]=checksum;d["errors"]=errors_out;d["threads"]=threads;d["cycles"]=cycles;d["name"]=name;d["use_table"]=table;d["gate_enabled"]=optb2_gate_enabled;d["score_enabled"]=optb2_score_enabled;d["scope"]="native scalar unary hot-loop; thread creation, joins and table generation excluded; not run_launch timing";return d;
 },py::arg("name"),py::arg("inputs"),py::arg("threads"),py::arg("cycles"),py::arg("use_table"));
