// B2-a: immutable, full-domain tables generated ONLY by the original functions.
#ifndef OPTB2_GENERATOR_SOURCE_SHA
#error "Build through the B2 CMake source identity manifest"
#endif
#ifdef __FAST_MATH__
#error "B2 requires strict floating point compilation"
#endif

constexpr const char* optb2_source_identity =
 "M/tin/tin.cpp=29092936af6682199ea9d055328c8d97f0fca3c9ffde0da94c3c0256df269a04;"
 "M/tin/primitives.inc=88649e232a2713344f1a5b141c70f29fb64b0e704204c37b2641c09845cd7532;"
 "M/shared/portable_bits.hpp=4309a826730758f4a4dbda24b73178f612a82d09bf50cd6b0671c86abd3132e4;"
 "M/shared/decode_reuse.inc=a9d4ac00c71d711fa5659f576236061c2155e2b667a55729ea497b7def97394c;"
 "primitive=half-binary64-RN-even/M-B-sat-v1;"
 "MSVC=/fp:strict,/fp:except-;GCC-Clang=-O2,-fno-fast-math,-ffp-contract=off;"
 "FP=FE_TONEAREST,IEEE-binary32-binary64,no-global-rounding-change,no-FTZ-DAZ-dependency;"
 "generator_source_manifest_sha256=" OPTB2_GENERATOR_SOURCE_SHA ";";

std::string optb2_build_identity(){
 std::string s=optb2_source_identity;
#if defined(_MSC_VER)
 s+="compiler=MSVC:"+std::to_string(_MSC_VER);
#elif defined(__clang__)
 s+=std::string("compiler=Clang:")+__clang_version__;
#else
 s+=std::string("compiler=GCC:")+__VERSION__;
#endif
 return s;
}
void optb2_fp_check(){
 if(std::fegetround()!=FE_TONEAREST || !std::numeric_limits<float>::is_iec559 ||
    !std::numeric_limits<double>::is_iec559 || sizeof(float)!=4 || sizeof(double)!=8)
  throw std::runtime_error("OPTB2_FP_ENVIRONMENT_MISMATCH");
}
struct Optb2Error {std::string cls,message;};
template<class T> struct Optb2UnaryTable {
 std::array<T,65536> values{};
 std::array<uint8_t,65536> outcomes{};
 std::vector<Optb2Error> errors;
 double generation_seconds=0;
 template<class F> void generate(F original){
  optb2_fp_check();auto start=std::chrono::steady_clock::now();
  for(uint32_t h=0;h<65536;++h){try{values[h]=original(uint16_t(h));}
   catch(const std::domain_error& e){
    size_t i=0;for(;i<errors.size();++i)if(errors[i].cls=="std::domain_error"&&errors[i].message==e.what())break;
    if(i==errors.size())errors.push_back({"std::domain_error",e.what()});
    if(i>=255)throw std::runtime_error("OPTB2_OUTCOME_CATALOG_OVERFLOW");
    outcomes[h]=uint8_t(i+1);
   }
  }
  generation_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 }
 T lookup(uint16_t h)const{
  auto outcome=outcomes[h];if(outcome){
   const auto& e=errors.at(size_t(outcome)-1);
   if(e.cls=="std::domain_error")throw std::domain_error(e.message);
   throw std::runtime_error("OPTB2_UNKNOWN_OUTCOME_CLASS");
  }return values[h];
 }
};
Optb2UnaryTable<uint8_t> optb2_gate_table;
Optb2UnaryTable<uint16_t> optb2_score_table;
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
bool optb2_gate_enabled=true,optb2_score_enabled=true;
#else
bool optb2_gate_enabled=false,optb2_score_enabled=false;
#endif

void optb2_put32(std::string& s,uint32_t v){for(int i=0;i<4;++i)s.push_back(char(v>>(8*i)));}
void optb2_put_string(std::string& s,const std::string& v){optb2_put32(s,uint32_t(v.size()));s+=v;}
template<class T> std::string optb2_export(const char* name,const Optb2UnaryTable<T>& t){
 // Header: magic[8], version u32, LE tag u32, count u32, value width u32,
 // payload length u32, identity length u32; all integers are little-endian.
 std::string s("B2UNARY\0",8);optb2_put32(s,1);optb2_put32(s,0x01020304);
 optb2_put32(s,65536);optb2_put32(s,sizeof(T));optb2_put32(s,65536*(sizeof(T)+1));
 optb2_put_string(s,std::string(name)+";"+optb2_build_identity());
 for(T v:t.values)for(size_t i=0;i<sizeof(T);++i)s.push_back(char(uint32_t(v)>>(8*i)));
 s.append(reinterpret_cast<const char*>(t.outcomes.data()),t.outcomes.size());
 optb2_put32(s,uint32_t(t.errors.size()));
 for(const auto& e:t.errors){optb2_put_string(s,e.cls);optb2_put_string(s,e.message);}
 return s;
}
template<class T> void optb2_import(const char* name,const std::string& b,Optb2UnaryTable<T>& t){
 optb2_fp_check();
 if(b.size()<32)throw std::invalid_argument("OPTB2_TABLE_TRUNCATED");
 auto u32=[&](size_t i){uint32_t v=0;for(int k=0;k<4;++k)v|=uint32_t(uint8_t(b[i+k]))<<(8*k);return v;};
 if(b.compare(0,8,std::string("B2UNARY\0",8)))throw std::invalid_argument("OPTB2_TABLE_MAGIC");
 if(u32(8)!=1)throw std::invalid_argument("OPTB2_TABLE_VERSION");
 if(u32(12)!=0x01020304)throw std::invalid_argument("OPTB2_TABLE_ENDIAN");
 if(u32(16)!=65536||u32(20)!=sizeof(T)||u32(24)!=65536*(sizeof(T)+1))throw std::invalid_argument("OPTB2_TABLE_LENGTH");
 const auto expected=optb2_export(name,t);
 if(b.size()!=expected.size())throw std::invalid_argument("OPTB2_TABLE_LENGTH");
 if(b!=expected)throw std::invalid_argument("OPTB2_TABLE_IDENTITY_MISMATCH");
 // Decode the verified file into a complete staging table, then publish under
 // AbiLease. A reload cannot mutate bytes while any launch worker is running.
 Optb2UnaryTable<T> loaded;size_t offset=32+u32(28);
 for(auto& v:loaded.values){uint32_t value=0;for(size_t k=0;k<sizeof(T);++k)value|=uint32_t(uint8_t(b[offset++]))<<(8*k);v=T(value);}
 for(auto& outcome:loaded.outcomes)outcome=uint8_t(b[offset++]);
 uint32_t count=u32(offset);offset+=4;
 auto getstring=[&](){uint32_t len=u32(offset);offset+=4;std::string s=b.substr(offset,len);offset+=len;return s;};
 for(uint32_t i=0;i<count;++i){auto cls=getstring();auto message=getstring();loaded.errors.push_back({cls,message});}
 t.values=loaded.values;t.outcomes=loaded.outcomes;t.errors=std::move(loaded.errors);
}
uint64_t optb2_shuffle_seed=0;
std::vector<int> optb2_last_order;
uint64_t optb2_last_seed=0,optb2_last_permutation=0;
int optb2_last_completed=0,optb2_last_claimed=0,optb2_last_launch=-1,optb2_last_threads=0;
uint64_t optb2_random(uint64_t& s){s+=0x9e3779b97f4a7c15ULL;uint64_t z=s;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
std::vector<int> optb2_task_order(int count){
 std::vector<int> order(count);std::iota(order.begin(),order.end(),0);
 uint64_t s=optb2_shuffle_seed;
 if(s)for(size_t i=order.size();i>1;--i)std::swap(order[i-1],order[optb2_random(s)%i]);
 uint64_t hash=14695981039346656037ULL;
 for(int v:order)for(int k=0;k<4;++k){hash^=uint8_t(uint32_t(v)>>(8*k));hash*=1099511628211ULL;}
 optb2_last_order=order;optb2_last_seed=optb2_shuffle_seed;optb2_last_permutation=hash;
 optb2_last_completed=0;optb2_last_claimed=0;return order;
}
