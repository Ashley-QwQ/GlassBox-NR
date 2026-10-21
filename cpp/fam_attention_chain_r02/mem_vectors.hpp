#pragma once
#include <vector>
#include <atomic>
#include <memory>
#include <type_traits>
#include <cstdlib>
#include <new>
#include <cstdint>
#include <cstddef>
// Module-local C++ allocation requests, active only during numerical execute.
// Python allocators, borrowed buffers, external CRT/OS stacks are excluded.
namespace chain_mem {
inline std::atomic<size_t> live[5]{},peak[5]{},total{0},high{0};
inline std::atomic<bool> active{false};
inline thread_local int group=4;
inline void rise(std::atomic<size_t>& p,size_t v){size_t old=p.load();while(v>old&&!p.compare_exchange_weak(old,v)){};}
struct Header {void* raw;size_t bytes;int category;bool counted;};
inline void* request(size_t n,size_t alignment){
 if(alignment<alignof(std::max_align_t))alignment=alignof(std::max_align_t);
 if(n>SIZE_MAX-alignment-sizeof(Header))throw std::bad_alloc();
 void* raw=std::malloc(n+alignment+sizeof(Header));if(!raw)throw std::bad_alloc();
 auto address=(reinterpret_cast<uintptr_t>(raw)+sizeof(Header)+alignment-1)&~uintptr_t(alignment-1);
 auto h=reinterpret_cast<Header*>(address)-1;*h={raw,n,group,active.load()};
 if(h->counted){rise(peak[group],live[group].fetch_add(n)+n);rise(high,total.fetch_add(n)+n);}
 return reinterpret_cast<void*>(address);
}
inline void release(void* p)noexcept{if(!p)return;auto h=reinterpret_cast<Header*>(p)-1;if(h->counted){live[h->category].fetch_sub(h->bytes);total.fetch_sub(h->bytes);}std::free(h->raw);}
inline void begin(){high=total.load();for(int i=0;i<5;i++)peak[i]=live[i].load();active=true;}
inline pybind11::dict finish(){active=false;namespace py=pybind11;py::dict d,b;const char* names[]={"code_and_ownership_vectors","half_accumulator_vectors","integer_address_vectors","program_and_worker_vectors","strings_nodes_and_other_cpp_heap"};for(int i=0;i<5;i++){py::dict x;x["peak_bytes"]=peak[i].load();x["live_end_bytes"]=live[i].load();b[names[i]]=x;}d["peak_allocated_bytes"]=high.load();d["live_end_bytes"]=total.load();d["buffers"]=b;d["status"]="CXX_MODULE_NEW_ALLOCATOR_MEASURED";d["scope"]="Numerical call's module-local C++ new/new[] requests, including vectors, strings and STL nodes. Borrowed arrays/workspace/weights/Python allocators excluded.";d["unaccounted"]="Allocations wholly inside external CRT/OS libraries, worker stacks and static storage; requested payload counts exclude allocator headers/alignment overhead";d["process_rss_included"]=false;return d;}
struct Tag {int saved;explicit Tag(int n):saved(group){group=n;}~Tag(){group=saved;}};
template<class T> constexpr int category(){return std::is_same_v<T,uint8_t>?0:std::is_same_v<T,uint16_t>?1:std::is_integral_v<T>?2:3;}
template<class T> struct Alloc {using value_type=T;Alloc()=default;template<class U>Alloc(const Alloc<U>&){}T* allocate(size_t n){Tag tag(category<T>());return std::allocator<T>{}.allocate(n);}void deallocate(T* p,size_t n){std::allocator<T>{}.deallocate(p,n);}template<class U>bool operator==(const Alloc<U>&)const{return true;}};
}
void* operator new(size_t n){return chain_mem::request(n,alignof(std::max_align_t));}
void* operator new[](size_t n){return chain_mem::request(n,alignof(std::max_align_t));}
void operator delete(void* p)noexcept{chain_mem::release(p);}
void operator delete[](void* p)noexcept{chain_mem::release(p);}
void operator delete(void* p,size_t)noexcept{chain_mem::release(p);}
void operator delete[](void* p,size_t)noexcept{chain_mem::release(p);}
void* operator new(size_t n,std::align_val_t a){return chain_mem::request(n,size_t(a));}
void* operator new[](size_t n,std::align_val_t a){return chain_mem::request(n,size_t(a));}
void operator delete(void* p,std::align_val_t)noexcept{chain_mem::release(p);}
void operator delete[](void* p,std::align_val_t)noexcept{chain_mem::release(p);}
void operator delete(void* p,size_t,std::align_val_t)noexcept{chain_mem::release(p);}
void operator delete[](void* p,size_t,std::align_val_t)noexcept{chain_mem::release(p);}
template<class T>using ChainVector=std::vector<T,chain_mem::Alloc<T>>;
