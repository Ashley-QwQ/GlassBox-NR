// Token-layout repack for launches 58 and 99 (address mapping only; no arithmetic on payload bytes).
//
// The payload is a sequence of 4-byte chunks, 256 chunks per token, rows*cols tokens.
// Two layouts of the same tokens are connected by one fixed address mapping:
//   * "tiled":  tokens grouped in 4x4 spatial tiles (tile index = (row>>2)*ceil(cols/4) + (col>>2));
//               a tile holds 16 tokens x 256 chunks = 4096 chunks = 16 KiB.
//   * "raster": tokens in raster order (token = row*cols + col), 16 consecutive tokens per 16 KiB block.
// Inside a 16 KiB block, chunk (slot, c) with slot in [0,16) and c in [0,256) lives at the position
// obtained by permuting the 12 bits of x = slot*256 + c (see kBitDest). Both layouts use the same permutation.
// Launch 58 reads the tiled layout and writes the raster layout; launch 99 is its exact inverse.
//
// Validation of this mapping (not of the code text): byte-for-byte against GPU workspace captures of both
// launches at 1080, 1440 and 4K geometry (three frames each), and against the previous implementation over all
// accepted (rows, cols) with rows, cols <= 64. See the release documentation for the receipts.
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace py = pybind11;
namespace {

constexpr uint32_t kChunksPerToken = 256;
constexpr uint32_t kBlockShift = 12;                 // 4096 chunks per block
constexpr uint32_t kBlockMask = (1u << kBlockShift) - 1u;
// Destination bit of each of the 12 index bits of x = slot*256 + c (index = source bit).
constexpr uint32_t kBitDest[12] = {2, 3, 1, 7, 8, 9, 10, 11, 4, 5, 6, 0};

uint32_t swizzle(uint32_t x) {
    uint32_t y = 0;
    for (uint32_t b = 0; b < 12; ++b) y |= ((x >> b) & 1u) << kBitDest[b];
    return y;
}

// Chunk index (in 4-byte units) of logical chunk g = token*256 + c in the tiled layout.
uint32_t tiled_chunk(uint32_t g, uint32_t cols, const uint16_t* sw) {
    const uint32_t token = g / kChunksPerToken, c = g % kChunksPerToken;
    const uint32_t row = token / cols, col = token % cols;
    const uint32_t tiles_per_row = (cols + 3) / 4;
    const uint32_t tile = (row >> 2) * tiles_per_row + (col >> 2);
    const uint32_t slot = ((row & 3u) << 2) | (col & 3u);
    return (tile << kBlockShift) | sw[(slot << 8) | c];
}
// Chunk index of logical chunk g in the raster layout.
uint32_t raster_chunk(uint32_t g, const uint16_t* sw) {
    const uint32_t token = g / kChunksPerToken, c = g % kChunksPerToken;
    return ((token >> 4) << kBlockShift) | sw[((token & 15u) << 8) | c];
}

// Accepted dimensions: both multiples of 4, below the tested range, and not a single 4-wide strip.
void check_dims(uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0 || uint64_t(rows) * cols > 1000000)
        throw std::invalid_argument("invalid repack rows or cols");
    if ((rows & 3u) || (cols & 3u))
        throw std::invalid_argument("repack rows and cols must be multiples of 4");
    if ((rows == 4) != (cols == 4))
        throw std::invalid_argument("repack does not support a single 4-wide strip");
    if (rows >= 16384 || cols >= 65536)
        throw std::invalid_argument("repack rows or cols outside the supported range");
}

uint32_t read32(const std::string& s,size_t off){
    const auto* p=reinterpret_cast<const uint8_t*>(s.data()+off);
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
uint64_t read64(const std::string& s,size_t off){
    return uint64_t(read32(s,off))|(uint64_t(read32(s,off+4))<<32);
}
std::vector<uint16_t> swizzle_table() {
    std::vector<uint16_t> t(1u << kBlockShift);
    for (uint32_t x = 0; x < t.size(); ++x) t[x] = static_cast<uint16_t>(swizzle(x));
    return t;
}
py::bytes execute(int launch,py::bytes data,py::bytes params,int threads,uint32_t queue_seed){
    std::string input=data,p=params;
    if(launch!=58&&launch!=99)throw std::invalid_argument("launch must be 58 or 99");
    if(p.size()!=24)throw std::invalid_argument("repack params must be 24 bytes");
    const uint32_t rows=read32(p,16),cols=read32(p,20);
    check_dims(rows,cols);
    const uint64_t n=uint64_t(rows)*cols*kChunksPerToken,bytes=n*4;
    if(input.size()!=bytes)throw std::invalid_argument("repack input length does not match params");
    const std::vector<uint16_t> sw=swizzle_table();
    std::vector<uint32_t> src(n),dst(n);
    std::vector<uint8_t> src_seen(n,0),dst_seen(n,0);
    for(uint64_t i=0;i<n;++i){
        const uint32_t t=tiled_chunk(static_cast<uint32_t>(i),cols,sw.data());
        const uint32_t r=raster_chunk(static_cast<uint32_t>(i),sw.data());
        if(uint64_t(t)>=n||uint64_t(r)>=n)throw std::invalid_argument("map offset out of bounds");
        if(src_seen[t]++||dst_seen[r]++)throw std::invalid_argument("map not bijective");
        // byte offsets
        src[i]=(launch==58?t:r)*4U;dst[i]=(launch==58?r:t)*4U;
    }
    std::string output(bytes,'\0');
    threads=std::max(1,std::min(threads,12));
    // Each destination has exactly one owner; queue_seed rotates the ownership schedule.
    auto work=[&](int tid){
        const uint64_t begin=(uint64_t(tid)+queue_seed%uint32_t(threads))%uint32_t(threads);
        for(uint64_t i=begin;i<n;i+=uint64_t(threads))
            std::memcpy(output.data()+dst[i],input.data()+src[i],4);
    };
    {py::gil_scoped_release release;
     std::vector<std::thread> pool;
     for(int t=1;t<threads;++t)pool.emplace_back(work,t);
     work(0);for(auto& t:pool)t.join();}
    return py::bytes(output);
}
}
PYBIND11_MODULE(dlssnr_fam_repack,m){
  m.def("execute",&execute,py::arg("launch"),py::arg("data"),py::arg("params"),py::arg("threads")=1,py::arg("queue_seed")=0);
  m.def("execute_roles",[](int launch,py::bytes data,py::bytes params,uint64_t workspace_base,uint64_t workspace_bytes,int threads,uint32_t queue_seed){
      std::string p=params;
      if(p.size()!=24)throw std::invalid_argument("repack params must be 24 bytes");
      uint64_t src=read64(p,0),dst=read64(p,8);
      if(src<workspace_base||dst<workspace_base)
          throw std::invalid_argument("repack pointer below workspace base");
      const size_t size=std::string(data).size();
      if(src-workspace_base>workspace_bytes||dst-workspace_base>workspace_bytes||
         size>workspace_bytes-(src-workspace_base)||size>workspace_bytes-(dst-workspace_base))
          throw std::invalid_argument("repack role outside workspace");
      py::dict roles;
      roles["output"]=execute(launch,data,params,threads,queue_seed);
      roles["input_workspace_offset"]=py::int_(src-workspace_base);
      roles["output_workspace_offset"]=py::int_(dst-workspace_base);
      return roles;
  },py::arg("launch"),py::arg("data"),py::arg("params"),py::arg("workspace_base"),py::arg("workspace_bytes"),
    py::arg("threads")=1,py::arg("queue_seed")=0);
  // Launch 58 runs a grid padded to a multiple of 32 tokens (ceil(T/32)*32 blocks of 256 threads); the unused
  // grid slots store zero words into the padding role that follows the logical output.
  // execute_roles() is kept unchanged for the logical-output evidence.
  m.def("execute_roles_padded",[](py::bytes data,py::bytes params,uint64_t workspace_base,
                                  uint64_t workspace_bytes,uint32_t grid_x,int threads,uint32_t queue_seed){
      std::string p=params;
      if(p.size()!=24)throw std::invalid_argument("repack params must be 24 bytes");
      const uint32_t rows=read32(p,16),cols=read32(p,20);
      const uint64_t tokens=uint64_t(rows)*cols;
      if(rows==0||cols==0||tokens>1000000)
          throw std::invalid_argument("invalid repack dimensions");
      const uint64_t padded=((tokens+31)/32)*32;
      if(grid_x!=padded)throw std::invalid_argument("L58 grid.x does not match padded token count");
      const uint64_t logical_bytes=tokens*1024,padded_bytes=(padded-tokens)*1024;
      if(std::string(data).size()!=logical_bytes)
          throw std::invalid_argument("L58 logical input length mismatch");
      const uint64_t src=read64(p,0),dst=read64(p,8);
      if(src<workspace_base||dst<workspace_base||
         src-workspace_base>workspace_bytes||dst-workspace_base>workspace_bytes||
         logical_bytes>workspace_bytes-(src-workspace_base)||
         logical_bytes+padded_bytes>workspace_bytes-(dst-workspace_base))
          throw std::invalid_argument("L58 padded role outside workspace");
      py::bytes logical=execute(58,data,params,threads,queue_seed);
      const std::vector<uint16_t> sw=swizzle_table();
      std::string padding(padded_bytes,'\x7f');
      std::vector<uint8_t> seen(padded_bytes/4,0);
      for(uint64_t g=tokens*kChunksPerToken;g<padded*kChunksPerToken;++g){
          const uint64_t d=uint64_t(raster_chunk(static_cast<uint32_t>(g),sw.data()))*4U;
          if(d<logical_bytes||d>=logical_bytes+padded_bytes)
              throw std::invalid_argument("L58 fallback store outside padding role");
          const size_t slot=size_t((d-logical_bytes)/4);
          if(seen[slot]++)throw std::invalid_argument("L58 duplicate padding store");
          std::memset(padding.data()+slot*4,0,4); // unused grid slots store a zero word
      }
      if(std::any_of(seen.begin(),seen.end(),[](uint8_t x){return x!=1;}))
          throw std::invalid_argument("L58 padding coverage has holes");
      py::dict roles;
      roles["output"]=logical;
      roles["padding"]=py::bytes(padding);
      roles["input_workspace_offset"]=py::int_(src-workspace_base);
      roles["output_workspace_offset"]=py::int_(dst-workspace_base);
      roles["padding_workspace_offset"]=py::int_(dst-workspace_base+logical_bytes);
      roles["logical_tokens"]=py::int_(tokens);
      roles["padded_tokens"]=py::int_(padded);
      return roles;
  },py::arg("data"),py::arg("params"),py::arg("workspace_base"),py::arg("workspace_bytes"),
    py::arg("grid_x"),py::arg("threads")=1,py::arg("queue_seed")=0);
}
