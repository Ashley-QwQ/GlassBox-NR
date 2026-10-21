#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dlssnr_optd {
inline constexpr uint32_t noise_k_count = 1u << 24;
inline constexpr const char* noise_membership_error = "MUFU noise outside measured membership";
class FormatError : public std::runtime_error {
public:
    std::string category;
    explicit FormatError(const char* c) : std::runtime_error(std::string("D noise format: ")+c), category(c) {}
};
inline uint32_t read_le32(const uint8_t* p) noexcept {
    return uint32_t(p[0]) | (uint32_t(p[1])<<8) | (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24);
}
inline int64_t floor_div_optd(int64_t n, int64_t d) noexcept {
    const int64_t q=n/d;
#ifdef D_OPTD_NEGATIVE_TRUNC_FLOOR
    return q;
#else
    return q - int64_t(n<0 && n%d!=0);
#endif
}
inline std::vector<uint8_t> load_bytes_optd(const std::string& path, size_t expected_bytes) {
    std::ifstream f(path, std::ios::binary|std::ios::ate);
    if (!f) throw FormatError("FILE_OPEN");
    const auto length=f.tellg();
    if (length<0 || uint64_t(length)!=expected_bytes) throw FormatError("FILE_LENGTH");
    std::vector<uint8_t> data(expected_bytes);
    f.seekg(0); f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!f || f.peek()!=std::char_traits<char>::eof()) throw FormatError("FILE_READ_LENGTH");
    return data;
}

// DNOISE01: packed LE metadata and lane-major zigzag residuals. The format
// has version/endian in its fixed magic and explicit LE contract, no native structs.
// Structural validation is deliberately separate from external SHA authentication.
class CompactTable {
    std::vector<uint8_t> data_;
    uint32_t n_=0, b_=0, nb_=0, payload_bytes_=0;
    size_t end_=0, width_=0, offset_=0, payload_=0;
    void validate(uint32_t expected_n) {
        if(data_.size()<24) throw FormatError("HEADER_TRUNCATED");
        if(std::memcmp(data_.data(),"DNOISE01",8)!=0) throw FormatError("MAGIC_VERSION");
        n_=read_le32(data_.data()+8); b_=read_le32(data_.data()+12);
        nb_=read_le32(data_.data()+16); payload_bytes_=read_le32(data_.data()+20);
        if(n_!=expected_n || n_==0 || n_>noise_k_count) throw FormatError("COUNT_ENDIAN");
        if(b_<8 || b_>512 || (b_&(b_-1)) || n_%b_ || nb_!=n_/b_) throw FormatError("BLOCK_LAYOUT");
        end_=24+size_t(nb_)*4; width_=24+size_t(nb_)*8;
        offset_=24+size_t(nb_)*9; payload_=24+size_t(nb_)*13;
        if(uint64_t(payload_)+payload_bytes_!=data_.size()) throw FormatError("EXACT_LENGTH");
        uint64_t cursor=0;
        for(uint32_t i=0;i<nb_;++i) {
            const uint32_t w=data_[width_+i];
            if(w>33) throw FormatError("WIDTH");
            if(read_le32(data_.data()+offset_+size_t(i)*4)!=cursor) throw FormatError("OFFSET_LAYOUT");
            cursor+=uint64_t(w)*b_/8;
            if(cursor>payload_bytes_) throw FormatError("OFFSET_BOUNDS");
        }
        if(cursor!=payload_bytes_) throw FormatError("PAYLOAD_LENGTH");
    }
public:
    CompactTable()=default;
    static CompactTable from_bytes(std::vector<uint8_t> data, uint32_t expected_n=noise_k_count) {
        CompactTable t; t.data_=std::move(data); t.validate(expected_n); return t;
    }
    static CompactTable load(const std::string& path,size_t expected_bytes,uint32_t expected_n=noise_k_count) {
        return from_bytes(load_bytes_optd(path,expected_bytes),expected_n);
    }
    const uint8_t* raw_data() const noexcept { return data_.data(); }
    size_t raw_size() const noexcept { return data_.size(); }
    const std::vector<uint8_t>& bytes() const noexcept { return data_; }
    uint32_t count() const noexcept { return n_; }
    uint32_t at(uint32_t k) const {
        if(k<1 || k>n_) throw std::domain_error(noise_membership_error);
        const uint32_t id=k-1, block=id/b_, lane=id%b_, w=data_[width_+block];
        const int64_t a=read_le32(data_.data()+24+size_t(block)*4);
        const int64_t z=read_le32(data_.data()+end_+size_t(block)*4);
        const uint64_t bit=uint64_t(lane)*w;
        const uint64_t byte=uint64_t(read_le32(data_.data()+offset_+size_t(block)*4))+(bit>>3);
        uint64_t window=0;
        // Last residual may have fewer than 5 remaining bytes. Missing bytes
        // are zero in the extraction window, never read outside the real owner.
        if(w) for(uint32_t j=0;j<5 && byte+j<payload_bytes_;++j)
            window|=uint64_t(data_[payload_+size_t(byte)+j])<<(j*8);
        const uint64_t zz=w ? ((window>>(bit&7))&((uint64_t(1)<<w)-1)) : 0;
        const int64_t residual=(zz&1) ? -int64_t(zz>>1)-1 : int64_t(zz>>1);
        const int64_t value=a+floor_div_optd((z-a)*int64_t(lane),b_-1)+residual;
        if(value<0 || value>int64_t(UINT32_MAX)) throw FormatError("DECODED_U32_RANGE");
        return uint32_t(value);
    }
};

// Proof-only standalone LG2 representation. Production fused L2 does not load it.
class LG2CompactTable {
    std::vector<uint8_t> data_;
    uint32_t n_=0,b_=0,nb_=0,last_=0;
    size_t end_=0,width_=0,offset_=0,payload_=0;
    void validate(uint32_t expected_n) {
        if(data_.size()<24) throw FormatError("HEADER_TRUNCATED");
        if(std::memcmp(data_.data(),"LG2DPR01",8)!=0) throw FormatError("MAGIC_VERSION");
        n_=read_le32(data_.data()+8); b_=read_le32(data_.data()+12);
        if(n_!=expected_n || !n_ || n_>noise_k_count) throw FormatError("COUNT_ENDIAN");
        if(b_<8 || b_>512 || (b_&(b_-1)) || n_%b_) throw FormatError("BLOCK_LAYOUT");
        if(read_le32(data_.data()+16)!=3) throw FormatError("PREDICTOR_VERSION");
        last_=read_le32(data_.data()+20); nb_=n_/b_;
        end_=24+size_t(nb_)*4; width_=24+size_t(nb_)*8;
        offset_=24+size_t(nb_)*9; payload_=28+size_t(nb_)*13;
        if(payload_>data_.size()) throw FormatError("METADATA_TRUNCATED");
        uint64_t cursor=0;
        for(uint32_t i=0;i<nb_;++i) {
            const uint32_t w=data_[width_+i]; if(w>32) throw FormatError("WIDTH");
            if(read_le32(data_.data()+offset_+size_t(i)*4)!=cursor) throw FormatError("OFFSET_LAYOUT");
            cursor+=uint64_t(w)*b_/8;
            if(cursor>data_.size()-payload_) throw FormatError("OFFSET_BOUNDS");
        }
        if(read_le32(data_.data()+offset_+size_t(nb_)*4)!=cursor) throw FormatError("OFFSET_LAYOUT");
        if(payload_+cursor!=data_.size()) throw FormatError("EXACT_LENGTH");
    }
public:
    static LG2CompactTable from_bytes(std::vector<uint8_t> d,uint32_t n=noise_k_count) {
        LG2CompactTable t;t.data_=std::move(d);t.validate(n);return t;
    }
    static LG2CompactTable load(const std::string& path,size_t size,uint32_t n=noise_k_count) {
        return from_bytes(load_bytes_optd(path,size),n);
    }
    const uint8_t* raw_data() const noexcept { return data_.data(); }
    size_t raw_size() const noexcept { return data_.size(); }
    uint32_t at(uint32_t k) const {
        if(k<1 || k>n_) throw std::domain_error(noise_membership_error);
        if(k==n_) return last_;
        const uint32_t id=k-1,block=id/b_,lane=id%b_,w=data_[width_+block];
        const int64_t a=read_le32(data_.data()+24+size_t(block)*4);
        const int64_t z=read_le32(data_.data()+end_+size_t(block)*4);
        const size_t off=read_le32(data_.data()+offset_+size_t(block)*4);
        uint64_t zz=0;
        for(uint32_t j=0;j<w;++j)
            zz|=uint64_t((data_[payload_+off+size_t(j)*(b_/8)+lane/8]>>(lane%8))&1)<<j;
        const int64_t r=(zz&1)? -int64_t(zz>>1)-1 : int64_t(zz>>1);
        const int64_t value=a-floor_div_optd((a-z)*int64_t(lane),b_-1)+r;
        if(value<0 || value>int64_t(UINT32_MAX)) throw FormatError("DECODED_U32_RANGE");
        return uint32_t(value);
    }
};
}
