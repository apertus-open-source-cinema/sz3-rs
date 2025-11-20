#include <cstdint>

#include "SZ3/api/sz.hpp"

enum SZ_DATA_TYPE {
    FLOAT = SZ_FLOAT,
    DOUBLE = SZ_DOUBLE,
    UINT8 = SZ_UINT8,
    INT8 = SZ_INT8,
    UINT16 = SZ_UINT16,
    INT16 = SZ_INT16,
    UINT32 = SZ_UINT32,
    INT32 = SZ_INT32,
    UINT64 = SZ_UINT64,
    INT64 = SZ_INT64,
};

struct SZ3_Config {
    char N;
    size_t * dims;
    size_t num;
    uint8_t cmprAlgo;
    uint8_t errorBoundMode;
    double absErrorBound;
    double relErrorBound;
    double psnrErrorBound;
    double l2normErrorBound;
    bool lorenzo;
    bool lorenzo2;
    bool regression;
    bool regression2;
    bool openmp;
    uint8_t dataType;
    int quantbinCnt;
    int blockSize;

    SZ3::Config into() {
        auto conf = SZ3::Config{};
        conf.N = N;
        conf.dims = std::vector<size_t>(dims, dims + N);
        conf.num = num;
        conf.cmprAlgo = cmprAlgo;
        conf.errorBoundMode = errorBoundMode;
        conf.absErrorBound = absErrorBound;
        conf.relErrorBound = relErrorBound;
        conf.psnrErrorBound = psnrErrorBound;
        conf.l2normErrorBound = l2normErrorBound;
        conf.lorenzo = lorenzo;
        conf.lorenzo2 = lorenzo2;
        conf.regression = regression;
        conf.regression2 = regression2;
        conf.openmp = openmp;
        conf.dataType = dataType;
        conf.quantbinCnt = quantbinCnt;
        conf.blockSize = blockSize;
        return conf;
    }

    SZ3_Config(SZ3::Config &conf) {
        dims = new size_t[conf.N];
        std::copy(conf.dims.begin(), conf.dims.end(), dims);
        N = conf.N;
        num = conf.num;
        cmprAlgo = conf.cmprAlgo;
        errorBoundMode = conf.errorBoundMode;
        absErrorBound = conf.absErrorBound;
        relErrorBound = conf.relErrorBound;
        psnrErrorBound = conf.psnrErrorBound;
        l2normErrorBound = conf.l2normErrorBound;
        lorenzo = conf.lorenzo;
        lorenzo2 = conf.lorenzo2;
        regression = conf.regression;
        regression2 = conf.regression2;
        openmp = conf.openmp;
        dataType = conf.dataType;
        quantbinCnt = conf.quantbinCnt;
        blockSize = conf.blockSize;
    }
};


#define func(ty) \
    size_t compress_ ## ty ## _size_bound(SZ3_Config config) { \
        return SZ3::SZ_compress_size_bound<ty>(config.into()); \
    } \
    size_t compress_ ## ty(SZ3_Config config, const ty * data, char * compressedData, size_t compressedCapacity) { \
        return SZ_compress<ty>(config.into(), data, compressedData, compressedCapacity); \
    } \
    void decompress_ ## ty(const char * compressedData, size_t compressedSize, ty * decompressedData) { \
        auto conf = SZ3::Config{}; \
        SZ_decompress<ty>(conf, compressedData, compressedSize, decompressedData); \
    }

func(float)
func(double)
func(uint8_t)
func(int8_t)
func(uint16_t)
func(int16_t)
func(uint32_t)
func(int32_t)
func(uint64_t)
func(int64_t)

SZ3_Config decompress_config(const char * compressedData, size_t compressedSize) {
    auto cmpDataPos = reinterpret_cast<const SZ3::uchar *>(compressedData);
    uint32_t magic;
    SZ3::read(magic, cmpDataPos);
    uint32_t ver;
    SZ3::read(ver, cmpDataPos);
    uint64_t cmpDataSize;
    SZ3::read(cmpDataSize,  cmpDataPos);
    auto cmpConfPos = cmpDataPos + cmpDataSize;
    auto conf = SZ3::Config{};
    conf.load(cmpConfPos);
    return SZ3_Config(conf);
}

void dealloc_size_t(size_t * data) {
    delete[] data;
}
