#include "SZ3/api/sz.hpp"

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
    uint8_t lossless;
    uint8_t encoder;
    uint8_t interpAlgo;
    int interpBlockSize;
    int quantbinCnt;
    int blockSize;
    int stride;
    int pred_dim;

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
        conf.lossless = lossless;
        conf.encoder = encoder;
        conf.interpAlgo = interpAlgo;
        conf.interpBlockSize = interpBlockSize;
        conf.quantbinCnt = quantbinCnt;
        conf.blockSize = blockSize;
        conf.stride = stride;
        conf.pred_dim = pred_dim;
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
        lossless = conf.lossless;
        encoder = conf.encoder;
        interpAlgo = conf.interpAlgo;
        interpBlockSize = conf.interpBlockSize;
        quantbinCnt = conf.quantbinCnt;
        blockSize = conf.blockSize;
        stride = conf.stride;
        pred_dim = conf.pred_dim;
    }
};


#define func(ty) \
    char * compress_ ## ty(SZ3_Config config, const ty * data, size_t &outSize) { \
        return SZ_compress(config.into(), data, outSize); \
    } \
    SZ3_Config decompress_ ## ty(const char * compressedData, size_t compressedSize, ty *&decompressedData) { \
        auto conf = SZ3::Config{}; \
        auto d = std::vector<char>(compressedData, compressedData + compressedSize); \
        SZ_decompress(conf, d.data(), d.size(), decompressedData); \
        return SZ3_Config(conf); \
    } \
    void dealloc_result_ ## ty(ty * result) { \
        delete [] result; \
    }

func(float)
func(double)
func(int32_t)
func(int64_t)

void dealloc_result(char * data) {
    delete[] data;
}

void dealloc_config_dims(size_t * data) {
    delete[] data;
}
