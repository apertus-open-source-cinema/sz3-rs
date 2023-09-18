#[derive(Clone, Debug, Copy)]
pub enum CompressionAlgorithm {
    Interpolation {
        interp_block_size: u32,
    },
    InterpolationLorenzo {
        interp_block_size: u32,
    },
    LorenzoRegression {
        lorenzo: bool,
        lorenzo_second_order: bool,
        regression: bool,
        regression_second_order: bool,
        prediction_dimension: Option<u32>,
    },
}

impl CompressionAlgorithm {
    fn decode(config: sz3_sys::SZ_Config) -> Self {
        match config.cmprAlgo as _ {
            sz3_sys::SZ_ALGO_ALGO_INTERP => Self::Interpolation {
                interp_block_size: config.interpBlockSize as _,
            },
            sz3_sys::SZ_ALGO_ALGO_INTERP_LORENZO => Self::InterpolationLorenzo {
                interp_block_size: config.interpBlockSize as _,
            },
            sz3_sys::SZ_ALGO_ALGO_LORENZO_REG => Self::LorenzoRegression {
                lorenzo: config.lorenzo,
                lorenzo_second_order: config.lorenzo2,
                regression: config.regression,
                regression_second_order: config.regression2,
                prediction_dimension: Some(config.pred_dim as _),
            },
            _ => unreachable!(),
        }
    }

    fn code(&self) -> u8 {
        (match self {
            Self::Interpolation { .. } => sz3_sys::SZ_ALGO_ALGO_INTERP,
            Self::InterpolationLorenzo { .. } => sz3_sys::SZ_ALGO_ALGO_INTERP_LORENZO,
            Self::LorenzoRegression { .. } => sz3_sys::SZ_ALGO_ALGO_LORENZO_REG,
        }) as _
    }

    fn interp_block_size(&self) -> u32 {
        match self {
            Self::Interpolation { interp_block_size }
            | Self::InterpolationLorenzo { interp_block_size } => *interp_block_size,
            _ => 32,
        }
    }

    fn lorenzo(&self) -> bool {
        match self {
            Self::LorenzoRegression { lorenzo, .. } => *lorenzo,
            _ => true,
        }
    }

    fn lorenzo_second_order(&self) -> bool {
        match self {
            Self::LorenzoRegression {
                lorenzo_second_order,
                ..
            } => *lorenzo_second_order,
            _ => true,
        }
    }

    fn regression(&self) -> bool {
        match self {
            Self::LorenzoRegression { regression, .. } => *regression,
            _ => true,
        }
    }

    fn regression_second_order(&self) -> bool {
        match self {
            Self::LorenzoRegression {
                regression_second_order,
                ..
            } => *regression_second_order,
            _ => true,
        }
    }

    fn encode_prediction_dimension(&self, num_dims: u32) -> u32 {
        if let Self::LorenzoRegression {
            prediction_dimension: Some(prediction_dimension),
            ..
        } = self
        {
            *prediction_dimension
        } else {
            num_dims
        }
    }

    pub fn interpolation() -> Self {
        Self::Interpolation {
            interp_block_size: 32,
        }
    }

    pub fn interpolation_custom(interp_block_size: u32) -> Self {
        Self::Interpolation { interp_block_size }
    }

    pub fn interpolation_lorenzo() -> Self {
        Self::InterpolationLorenzo {
            interp_block_size: 32,
        }
    }

    pub fn interpolation_lorenzo_custom(interp_block_size: u32) -> Self {
        Self::InterpolationLorenzo { interp_block_size }
    }

    pub fn lorenzo_regression() -> Self {
        Self::LorenzoRegression {
            lorenzo: true,
            lorenzo_second_order: false,
            regression: true,
            regression_second_order: false,
            prediction_dimension: None,
        }
    }

    pub fn lorenzo_regression_custom(
        lorenzo: Option<bool>,
        lorenzo_second_order: Option<bool>,
        regression: Option<bool>,
        regression_second_order: Option<bool>,
        prediction_dimension: Option<u32>,
    ) -> Self {
        Self::LorenzoRegression {
            lorenzo: lorenzo.unwrap_or(true),
            lorenzo_second_order: lorenzo_second_order.unwrap_or(false),
            regression: regression.unwrap_or(true),
            regression_second_order: regression_second_order.unwrap_or(true),
            prediction_dimension,
        }
    }

    fn prediction_dimension(&self) -> Option<u32> {
        match self {
            Self::LorenzoRegression {
                prediction_dimension: Some(prediction_dimension),
                ..
            } => Some(*prediction_dimension),
            _ => None,
        }
    }
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        CompressionAlgorithm::interpolation_lorenzo()
    }
}

#[derive(Clone, Debug, Copy)]
pub enum ErrorBound {
    Absolute(f64),
    Relative(f64),
    PSNR(f64),
    L2Norm(f64),
    AbsoluteAndRelative {
        absolute_bound: f64,
        relative_bound: f64,
    },
    AbsoluteOrRelative {
        absolute_bound: f64,
        relative_bound: f64,
    },
}

impl ErrorBound {
    fn decode(config: sz3_sys::SZ_Config) -> Self {
        match config.errorBoundMode as _ {
            sz3_sys::SZ_EB_EB_ABS => Self::Absolute(config.absErrorBound),
            sz3_sys::SZ_EB_EB_REL => Self::Relative(config.relErrorBound),
            sz3_sys::SZ_EB_EB_PSNR => Self::PSNR(config.psnrErrorBound),
            sz3_sys::SZ_EB_EB_L2NORM => Self::L2Norm(config.l2normErrorBound),
            sz3_sys::SZ_EB_EB_ABS_OR_REL => Self::AbsoluteOrRelative {
                absolute_bound: config.absErrorBound,
                relative_bound: config.relErrorBound,
            },
            sz3_sys::SZ_EB_EB_ABS_AND_REL => Self::AbsoluteAndRelative {
                absolute_bound: config.absErrorBound,
                relative_bound: config.relErrorBound,
            },
            _ => unreachable!(),
        }
    }

    fn code(&self) -> u8 {
        (match self {
            Self::Absolute(_) => sz3_sys::SZ_EB_EB_ABS,
            Self::Relative(_) => sz3_sys::SZ_EB_EB_REL,
            Self::PSNR(_) => sz3_sys::SZ_EB_EB_PSNR,
            Self::L2Norm(_) => sz3_sys::SZ_EB_EB_L2NORM,
            Self::AbsoluteAndRelative { .. } => sz3_sys::SZ_EB_EB_ABS_AND_REL,
            Self::AbsoluteOrRelative { .. } => sz3_sys::SZ_EB_EB_ABS_OR_REL,
        }) as _
    }

    fn abs_bound(&self) -> f64 {
        match self {
            Self::Absolute(bound) => *bound,
            Self::AbsoluteOrRelative { absolute_bound, .. } => *absolute_bound,
            Self::AbsoluteAndRelative { absolute_bound, .. } => *absolute_bound,
            _ => 0.0,
        }
    }

    fn rel_bound(&self) -> f64 {
        match self {
            Self::Relative(bound) => *bound,
            Self::AbsoluteOrRelative { relative_bound, .. } => *relative_bound,
            Self::AbsoluteAndRelative { relative_bound, .. } => *relative_bound,
            _ => 0.0,
        }
    }

    fn l2norm_bound(&self) -> f64 {
        match self {
            Self::L2Norm(bound) => *bound,
            _ => 0.0,
        }
    }

    fn psnr_bound(&self) -> f64 {
        match self {
            Self::PSNR(bound) => *bound,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub enum InterpolationAlgorithm {
    Linear,
    Cubic,
}

impl InterpolationAlgorithm {
    fn decode(config: sz3_sys::SZ_Config) -> Self {
        match config.interpAlgo as _ {
            sz3_sys::SZ_INTERP_ALGO_INTERP_ALGO_LINEAR => Self::Linear,
            sz3_sys::SZ_INTERP_ALGO_INTERP_ALGO_CUBIC => Self::Cubic,
            _ => unreachable!(),
        }
    }

    fn code(&self) -> u8 {
        (match self {
            Self::Linear => sz3_sys::SZ_INTERP_ALGO_INTERP_ALGO_LINEAR,
            Self::Cubic => sz3_sys::SZ_INTERP_ALGO_INTERP_ALGO_CUBIC,
        }) as _
    }
}

impl Default for InterpolationAlgorithm {
    fn default() -> Self {
        InterpolationAlgorithm::Cubic
    }
}

#[derive(Clone, Debug, Copy)]
pub enum LossLess {
    LossLessBypass,
    ZSTD,
}

impl LossLess {
    fn decode(config: sz3_sys::SZ_Config) -> Self {
        match config.lossless {
            0 => LossLess::LossLessBypass,
            1 => LossLess::ZSTD,
            _ => unreachable!(),
        }
    }

    fn code(&self) -> u8 {
        match self {
            Self::LossLessBypass => 0,
            Self::ZSTD => 1,
        }
    }
}

impl Default for LossLess {
    fn default() -> Self {
        LossLess::ZSTD
    }
}

#[derive(Clone, Debug, Copy)]
pub enum Encoder {
    SkipEncoder,
    HuffmanEncoder,
    ArithmeticEncoder,
}

impl Encoder {
    fn decode(config: sz3_sys::SZ_Config) -> Self {
        match config.encoder {
            0 => Self::SkipEncoder,
            1 => Self::HuffmanEncoder,
            2 => Self::ArithmeticEncoder,
            _ => unreachable!(),
        }
    }

    fn code(&self) -> u8 {
        match self {
            Self::SkipEncoder => 0,
            Self::HuffmanEncoder => 1,
            Self::ArithmeticEncoder => 2,
        }
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Encoder::HuffmanEncoder
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    compression_algorithm: CompressionAlgorithm,
    error_bound: ErrorBound,
    openmp: bool,
    lossless: LossLess,
    encoder: Encoder,
    interpolation_algorithm: InterpolationAlgorithm,
    quantization_bincount: u32,
    block_size: Option<u32>,
}

#[derive(Debug)]
enum OwnedOrBorrowed<'a, A: ?Sized, B> {
    Borrowed(&'a A),
    Owned(B),
}

impl<'a, A: ?Sized, B: Clone> Clone for OwnedOrBorrowed<'a, A, B> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(a) => Self::Borrowed(a),
            Self::Owned(b) => Self::Owned(b.clone()),
        }
    }
}

impl<'a, A: ?Sized, B, T: 'a> std::ops::Deref for OwnedOrBorrowed<'a, A, B>
where
    B: std::ops::Deref<Target = T>,
    T: std::ops::Deref<Target = A>,
{
    type Target = A;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(owned) => &**owned,
            Self::Borrowed(borrowed) => *borrowed,
        }
    }
}

pub trait SZ3Compressible: private::Sealed + std::ops::Sub<Output = Self> + Sized {}
impl SZ3Compressible for f32 {}
impl SZ3Compressible for f64 {}
impl SZ3Compressible for i32 {}
impl SZ3Compressible for i64 {}

mod private {
    pub trait Sealed {
        unsafe fn compress(
            config: sz3_sys::SZ_Config,
            data: *const Self,
            len: *mut sz3_sys::size_t,
        ) -> *mut i8;
        unsafe fn decompress(
            compressed_data: *const i8,
            compressed_len: sz3_sys::size_t,
            uncompressed: *mut *mut Self,
        ) -> sz3_sys::SZ_Config;
        unsafe fn dealloc(data: *mut Self);
    }
    impl Sealed for f32 {
        unsafe fn compress(
            config: sz3_sys::SZ_Config,
            data: *const Self,
            len: *mut sz3_sys::size_t,
        ) -> *mut i8 {
            sz3_sys::compress_float(config, data, len) as _
        }
        unsafe fn decompress(
            compressed_data: *const i8,
            compressed_len: sz3_sys::size_t,
            uncompressed: *mut *mut Self,
        ) -> sz3_sys::SZ_Config {
            sz3_sys::decompress_float(compressed_data as _, compressed_len, uncompressed)
        }
        unsafe fn dealloc(data: *mut Self) {
            sz3_sys::dealloc_result_float(data)
        }
    }
    impl Sealed for f64 {
        unsafe fn compress(
            config: sz3_sys::SZ_Config,
            data: *const Self,
            len: *mut sz3_sys::size_t,
        ) -> *mut i8 {
            sz3_sys::compress_double(config, data, len) as _
        }
        unsafe fn decompress(
            compressed_data: *const i8,
            compressed_len: sz3_sys::size_t,
            uncompressed: *mut *mut Self,
        ) -> sz3_sys::SZ_Config {
            sz3_sys::decompress_double(compressed_data as _, compressed_len, uncompressed)
        }
        unsafe fn dealloc(data: *mut Self) {
            sz3_sys::dealloc_result_double(data)
        }
    }
    impl Sealed for i32 {
        unsafe fn compress(
            config: sz3_sys::SZ_Config,
            data: *const Self,
            len: *mut sz3_sys::size_t,
        ) -> *mut i8 {
            sz3_sys::compress_int32_t(config, data, len) as _
        }
        unsafe fn decompress(
            compressed_data: *const i8,
            compressed_len: sz3_sys::size_t,
            uncompressed: *mut *mut Self,
        ) -> sz3_sys::SZ_Config {
            sz3_sys::decompress_int32_t(compressed_data as _, compressed_len, uncompressed)
        }
        unsafe fn dealloc(data: *mut Self) {
            sz3_sys::dealloc_result_int32_t(data)
        }
    }
    impl Sealed for i64 {
        unsafe fn compress(
            config: sz3_sys::SZ_Config,
            data: *const Self,
            len: *mut sz3_sys::size_t,
        ) -> *mut i8 {
            sz3_sys::compress_int64_t(config, data, len) as _
        }
        unsafe fn decompress(
            compressed_data: *const i8,
            compressed_len: sz3_sys::size_t,
            uncompressed: *mut *mut Self,
        ) -> sz3_sys::SZ_Config {
            sz3_sys::decompress_int64_t(compressed_data as _, compressed_len, uncompressed)
        }
        unsafe fn dealloc(data: *mut Self) {
            sz3_sys::dealloc_result_int64_t(data)
        }
    }
}

#[derive(Clone, Debug)]
pub struct DimensionedData<'a, V: SZ3Compressible> {
    data: OwnedOrBorrowed<'a, [V], SZ3DecompressionResult<V>>,
    dims: Vec<sz3_sys::size_t>,
}

#[derive(Clone, Debug)]
pub struct DimensionedDataBuilder<'a, V> {
    data: &'a [V],
    dims: Vec<sz3_sys::size_t>,
    remainder: usize,
}

impl<V: SZ3Compressible> DimensionedData<'static, V> {
    fn from_raw(config: sz3_sys::SZ_Config, data: SZ3DecompressionResult<V>) -> Self {
        let dims = (0..config.N)
            .map(|i| unsafe { std::ptr::read(config.dims.add(i as _)) })
            .collect();
        Self {
            data: OwnedOrBorrowed::Owned(data),
            dims,
        }
    }
}

impl<'a, V: SZ3Compressible> DimensionedData<'a, V> {
    pub fn build<T: std::ops::Deref<Target = [V]>>(data: &'a T) -> DimensionedDataBuilder<'a, V> {
        DimensionedDataBuilder {
            data: &*data,
            dims: vec![],
            remainder: data.len(),
        }
    }

    pub fn data(&self) -> &[V] {
        &*self.data
    }

    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn as_ptr(&self) -> *const V {
        self.data.as_ptr()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SZ3Error {
    #[error("invalid dimension specification for data of length {len}: already specified dimensions {dims:?}, and wanted to add dimension with length {wanted}, but this does not divide {remainder} cleanly")]
    InvalidDimensionSize {
        dims: Vec<sz3_sys::size_t>,
        len: usize,
        wanted: usize,
        remainder: usize,
    },
    #[error("dimension with size one has no use")]
    OneSizedDimension,
    #[error("dimension specification {dims:?} for data of length {len} does not cover whole space, missing a dimension of {remainder}")]
    UnderSpecifiedDimensions {
        dims: Vec<sz3_sys::size_t>,
        len: usize,
        remainder: usize,
    },
    #[error("prediction dimension cannot be zero (it is one based)")]
    PredictionDimensionZero,
    #[error("wanted to predict along dimension {prediction_dimension}, but data only has {data_dimensions} dimensions")]
    PredictionDimensionDataDimensionsMismatch {
        prediction_dimension: u32,
        data_dimensions: u32,
    },
}

type Result<T> = std::result::Result<T, SZ3Error>;

impl<'a, V: SZ3Compressible> DimensionedDataBuilder<'a, V> {
    pub fn dim(mut self, length: usize) -> Result<Self> {
        if length == 1 {
            Err(SZ3Error::OneSizedDimension)
        } else if self.remainder.rem_euclid(length) != 0 {
            Err(SZ3Error::InvalidDimensionSize {
                dims: self.dims,
                len: self.data.len(),
                wanted: length,
                remainder: self.remainder,
            })
        } else {
            self.dims.push(length as _);
            self.remainder /= length;
            Ok(self)
        }
    }

    pub fn remainder_dim(self) -> Result<DimensionedData<'a, V>> {
        let remainder = self.remainder;
        self.dim(remainder)?.finish()
    }

    pub fn finish(self) -> Result<DimensionedData<'a, V>> {
        if self.remainder != 1 {
            Err(SZ3Error::UnderSpecifiedDimensions {
                dims: self.dims,
                len: self.data.len(),
                remainder: self.remainder,
            })
        } else {
            Ok(DimensionedData {
                data: OwnedOrBorrowed::Borrowed(self.data),
                dims: self.dims,
            })
        }
    }
}

impl Config {
    fn from_decompressed(config: sz3_sys::SZ_Config) -> Self {
        Self {
            compression_algorithm: CompressionAlgorithm::decode(config),
            error_bound: ErrorBound::decode(config),
            openmp: config.openmp,
            lossless: LossLess::decode(config),
            encoder: Encoder::decode(config),
            interpolation_algorithm: InterpolationAlgorithm::decode(config),
            quantization_bincount: config.quantbinCnt as _,
            block_size: Some(config.blockSize as _),
        }
    }
}

impl Config {
    pub fn new(error_bound: ErrorBound) -> Self {
        Self {
            compression_algorithm: Default::default(),
            error_bound,
            openmp: false,
            encoder: Default::default(),
            lossless: Default::default(),
            interpolation_algorithm: Default::default(),
            quantization_bincount: 65536,
            block_size: None,
        }
    }

    pub fn compression_algorithm(mut self, compression_algorithm: CompressionAlgorithm) -> Self {
        self.compression_algorithm = compression_algorithm;
        self
    }

    pub fn error_bound(mut self, error_bound: ErrorBound) -> Self {
        self.error_bound = error_bound;
        self
    }

    #[cfg(feature = "openmp")]
    pub fn openmp(mut self, openmp: bool) -> Self {
        self.openmp = openmp;
        self
    }

    pub fn lossless(mut self, lossless: LossLess) -> Self {
        self.lossless = lossless;
        self
    }

    pub fn encoder(mut self, encoder: Encoder) -> Self {
        self.encoder = encoder;
        self
    }

    pub fn interpolation_algorithm(
        mut self,
        interpolation_algorithm: InterpolationAlgorithm,
    ) -> Self {
        self.interpolation_algorithm = interpolation_algorithm;
        self
    }

    pub fn quantization_bincount(mut self, quantization_bincount: u32) -> Self {
        self.quantization_bincount = quantization_bincount;
        self
    }

    pub fn block_size(mut self, block_size: u32) -> Self {
        self.block_size = Some(block_size);
        self
    }

    pub fn automatic_block_size(mut self) -> Self {
        self.block_size = None;
        self
    }
}

#[derive(Debug)]
pub struct SZ3CompressionResult {
    data: *mut u8,
    len: usize,
}

impl Drop for SZ3CompressionResult {
    fn drop(&mut self) {
        unsafe { sz3_sys::dealloc_result(self.data as _) };
    }
}

impl std::ops::Deref for SZ3CompressionResult {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}

pub fn compress<V: SZ3Compressible>(
    data: &DimensionedData<'_, V>,
    error_bound: ErrorBound,
) -> Result<SZ3CompressionResult> {
    let config = Config::new(error_bound);
    compress_with_config(data, &config)
}

pub fn compress_with_config<V: SZ3Compressible>(
    data: &DimensionedData<'_, V>,
    config: &Config,
) -> Result<SZ3CompressionResult> {
    if let Some(prediction_dimension) = config.compression_algorithm.prediction_dimension() {
        let data_dimensions = data.num_dims() as u32;
        if prediction_dimension == 0 {
            return Err(SZ3Error::PredictionDimensionZero);
        } else if prediction_dimension > data_dimensions {
            return Err(SZ3Error::PredictionDimensionDataDimensionsMismatch {
                prediction_dimension,
                data_dimensions,
            });
        }
    }

    let block_size = config.block_size.unwrap_or(match data.num_dims() {
        1 => 128,
        2 => 16,
        _ => 6,
    });

    let raw_config = sz3_sys::SZ_Config {
        N: data.num_dims() as _,
        dims: data.dims.as_ptr() as _,
        num: data.len() as _,
        errorBoundMode: config.error_bound.code(),
        absErrorBound: config.error_bound.abs_bound(),
        relErrorBound: config.error_bound.rel_bound(),
        l2normErrorBound: config.error_bound.l2norm_bound(),
        psnrErrorBound: config.error_bound.psnr_bound(),
        cmprAlgo: config.compression_algorithm.code(),
        lorenzo: config.compression_algorithm.lorenzo(),
        lorenzo2: config.compression_algorithm.lorenzo_second_order(),
        regression: config.compression_algorithm.regression(),
        regression2: config.compression_algorithm.regression_second_order(),
        interpBlockSize: config.compression_algorithm.interp_block_size() as _,
        pred_dim: config
            .compression_algorithm
            .encode_prediction_dimension(data.num_dims() as _) as _,
        openmp: config.openmp,
        lossless: config.lossless.code(),
        encoder: config.encoder.code(),
        interpAlgo: config.interpolation_algorithm.code(),
        blockSize: block_size as _,
        quantbinCnt: config.quantization_bincount as _,
        stride: block_size as _,
    };

    let mut len: sz3_sys::size_t = 0;
    let data = unsafe { V::compress(raw_config, data.as_ptr() as _, &mut len) };

    Ok(SZ3CompressionResult {
        data: data as _,
        len: len as _,
    })
}

type SZ3DecompressionResult<T> = std::rc::Rc<SZ3DecompressionResultInner<T>>;

#[derive(Debug)]
struct SZ3DecompressionResultInner<T: SZ3Compressible> {
    data: *mut T,
    len: usize,
}

impl<T: SZ3Compressible> Drop for SZ3DecompressionResultInner<T> {
    fn drop(&mut self) {
        unsafe { T::dealloc(self.data) };
    }
}

impl<T: SZ3Compressible> std::ops::Deref for SZ3DecompressionResultInner<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}

pub fn decompress<V: SZ3Compressible, T: std::ops::Deref<Target = [u8]>>(
    compressed_data: T,
) -> (Config, DimensionedData<'static, V>) {
    let mut destination_ptr = std::ptr::null_mut();
    let config = unsafe {
        V::decompress(
            compressed_data.as_ptr() as _,
            compressed_data.len() as _,
            &mut destination_ptr,
        )
    };

    let data = std::rc::Rc::new(SZ3DecompressionResultInner {
        data: destination_ptr,
        len: config.num as _,
    });

    let decoded = Config::from_decompressed(config);
    let data = DimensionedData::from_raw(config, data);

    unsafe {
        sz3_sys::dealloc_config_dims(config.dims);
    }

    (decoded, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_DATA: [f32; 8847360] = unsafe {
        align_data::include_transmute!("../test_data/darkframe_1.0x_10.872624ms_1024n.blob")
    };

    fn test_data<T: SZ3Compressible + From<f32>>() -> Vec<T> {
        bytemuck::cast_slice(&TEST_DATA[..])
            .iter()
            .map(|v| T::from(*v))
            .take(64 * 64 * 64)
            .collect()
    }

    fn check_error_bound<T: SZ3Compressible + Copy + 'static>(
        data: &'_ DimensionedData<'_, T>,
        config: &Config,
        error_bound: ErrorBound,
    ) -> Result<()>
    where
        f64: From<T>,
    {
        let config = config.clone().error_bound(error_bound.clone());
        let (_decompressed_config, decompressed_data) =
            decompress::<T, _>(&*compress_with_config(&data, &config)?);
        let min = data
            .data()
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::MAX, f64::min);
        let max = data
            .data()
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::MIN, f64::max);
        let range = max - min;

        match error_bound {
            ErrorBound::Absolute(absolute_bound) => {
                for (orig, compressed) in data.data().iter().zip(decompressed_data.data()) {
                    assert!((f64::from(*orig) - f64::from(*compressed)).abs() <= absolute_bound)
                }
            }
            ErrorBound::Relative(relative_bound) => {
                for (orig, compressed) in data.data().iter().zip(decompressed_data.data()) {
                    let orig = f64::from(*orig);
                    let compressed = f64::from(*compressed);
                    assert!((orig - compressed).abs() / range < relative_bound)
                }
            }
            ErrorBound::PSNR(psnr_bound) => {
                let mse: f64 = data
                    .data()
                    .iter()
                    .zip(decompressed_data.data())
                    .map(|(a, b)| {
                        let diff = f64::from(*a) - f64::from(*b);
                        diff * diff
                    })
                    .sum();
                let psnr = 20. * (max - min).log10() - 10. * mse.log10();
                assert!(psnr < psnr_bound);
            }
            ErrorBound::L2Norm(l2norm_bound) => {
                let mse: f64 = data
                    .data()
                    .iter()
                    .zip(decompressed_data.data())
                    .map(|(a, b)| {
                        let diff = f64::from(*a) - f64::from(*b);
                        diff * diff
                    })
                    .sum();
                let l2norm = mse.sqrt();
                // random fudge factor because its not always really good at matching
                assert!(l2norm < l2norm_bound * 1.005);
            }
            ErrorBound::AbsoluteAndRelative {
                absolute_bound,
                relative_bound,
            } => {
                for (orig, compressed) in data.data().iter().zip(decompressed_data.data()) {
                    let orig = f64::from(*orig);
                    let compressed = f64::from(*compressed);
                    let abs = (orig - compressed).abs();
                    let rel = abs / range;
                    assert!((rel < relative_bound) && (abs < absolute_bound));
                }
            }
            ErrorBound::AbsoluteOrRelative {
                absolute_bound,
                relative_bound,
            } => {
                for (orig, compressed) in data.data().iter().zip(decompressed_data.data()) {
                    let orig = f64::from(*orig);
                    let compressed = f64::from(*compressed);
                    let abs = (orig - compressed).abs();
                    let rel = abs / range;
                    assert!((rel < relative_bound) || (abs < absolute_bound));
                }
            }
        }

        Ok(())
    }

    macro_rules! foreach_combination {
        (([$($values:tt),*], $the_rest:tt); $to_call:ident, $accum:tt) => {
            $(
                foreach_combination!(@concat $the_rest; $to_call, $accum, $values);
            )*
        };

        (([$($values:tt),*]); $to_call:ident, $accum:tt) => {
            $(
                foreach_combination!(@concat_call $to_call $accum, $values);
            )*
        };

        (@concat $the_rest:tt; $to_call:ident, ($($accum:tt),*), $value:tt) => {
            foreach_combination!($the_rest; $to_call, ($($accum,)* $value));
        };

        (@concat_call $to_call:ident ($($accum:tt),*), $value:tt) => {
            $to_call!($($accum,)* $value);
        };
    }

    macro_rules! gen_test {
        (($lossless_name:ident, $lossless:expr), ($openmp:expr, $openmp_cfg:meta), ($eb_name:ident, $eb:expr), ($ca_name: ident, $ca:expr), ($ia_name:ident, $ia:expr), $qb:expr, $block_size:expr) => {
            paste::paste! {
                #[$openmp_cfg]
                #[test]
                fn [<test_ $lossless_name _ $openmp _ $eb_name _ $ca_name _ $ia_name _ $qb _ $block_size>]() -> Result<()> {
                    let data = test_data::<f32>();
                    let data = DimensionedData::build(&data)
                        .dim(64)?
                        .dim(64)?
                        .remainder_dim()?;
                    let config = Config::new($eb)
                        .lossless($lossless)
                        .error_bound($eb)
                        .compression_algorithm($ca)
                        .interpolation_algorithm($ia)
                        .quantization_bincount($qb)
                        .block_size($block_size);
                    #[cfg(feature = "openmp")]
                    let config = config.openmp($openmp);

                    check_error_bound(&data, &config, $eb)?;

                    Ok(())
                }
            }
        }
    }

    foreach_combination!(
        ([(lossless_bypass, LossLess::LossLessBypass), (zstd, LossLess::ZSTD)],
        ([(true, cfg(feature = "openmp")), (false, cfg(all()))],
        ([
            (absolute_1_0, ErrorBound::Absolute(1.)),
            (absolute_0_1, ErrorBound::Absolute(0.1)),
            (absolute_0_01, ErrorBound::Absolute(0.01)),
            (relative_0_01, ErrorBound::Relative(0.01)),
            (relative_0_001, ErrorBound::Relative(0.001)),
            (relative_0_0001, ErrorBound::Relative(0.0001)),
            (psnr_120, ErrorBound::PSNR(120.0)),
            (psnr_100, ErrorBound::PSNR(100.0)),
            (psnr_80, ErrorBound::PSNR(80.0)),
            (l2norm_40, ErrorBound::L2Norm(40.0)),
            (l2norm_30, ErrorBound::L2Norm(30.0)),
            (l2norm_20, ErrorBound::L2Norm(20.0)),
            (abs_and_rel_0_1_0_001, ErrorBound::AbsoluteAndRelative {
                absolute_bound: 0.1,
                relative_bound: 0.001,
            }),
            (abs_or_rel_0_1_0_001, ErrorBound::AbsoluteOrRelative {
                absolute_bound: 0.1,
                relative_bound: 0.001,
            })
        ],
        ([
            (interpolation, CompressionAlgorithm::interpolation()),
            (interpolation_8, CompressionAlgorithm::interpolation_custom(8)),
            (interpolation_lorenzo, CompressionAlgorithm::interpolation_lorenzo()),
            (interpolation_lorenzo_8, CompressionAlgorithm::interpolation_lorenzo_custom(8)),
            (lorenzo_reg, CompressionAlgorithm::lorenzo_regression()),
            (lorenzo_reg_all, CompressionAlgorithm::lorenzo_regression_custom(
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                None
            ))
        ],
        ([(linear, InterpolationAlgorithm::Linear), (cubic, InterpolationAlgorithm::Cubic)],
        ([65536, 256, 2097152],
        ([2, 4, 8, 16])))))));
        gen_test, ());
}
