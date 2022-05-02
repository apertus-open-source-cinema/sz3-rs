# sz3-rs
High level bindings to the [SZ3](https://github.com/szcompressor/SZ3) lossy floating point compressor.

## Usage
```rust
let data = vec![0; 64 * 64 * 64];
let data = DimensionedData::build(&data)
    .dim(64)?
    .dim(64)?
    .remainder_dim()?;
let config = Config::new(ErrorBound::Absolute(0.02));
let compressed = compress_with_config(&data, &config);
let decompressed = decompress::<f32, _>(&*compressed);
```
