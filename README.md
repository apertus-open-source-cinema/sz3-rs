# sz3-rs
High level bindings to the [SZ3](https://github.com/szcompressor/SZ3) lossy floating point compressor.

## Usage
```rust
use sz3::{compress_with_config, decompress, Config, DimensionedData, ErrorBound, SZ3Error};

fn main() -> Result<(), SZ3Error> {
    let data = vec![0; 64 * 64 * 64];
    let data = DimensionedData::build(&data)
        .dim(64)?
        .dim(64)?
        .remainder_dim()?;
    
    let config = Config::new(ErrorBound::Absolute(0.02));
    
    let compressed = compress_with_config(&data, &config)?;
    let decompressed = decompress::<f32, _>(compressed)?;
    
    Ok(())
}
```
