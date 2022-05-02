use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

#[derive(Debug)]
struct IgnoreMacros(HashSet<String>);

impl bindgen::callbacks::ParseCallbacks for IgnoreMacros {
    fn will_parse_macro(&self, name: &str) -> bindgen::callbacks::MacroParsingBehavior {
        if self.0.contains(name) {
            bindgen::callbacks::MacroParsingBehavior::Ignore
        } else {
            bindgen::callbacks::MacroParsingBehavior::Default
        }
    }
}

fn main() -> Result<(), std::env::VarError> {
    let zstd_root = env::var("DEP_ZSTD_ROOT")?;

    let ignored_macros = IgnoreMacros(
        vec![
            "FP_INFINITE".into(),
            "FP_NAN".into(),
            "FP_NORMAL".into(),
            "FP_SUBNORMAL".into(),
            "FP_ZERO".into(),
        ]
        .into_iter()
        .collect(),
    );

    println!("cargo:rerun-if-changed=wrapper.hpp");
    let bindings = bindgen::Builder::default()
        .clang_arg("-I.")
        .clang_arg(format!("-I{}/include", zstd_root))
        .clang_arg("-fopenmp")
        .header("wrapper.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .parse_callbacks(Box::new(ignored_macros))
        .allowlist_type("SZ_Config")
        .allowlist_type("SZ::EB")
        .allowlist_type("SZ::ALGO")
        .allowlist_type("SZ::INTERP_ALGO")
        .allowlist_function("compress_float")
        .allowlist_function("compress_double")
        .allowlist_function("compress_int32_t")
        .allowlist_function("compress_int64_t")
        .allowlist_function("decompress_float")
        .allowlist_function("decompress_double")
        .allowlist_function("decompress_int32_t")
        .allowlist_function("decompress_int64_t")
        .allowlist_function("dealloc_result_float")
        .allowlist_function("dealloc_result_double")
        .allowlist_function("dealloc_result_int32_t")
        .allowlist_function("dealloc_result_int64_t")
        .allowlist_function("dealloc_config_dims")
        .allowlist_function("dealloc_result")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    let mut build = cc::Build::new();
    env::var("DEP_OPENMP_FLAG")
        .unwrap()
        .split(' ')
        .for_each(|f| {
            build.flag(f);
        });
    build
        .cpp(true)
        .warnings(false)
        .flag("-fopenmp")
        .include(".")
        .include(format!("{}/include", zstd_root))
        .file("lib.cpp")
        .compile("sz3");

    println!("cargo:rustc-link-lib=static=zstd");
    if let Some(link) = env::var_os("DEP_OPENMP_CARGO_LINK_INSTRUCTIONS") {
        for i in env::split_paths(&link) {
            println!("cargo:{}", i.display());
        }
    }

    Ok(())
}
