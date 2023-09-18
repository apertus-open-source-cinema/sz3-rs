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
        .clang_arg("-x").clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg("-I.")
        .clang_arg(format!("-I{}/include", zstd_root))
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

    build
        .cpp(true)
        .warnings(false)
        .flag("-std=c++17")
        .include(".")
        .include(format!("{}/include", zstd_root))
        .file("lib.cpp");

    if cfg!(feature = "openmp") {
        env::var("DEP_OPENMP_FLAG")  // set by openmp-sys
            .unwrap()
            .split(' ')
            .for_each(|f| {
                build.flag(f);
            });
    }

    build.compile("sz3");

    println!("cargo:rustc-link-lib=static=zstd");

    if cfg!(feature = "openmp") {
        if let Some(link) = env::var_os("DEP_OPENMP_CARGO_LINK_INSTRUCTIONS") {
            for i in env::split_paths(&link) {
                if i.as_os_str().len() == 0 {
                    continue
                }
                println!("cargo:{}", i.display());
            }
        }
    }

    Ok(())
}
