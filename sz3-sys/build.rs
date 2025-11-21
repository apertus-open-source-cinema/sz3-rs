use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

fn main() {
    let zstd_root = env::var("DEP_ZSTD_ROOT")
        .map(PathBuf::from)
        .expect("missing zstd dependency");

    // use cmake to configure (but not compile) the SZ3 build
    let mut config = cmake::Config::new("SZ3");
    config.define("BUILD_SHARED_LIBS", "OFF");
    config.define("BUILD_TESTING", "OFF");
    config.build_arg("--version");
    let sz3_root = config.build().join("build");

    // copy the SZ3 source to the pre-configured build directory
    copy_dir_all(Path::new("SZ3").join("include"), sz3_root.join("include"))
        .expect("failed to copy SZ3 source");

    println!("cargo:rerun-if-changed=wrapper.hpp");
    println!("cargo:rerun-if-changed=SZ3");

    let cargo_callbacks = bindgen::CargoCallbacks::new();
    let bindings = bindgen::Builder::default()
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{}", sz3_root.join("include").display()))
        .clang_arg(format!("-I{}", zstd_root.join("include").display()))
        .header("wrapper.hpp")
        .parse_callbacks(Box::new(cargo_callbacks))
        .enable_cxx_namespaces()
        .allowlist_type("SZ3_Config")
        .allowlist_type("SZ3::EB")
        .allowlist_type("SZ3::ALGO")
        .allowlist_type("SZ_DATA_TYPE")
        .allowlist_function("dealloc_size_t")
        .allowlist_function("decompress_config")
        .allowlist_item("impl_.*")
        .use_core()
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++17")
        .flag_if_supported("/bigobj") // required on Windows
        .include(sz3_root.join("include"))
        .include(zstd_root.join("include"))
        .file("lib.cpp")
        .warnings(false);

    if cfg!(feature = "openmp") {
        env::var("DEP_OPENMP_FLAG") // set by openmp-sys
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
                if i.as_os_str().is_empty() {
                    continue;
                }
                println!("cargo:{}", i.display());
            }
        }
    }
}

// https://stackoverflow.com/a/65192210
fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
    fs::create_dir_all(&dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let dst = dst.as_ref().join(entry.file_name());

        if entry.file_type()?.is_dir() {
            copy_dir_all(entry.path(), dst)?;
        } else {
            let src = fs::metadata(entry.path())?;
            fs::copy(entry.path(), &dst)?;
            // also copy over the `accessed` and `modified` timestamps
            fs::File::options().write(true).open(dst)?.set_times(
                fs::FileTimes::new()
                    .set_accessed(src.accessed()?)
                    .set_modified(src.modified()?),
            )?;
        }
    }

    Ok(())
}
