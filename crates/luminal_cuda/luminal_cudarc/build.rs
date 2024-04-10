use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(not(feature = "ci-check"))]
    link_cuda();
}

#[allow(unused)]
fn link_cuda() {
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    let candidates: Vec<PathBuf> = root_candidates().collect();

    let toolkit_root = root_candidates()
        .find(|path| path.join("include").join("cuda.h").is_file())
        .unwrap_or_else(|| {
            panic!(
                "Unable to find `include/cuda.h` under any of: {:?}. Set the `CUDA_ROOT` environment variable to `$CUDA_ROOT/include/cuda.h` to override path.",
                candidates
            )
        });

    for path in lib_candidates(&toolkit_root) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    #[cfg(feature = "driver")]
    println!("cargo:rustc-link-lib=dylib=cuda");
    #[cfg(feature = "nccl")]
    println!("cargo:rustc-link-lib=dylib=nccl");

    #[cfg(feature = "static-linking")]
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        #[cfg(any(feature = "cublas", feature = "cublaslt"))]
        {
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=static=cublasLt_static");
        }
        #[cfg(feature = "cublas")]
        println!("cargo:rustc-link-lib=static=cublas_static");
        #[cfg(feature = "curand")]
        {
            println!("cargo:rustc-link-lib=dylib=culibos");
            println!("cargo:rustc-link-lib=static=curand_static");
        }
        #[cfg(feature = "nvrtc")]
        {
            println!("cargo:rustc-link-lib=static=nvrtc_static");
            println!("cargo:rustc-link-lib=static=nvptxcompiler_static");
            println!("cargo:rustc-link-lib=static=nvrtc-builtins_static");
        }
    }
    #[cfg(not(feature = "static-linking"))]
    {
        #[cfg(feature = "nvrtc")]
        println!("cargo:rustc-link-lib=dylib=nvrtc");
        #[cfg(feature = "curand")]
        println!("cargo:rustc-link-lib=dylib=curand");
        #[cfg(feature = "cublas")]
        println!("cargo:rustc-link-lib=dylib=cublas");
        #[cfg(any(feature = "cublas", feature = "cublaslt"))]
        println!("cargo:rustc-link-lib=dylib=cublasLt");
    }

    #[cfg(feature = "cudnn")]
    {
        let cudnn_root = root_candidates()
            .find(|path| path.join("include").join("cudnn.h").is_file())
            .unwrap_or_else(|| {
                panic!(
                    "Unable to find `include/cudnn.h` under any of: {:?}. Set the `CUDNN_LIB` environment variable to `$CUDNN_LIB/include/cudnn.h` to override path.",
                    candidates
                )
            });

        for path in lib_candidates(&cudnn_root) {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
    #[cfg(feature = "cudnn")]
    println!("cargo:rustc-link-lib=dylib=cudnn");
}

fn root_candidates() -> impl Iterator<Item = PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::into);
    env_vars.chain(roots).map(Into::<PathBuf>::into)
}

fn lib_candidates(root: &Path) -> Vec<PathBuf> {
    [
        "lib",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ]
    .iter()
    .map(|&p| root.join(p))
    .filter(|p| p.is_dir())
    .collect()
}
