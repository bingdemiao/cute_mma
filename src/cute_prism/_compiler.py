"""JIT compilation of Prism CUDA kernels.

Compiles per-variant .so files keyed by (backend, kernel_type, group_size, reconn_sz, comp_params),
caching them on disk under ~/.cache/cute_prism/. Each kernel type (fwd, bwd_dadr, bwd_db)
is compiled independently for faster incremental builds.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from ._config import BwdDAdRCompParams, BwdDBCompParams, CompParams

KernelType = Literal["fwd", "bwd_dadr", "bwd_db"]

# Root of the cute_mma source tree (three levels up: _compiler.py -> cute_prism -> src -> repo root)
_SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent


class CompilationError(RuntimeError):
    """Raised when kernel compilation fails."""

    def __init__(self, summary: str, full_log: str):
        self.summary = summary
        self.full_log = full_log
        super().__init__(summary)


def _cache_root() -> Path:
    return Path(os.environ.get("CUTE_PRISM_CACHE_DIR",
                                   Path.home() / ".cache" / "cute_prism"))


def _source_root() -> Path:
    root = Path(os.environ.get("CUTE_PRISM_SOURCE_DIR", _SOURCE_ROOT))
    if not (root / "cute_prism_coop_pc.cu").exists():
        raise RuntimeError(
            f"Cannot find cute_mma source tree at {root}. "
            f"Set CUTE_PRISM_SOURCE_DIR to the cute_mma root directory."
        )
    return root


def _source_files_for_kernel(src: Path, kernel_type: KernelType) -> list[Path]:
    """List source files for a specific kernel type."""
    common = [
        src / "cute_prism_coop_pc.hpp",
        src / "cute_prism_dtype.hpp",
        src / "cute_prism_util.hpp",
        src / "z_curve.hpp",
        src / "common.hpp",
        src / "torch_ext" / "prism_shuffle.cuh",
    ]
    if kernel_type == "fwd":
        return common + [
            src / "cute_prism_coop_pc.cu",
            src / "torch_ext" / "prism_fwd_torch.cu",
            src / "torch_ext" / "prism_fwd_bind.cpp",
        ]
    elif kernel_type == "bwd_dadr":
        return common + [
            src / "cute_prism_backward_dadr.cu",
            src / "cute_prism_backward_dadr.hpp",
            src / "torch_ext" / "prism_bwd_dadr_torch.cu",
            src / "torch_ext" / "prism_bwd_dadr_bind.cpp",
        ]
    elif kernel_type == "bwd_db":
        return common + [
            src / "cute_prism_backward_db.cu",
            src / "cute_prism_backward_db.hpp",
            src / "torch_ext" / "prism_bwd_db_torch.cu",
            src / "torch_ext" / "prism_bwd_db_bind.cpp",
        ]
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")


def _source_files_cublas(src: Path) -> list[Path]:
    """List source files for the cuBLAS backend."""
    return [
        src / "common.hpp",
        src / "torch_ext" / "cublas_prism_torch.cu",
        src / "torch_ext" / "cublas_prism_torch_bind.cpp",
    ]


def _compute_source_hash(src: Path, backend: str, kernel_type: KernelType = "fwd") -> str:
    h = hashlib.sha256()
    if backend == "cublas":
        files = _source_files_cublas(src)
    else:
        files = _source_files_for_kernel(src, kernel_type)
    for f in sorted(files):
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _params_cache_key(
    kernel_type: KernelType,
    comp_params: CompParams | None = None,
    bwd_dadr_params: BwdDAdRCompParams | None = None,
    bwd_db_params: BwdDBCompParams | None = None,
) -> str:
    """Get the cache key for the relevant params based on kernel type."""
    if kernel_type == "fwd":
        return (comp_params or CompParams()).cache_key()
    elif kernel_type == "bwd_dadr":
        return (bwd_dadr_params or BwdDAdRCompParams()).cache_key()
    elif kernel_type == "bwd_db":
        return (bwd_db_params or BwdDBCompParams()).cache_key()
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")


def _build_dir_name(
    group_size: int,
    reconn_sz: int,
    backend: str,
    kernel_type: KernelType,
    cache_key: str,
    gated: bool = False,
    internal_bias: bool = False,
    dtype: str = "fp16",
    dropout: bool = False,
) -> str:
    if backend == "cublas":
        return "cublas"
    gated_suffix = "_gated" if gated else ""
    bias_suffix = "_bias" if internal_bias else ""
    dtype_suffix = "_bf16" if dtype == "bf16" else ""
    drop_suffix = "_drop" if dropout else ""
    return f"cute_{kernel_type}_g{group_size}_r{reconn_sz}_{cache_key}{gated_suffix}{bias_suffix}{dtype_suffix}{drop_suffix}"


def _target_name(
    group_size: int,
    reconn_sz: int,
    backend: str,
    kernel_type: KernelType = "fwd",
    gated: bool = False,
    internal_bias: bool = False,
    dtype: str = "fp16",
    dropout: bool = False,
) -> str:
    if backend == "cublas":
        return "cublas_prism"
    gated_suffix = "_gated" if gated else ""
    bias_suffix = "_bias" if internal_bias else ""
    dtype_suffix = "_bf16" if dtype == "bf16" else ""
    drop_suffix = "_drop" if dropout else ""
    return f"cute_{kernel_type}_g{group_size}_r{reconn_sz}{gated_suffix}{bias_suffix}{dtype_suffix}{drop_suffix}"


def _compose_config_header(
    kernel_type: KernelType,
    comp_params: CompParams | None = None,
    bwd_dadr_params: BwdDAdRCompParams | None = None,
    bwd_db_params: BwdDBCompParams | None = None,
) -> str:
    """Compose prism_config.hpp containing CurrKernelParams plus relevant params struct."""
    parts = ["#pragma once", "#include <cute/tensor.hpp>", "namespace cute {", ""]

    if kernel_type == "fwd":
        params = comp_params or CompParams()
        parts.append(params.to_header())
    elif kernel_type == "bwd_dadr":
        params = bwd_dadr_params or BwdDAdRCompParams()
        parts.append(params.to_header())
    elif kernel_type == "bwd_db":
        params = bwd_db_params or BwdDBCompParams()
        parts.append(params.to_header())

    parts.extend(["", "}"])
    return "\n".join(parts) + "\n"


def _generate_cmake_cute(
    cache_dir: Path,
    src: Path,
    group_size: int,
    reconn_sz: int,
    kernel_type: KernelType,
    gated: bool = False,
    internal_bias: bool = False,
    dtype: str = "fp16",
    dropout: bool = False,
) -> None:
    target = _target_name(group_size, reconn_sz, "cute", kernel_type, gated, internal_bias, dtype, dropout)
    torch_ext = src / "torch_ext"

    # Select source files per kernel type
    if kernel_type == "fwd":
        sources = f"""\
add_library({target} SHARED
    {torch_ext / "prism_fwd_bind.cpp"}
    {torch_ext / "prism_fwd_torch.cu"}
    {src / "cute_prism_coop_pc.cu"}
)"""
    elif kernel_type == "bwd_dadr":
        sources = f"""\
add_library({target} SHARED
    {torch_ext / "prism_bwd_dadr_bind.cpp"}
    {torch_ext / "prism_bwd_dadr_torch.cu"}
    {src / "cute_prism_backward_dadr.cu"}
)"""
    elif kernel_type == "bwd_db":
        sources = f"""\
add_library({target} SHARED
    {torch_ext / "prism_bwd_db_bind.cpp"}
    {torch_ext / "prism_bwd_db_torch.cu"}
    {src / "cute_prism_backward_db.cu"}
)"""

    content = f"""\
cmake_minimum_required(VERSION 3.18)
project({target} LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python first (needed for arch detection and PyTorch)
find_program(PYTHON_HINT NAMES python python3)
if(PYTHON_HINT)
    set(Python_EXECUTABLE "${{PYTHON_HINT}}" CACHE FILEPATH "Python interpreter")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Detect GPU architecture via PyTorch
execute_process(
    COMMAND ${{Python_EXECUTABLE}} -c "import torch; print(torch.cuda.get_device_capability()[0]*10+torch.cuda.get_device_capability()[1])"
    OUTPUT_VARIABLE DETECTED_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE DETECT_RESULT
)
if(DETECT_RESULT EQUAL 0)
    set(CMAKE_CUDA_ARCHITECTURES ${{DETECTED_ARCH}})
else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${{INSTALLED_GPU_CCS_1}}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${{INSTALLED_GPU_CCS_2}}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${{INSTALLED_GPU_CCS_3}}")
    set(CMAKE_CUDA_ARCHITECTURES ${{CUDA_ARCH_LIST}})
endif()

set(CMAKE_CUDA_FLAGS "${{CMAKE_CUDA_FLAGS}} --expt-relaxed-constexpr --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS "${{CMAKE_CUDA_FLAGS}} -lineinfo --use_fast_math -O3")

list(APPEND CMAKE_MODULE_PATH "{src / 'cmake'}")
find_package(Cutlass REQUIRED)

execute_process(
    COMMAND ${{Python_EXECUTABLE}} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
)
if(NOT TORCH_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch. Ensure torch is installed.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${{TORCH_CMAKE_PREFIX}}")
find_package(Torch REQUIRED)

execute_process(
    COMMAND ${{Python_EXECUTABLE}} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

{sources}

set_target_properties({target} PROPERTIES
    PREFIX ""
    SUFFIX "${{PYTHON_EXT_SUFFIX}}"
)

target_include_directories({target} PRIVATE
    {src}
    {cache_dir}
    ${{CUTLASS_INCLUDE_DIRS}}
    ${{TORCH_INCLUDE_DIRS}}
    ${{Python_INCLUDE_DIRS}}
)

target_compile_definitions({target} PRIVATE
    PRISM_GROUP_SIZE={group_size}
    PRISM_RECONN_SIZE={reconn_sz}
    PRISM_GATED={'1' if gated else '0'}
    PRISM_INTERNAL_BIAS={'1' if internal_bias else '0'}
    PRISM_DTYPE={'1' if dtype == 'bf16' else '0'}
    PRISM_DROPOUT={'1' if dropout else '0'}
    TORCH_EXTENSION_NAME={target}
)

execute_process(
    COMMAND ${{Python_EXECUTABLE}} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_link_libraries({target} PRIVATE
    ${{TORCH_LIBRARIES}}
    "${{TORCH_LIB_DIR}}/libtorch_python.so"
    ${{Python_LIBRARIES}}
)

set_target_properties({target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
"""
    (cache_dir / "CMakeLists.txt").write_text(content)


def _generate_cmake_cublas(
    cache_dir: Path,
    src: Path,
) -> None:
    target = "cublas_prism"
    torch_ext = src / "torch_ext"

    content = f"""\
cmake_minimum_required(VERSION 3.18)
project({target} LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python first (needed for arch detection)
find_program(PYTHON_HINT NAMES python python3)
if(PYTHON_HINT)
    set(Python_EXECUTABLE "${{PYTHON_HINT}}" CACHE FILEPATH "Python interpreter")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Detect GPU architecture via PyTorch
execute_process(
    COMMAND ${{Python_EXECUTABLE}} -c "import torch; print(torch.cuda.get_device_capability()[0]*10+torch.cuda.get_device_capability()[1])"
    OUTPUT_VARIABLE DETECTED_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE DETECT_RESULT
)
if(DETECT_RESULT EQUAL 0)
    set(CMAKE_CUDA_ARCHITECTURES ${{DETECTED_ARCH}})
else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${{INSTALLED_GPU_CCS_1}}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${{INSTALLED_GPU_CCS_2}}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${{INSTALLED_GPU_CCS_3}}")
    set(CMAKE_CUDA_ARCHITECTURES ${{CUDA_ARCH_LIST}})
endif()

set(CMAKE_CUDA_FLAGS "${{CMAKE_CUDA_FLAGS}} -O3")

execute_process(
    COMMAND ${{Python_EXECUTABLE}} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
)
if(NOT TORCH_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch. Ensure torch is installed.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${{TORCH_CMAKE_PREFIX}}")
find_package(Torch REQUIRED)

execute_process(
    COMMAND ${{Python_EXECUTABLE}} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library({target} SHARED
    {torch_ext / "cublas_prism_torch_bind.cpp"}
    {torch_ext / "cublas_prism_torch.cu"}
)

set_target_properties({target} PROPERTIES
    PREFIX ""
    SUFFIX "${{PYTHON_EXT_SUFFIX}}"
)

target_include_directories({target} PRIVATE
    {src}
    ${{TORCH_INCLUDE_DIRS}}
    ${{Python_INCLUDE_DIRS}}
)

target_compile_definitions({target} PRIVATE
    TORCH_EXTENSION_NAME={target}
)

execute_process(
    COMMAND ${{Python_EXECUTABLE}} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries({target} PRIVATE
    ${{TORCH_LIBRARIES}}
    "${{TORCH_LIB_DIR}}/libtorch_python.so"
    ${{Python_LIBRARIES}}
    CUDA::cublas
)

set_target_properties({target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
"""
    (cache_dir / "CMakeLists.txt").write_text(content)



def _parse_build_errors(output: str) -> str:
    """Extract key error messages from build output."""
    lines = output.splitlines()
    errors = []
    for i, line in enumerate(lines):
        lower = line.lower()
        if "error" in lower or "cute_static_assert" in lower or "static_assert" in lower:
            start = max(0, i - 1)
            end = min(len(lines), i + 3)
            errors.extend(lines[start:end])
            errors.append("---")
    if errors:
        return "\n".join(errors)
    return "Unknown compilation error (see full log above)"


def compile_kernel(
    group_size: int,
    reconn_sz: int,
    backend: str = "cute",
    comp_params: CompParams | None = None,
    gated: bool = False,
    kernel_type: KernelType = "fwd",
    bwd_db_params: BwdDBCompParams | None = None,
    bwd_dadr_params: BwdDAdRCompParams | None = None,
    parallel_build: bool = False,
    internal_bias: bool = False,
    dtype: str = "fp16",
    dropout: bool = False,
) -> Path:
    """Compile (or retrieve from cache) a kernel .so for the given parameters.

    Args:
        parallel_build: If True, use ``-j1`` so each worker spawns exactly one
            nvcc process. Use this when multiple compilations run concurrently
            (e.g. during autotuning) to avoid OOM from many nvcc processes.

    Returns the path to the compiled .so file.
    """
    if backend == "cublas":
        # cuBLAS has a single module for all operations
        kernel_type = "fwd"  # doesn't matter, single build

    cache_key = _params_cache_key(kernel_type, comp_params, bwd_dadr_params, bwd_db_params)

    src = _source_root()
    build_dir_name = _build_dir_name(
        group_size, reconn_sz, backend, kernel_type, cache_key, gated, internal_bias, dtype, dropout
    )
    cache = _cache_root() / "builds" / build_dir_name
    cache.mkdir(parents=True, exist_ok=True)

    target = _target_name(group_size, reconn_sz, backend, kernel_type, gated, internal_bias, dtype, dropout)
    build_dir = cache / "build"

    lock_path = cache / ".lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # Check if a valid cached build exists
        source_hash = _compute_source_hash(src, backend, kernel_type)
        hash_file = cache / ".source_hash"
        config_hash_file = cache / ".config_hash"

        needs_rebuild = True
        if hash_file.exists() and config_hash_file.exists():
            cached_src_hash = hash_file.read_text().strip()
            cached_cfg_hash = config_hash_file.read_text().strip()
            if cached_src_hash == source_hash and cached_cfg_hash == cache_key:
                so_files = list(build_dir.glob(f"{target}.*"))
                if so_files:
                    needs_rebuild = False

        if not needs_rebuild:
            so_path = list(build_dir.glob(f"{target}.*"))[0]
            return so_path

        # Generate config header and CMakeLists.txt
        if backend == "cute":
            header = _compose_config_header(kernel_type, comp_params, bwd_dadr_params, bwd_db_params)
            (cache / "prism_config.hpp").write_text(header)
            _generate_cmake_cute(cache, src, group_size, reconn_sz, kernel_type, gated, internal_bias, dtype, dropout)
        elif backend == "cublas":
            _generate_cmake_cublas(cache, src)

        # Configure
        # When running inside autotuning (parallel_build=True), use -j1 so each
        # worker spawns exactly one nvcc process — total concurrent nvcc processes
        # equals CUTE_PRISM_COMPILE_WORKERS.  For standalone compilation, use all
        # available cores for maximum single-build speed.
        nproc = 1 if parallel_build else (os.cpu_count() or 4)
        python_exe = sys.executable

        print(f"[cute_prism] Compiling {backend}/{kernel_type} kernel for group_size={group_size}, reconn_sz={reconn_sz}...")

        configure_cmd = [
            "cmake",
            "-S", str(cache),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={python_exe}",
        ]
        result = subprocess.run(
            configure_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = result.stdout + "\n" + result.stderr
            summary = _parse_build_errors(output)
            raise CompilationError(
                f"CMake configuration failed for {backend}/{kernel_type} backend, "
                f"group_size={group_size}, reconn_sz={reconn_sz}:\n{summary}",
                output,
            )

        # Build
        build_cmd = [
            "cmake",
            "--build", str(build_dir),
            "--target", target,
            f"-j{nproc}",
        ]
        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = result.stdout + "\n" + result.stderr
            summary = _parse_build_errors(output)
            raise CompilationError(
                f"Kernel compilation failed for {backend}/{kernel_type} backend, "
                f"group_size={group_size}, reconn_sz={reconn_sz}:\n{summary}",
                output,
            )

        # Record hashes
        hash_file.write_text(source_hash)
        config_hash_file.write_text(cache_key)

        so_files = list(build_dir.glob(f"{target}.*"))
        if not so_files:
            raise CompilationError(
                f"Build succeeded but no .so file found for target {target}",
                result.stdout + "\n" + result.stderr,
            )

        print(f"[cute_prism] Compilation successful: {so_files[0].name}")
        return so_files[0]

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def cleanup_non_best_builds(
    group_size: int,
    reconn_sz: int,
    kernel_type: KernelType,
    best_cache_key: str,
    gated: bool = False,
) -> None:
    """Delete build directories for non-best configs to save disk space."""
    builds_root = _cache_root() / "builds"
    if not builds_root.exists():
        return

    gated_suffix = "_gated" if gated else ""
    prefix = f"cute_{kernel_type}_g{group_size}_r{reconn_sz}_"
    best_name = f"{prefix}{best_cache_key}{gated_suffix}"

    for d in builds_root.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix) or d.name == best_name:
            continue
        # Only clean dirs matching the same gated mode
        is_gated_dir = d.name.endswith("_gated")
        if is_gated_dir == gated:
            shutil.rmtree(d, ignore_errors=True)


def clear_cache() -> None:
    """Remove all cached kernel builds."""
    cache = _cache_root()
    if cache.exists():
        shutil.rmtree(cache)
        print(f"[cute_prism] Cleared cache at {cache}")
