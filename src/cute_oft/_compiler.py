"""JIT compilation of OFT CUDA kernels.

Compiles per-variant .so files keyed by (backend, group_size, reconn_sz, comp_params),
caching them on disk under ~/.cache/cute_oft/. Handles file locking to
prevent concurrent-build races.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from ._config import BwdDAdRCompParams, BwdDBCompParams, CompParams

# Root of the cute_mma source tree (three levels up: _compiler.py -> cute_oft -> src -> repo root)
_SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent


class CompilationError(RuntimeError):
    """Raised when kernel compilation fails."""

    def __init__(self, summary: str, full_log: str):
        self.summary = summary
        self.full_log = full_log
        super().__init__(summary)


def _cache_root() -> Path:
    return Path(os.environ.get("CUTE_OFT_CACHE_DIR", Path.home() / ".cache" / "cute_oft"))


def _source_root() -> Path:
    root = Path(os.environ.get("CUTE_OFT_SOURCE_DIR", _SOURCE_ROOT))
    if not (root / "cute_oft_coop_pc.cu").exists():
        raise RuntimeError(
            f"Cannot find cute_mma source tree at {root}. "
            f"Set CUTE_OFT_SOURCE_DIR to the cute_mma root directory."
        )
    return root


def _source_files(src: Path, backend: str) -> list[Path]:
    """List all source files that affect the compiled kernel."""
    if backend == "cute":
        return [
            src / "cute_oft_coop_pc.cu",
            src / "cute_oft_coop_pc.hpp",
            src / "cute_oft_backward.cu",
            src / "cute_oft_backward.hpp",
            src / "cute_oft_util.hpp",
            src / "z_curve.hpp",
            src / "common.hpp",
            src / "torch_ext" / "oft_torch.cu",
            src / "torch_ext" / "oft_torch_bind.cpp",
        ]
    elif backend == "cublas":
        return [
            src / "common.hpp",
            src / "torch_ext" / "cublas_oft_torch.cu",
            src / "torch_ext" / "cublas_oft_torch_bind.cpp",
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _compute_source_hash(src: Path, backend: str) -> str:
    h = hashlib.sha256()
    for f in sorted(_source_files(src, backend)):
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _build_dir_name(group_size: int, reconn_sz: int, backend: str, comp: CompParams, gated: bool = False) -> str:
    if backend == "cute":
        gated_suffix = "_gated" if gated else ""
        return f"cute_g{group_size}_r{reconn_sz}_{comp.cache_key()}{gated_suffix}"
    elif backend == "cublas":
        # cuBLAS supports dynamic hyperparameters — single build for all configs
        return "cublas"
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _target_name(group_size: int, reconn_sz: int, backend: str) -> str:
    if backend == "cublas":
        return "cublas_oft"
    return f"{backend}_oft_g{group_size}_r{reconn_sz}"


def _generate_cmake_cute(
    cache_dir: Path,
    src: Path,
    group_size: int,
    reconn_sz: int,
    gated: bool = False,
) -> None:
    target = _target_name(group_size, reconn_sz, "cute")
    torch_ext = src / "torch_ext"
    cmake_module_path = src / "cmake"

    content = f"""\
cmake_minimum_required(VERSION 3.18)
project({target} LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Auto-detect CUDA architecture from current GPU
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${{INSTALLED_GPU_CCS_1}}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${{INSTALLED_GPU_CCS_2}}")
string(REPLACE "." "" CUDA_ARCH_LIST "${{INSTALLED_GPU_CCS_3}}")
set(CMAKE_CUDA_ARCHITECTURES ${{CUDA_ARCH_LIST}})

set(CMAKE_CUDA_FLAGS "${{CMAKE_CUDA_FLAGS}} --expt-relaxed-constexpr --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS "${{CMAKE_CUDA_FLAGS}} -lineinfo --use_fast_math -O3")

list(APPEND CMAKE_MODULE_PATH "{cmake_module_path}")
find_package(Cutlass REQUIRED)

# Find Python and PyTorch
find_program(PYTHON_HINT NAMES python python3)
if(PYTHON_HINT)
    set(Python_EXECUTABLE "${{PYTHON_HINT}}" CACHE FILEPATH "Python interpreter")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development)

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
    {torch_ext / "oft_torch_bind.cpp"}
    {torch_ext / "oft_torch.cu"}
    {src / "cute_oft_coop_pc.cu"}
    {src / "cute_oft_backward.cu"}
)

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
    OFT_GROUP_SIZE={group_size}
    OFT_RECONN_SIZE={reconn_sz}
    OFT_GATED={'1' if gated else '0'}
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
    target = "cublas_oft"
    torch_ext = src / "torch_ext"

    content = f"""\
cmake_minimum_required(VERSION 3.18)
project({target} LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Auto-detect CUDA architecture from current GPU
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${{INSTALLED_GPU_CCS_1}}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${{INSTALLED_GPU_CCS_2}}")
string(REPLACE "." "" CUDA_ARCH_LIST "${{INSTALLED_GPU_CCS_3}}")
set(CMAKE_CUDA_ARCHITECTURES ${{CUDA_ARCH_LIST}})

set(CMAKE_CUDA_FLAGS "${{CMAKE_CUDA_FLAGS}} -O3")

# Find Python and PyTorch
find_program(PYTHON_HINT NAMES python python3)
if(PYTHON_HINT)
    set(Python_EXECUTABLE "${{PYTHON_HINT}}" CACHE FILEPATH "Python interpreter")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development)

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
    {torch_ext / "cublas_oft_torch_bind.cpp"}
    {torch_ext / "cublas_oft_torch.cu"}
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


def _compose_config_header(
    comp_params: CompParams,
    bwd_db_params: BwdDBCompParams | None = None,
    bwd_dadr_params: BwdDAdRCompParams | None = None,
) -> str:
    """Compose the full oft_config.hpp with forward + backward params."""
    if bwd_db_params is None:
        bwd_db_params = BwdDBCompParams()
    if bwd_dadr_params is None:
        bwd_dadr_params = BwdDAdRCompParams()

    return f"""\
#pragma once
#include <cute/tensor.hpp>
namespace cute {{

{comp_params.to_header()}

{bwd_db_params.to_header()}

{bwd_dadr_params.to_header()}

}}
"""


def compile_kernel(
    group_size: int,
    reconn_sz: int,
    backend: str = "cute",
    comp_params: CompParams | None = None,
    gated: bool = False,
    bwd_db_params: BwdDBCompParams | None = None,
    bwd_dadr_params: BwdDAdRCompParams | None = None,
) -> Path:
    """Compile (or retrieve from cache) a kernel .so for the given parameters.

    Returns the path to the compiled .so file.
    """
    if comp_params is None:
        comp_params = CompParams()
    if bwd_db_params is None:
        bwd_db_params = BwdDBCompParams()
    if bwd_dadr_params is None:
        bwd_dadr_params = BwdDAdRCompParams()

    src = _source_root()
    cache = _cache_root() / "builds" / _build_dir_name(group_size, reconn_sz, backend, comp_params, gated)
    cache.mkdir(parents=True, exist_ok=True)

    target = _target_name(group_size, reconn_sz, backend)
    build_dir = cache / "build"

    lock_path = cache / ".lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # Check if a valid cached build exists
        source_hash = _compute_source_hash(src, backend)
        hash_file = cache / ".source_hash"
        config_hash_file = cache / ".config_hash"

        config_key = comp_params.cache_key() if backend == "cute" else "default"

        needs_rebuild = True
        if hash_file.exists() and config_hash_file.exists():
            cached_src_hash = hash_file.read_text().strip()
            cached_cfg_hash = config_hash_file.read_text().strip()
            if cached_src_hash == source_hash and cached_cfg_hash == config_key:
                so_files = list(build_dir.glob(f"{target}.*"))
                if so_files:
                    needs_rebuild = False

        if not needs_rebuild:
            so_path = list(build_dir.glob(f"{target}.*"))[0]
            return so_path

        # Generate config header and CMakeLists.txt
        if backend == "cute":
            header = _compose_config_header(comp_params, bwd_db_params, bwd_dadr_params)
            (cache / "oft_config.hpp").write_text(header)
            _generate_cmake_cute(cache, src, group_size, reconn_sz, gated)
        elif backend == "cublas":
            _generate_cmake_cublas(cache, src)

        # Configure
        nproc = os.cpu_count() or 4
        python_exe = sys.executable

        print(f"[cute_oft] Compiling {backend} kernel for group_size={group_size}, reconn_sz={reconn_sz}...")

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
                f"CMake configuration failed for {backend} backend, "
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
                f"Kernel compilation failed for {backend} backend, "
                f"group_size={group_size}, reconn_sz={reconn_sz}:\n{summary}",
                output,
            )

        # Record hashes
        hash_file.write_text(source_hash)
        config_hash_file.write_text(config_key)

        so_files = list(build_dir.glob(f"{target}.*"))
        if not so_files:
            raise CompilationError(
                f"Build succeeded but no .so file found for target {target}",
                result.stdout + "\n" + result.stderr,
            )

        print(f"[cute_oft] Compilation successful: {so_files[0].name}")
        return so_files[0]

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def clear_cache() -> None:
    """Remove all cached kernel builds."""
    import shutil

    cache = _cache_root()
    if cache.exists():
        shutil.rmtree(cache)
        print(f"[cute_oft] Cleared cache at {cache}")
