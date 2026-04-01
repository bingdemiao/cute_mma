"""Dynamic loading and caching of compiled Prism kernel modules."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from types import ModuleType
from typing import Literal

from ._compiler import KernelType, compile_kernel
from ._config import BwdDAdRCompParams, BwdDBCompParams, CompParams

# In-process cache: cache_key -> loaded module
_module_cache: dict[str, ModuleType] = {}
_cache_lock = threading.Lock()


def get_or_compile(
    group_size: int,
    reconn_sz: int,
    backend: str = "cute",
    comp_params: CompParams | None = None,
    gated: bool = False,
    kernel_type: KernelType = "fwd",
    bwd_dadr_params: BwdDAdRCompParams | None = None,
    bwd_db_params: BwdDBCompParams | None = None,
) -> ModuleType:
    """Get a compiled kernel module, compiling if necessary.

    Returns a module with the appropriate function(s) for the kernel type:
    - fwd: forward(A, B, R, group_size, reconn_sz)
    - bwd_dadr: backward_dA_dR(dC, A, B, R, group_size, reconn_sz)
    - bwd_db: backward_dB(dC, A, R, group_size, reconn_sz)
    """
    if backend == "cublas":
        # cuBLAS has a single module for all operations
        key = "cublas_prism"
        module_name = key
    else:
        # Build cache key from kernel_type + relevant params
        from ._compiler import _params_cache_key
        params_key = _params_cache_key(kernel_type, comp_params, bwd_dadr_params, bwd_db_params)
        gated_suffix = "_gated" if gated else ""
        key = f"cute_{kernel_type}_g{group_size}_r{reconn_sz}_{params_key}{gated_suffix}"
        module_name = f"cute_{kernel_type}_g{group_size}_r{reconn_sz}{gated_suffix}"

    with _cache_lock:
        if key not in _module_cache:
            so_path = compile_kernel(
                group_size, reconn_sz, backend, comp_params,
                gated=gated, kernel_type=kernel_type,
                bwd_dadr_params=bwd_dadr_params, bwd_db_params=bwd_db_params,
            )
            module = _load_so(so_path, module_name)
            _module_cache[key] = module

        return _module_cache[key]


def _load_so(so_path: Path, module_name: str) -> ModuleType:
    """Load a compiled .so as a Python module."""
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec from {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
