"""Dynamic loading and caching of compiled OFT kernel modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from ._compiler import compile_kernel
from ._config import CompParams

# In-process cache: cache_key -> loaded module
_module_cache: dict[str, ModuleType] = {}


def get_or_compile(
    group_size: int,
    reconn_sz: int,
    backend: str = "cute",
    comp_params: CompParams | None = None,
) -> ModuleType:
    """Get a compiled kernel module, compiling if necessary.

    Returns a module with a `forward(A, B, R, group_size, reconn_sz)` function.
    """
    if comp_params is None:
        comp_params = CompParams()

    if backend == "cublas":
        # cuBLAS supports dynamic hyperparameters — single compiled module
        key = "cublas_oft"
    else:
        key = f"cute_oft_g{group_size}_r{reconn_sz}"

    if key not in _module_cache:
        so_path = compile_kernel(group_size, reconn_sz, backend, comp_params)
        module = _load_so(so_path, key)
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
