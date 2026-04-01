"""Autotuning for Prism kernel parameters.

Searches over a space of tile sizes, pipeline depths, and warp layouts to find
the fastest parameters for each kernel type (forward, backward dA+dR, backward dB).
Results are cached to disk so that repeated calls with the same shape skip benchmarking.

Search space protocol:
- Iterable of CompParam items or (CompParam, callback) tuples
- callback(config, time_ms) is called after benchmarking (time_ms is None if failed)
- For already-tested configs: if callback is provided, it is called with cached time
- Supports generators for adaptive/early-stopping strategies
"""

from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import torch

from ._compiler import KernelType
from ._config import BwdDAdRCompParams, BwdDBCompParams, CompParams
from ._validate import (
    check_smem_limit,
    compute_smem_bytes,
    compute_smem_bytes_bwd_dadr,
    compute_smem_bytes_bwd_db,
)

T = TypeVar("T", CompParams, BwdDAdRCompParams, BwdDBCompParams)


# ---------------------------------------------------------------------------
# Default search spaces
# ---------------------------------------------------------------------------

def default_search_space(
    group_size: int,
    reconn_sz: int,
    device: int = 0,
) -> list[CompParams]:
    """Generate candidate CompParams configurations for the forward kernel."""
    # Warp layout presets: (warp_layout_ar, warp_layout_arb, min_bM, min_bN)
    warp_presets: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = [
        ((2,), (2, 2), 16, 16),    # 6 warps, 192 threads — general purpose
        ((4,), (2, 2), 16, 16),    # 8 warps, 256 threads — more AR producers
        ((2,), (4, 4), 32, 32),    # 18 warps, 576 threads — more ARB consumers
    ]

    candidates: list[CompParams] = []
    for bM in (128, 256):
        for bN in (128, 256):
            if bN % group_size != 0 and group_size % bN != 0:
                continue
            for bK in (16, 32):
                if bK % reconn_sz != 0 and reconn_sz % bK != 0:
                    continue
                for c_width in range(reconn_sz, bK + 1, reconn_sz):
                    if bK % c_width != 0:
                        continue
                    for wl_ar, wl_arb, min_bM, min_bN in warp_presets:
                        if bM < min_bM or bN < min_bN:
                            continue
                        warp_tile_n = bN // wl_arb[1] if len(wl_arb) > 1 else bN
                        if warp_tile_n > group_size:
                            continue
                        params = CompParams(
                            bM=bM, bN=bN, bK=bK,
                            c_width=c_width,
                            bP_a_r=2, bP_ar=2, bP_b=2,
                            warp_layout_ar=wl_ar, warp_layout_arb=wl_arb,
                        )
                        try:
                            smem = compute_smem_bytes(
                                bM, bN, bK, group_size, reconn_sz,
                                c_width, 2, 2, 2,
                            )
                            check_smem_limit(smem, device)
                        except (RuntimeError, ValueError):
                            continue
                        candidates.append(params)
    return candidates


def default_search_space_bwd_dadr(
    group_size: int,
    reconn_sz: int,
    device: int = 0,
) -> list[BwdDAdRCompParams]:
    """Generate candidate configs for the PC (producer-consumer) dAdR kernel.

    The dH GEMM reduces over N (group dimension), so bN should be small
    while bM and bK should be large.  Producer warps (2D) do the heavy GEMM;
    consumer warps (1D along K) handle the lightweight dA/dR epilogue.
    """
    # (consumer, producer) — producer gets more warps for the heavy GEMM
    warp_presets: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
        ((2,), (2, 2)),    # 2 consumer + 4 producer = 6 warps
        ((2,), (2, 4)),    # 2 consumer + 6 producer = 8 warps
        ((4,), (2, 4)),    # 4 consumer + 8 producer = 12 warps
    ]

    candidates: list[BwdDAdRCompParams] = []
    for bM in (64, 128, 256):
        for bK in (128, 256):
            for bN in (16, 32):
                if group_size % bN != 0:
                    continue
                for bP_dc_b in (2,):
                    for bP_dh in (1,):
                        for n_buf_slots in (4, 8):
                            for wl_arb, wl_ar in warp_presets:
                                n_cons = _warp_count(wl_arb)
                                warp_k = bK // n_cons
                                if warp_k < reconn_sz or warp_k % reconn_sz != 0:
                                    continue
                                params = BwdDAdRCompParams(
                                    bM=bM, bK=bK, bN=bN,
                                    bP_dc_b=bP_dc_b, bP_dh=bP_dh,
                                    n_buf_slots=n_buf_slots,
                                    warp_layout_arb=wl_arb,
                                    warp_layout_ar=wl_ar,
                                )
                                try:
                                    smem = compute_smem_bytes_bwd_dadr(
                                        bM, bK, bN, group_size, reconn_sz,
                                        bP_dc_b=bP_dc_b, bP_dh=bP_dh,
                                    )
                                    check_smem_limit(smem, device)
                                except (RuntimeError, ValueError):
                                    continue
                                candidates.append(params)
    return candidates


def _warp_count(layout: tuple[int, ...]) -> int:
    """Compute total warp count from a warp layout shape tuple."""
    result = 1
    for s in layout:
        result *= s
    return result


def default_search_space_bwd_db(
    group_size: int,
    reconn_sz: int,
    device: int = 0,
) -> list[BwdDBCompParams]:
    """Generate candidate BwdDBCompParams configurations."""
    # Warp presets: (warp_layout_ar, warp_layout_arb)
    # dB: ar = AR producer (1D), arb = heavy consumer (2D: warp_along_N, warp_along_K)
    warp_presets: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
        ((2,), (1, 2)),  # 2 AR, 2 consumer (split K)
        ((2,), (2, 2)),  # 2 AR, 4 consumer (split N and K)
        ((2,), (2, 4)),  # 2 AR, 8 consumer
    ]

    candidates: list[BwdDBCompParams] = []
    for bM in (16, 32):
        for bN in (128, 256):  # bN = group_size for now; multi-group later
            for bK in (128, 256):
                if bK % reconn_sz != 0:
                    continue
                for c_width in (16,):
                    if bM % c_width != 0:
                        continue
                    for bP_a in (2, 3):
                        for bP_ar in (2,):
                            for bP_dc in (2, 3):
                                for wl_ar, wl_arb in warp_presets:
                                    n_producer_warps = _warp_count(wl_ar)
                                    n_consumer_warps = _warp_count(wl_arb)
                                    warp_along_n = wl_arb[0] if len(wl_arb) >= 2 else 1
                                    warp_along_k = wl_arb[1] if len(wl_arb) >= 2 else wl_arb[0]
                                    # Validate tile divisibility
                                    if bM % n_producer_warps != 0:
                                        continue
                                    if bN % warp_along_n != 0:
                                        continue
                                    if bK % warp_along_k != 0:
                                        continue
                                    warp_tile_n = bN // warp_along_n
                                    if warp_tile_n > group_size:
                                        continue
                                    if warp_tile_n < 8:
                                        continue
                                    params = BwdDBCompParams(
                                        bM=bM, bN=bN, bK=bK, c_width=c_width,
                                        bP_a=bP_a, bP_ar=bP_ar, bP_dc=bP_dc,
                                        warp_layout_ar=wl_ar, warp_layout_arb=wl_arb,
                                    )
                                    try:
                                        smem = compute_smem_bytes_bwd_db(
                                            bM, bK, group_size, reconn_sz,
                                            bP_a, bP_ar, bP_dc,
                                        )
                                        check_smem_limit(smem, device)
                                    except (RuntimeError, ValueError):
                                        continue
                                    candidates.append(params)
    return candidates


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _problem_key(
    M: int, N: int, K: int,
    group_size: int, reconn_sz: int,
    device_name: str,
    kernel_type: KernelType = "fwd",
    gated: bool = False,
) -> str:
    """Cache key for a specific problem shape + device + kernel type."""
    suffix = "_gated" if gated else ""
    return f"{device_name}_{kernel_type}_M{M}_N{N}_K{K}_g{group_size}_r{reconn_sz}{suffix}"


def _cache_path() -> Path:
    import os
    root = Path(os.environ.get("CUTE_PRISM_CACHE_DIR",
                                   Path.home() / ".cache" / "cute_prism"))
    return root / "autotune_results.json"


def _load_cache() -> dict:
    path = _cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def _compile_failed_path() -> Path:
    import os
    root = Path(os.environ.get("CUTE_PRISM_CACHE_DIR",
                                   Path.home() / ".cache" / "cute_prism"))
    return root / "compile_failed.json"


def _load_compile_failed() -> dict[str, dict]:
    """Load the global compile-failed registry.

    Returns dict mapping compile_context_key -> {"source_hash": str, "keys": [str]}
    """
    path = _compile_failed_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_compile_failed(data: dict[str, dict]) -> None:
    path = _compile_failed_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _compile_context_key(
    group_size: int, reconn_sz: int,
    kernel_type: KernelType, gated: bool,
) -> str:
    """Key for compile-failed registry (shape-independent)."""
    suffix = "_gated" if gated else ""
    return f"{kernel_type}_g{group_size}_r{reconn_sz}{suffix}"


def _current_source_hash(kernel_type: KernelType) -> str:
    """Get the current source hash for a kernel type."""
    from ._compiler import _compute_source_hash, _source_root
    src = _source_root()
    return _compute_source_hash(src, "cute", kernel_type)


def _load_known_bad(ctx_key: str, kernel_type: KernelType) -> set[str]:
    """Load known-bad config keys for a context, checking source hash freshness."""
    data = _load_compile_failed()
    entry = data.get(ctx_key)
    if entry is None:
        return set()
    # If source has changed, these failures may no longer apply
    if entry.get("source_hash") != _current_source_hash(kernel_type):
        return set()
    return set(entry.get("keys", []))


# Lock for thread-safe incremental writes to compile_failed.json
_compile_failed_lock = threading.Lock()


def _record_compile_failure(
    ctx_key: str, kernel_type: KernelType, cache_key: str,
) -> None:
    """Immediately record a single compile failure to the persistent registry.

    This is called from the producer thread so that failures are persisted
    even if the autotuning process is interrupted before completion.
    """
    with _compile_failed_lock:
        data = _load_compile_failed()
        src_hash = _current_source_hash(kernel_type)
        entry = data.get(ctx_key, {})
        # Reset if source changed
        if entry.get("source_hash") != src_hash:
            entry = {"source_hash": src_hash, "keys": []}
        existing = set(entry.get("keys", []))
        if cache_key not in existing:
            existing.add(cache_key)
            entry["keys"] = sorted(existing)
            entry["source_hash"] = src_hash
            data[ctx_key] = entry
            _save_compile_failed(data)


def _parse_problem_key(key: str) -> dict | None:
    """Parse a _problem_key string back into its components.

    Returns dict with M, N, K, group_size, reconn_sz, device_name,
    kernel_type, gated — or None if parsing fails.
    """
    import re
    # Format: {device_name}_{kernel_type}_M{M}_N{N}_K{K}_g{group_size}_r{reconn_sz}[_gated]
    # kernel_type is one of: fwd, bwd_dadr, bwd_db
    # device_name can contain underscores, so we match from the right
    pattern = r'^(.+?)_(fwd|bwd_dadr|bwd_db)_M(\d+)_N(\d+)_K(\d+)_g(\d+)_r(\d+)(_gated)?$'
    m = re.match(pattern, key)
    if not m:
        return None
    return {
        "device_name": m.group(1),
        "kernel_type": m.group(2),
        "M": int(m.group(3)),
        "N": int(m.group(4)),
        "K": int(m.group(5)),
        "group_size": int(m.group(6)),
        "reconn_sz": int(m.group(7)),
        "gated": m.group(8) is not None,
    }


def get_autotune_cache() -> dict:
    """Inspect the autotune results cache.

    Returns nested dict::

        {
            (M, N, K, group_size, reconn_sz): {
                (kernel_type, gated): {
                    config_cache_key: time_ms | None,
                    ...
                },
            },
        }
    """
    cache = _load_cache()
    result: dict = {}
    for key, entry in cache.items():
        parsed = _parse_problem_key(key)
        if parsed is None:
            continue
        shape_key = (
            parsed["M"], parsed["N"], parsed["K"],
            parsed["group_size"], parsed["reconn_sz"],
        )
        kernel_key = (parsed["kernel_type"], parsed["gated"])
        configs: dict[str, float | None] = {}
        for t in entry.get("tested", []):
            configs[t["cache_key"]] = t.get("time_ms")
        result.setdefault(shape_key, {})[kernel_key] = configs
    return result


def clear_autotune_cache(
    M: int | None = None,
    N: int | None = None,
    K: int | None = None,
    group_size: int | None = None,
    reconn_sz: int | None = None,
    device: int | None = None,
    kernel_type: KernelType | None = None,
    gated: bool | None = None,
    keep_best_kernels: bool = True,
) -> None:
    """Clear autotune results from the cache.

    **Mode 1 — Specific entry** (all parameters provided):
    Removes that single entry from autotune_results.json. Does NOT delete
    compiled kernel .so files (they may be reused for other MNK shapes).

    **Mode 2 — Full wipe** (no parameters provided):
    If ``keep_best_kernels=True``: deletes autotune_results.json but keeps
    compiled .so files that are referenced as "best" by any entry. Deletes
    all other compiled kernel builds.
    If ``keep_best_kernels=False``: deletes everything — autotune_results.json
    and all compiled kernel builds.

    Raises ``ValueError`` if some but not all of the identifying parameters
    are provided.
    """
    import shutil

    identifying = [M, N, K, group_size, reconn_sz, device, kernel_type, gated]
    n_provided = sum(x is not None for x in identifying)

    if 0 < n_provided < len(identifying):
        raise ValueError(
            "Either provide all of M, N, K, group_size, reconn_sz, device, "
            "kernel_type, gated for a specific entry, or none for a full wipe."
        )

    if n_provided == len(identifying):
        # Mode 1: remove specific entry
        assert M is not None and N is not None and K is not None
        assert group_size is not None and reconn_sz is not None
        assert device is not None and kernel_type is not None and gated is not None
        device_name = torch.cuda.get_device_properties(device).name
        key = _problem_key(M, N, K, group_size, reconn_sz, device_name, kernel_type, gated)
        cache = _load_cache()
        if key in cache:
            del cache[key]
            _save_cache(cache)
        return

    # Mode 2: full wipe
    # Also clear the compile-failed registry
    cf_path = _compile_failed_path()
    if cf_path.exists():
        cf_path.unlink()

    cache_path = _cache_path()

    if keep_best_kernels:
        # Collect best cache keys before deleting
        cache = _load_cache()
        best_keys: set[tuple[int, int, str, bool, str]] = set()
        for key, entry in cache.items():
            parsed = _parse_problem_key(key)
            if parsed is None:
                continue
            best = entry.get("best")
            if best is not None:
                from ._config import CompParams as _CP, BwdDAdRCompParams as _DADR, BwdDBCompParams as _DB
                kt = parsed["kernel_type"]
                params_cls = {"fwd": _CP, "bwd_dadr": _DADR, "bwd_db": _DB}[kt]
                config = params_cls.from_dict(best["params"])
                best_keys.add((
                    parsed["group_size"], parsed["reconn_sz"],
                    kt, parsed["gated"], config.cache_key(),
                ))

        # Delete autotune results
        if cache_path.exists():
            cache_path.unlink()

        # Delete non-best build dirs
        builds_root = cache_path.parent / "builds"
        if builds_root.exists():
            for d in builds_root.iterdir():
                if not d.is_dir():
                    continue
                # Check if this build dir matches any best key
                keep = False
                for gs, rs, kt, g, ck in best_keys:
                    gated_suffix = "_gated" if g else ""
                    best_name = f"cute_{kt}_g{gs}_r{rs}_{ck}{gated_suffix}"
                    if d.name == best_name:
                        keep = True
                        break
                if not keep:
                    shutil.rmtree(d, ignore_errors=True)
    else:
        # Delete everything
        if cache_path.exists():
            cache_path.unlink()
        builds_root = cache_path.parent / "builds"
        if builds_root.exists():
            shutil.rmtree(builds_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Core autotuning engine (shared by all three autotune functions)
# ---------------------------------------------------------------------------

def _compile_one(
    config: T,
    group_size: int,
    reconn_sz: int,
    kernel_type: KernelType,
    gated: bool = False,
) -> bool:
    """Compile a single config. Returns True on success, False on failure."""
    from ._compiler import compile_kernel
    kwargs: dict = dict(
        group_size=group_size, reconn_sz=reconn_sz,
        backend="cute", gated=gated, kernel_type=kernel_type,
        parallel_build=True,
    )
    if kernel_type == "fwd":
        kwargs["comp_params"] = config
    elif kernel_type == "bwd_dadr":
        kwargs["bwd_dadr_params"] = config
    elif kernel_type == "bwd_db":
        kwargs["bwd_db_params"] = config
    try:
        compile_kernel(**kwargs)
        return True
    except Exception:
        return False



def _benchmark_one(
    config: T,
    kernel_type: KernelType,
    tensors: dict,
    group_size: int,
    reconn_sz: int,
    warmup: int,
    repeat: int,
    gated: bool = False,
    device: int = 0,
) -> float | None:
    """Benchmark a single config. Returns median time in ms, or None on failure."""
    from ._loader import get_or_compile

    kwargs: dict = dict(
        group_size=group_size, reconn_sz=reconn_sz,
        backend="cute", gated=gated, kernel_type=kernel_type,
    )
    if kernel_type == "fwd":
        kwargs["comp_params"] = config
    elif kernel_type == "bwd_dadr":
        kwargs["bwd_dadr_params"] = config
    elif kernel_type == "bwd_db":
        kwargs["bwd_db_params"] = config

    try:
        module = get_or_compile(**kwargs)
    except Exception:
        return None

    # Build the call args per kernel type
    if kernel_type == "fwd":
        call = lambda: module.forward(tensors["A"], tensors["B"], tensors["R"], group_size, reconn_sz)
    elif kernel_type == "bwd_dadr":
        call = lambda: module.backward_dA_dR(
            tensors["dC"], tensors["A"], tensors["B"], tensors["R"],
            group_size, reconn_sz)
    elif kernel_type == "bwd_db":
        call = lambda: module.backward_dB(
            tensors["dC"], tensors["A"], tensors["R"],
            group_size, reconn_sz)
    else:
        return None

    with torch.cuda.device(device):
        # Warmup
        for _ in range(warmup):
            try:
                call()
            except Exception:
                return None

        torch.cuda.synchronize()

        # Timed runs
        times: list[float] = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def _autotune_generic(
    kernel_type: KernelType,
    params_cls: type[T],
    M: int, N: int, K: int,
    group_size: int,
    reconn_sz: int,
    device: int | list[int],
    search_space: Iterable[T | tuple[T, Callable]] | None,
    default_space_fn: Callable,
    warmup: int,
    repeat: int,
    use_cache: bool,
    verbose: bool,
    gated: bool,
    make_tensors: Callable,
    force_rebenchmark: bool = False,
) -> T:
    """Generic autotuning engine shared by all three autotune functions.

    Uses a producer-consumer pipeline: compilation threads push completed
    configs to a queue, and benchmark worker threads (one per GPU) pull
    from the queue and benchmark as configs become available.  This
    overlaps compilation and benchmarking for better resource utilisation.
    """
    # Normalise device to a list
    devices: list[int] = [device] if isinstance(device, int) else list(device)
    if not devices:
        raise ValueError("device must be a non-empty int or list[int]")

    # Use the first device for cache key / smem checks
    primary_device = devices[0]
    device_name = torch.cuda.get_device_properties(primary_device).name
    key = _problem_key(M, N, K, group_size, reconn_sz, device_name, kernel_type, gated)

    # Load existing cache entry
    cache_data = _load_cache() if use_cache else {}
    entry = cache_data.get(key, {"best": None, "tested": []})
    tested_keys = {t["cache_key"] for t in entry.get("tested", [])}

    # Build lookup for cached results
    tested_by_key = {t["cache_key"]: t for t in entry.get("tested", [])}

    # Load global compile-failed registry (auto-invalidated on source changes)
    ctx_key = _compile_context_key(group_size, reconn_sz, kernel_type, gated)
    known_bad: set[str] = _load_known_bad(ctx_key, kernel_type)

    # If no search space provided and we have a cached best, return it
    if search_space is None and entry.get("best") is not None and not force_rebenchmark:
        cached_best = entry["best"]
        params = params_cls.from_dict(cached_best["params"])
        if verbose:
            time_str = f"{cached_best['time_ms']:.3f} ms" if cached_best["time_ms"] is not None else "FAILED"
            print(f"[cute_prism] Using cached autotune result for {kernel_type} "
                  f"M={M}, N={N}, K={K}: {time_str}")
        return params

    if search_space is None:
        search_space = default_space_fn(group_size, reconn_sz, primary_device)

    # Iterate search space lazily, separating configs and callbacks
    configs_to_test: list[T] = []
    callbacks: dict[str, Callable] = {}
    skipped_count = 0

    for item in search_space:
        if isinstance(item, tuple):
            config, callback = item
        else:
            config, callback = item, None

        ck = config.cache_key()

        if callback is not None:
            callbacks[ck] = callback

        if ck in tested_keys and not force_rebenchmark:
            # Already tested — report cached result through callback
            skipped_count += 1
            if callback is not None:
                cached_entry = tested_by_key.get(ck)
                if cached_entry is not None:
                    callback(config, cached_entry.get("time_ms"))
            continue

        if ck in known_bad:
            # Known compile failure — skip without recompiling
            skipped_count += 1
            if callback is not None:
                callback(config, None)
            continue

        configs_to_test.append(config)

    if not configs_to_test and entry.get("best") is not None:
        if verbose:
            print(f"[cute_prism] All {skipped_count} configs already tested for {kernel_type}, "
                  f"using cached best.")
        return params_cls.from_dict(entry["best"]["params"])

    if not configs_to_test and not entry.get("best"):
        raise ValueError(
            f"No valid configurations in the search space for {kernel_type}."
        )

    if verbose:
        total = len(configs_to_test) + skipped_count
        print(f"[cute_prism] Autotuning {len(configs_to_test)} {kernel_type} configurations "
              f"for M={M}, N={N}, K={K} ({skipped_count} cached, skipped)...")
        if len(devices) > 1:
            print(f"[cute_prism]   Benchmarking on {len(devices)} GPUs: {devices}")

    # -------------------------------------------------------------------
    # Producer-consumer pipeline
    # -------------------------------------------------------------------
    import os
    max_compile_workers = int(os.environ.get("CUTE_PRISM_COMPILE_WORKERS", 4))
    n_compile_workers = min(len(configs_to_test), max_compile_workers)
    n_bench_workers = len(devices)

    # Shared queue: items are (index, config, success) or None (sentinel)
    work_queue: queue.Queue[tuple[int, T, bool] | None] = queue.Queue()

    # Thread-safe results collection
    new_results: list[dict | None] = [None] * len(configs_to_test)
    results_lock = threading.Lock()

    # Counters for progress reporting
    compile_done = 0
    compile_fail = 0
    bench_done = 0
    total_compilable = len(configs_to_test)
    total_compiled_ok = 0
    counter_lock = threading.Lock()

    # --- Producer: compile configs and push to queue ---
    def _producer() -> None:
        nonlocal compile_done, compile_fail, total_compiled_ok

        with ThreadPoolExecutor(max_workers=n_compile_workers) as pool:
            futures = {
                pool.submit(_compile_one, config, group_size, reconn_sz, kernel_type, gated): i
                for i, config in enumerate(configs_to_test)
            }
            for future in as_completed(futures):
                idx = futures[future]
                success = future.result()
                work_queue.put((idx, configs_to_test[idx], success))

                with counter_lock:
                    compile_done += 1
                    if not success:
                        compile_fail += 1
                        # Record failure immediately so it persists even
                        # if autotuning is interrupted later.
                        ck = configs_to_test[idx].cache_key()
                        _record_compile_failure(ctx_key, kernel_type, ck)
                    else:
                        total_compiled_ok += 1
                    if verbose and compile_done % max(1, total_compilable // 5) == 0:
                        print(f"[cute_prism]   Compiled {compile_done}/{total_compilable} "
                              f"({compile_fail} failed)")

        if verbose:
            with counter_lock:
                ok = total_compiled_ok
            print(f"[cute_prism] Compilation done: {ok} succeeded, "
                  f"{total_compilable - ok} failed")

        # Push one sentinel per benchmark worker
        for _ in range(n_bench_workers):
            work_queue.put(None)

    # --- Consumer: benchmark on a specific GPU ---
    def _consumer(device_id: int) -> None:
        nonlocal bench_done
        tensors = make_tensors(M, N, K, group_size, reconn_sz, device_id)

        while True:
            item = work_queue.get()
            if item is None:
                # Sentinel — clean up and exit
                del tensors
                return

            idx, config, compiled_ok = item
            ck = config.cache_key()

            if not compiled_ok:
                result_entry = {
                    "params": config.to_dict(),
                    "cache_key": ck,
                    "time_ms": None,
                }
                with results_lock:
                    new_results[idx] = result_entry
                cb = callbacks.get(ck)
                if cb is not None:
                    cb(config, None)
                continue

            time_ms = _benchmark_one(
                config, kernel_type, tensors,
                group_size, reconn_sz, warmup, repeat, gated,
                device=device_id,
            )

            result_entry = {
                "params": config.to_dict(),
                "cache_key": ck,
                "time_ms": time_ms,
            }
            with results_lock:
                new_results[idx] = result_entry

            cb = callbacks.get(ck)
            if cb is not None:
                cb(config, time_ms)

            if verbose:
                with counter_lock:
                    bench_done += 1
                    bd = bench_done
                    tc = total_compiled_ok
                time_str = f"{time_ms:.3f} ms" if time_ms is not None else "FAILED"
                gpu_tag = f" [GPU {device_id}]" if n_bench_workers > 1 else ""
                print(f"  [{bd}/{tc}] {_config_summary(config)} — {time_str}{gpu_tag}")

    # Launch producer and consumer threads
    producer_thread = threading.Thread(target=_producer, daemon=True)
    consumer_threads = [
        threading.Thread(target=_consumer, args=(dev,), daemon=True)
        for dev in devices
    ]

    producer_thread.start()
    for t in consumer_threads:
        t.start()

    producer_thread.join()
    for t in consumer_threads:
        t.join()

    # -------------------------------------------------------------------
    # Collect results
    # -------------------------------------------------------------------
    final_results = [r for r in new_results if r is not None]

    # Merge new results into cache entry
    if force_rebenchmark:
        # Replace old entries with new results for the same cache_key
        old_by_key = {t["cache_key"]: t for t in entry.get("tested", [])}
        for r in final_results:
            old_by_key[r["cache_key"]] = r
        all_tested = list(old_by_key.values())
    else:
        all_tested = entry.get("tested", []) + final_results

    # Find overall best (excluding None/failed)
    valid = [t for t in all_tested if t.get("time_ms") is not None]
    if not valid:
        if entry.get("best") is not None:
            # Keep existing best
            best_entry = entry["best"]
        else:
            raise RuntimeError(f"All {kernel_type} configurations failed during benchmarking.")
    else:
        best_entry = min(valid, key=lambda t: t["time_ms"])

    # NOTE: Compile failures are now recorded incrementally in the producer
    # thread via _record_compile_failure(), so they persist even if
    # autotuning is interrupted before reaching this point.

    # Update cache
    if use_cache:
        cache_data = _load_cache()
        cache_data[key] = {
            "best": {"params": best_entry["params"], "time_ms": best_entry["time_ms"]},
            "tested": all_tested,
        }
        _save_cache(cache_data)

    best_config = params_cls.from_dict(best_entry["params"])

    if verbose:
        time_str = f"{best_entry['time_ms']:.3f} ms" if best_entry["time_ms"] is not None else "N/A"
        print(f"[cute_prism] Best {kernel_type} config: {_config_summary(best_config)} — {time_str}")

    # Clean up tensors
    torch.cuda.empty_cache()

    return best_config


def _config_summary(config) -> str:
    """One-line summary of a config for progress output."""
    if isinstance(config, CompParams):
        return (
            f"bM={config.bM} bN={config.bN} bK={config.bK} "
            f"c={config.c_width} P={config.bP_a_r}/{config.bP_ar}/{config.bP_b}"
        )
    elif isinstance(config, BwdDAdRCompParams):
        return (
            f"bM={config.bM} bK={config.bK} bN={config.bN} "
            f"bufs={config.n_buf_slots}"
        )
    elif isinstance(config, BwdDBCompParams):
        return (
            f"bM={config.bM} bK={config.bK} "
            f"P={config.bP_a}/{config.bP_ar}/{config.bP_dc}"
        )
    return str(config)


# ---------------------------------------------------------------------------
# Tensor factories for benchmarking
# ---------------------------------------------------------------------------

def _make_fwd_tensors(M, N, K, group_size, reconn_sz, device):
    n_groups = N // group_size
    return {
        "A": torch.randn(M, K, dtype=torch.float16, device=f"cuda:{device}"),
        "B": torch.randn(N, K, dtype=torch.float16, device=f"cuda:{device}"),
        "R": torch.randn(n_groups * reconn_sz, K, dtype=torch.float16, device=f"cuda:{device}"),
    }


def _make_bwd_dadr_tensors(M, N, K, group_size, reconn_sz, device):
    n_groups = N // group_size
    return {
        "dC": torch.randn(M, N, dtype=torch.float16, device=f"cuda:{device}"),
        "A": torch.randn(M, K, dtype=torch.float16, device=f"cuda:{device}"),
        "B": torch.randn(N, K, dtype=torch.float16, device=f"cuda:{device}"),
        "R": torch.randn(n_groups * reconn_sz, K, dtype=torch.float16, device=f"cuda:{device}"),
    }


def _make_bwd_db_tensors(M, N, K, group_size, reconn_sz, device):
    n_groups = N // group_size
    return {
        "dC": torch.randn(M, N, dtype=torch.float16, device=f"cuda:{device}"),
        "A": torch.randn(M, K, dtype=torch.float16, device=f"cuda:{device}"),
        "R": torch.randn(n_groups * reconn_sz, K, dtype=torch.float16, device=f"cuda:{device}"),
    }


# ---------------------------------------------------------------------------
# Public autotune functions
# ---------------------------------------------------------------------------

def autotune(
    M: int,
    N: int,
    K: int,
    group_size: int = 256,
    reconn_sz: int = 8,
    device: int | list[int] = 0,
    search_space: Iterable[CompParams | tuple[CompParams, Callable]] | None = None,
    warmup: int = 3,
    repeat: int = 10,
    use_cache: bool = True,
    verbose: bool = True,
    gated: bool = False,
    force_rebenchmark: bool = False,
) -> CompParams:
    """Find the best CompParams for the forward kernel.

    Args:
        device: GPU device index or list of device indices for multi-GPU
            benchmarking.  When multiple devices are given, benchmark work
            is distributed across them for faster autotuning.
        force_rebenchmark: When True, re-benchmark all configs even if cached
            results exist.  Useful when previous measurements seem unreliable.
    """
    return _autotune_generic(
        kernel_type="fwd",
        params_cls=CompParams,
        M=M, N=N, K=K,
        group_size=group_size, reconn_sz=reconn_sz,
        device=device,
        search_space=search_space,
        default_space_fn=default_search_space,
        warmup=warmup, repeat=repeat,
        use_cache=use_cache, verbose=verbose, gated=gated,
        make_tensors=_make_fwd_tensors,
        force_rebenchmark=force_rebenchmark,
    )


def autotune_bwd_dadr(
    M: int,
    N: int,
    K: int,
    group_size: int = 256,
    reconn_sz: int = 8,
    device: int | list[int] = 0,
    search_space: Iterable[BwdDAdRCompParams | tuple[BwdDAdRCompParams, Callable]] | None = None,
    warmup: int = 3,
    repeat: int = 10,
    use_cache: bool = True,
    verbose: bool = True,
    gated: bool = False,
    force_rebenchmark: bool = False,
) -> BwdDAdRCompParams:
    """Find the best BwdDAdRCompParams for the backward dA+dR kernel.

    Args:
        device: GPU device index or list of device indices for multi-GPU
            benchmarking.
        force_rebenchmark: When True, re-benchmark all configs even if cached
            results exist.
    """
    return _autotune_generic(
        kernel_type="bwd_dadr",
        params_cls=BwdDAdRCompParams,
        M=M, N=N, K=K,
        group_size=group_size, reconn_sz=reconn_sz,
        device=device,
        search_space=search_space,
        default_space_fn=default_search_space_bwd_dadr,
        warmup=warmup, repeat=repeat,
        use_cache=use_cache, verbose=verbose, gated=gated,
        make_tensors=_make_bwd_dadr_tensors,
        force_rebenchmark=force_rebenchmark,
    )


def autotune_bwd_db(
    M: int,
    N: int,
    K: int,
    group_size: int = 256,
    reconn_sz: int = 8,
    device: int | list[int] = 0,
    search_space: Iterable[BwdDBCompParams | tuple[BwdDBCompParams, Callable]] | None = None,
    warmup: int = 3,
    repeat: int = 10,
    use_cache: bool = True,
    verbose: bool = True,
    gated: bool = False,
    force_rebenchmark: bool = False,
) -> BwdDBCompParams:
    """Find the best BwdDBCompParams for the backward dB kernel.

    Args:
        device: GPU device index or list of device indices for multi-GPU
            benchmarking.
        force_rebenchmark: When True, re-benchmark all configs even if cached
            results exist.
    """
    return _autotune_generic(
        kernel_type="bwd_db",
        params_cls=BwdDBCompParams,
        M=M, N=N, K=K,
        group_size=group_size, reconn_sz=reconn_sz,
        device=device,
        search_space=search_space,
        default_space_fn=default_search_space_bwd_db,
        warmup=warmup, repeat=repeat,
        use_cache=use_cache, verbose=verbose, gated=gated,
        make_tensors=_make_bwd_db_tensors,
        force_rebenchmark=force_rebenchmark,
    )
