"""Autotuning for OFT kernel CompParams.

Searches over a space of tile sizes, pipeline depths, and warp layouts to find
the fastest CompParams for a given problem shape. Results are cached to disk
so that repeated calls with the same shape skip benchmarking.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import torch

from ._config import CompParams
from ._validate import compute_smem_bytes, check_smem_limit


def default_search_space(
    group_size: int,
    reconn_sz: int,
    device: int = 0,
) -> list[CompParams]:
    """Generate candidate CompParams configurations.

    Uses kernel architecture constraints to produce a focused search space:
    - Tile sizes (bM, bN) of 128 or 256 cover practical problem shapes.
    - bK of 16 or 32 balances K-loop iterations vs shared memory.
    - c_width must satisfy: c_width % reconn_sz == 0 AND bK % c_width == 0.
    - Pipeline depths are fixed at 2 (double-buffering); depth 3 rarely
      justifies the extra shared memory for the marginal latency hiding.
    - Warp layouts: (2,)/(2,2) is the general-purpose choice (192 threads).
      (4,)/(4,4) is only viable with large tiles (640 threads, more parallelism).
    - Configs exceeding the device shared memory limit are pruned.
    """
    # Warp layout presets: (warp_layout1, warp_layout2, min_bM, min_bN)
    # The min values come from: bM >= wl2_rows * 8, bN >= wl2_cols * 8,
    # and the warp N-tile (bN / wl2_cols) must be <= group_size.
    warp_presets: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = [
        ((2,), (2, 2), 16, 16),    # 6 warps, 192 threads — general purpose
        ((4,), (2, 2), 16, 16),    # 8 warps, 256 threads — more AR producers
        ((2,), (4, 4), 32, 32),    # 18 warps, 576 threads — more ARB consumers
    ]

    candidates: list[CompParams] = []
    for bM in (128, 256):
        for bN in (128, 256):
            # Kernel requires: bN % group_size == 0 or group_size % bN == 0
            if bN % group_size != 0 and group_size % bN != 0:
                continue

            for bK in (16, 32):
                # bK must align with reconn_sz
                if bK % reconn_sz != 0 and reconn_sz % bK != 0:
                    continue

                # c_width must be a multiple of reconn_sz AND must divide bK
                for c_width in range(reconn_sz, bK + 1, reconn_sz):
                    if bK % c_width != 0:
                        continue

                    for wl1, wl2, min_bM, min_bN in warp_presets:
                        if bM < min_bM or bN < min_bN:
                            continue
                        # Warp N-tile must fit within a single group
                        warp_tile_n = bN // wl2[1] if len(wl2) > 1 else bN
                        if warp_tile_n > group_size:
                            continue

                        params = CompParams(
                            bM=bM, bN=bN, bK=bK,
                            c_width=c_width,
                            bP_a_r=2, bP_ar=2, bP_b=2,
                            warp_layout1=wl1, warp_layout2=wl2,
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


def _problem_key(
    M: int, N: int, K: int,
    group_size: int, reconn_sz: int,
    device_name: str,
    gated: bool = False,
) -> str:
    """Cache key for a specific problem shape + device."""
    suffix = "_gated" if gated else ""
    return f"{device_name}_M{M}_N{N}_K{K}_g{group_size}_r{reconn_sz}{suffix}"


def _cache_path() -> Path:
    import os
    root = Path(os.environ.get("CUTE_OFT_CACHE_DIR", Path.home() / ".cache" / "cute_oft"))
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


def _benchmark_config(
    config: CompParams,
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int,
    reconn_sz: int,
    warmup: int,
    repeat: int,
    gated: bool = False,
) -> float | None:
    """Benchmark a single CompParams config. Returns median time in ms, or None on failure."""
    from ._loader import get_or_compile

    try:
        module = get_or_compile(group_size, reconn_sz, "cute", config, gated=gated)
    except Exception:
        return None

    # Warmup
    for _ in range(warmup):
        try:
            module.forward(A, B, R, group_size, reconn_sz)
        except Exception:
            return None

    torch.cuda.synchronize()

    # Timed runs using CUDA events
    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module.forward(A, B, R, group_size, reconn_sz)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def _compile_one(
    config: CompParams,
    group_size: int,
    reconn_sz: int,
    gated: bool = False,
) -> bool:
    """Compile a single config. Returns True on success, False on failure."""
    from ._compiler import compile_kernel
    try:
        compile_kernel(group_size, reconn_sz, "cute", config, gated=gated)
        return True
    except Exception:
        return False


def _parallel_compile(
    configs: Sequence[CompParams],
    group_size: int,
    reconn_sz: int,
    verbose: bool = True,
    gated: bool = False,
) -> list[bool]:
    """Compile all configs in parallel using threads.

    Returns a list of booleans indicating which configs compiled successfully.
    """
    import os
    # Cap workers to avoid OOM — each nvcc compilation is memory-heavy (~1-2 GB).
    max_parallel = int(os.environ.get("CUTE_OFT_COMPILE_WORKERS", 4))
    n_workers = min(len(configs), max_parallel)
    results: list[bool] = [False] * len(configs)

    if verbose:
        print(f"[cute_oft] Compiling {len(configs)} kernel variants ({n_workers} parallel workers)...")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_compile_one, config, group_size, reconn_sz, gated): i
            for i, config in enumerate(configs)
        }
        done_count = 0
        fail_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            success = future.result()
            results[idx] = success
            done_count += 1
            if not success:
                fail_count += 1
            if verbose and done_count % max(1, len(configs) // 5) == 0:
                print(f"[cute_oft]   Compiled {done_count}/{len(configs)} "
                      f"({fail_count} failed)")

    if verbose:
        ok = sum(results)
        print(f"[cute_oft] Compilation done: {ok} succeeded, {len(configs) - ok} failed")

    return results


def autotune(
    M: int,
    N: int,
    K: int,
    group_size: int = 256,
    reconn_sz: int = 8,
    device: int = 0,
    search_space: Sequence[CompParams] | None = None,
    warmup: int = 3,
    repeat: int = 10,
    use_cache: bool = True,
    verbose: bool = True,
    gated: bool = False,
) -> CompParams:
    """Find the best CompParams for the given problem dimensions.

    Args:
        M: Number of rows in A (and output).
        N: Number of rows in B (output columns).
        K: Shared dimension.
        group_size: OFT group size.
        reconn_sz: OFT reconnection block size.
        device: CUDA device index.
        search_space: List of CompParams to try. Uses default_search_space() if None.
        warmup: Warmup iterations per config.
        repeat: Timed iterations per config.
        use_cache: Whether to load/save results from disk cache.
        verbose: Print progress information.

    Returns:
        The CompParams with the lowest median execution time.
    """
    device_name = torch.cuda.get_device_properties(device).name

    # Check cache
    if use_cache:
        cache = _load_cache()
        key = _problem_key(M, N, K, group_size, reconn_sz, device_name, gated)
        if key in cache:
            cached = cache[key]
            params = CompParams.from_dict(cached["params"])
            if verbose:
                print(
                    f"[cute_oft] Using cached autotune result for "
                    f"M={M}, N={N}, K={K}: {cached['time_ms']:.3f} ms"
                )
            return params

    if search_space is None:
        search_space = default_search_space(group_size, reconn_sz, device)

    if not search_space:
        raise ValueError("No valid configurations in the search space for the given parameters.")

    if verbose:
        print(f"[cute_oft] Autotuning {len(search_space)} configurations for M={M}, N={N}, K={K}...")

    # Phase 1: Compile all candidate kernels in parallel.
    # Each config gets its own build directory with file locking, so this is safe.
    # subprocess.run releases the GIL, so ThreadPoolExecutor gives true parallelism.
    compiled = _parallel_compile(search_space, group_size, reconn_sz, verbose, gated=gated)

    # Phase 2: Benchmark compiled configs sequentially on the GPU.
    n_groups = N // group_size
    A = torch.randn(M, K, dtype=torch.float16, device=f"cuda:{device}")
    B = torch.randn(N, K, dtype=torch.float16, device=f"cuda:{device}")
    R = torch.randn(n_groups * reconn_sz, K, dtype=torch.float16, device=f"cuda:{device}")

    best_time = float("inf")
    best_config = search_space[0]
    bench_idx = 0

    for i, config in enumerate(search_space):
        if not compiled[i]:
            continue
        bench_idx += 1
        time_ms = _benchmark_config(
            config, A, B, R, group_size, reconn_sz, warmup, repeat,
            gated=gated,
        )
        if time_ms is None:
            if verbose:
                print(f"  [{bench_idx}/{sum(compiled)}] {_config_summary(config)} — FAILED")
            continue

        if verbose:
            marker = " *" if time_ms < best_time else ""
            print(f"  [{bench_idx}/{sum(compiled)}] {_config_summary(config)} — {time_ms:.3f} ms{marker}")

        if time_ms < best_time:
            best_time = time_ms
            best_config = config

    if best_time == float("inf"):
        raise RuntimeError("All configurations failed during benchmarking.")

    if verbose:
        print(f"[cute_oft] Best config: {_config_summary(best_config)} — {best_time:.3f} ms")

    # Save to cache
    if use_cache:
        cache = _load_cache()
        key = _problem_key(M, N, K, group_size, reconn_sz, device_name, gated)
        cache[key] = {
            "params": best_config.to_dict(),
            "time_ms": best_time,
        }
        _save_cache(cache)

    # Clean up benchmark tensors
    del A, B, R
    torch.cuda.empty_cache()

    return best_config


def _config_summary(config: CompParams) -> str:
    """One-line summary of a CompParams for progress output."""
    return (
        f"bM={config.bM} bN={config.bN} bK={config.bK} "
        f"c={config.c_width} P={config.bP_a_r}/{config.bP_ar}/{config.bP_b}"
    )
