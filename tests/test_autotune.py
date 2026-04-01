"""Tests for the autotuning framework.

Usage:
    python test_autotune.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import cute_prism
from cute_prism._config import CompParams
from cute_prism._autotune import default_search_space, autotune, _compile_one


def oft_reference(A, B, R, group_size, reconn_sz):
    """Reference implementation (AR mode)."""
    M, K = A.shape
    N = B.shape[0]
    n_groups = N // group_size
    n_blocks = K // reconn_sz

    C = torch.zeros(M, N, dtype=A.dtype, device=A.device)
    for g in range(n_groups):
        AR = torch.zeros_like(A)
        for b in range(n_blocks):
            R_block = R[g * reconn_sz:(g + 1) * reconn_sz, b * reconn_sz:(b + 1) * reconn_sz]
            AR[:, b * reconn_sz:(b + 1) * reconn_sz] = A[:, b * reconn_sz:(b + 1) * reconn_sz] @ R_block.T
        B_group = B[g * group_size:(g + 1) * group_size, :]
        C[:, g * group_size:(g + 1) * group_size] = AR @ B_group.T
    return C


def test_search_space_generation():
    """Verify search space is non-empty and configs satisfy constraints."""
    print("--- test_search_space_generation ---")

    for group_size, reconn_sz in [(256, 8), (128, 16)]:
        configs = default_search_space(group_size, reconn_sz)
        print(f"  group_size={group_size}, reconn_sz={reconn_sz}: {len(configs)} configs")
        assert len(configs) > 0, f"Empty search space for gs={group_size}, rs={reconn_sz}"

        for c in configs:
            # c_width must be a multiple of reconn_sz
            assert c.c_width % reconn_sz == 0, f"c_width={c.c_width} not multiple of reconn_sz={reconn_sz}"
            # bK must be divisible by c_width
            assert c.bK % c.c_width == 0, f"bK={c.bK} not divisible by c_width={c.c_width}"
            # bK must align with reconn_sz
            assert c.bK % reconn_sz == 0 or reconn_sz % c.bK == 0
            # Pipeline depths are fixed at 2
            assert c.bP_a_r == 2 and c.bP_ar == 2 and c.bP_b == 2

    print("  PASSED\n")


def test_comp_params_serialization():
    """Verify CompParams round-trips through dict serialization."""
    print("--- test_comp_params_serialization ---")

    params = CompParams(bM=256, bN=128, bK=32, c_width=16, warp_layout_ar=(4,), warp_layout_arb=(2, 2))
    d = params.to_dict()
    restored = CompParams.from_dict(d)
    assert params == restored, f"Round-trip failed: {params} != {restored}"

    # Simulate JSON round-trip (tuples become lists)
    import json
    j = json.loads(json.dumps(d))
    restored2 = CompParams.from_dict(j)
    assert params == restored2, f"JSON round-trip failed: {params} != {restored2}"

    print("  PASSED\n")


def test_parallel_compilation():
    """Verify compilation works for a small set of configs."""
    print("--- test_parallel_compilation ---")

    configs = default_search_space(256, 8)
    # Take a small subset for testing
    subset = configs[:3]
    print(f"  Compiling {len(subset)} configs...")
    results = [_compile_one(c, group_size=256, reconn_sz=8, kernel_type="fwd") for c in subset]
    n_ok = sum(results)
    print(f"  {n_ok}/{len(subset)} compiled successfully")
    assert n_ok > 0, "No configs compiled successfully"
    print("  PASSED\n")


def test_autotune_returns_valid_config():
    """Verify autotune returns a config that produces correct results."""
    print("--- test_autotune_returns_valid_config ---")

    M, N, K = 1024, 1024, 256
    group_size, reconn_sz = 256, 8

    # Run autotune with a small subset to keep test fast
    configs = default_search_space(group_size, reconn_sz)[:3]
    best = autotune(
        M, N, K, group_size, reconn_sz,
        search_space=configs,
        warmup=1, repeat=3,
        use_cache=False, verbose=True,
    )
    print(f"  Best config: bM={best.bM} bN={best.bN} bK={best.bK} c={best.c_width}")

    # Verify the autotuned config produces correct results
    torch.manual_seed(42)
    n_groups = N // group_size
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda") / (K ** 0.5)
    R = torch.randn(n_groups * reconn_sz, K, dtype=torch.float16, device="cuda")

    C_tuned = cute_prism.forward(A, B, R, group_size, reconn_sz, comp_params=best)
    C_ref = oft_reference(A, B, R, group_size, reconn_sz)

    rel_err = (C_tuned.float() - C_ref.float()).abs()
    safe = C_ref.float().abs() > 1e-4
    if safe.any():
        mean_err = (rel_err[safe] / C_ref.float().abs()[safe]).mean().item()
    else:
        mean_err = 0.0

    print(f"  Mean relative error vs pytorch reference: {mean_err:.6f}")
    assert mean_err < 0.01, f"Autotuned kernel too inaccurate: mean_rel_err={mean_err}"
    print("  PASSED\n")


def test_forward_auto_mode():
    """Verify forward(..., comp_params='auto') works end-to-end."""
    print("--- test_forward_auto_mode ---")

    M, N, K = 256, 256, 32
    group_size, reconn_sz = 256, 8

    torch.manual_seed(0)
    n_groups = N // group_size
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(n_groups * reconn_sz, K, dtype=torch.float16, device="cuda")

    C = cute_prism.forward(A, B, R, group_size, reconn_sz, comp_params="auto")
    C_ref = oft_reference(A, B, R, group_size, reconn_sz)

    rel_err = (C.float() - C_ref.float()).abs()
    safe = C_ref.float().abs() > 1e-4
    if safe.any():
        mean_err = (rel_err[safe] / C_ref.float().abs()[safe]).mean().item()
    else:
        mean_err = 0.0

    print(f"  Mean relative error: {mean_err:.6f}")
    assert mean_err < 0.01, f"Auto-tuned forward too inaccurate: {mean_err}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_search_space_generation()
    test_comp_params_serialization()
    test_parallel_compilation()
    test_autotune_returns_valid_config()
    test_forward_auto_mode()
    print("All autotune tests passed!")
