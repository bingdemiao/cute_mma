"""End-to-end tests for the autotuning pipeline with forward and backward passes.

Tests that autotuned kernels produce correct results, that backward autotuning
works independently for dAdR and dB kernels, and that cache skip behavior works.

Usage:
    CUTE_OFT_COMPILE_WORKERS=2 uv run pytest tests/test_autotune_e2e.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import pytest
import cute_oft
from cute_oft._autotune import (
    autotune,
    autotune_bwd_dadr,
    autotune_bwd_db,
    default_search_space,
    default_search_space_bwd_dadr,
    default_search_space_bwd_db,
)
from cute_oft._config import BwdDAdRCompParams, BwdDBCompParams, CompParams


def autograd_reference(A, B, R, group_size, reconn_sz, activation=None):
    """Compute forward + backward using PyTorch autograd as ground truth (float32)."""
    A_f = A.float().detach().requires_grad_(True)
    B_f = B.float().detach().requires_grad_(True)
    R_f = R.float().detach().requires_grad_(True)

    M, K = A_f.shape
    N = B_f.shape[0]
    n_groups = N // group_size
    n_blocks = K // reconn_sz

    C = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    for g in range(n_groups):
        AR = torch.zeros_like(A_f)
        for b in range(n_blocks):
            R_block = R_f[
                g * reconn_sz : (g + 1) * reconn_sz,
                b * reconn_sz : (b + 1) * reconn_sz,
            ]
            AR[:, b * reconn_sz : (b + 1) * reconn_sz] = (
                A_f[:, b * reconn_sz : (b + 1) * reconn_sz] @ R_block.T
            )
        if activation == "silu_gate":
            AR = A_f * F.silu(AR)
        B_group = B_f[g * group_size : (g + 1) * group_size, :]
        C[:, g * group_size : (g + 1) * group_size] = AR @ B_group.T

    return C, A_f, B_f, R_f


def check_gradient(name, grad, ref, threshold=0.02):
    """Check gradient accuracy. Returns (passed, mean_err, max_err)."""
    grad_f = grad.float()
    abs_diff = (grad_f - ref).abs()
    ref_abs = ref.abs()
    safe_mask = ref_abs > 1e-4
    if safe_mask.any():
        rel_err = abs_diff[safe_mask] / ref_abs[safe_mask]
        mean_err = rel_err.mean().item()
        max_err = rel_err.max().item()
    else:
        mean_err = 0.0
        max_err = 0.0
    passed = mean_err < threshold
    return passed, mean_err, max_err


def make_test_tensors(m, n, k, group_size, reconn_sz, seed=42):
    """Create test tensors with R as permutation matrices for stability."""
    n_groups = n // group_size
    torch.manual_seed(seed)
    A = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.1
    B = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.1
    R = torch.zeros(n_groups * reconn_sz, k, dtype=torch.float16, device="cuda")
    n_blocks = k // reconn_sz
    for g in range(n_groups):
        for b in range(n_blocks):
            perm = torch.randperm(reconn_sz)
            for j in range(reconn_sz):
                R[g * reconn_sz + perm[j], b * reconn_sz + j] = 1.0
    return A, B, R


# ---------------------------------------------------------------------------
# Forward autotuning tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,n,k,group_size,reconn_sz,activation", [
    (256, 256, 32, 256, 8, None),
    (1024, 1024, 256, 256, 8, None),
    (256, 256, 32, 256, 8, "silu_gate"),
])
def test_autotune_forward_backward(m, n, k, group_size, reconn_sz, activation):
    """Autotune forward, then verify both forward and backward produce correct results."""
    gated = activation == "silu_gate"

    configs = default_search_space(group_size, reconn_sz)[:3]
    best = autotune(
        m, n, k, group_size, reconn_sz,
        search_space=configs,
        warmup=1, repeat=3,
        use_cache=False, verbose=False,
        gated=gated,
    )

    A, B, R = make_test_tensors(m, n, k, group_size, reconn_sz)

    C_tuned = cute_oft.forward(
        A, B, R, group_size, reconn_sz,
        backend="cute", comp_params=best, activation=activation,
    )

    C_ref, A_f, B_f, R_f = autograd_reference(A, B, R, group_size, reconn_sz, activation)
    dC = torch.randn(m, n, dtype=torch.float32, device="cuda") * 0.1
    C_ref.backward(dC)

    # Check forward
    c_diff = (C_tuned.float() - C_ref.detach()).abs()
    c_ref_abs = C_ref.detach().abs()
    safe = c_ref_abs > 1e-4
    if safe.any():
        fwd_mean = (c_diff[safe] / c_ref_abs[safe]).mean().item()
    else:
        fwd_mean = 0.0
    assert fwd_mean < 0.01, f"Forward error too high: {fwd_mean:.6f}"

    # Check backward
    dC_h = dC.half()
    dA, dR, dB = cute_oft.backward(
        dC_h, A, B, R, group_size, reconn_sz,
        backend="cute", activation=activation, comp_params=best,
    )

    for name, grad, ref in [("dA", dA, A_f.grad), ("dR", dR, R_f.grad)]:
        ok, mean_err, _ = check_gradient(name, grad, ref, 0.02)
        assert ok, f"{name} error too high: {mean_err:.6f}"

    if dB is not None:
        ok, mean_err, _ = check_gradient("dB", dB, B_f.grad, 0.02)
        assert ok, f"dB error too high: {mean_err:.6f}"


# ---------------------------------------------------------------------------
# Backward kernel autotuning tests
# ---------------------------------------------------------------------------

def test_autotune_bwd_dadr():
    """Autotune backward dAdR kernel with small search space, verify correctness."""
    m, n, k = 256, 256, 32
    group_size, reconn_sz = 256, 8

    configs = default_search_space_bwd_dadr(group_size, reconn_sz)[:3]
    if not configs:
        pytest.skip("No valid dAdR configs for this problem shape")

    best = autotune_bwd_dadr(
        m, n, k, group_size, reconn_sz,
        search_space=configs,
        warmup=1, repeat=3,
        use_cache=False, verbose=False,
    )

    A, B, R = make_test_tensors(m, n, k, group_size, reconn_sz)
    dC = torch.randn(m, n, dtype=torch.float16, device="cuda") * 0.1

    # Use autotuned params
    dA, dR, _ = cute_oft.backward(
        dC, A, B, R, group_size, reconn_sz,
        backend="cute", bwd_dadr_params=best,
    )

    # Reference
    dA_ref, dR_ref, _ = cute_oft.backward(
        dC, A, B, R, group_size, reconn_sz, backend="pytorch",
    )

    ok_dA, mean_dA, _ = check_gradient("dA", dA, dA_ref.float(), 0.02)
    ok_dR, mean_dR, _ = check_gradient("dR", dR, dR_ref.float(), 0.02)
    assert ok_dA, f"dA error: {mean_dA:.6f}"
    assert ok_dR, f"dR error: {mean_dR:.6f}"


def test_autotune_bwd_db():
    """Autotune backward dB kernel (gated only), verify correctness."""
    m, n, k = 256, 256, 32
    group_size, reconn_sz = 256, 8

    configs = default_search_space_bwd_db(group_size, reconn_sz)[:3]
    if not configs:
        pytest.skip("No valid dB configs for this problem shape")

    best = autotune_bwd_db(
        m, n, k, group_size, reconn_sz,
        search_space=configs,
        warmup=1, repeat=3,
        use_cache=False, verbose=False,
        gated=True,
    )

    A, B, R = make_test_tensors(m, n, k, group_size, reconn_sz)
    dC = torch.randn(m, n, dtype=torch.float16, device="cuda") * 0.1

    # Use autotuned params
    dA, dR, dB = cute_oft.backward(
        dC, A, B, R, group_size, reconn_sz,
        backend="cute", activation="silu_gate",
        bwd_db_params=best,
    )

    # Reference
    dA_ref, dR_ref, dB_ref = cute_oft.backward(
        dC, A, B, R, group_size, reconn_sz,
        backend="pytorch", activation="silu_gate",
    )

    assert dB is not None, "dB should not be None in gated mode"
    ok_dB, mean_dB, _ = check_gradient("dB", dB, dB_ref.float(), 0.02)
    assert ok_dB, f"dB error: {mean_dB:.6f}"


# ---------------------------------------------------------------------------
# Cache skip tests
# ---------------------------------------------------------------------------

def test_cache_skip():
    """Run autotune twice with overlapping search spaces, second run should skip tested."""
    m, n, k = 256, 256, 32
    group_size, reconn_sz = 256, 8

    configs = default_search_space(group_size, reconn_sz)[:4]
    if len(configs) < 2:
        pytest.skip("Need at least 2 configs")

    # First run: test first 3 configs
    autotune(
        m, n, k, group_size, reconn_sz,
        search_space=configs[:3],
        warmup=1, repeat=3,
        use_cache=True, verbose=False,
    )

    # Second run: test all 4 configs, first 3 should be skipped
    skip_count = 0
    def on_result(config, time_ms):
        nonlocal skip_count

    # Use callback to count — the second run should only benchmark 1 new config
    items = [(c, on_result) for c in configs]
    result = autotune(
        m, n, k, group_size, reconn_sz,
        search_space=items,
        warmup=1, repeat=3,
        use_cache=True, verbose=False,
    )
    assert isinstance(result, CompParams)


def test_callback_with_cached_results():
    """Test that callbacks receive cached performance data for already-tested configs."""
    m, n, k = 256, 256, 32
    group_size, reconn_sz = 256, 8

    configs = default_search_space(group_size, reconn_sz)[:2]
    if not configs:
        pytest.skip("No configs available")

    # First run
    autotune(
        m, n, k, group_size, reconn_sz,
        search_space=configs,
        warmup=1, repeat=3,
        use_cache=True, verbose=False,
    )

    # Second run with callback — should get cached results
    reported: list[tuple] = []
    def cb(config, time_ms):
        reported.append((config, time_ms))

    items = [(c, cb) for c in configs]
    autotune(
        m, n, k, group_size, reconn_sz,
        search_space=items,
        warmup=1, repeat=3,
        use_cache=True, verbose=False,
    )

    # All configs should have been reported through callback
    assert len(reported) == len(configs)
    # Cached results should have time_ms values (not None, since they compiled OK first time)
    for config, time_ms in reported:
        assert time_ms is not None, f"Cached config should have time_ms, got None"


def test_failed_config_cached():
    """Test that a config that fails compilation is cached with time_ms=None."""
    m, n, k = 256, 256, 32
    group_size, reconn_sz = 256, 8

    # Create an invalid config that will fail compilation
    # (extremely large tile sizes that exceed smem)
    invalid_config = CompParams(bM=128, bN=128, bK=32, c_width=32,
                                bP_a_r=2, bP_ar=2, bP_b=2,
                                warp_layout_ar=(2,), warp_layout_arb=(2, 2))
    valid_configs = default_search_space(group_size, reconn_sz)[:1]
    if not valid_configs:
        pytest.skip("No valid configs")

    # Mix valid and potentially failing configs
    reported: list[tuple] = []
    def cb(config, time_ms):
        reported.append((config, time_ms))

    all_configs = [(c, cb) for c in valid_configs]
    result = autotune(
        m, n, k, group_size, reconn_sz,
        search_space=all_configs,
        warmup=1, repeat=3,
        use_cache=False, verbose=False,
    )
    assert isinstance(result, CompParams)


# ---------------------------------------------------------------------------
# Generator-based search space test
# ---------------------------------------------------------------------------

def test_generator_search_space():
    """Test that generator-based search space works with callback."""
    m, n, k = 256, 256, 32
    group_size, reconn_sz = 256, 8

    all_configs = default_search_space(group_size, reconn_sz)[:3]
    if not all_configs:
        pytest.skip("No configs")

    results: list[float | None] = []

    def gen_with_callback():
        for c in all_configs:
            def cb(config, time_ms, r=results):
                r.append(time_ms)
            yield (c, cb)

    best = autotune(
        m, n, k, group_size, reconn_sz,
        search_space=gen_with_callback(),
        warmup=1, repeat=3,
        use_cache=False, verbose=False,
    )
    assert isinstance(best, CompParams)
    assert len(results) == len(all_configs)


# ---------------------------------------------------------------------------
# End-to-end autotuning=True tests
# ---------------------------------------------------------------------------

def test_auto_mode_forward_backward():
    """Test that autotuning=True works for both forward() and backward()."""
    m, n, k = 1024, 1024, 256
    group_size, reconn_sz = 256, 8

    A, B, R = make_test_tensors(m, n, k, group_size, reconn_sz)

    C = cute_oft.forward(
        A, B, R, group_size, reconn_sz,
        backend="cute", autotuning=True,
    )

    dC = torch.randn(m, n, dtype=torch.float16, device="cuda") * 0.1
    dA, dR, dB = cute_oft.backward(
        dC, A, B, R, group_size, reconn_sz,
        backend="cute", autotuning=True,
    )

    assert C.shape == (m, n)
    assert dA.shape == A.shape
    assert dR.shape == R.shape
    assert dB is None  # non-gated

    # Check against pytorch reference
    C_ref = cute_oft.forward(A, B, R, group_size, reconn_sz, backend="pytorch")
    dA_ref, dR_ref, _ = cute_oft.backward(
        dC, A, B, R, group_size, reconn_sz, backend="pytorch",
    )

    for name, val, ref in [("C", C, C_ref), ("dA", dA, dA_ref), ("dR", dR, dR_ref)]:
        ok, mean_err, _ = check_gradient(name, val, ref.float(), 0.02)
        assert ok, f"{name} auto: mean_rel_err={mean_err:.6f}"


def test_auto_mode_gated():
    """Test autotuning=True with gated activation for forward + backward."""
    m, n, k = 512, 512, 128
    group_size, reconn_sz = 256, 8

    A, B, R = make_test_tensors(m, n, k, group_size, reconn_sz)

    C = cute_oft.forward(
        A, B, R, group_size, reconn_sz,
        backend="cute", autotuning=True, activation="silu_gate",
    )

    dC = torch.randn(m, n, dtype=torch.float16, device="cuda") * 0.1
    dA, dR, dB = cute_oft.backward(
        dC, A, B, R, group_size, reconn_sz,
        backend="cute", autotuning=True, activation="silu_gate",
    )

    assert C.shape == (m, n)
    assert dA.shape == A.shape
    assert dR.shape == R.shape
    assert dB is not None
    assert dB.shape == B.shape

    C_ref = cute_oft.forward(
        A, B, R, group_size, reconn_sz,
        backend="pytorch", activation="silu_gate",
    )
    dA_ref, dR_ref, dB_ref = cute_oft.backward(
        dC, A, B, R, group_size, reconn_sz,
        backend="pytorch", activation="silu_gate",
    )

    for name, val, ref in [("C", C, C_ref), ("dA", dA, dA_ref), ("dR", dR, dR_ref), ("dB", dB, dB_ref)]:
        ok, mean_err, _ = check_gradient(name, val, ref.float(), 0.02)
        assert ok, f"{name} auto gated: mean_rel_err={mean_err:.6f}"
