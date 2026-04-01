"""Correctness test for the OFT PyTorch extension.

Tests all backends (cute, cublas, pytorch) and modes (ar, rw) against a PyTorch reference.

Usage:
    python test_oft.py                     # Run all backends and modes
    python test_oft.py --backend cute      # Only test cute backend
    python test_oft.py --backend cublas --mode rw  # Only cublas RW mode
"""

import argparse
import torch
import sys
import os

# Add the Python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cute_prism


def oft_reference(A, B, R, group_size, reconn_sz, activation=None):
    """Reference implementation (AR mode)."""
    M, K = A.shape
    N = B.shape[0]
    n_groups = N // group_size
    n_blocks = K // reconn_sz

    C = torch.zeros(M, N, dtype=A.dtype, device=A.device)

    for g in range(n_groups):
        AR = torch.zeros_like(A)
        for b in range(n_blocks):
            R_block = R[
                g * reconn_sz : (g + 1) * reconn_sz,
                b * reconn_sz : (b + 1) * reconn_sz,
            ]
            AR[:, b * reconn_sz : (b + 1) * reconn_sz] = (
                A[:, b * reconn_sz : (b + 1) * reconn_sz] @ R_block.T
            )

        if activation == "silu_gate":
            AR = A * torch.nn.functional.silu(AR)
        B_group = B[g * group_size : (g + 1) * group_size, :]
        C[:, g * group_size : (g + 1) * group_size] = AR @ B_group.T

    return C


def make_test_tensors(m, n, k, group_size, reconn_sz):
    """Create test tensors with R as permutation matrices."""
    n_groups = n // group_size

    torch.manual_seed(42)
    A = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(n, k, dtype=torch.float16, device="cuda") / (k ** 0.5)

    R = torch.zeros(n_groups * reconn_sz, k, dtype=torch.float16, device="cuda")
    n_blocks = k // reconn_sz
    for g in range(n_groups):
        for b in range(n_blocks):
            perm = torch.randperm(reconn_sz)
            for j in range(reconn_sz):
                R[g * reconn_sz + perm[j], b * reconn_sz + j] = 1.0

    return A, B, R


def test_oft(m, n, k, group_size=256, reconn_sz=8, backend="cute", mode="ar", error_threshold=5e-3, activation=None):
    """Test a backend against the reference."""
    A, B, R = make_test_tensors(m, n, k, group_size, reconn_sz)

    C_ref = oft_reference(A, B, R, group_size, reconn_sz, activation=activation)
    C_kernel = cute_prism.forward(A, B, R, group_size, reconn_sz, backend=backend, mode=mode, activation=activation)

    # Check for NaN/Inf
    nan_count = C_kernel.isnan().sum().item()
    inf_count = C_kernel.isinf().sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"[FAIL] {backend}/{mode} M={m}, N={n}, K={k}: kernel produced {nan_count} NaN, {inf_count} Inf")
        return False

    # Per-element relative error
    ref_f = C_ref.float()
    kernel_f = C_kernel.float()
    abs_diff = (kernel_f - ref_f).abs()
    ref_abs = ref_f.abs()

    safe_mask = ref_abs > 1e-4
    rel_err = torch.zeros_like(abs_diff)
    rel_err[safe_mask] = abs_diff[safe_mask] / ref_abs[safe_mask]

    mean_error = rel_err[safe_mask].mean().item() if safe_mask.any() else 0.0
    max_error = rel_err[safe_mask].max().item() if safe_mask.any() else 0.0
    error_count = (rel_err > error_threshold).sum().item()
    error_rate = error_count / C_ref.numel()

    max_acceptable_mean = 0.01
    passed = mean_error < max_acceptable_mean and nan_count == 0
    status = "PASS" if passed else "FAIL"
    act_str = f" act={activation}" if activation else ""
    print(
        f"[{status}] {backend:8s}/{mode:2s} M={m}, N={n}, K={k}, gs={group_size}, rs={reconn_sz}{act_str}: "
        f"mean_rel_err={mean_error:.6f}, max_rel_err={max_error:.6f}, "
        f"error_rate={error_rate*100:.2f}% (at {error_threshold:.0e} threshold)"
    )
    return passed


# Backend -> supported modes
ALL_CONFIGS = {
    "pytorch": ["ar", "rw"],
    "cublas":  ["ar", "rw"],
    "cute":    ["ar"],
}

# Backends that support gated activation (AR mode only)
GATED_BACKENDS = ["pytorch", "cublas", "cute"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OFT kernel correctness tests")
    parser.add_argument("--backend", choices=list(ALL_CONFIGS.keys()),
                        help="Test only this backend (default: all)")
    parser.add_argument("--mode", choices=["ar", "rw"],
                        help="Test only this mode (default: all supported modes)")
    parser.add_argument("--activation", choices=["silu_gate"],
                        help="Test only this activation (default: test both None and silu_gate)")
    args = parser.parse_args()

    # Build list of (backend, mode) pairs to test
    if args.backend:
        backends = {args.backend: ALL_CONFIGS[args.backend]}
    else:
        backends = ALL_CONFIGS

    all_passed = True

    for backend, supported_modes in backends.items():
        modes = [args.mode] if args.mode and args.mode in supported_modes else supported_modes
        if args.mode and args.mode not in supported_modes:
            print(f"\nSkipping {backend}: mode '{args.mode}' not supported")
            continue

        for mode in modes:
            print(f"\n--- Backend: {backend}, Mode: {mode} ---\n")

            all_passed &= test_oft(256, 256, 32, backend=backend, mode=mode)
            all_passed &= test_oft(1024, 1024, 256, backend=backend, mode=mode)
            all_passed &= test_oft(4096, 4096, 1024, backend=backend, mode=mode)

            # Test with different hyperparameters
            all_passed &= test_oft(1024, 1024, 256, group_size=128, reconn_sz=16, backend=backend, mode=mode)

    # Test gated activation (silu_gate) — AR mode only
    gated_backends = [args.backend] if args.backend else GATED_BACKENDS
    if args.mode != "rw" and args.activation != "silu_gate" or args.activation == "silu_gate":
        for backend in gated_backends:
            if args.activation == "silu_gate" or args.activation is None:
                print(f"\n--- Backend: {backend}, Mode: ar, Activation: silu_gate ---\n")

                all_passed &= test_oft(256, 256, 32, backend=backend, activation="silu_gate")
                all_passed &= test_oft(1024, 1024, 256, backend=backend, activation="silu_gate")
                all_passed &= test_oft(4096, 4096, 1024, backend=backend, activation="silu_gate")

                # Test with different hyperparameters
                all_passed &= test_oft(1024, 1024, 256, group_size=128, reconn_sz=16,
                                       backend=backend, activation="silu_gate")

    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
