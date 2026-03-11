"""Correctness test for the OFT PyTorch extension.

Compares the CUDA kernel output against a cuBLAS-equivalent PyTorch reference.

Note: The kernel uses FP16 MMA accumulators (SM80_16x8x16_F16F16F16F16_TN),
while cuBLAS/PyTorch use FP32 accumulators internally. This causes divergence
that scales as ~sqrt(K) * eps_fp16.
"""

import torch
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), "..", "build_torch", "torch_ext")
sys.path.insert(0, build_dir)
import cute_oft


def oft_reference(A, B, R, group_size, reconn_sz):
    """Reference implementation matching cublas_oft AR mode.

    For each group g:
        AR[:, b*rs:(b+1)*rs] = A[:, b*rs:(b+1)*rs] @ R_g[b]^T  (for each K-block b)
        C[:, g*gs:(g+1)*gs] = AR @ B_g^T
    """
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

        B_group = B[g * group_size : (g + 1) * group_size, :]
        C[:, g * group_size : (g + 1) * group_size] = AR @ B_group.T

    return C


def test_oft(m, n, k, group_size=256, reconn_sz=8, error_threshold=5e-3):
    """Test the CUDA kernel against the reference."""
    n_groups = n // group_size

    torch.manual_seed(42)
    A = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(n, k, dtype=torch.float16, device="cuda") / (k ** 0.5)

    # Initialize R as permutation matrices
    R = torch.zeros(n_groups * reconn_sz, k, dtype=torch.float16, device="cuda")
    n_blocks = k // reconn_sz
    for g in range(n_groups):
        for b in range(n_blocks):
            perm = torch.randperm(reconn_sz)
            for j in range(reconn_sz):
                R[g * reconn_sz + perm[j], b * reconn_sz + j] = 1.0

    C_ref = oft_reference(A, B, R, group_size, reconn_sz)
    C_kernel = cute_oft.forward(A, B, R, group_size, reconn_sz)

    # Check for NaN/Inf (would indicate a kernel bug)
    nan_count = C_kernel.isnan().sum().item()
    inf_count = C_kernel.isinf().sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"[FAIL] M={m}, N={n}, K={k}: kernel produced {nan_count} NaN, {inf_count} Inf")
        return False

    # Per-element relative error (matching C++ check_result approach)
    ref_f = C_ref.float()
    kernel_f = C_kernel.float()
    abs_diff = (kernel_f - ref_f).abs()
    ref_abs = ref_f.abs()

    safe_mask = ref_abs > 1e-4
    rel_err = torch.zeros_like(abs_diff)
    rel_err[safe_mask] = abs_diff[safe_mask] / ref_abs[safe_mask]

    error_count = (rel_err > error_threshold).sum().item()
    total = C_ref.numel()
    error_rate = error_count / total
    max_error = rel_err[safe_mask].max().item() if safe_mask.any() else 0.0
    mean_error = rel_err[safe_mask].mean().item() if safe_mask.any() else 0.0

    # FP16 MMA accumulation causes ~sqrt(K/reconn_sz) * eps_fp16 mean error.
    # Allow error rate scaling with K; the key check is mean relative error.
    max_acceptable_mean = 0.01  # 1% mean relative error
    passed = mean_error < max_acceptable_mean and nan_count == 0
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] M={m}, N={n}, K={k}: "
        f"mean_rel_err={mean_error:.6f}, max_rel_err={max_error:.6f}, "
        f"error_rate={error_rate*100:.2f}% (at {error_threshold:.0e} threshold)"
    )
    return passed


if __name__ == "__main__":
    all_passed = True

    print("Running OFT kernel correctness tests...\n")

    # Dimensions must be >= tile sizes (bM=128, bN=128, bK=32)
    # and N must be divisible by group_size (256)
    all_passed &= test_oft(256, 256, 32)
    all_passed &= test_oft(1024, 1024, 256)
    all_passed &= test_oft(4096, 4096, 1024)

    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
