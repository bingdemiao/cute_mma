"""Correctness test for the Prism backward pass.

Tests backward gradients against PyTorch autograd reference.

Usage:
    python test_backward.py
    python test_backward.py --activation silu_gate
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn.functional as F
import cute_prism


def autograd_reference(A, B, R, group_size, reconn_sz, activation=None):
    """Compute forward + backward using PyTorch autograd as ground truth.

    Uses float32 for numerical stability in the reference.
    """
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


def test_backward(m, n, k, group_size=256, reconn_sz=8, backend="pytorch",
                  activation=None, error_threshold=0.02):
    """Test backward pass against autograd reference."""
    n_groups = n // group_size

    torch.manual_seed(42)
    A = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.1
    B = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.1
    R = torch.zeros(n_groups * reconn_sz, k, dtype=torch.float16, device="cuda")
    n_blocks = k // reconn_sz
    for g in range(n_groups):
        for b in range(n_blocks):
            perm = torch.randperm(reconn_sz)
            for j in range(reconn_sz):
                R[g * reconn_sz + perm[j], b * reconn_sz + j] = 1.0

    # Autograd reference (float32)
    C_ref, A_f, B_f, R_f = autograd_reference(A, B, R, group_size, reconn_sz, activation)
    dC = torch.randn(m, n, dtype=torch.float32, device="cuda") * 0.1
    C_ref.backward(dC)
    dA_ref = A_f.grad
    dB_ref = B_f.grad
    dR_ref = R_f.grad

    # Our backward (float16 inputs)
    dC_h = dC.half()
    dA, dR, dB = cute_prism.backward(
        dC_h, A, B, R, group_size, reconn_sz,
        backend=backend, activation=activation,
    )

    passed = True
    act_str = f" act={activation}" if activation else ""

    grad_pairs = [("dA", dA, dA_ref), ("dR", dR, dR_ref)]
    if dB is not None:
        grad_pairs.append(("dB", dB, dB_ref))
    else:
        print(f"  [SKIP] dB {backend:8s} M={m}, N={n}, K={k}, "
              f"gs={group_size}, rs={reconn_sz}{act_str}: "
              f"dB=None (non-gated mode, B frozen)")

    for name, grad, ref in grad_pairs:
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

        ok = mean_err < error_threshold
        passed &= ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {name:2s} {backend:8s} M={m}, N={n}, K={k}, "
            f"gs={group_size}, rs={reconn_sz}{act_str}: "
            f"mean_rel_err={mean_err:.6f}, max_rel_err={max_err:.6f}"
        )

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prism backward correctness tests")
    parser.add_argument("--backend", default="pytorch",
                        help="Backend to test (default: pytorch)")
    parser.add_argument("--activation", choices=["silu_gate"],
                        help="Test only this activation")
    args = parser.parse_args()

    all_passed = True
    activations = [args.activation] if args.activation else [None, "silu_gate"]

    for activation in activations:
        act_label = activation or "none"
        print(f"\n--- Backend: {args.backend}, Activation: {act_label} ---\n")

        all_passed &= test_backward(256, 256, 32, backend=args.backend, activation=activation)
        all_passed &= test_backward(512, 512, 128, backend=args.backend, activation=activation)
        all_passed &= test_backward(1024, 1024, 256, backend=args.backend, activation=activation)
        all_passed &= test_backward(512, 512, 128, group_size=128, reconn_sz=16,
                                    backend=args.backend, activation=activation)

    print()
    if all_passed:
        print("All backward tests passed!")
    else:
        print("Some backward tests FAILED!")
        sys.exit(1)
