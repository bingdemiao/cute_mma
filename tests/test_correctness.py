"""Correctness tests for all OFT kernel variants.

Usage:
    uv run python tests/test_correctness.py                    # run all tests
    uv run python tests/test_correctness.py --backend cute     # test only cute backend
    uv run python tests/test_correctness.py --kernel fwd       # test only forward
    uv run python tests/test_correctness.py --kernel bwd_dadr  # test only backward dAdR
    uv run python tests/test_correctness.py --kernel bwd_db    # test only backward dB
    uv run python tests/test_correctness.py --quick            # small sizes only
    uv run python tests/test_correctness.py -v                 # verbose output
"""

import argparse
import sys
import torch

# Must set before importing cute_oft
import os
os.environ.setdefault("CUTE_OFT_COMPILE_WORKERS", "2")

import cute_oft


def check_close(name, val, ref, rtol=0.005):
    """Check relative error, return (pass, rel_error)."""
    if val is None and ref is None:
        return True, 0.0
    if val is None or ref is None:
        return False, float("inf")
    rel = ((val - ref).abs() / (ref.abs().max() + 1e-6)).max().item()
    return rel < rtol, rel


def test_forward(backend, gs, rs, sizes, n_seeds=5, verbose=False):
    passed, failed = 0, 0
    for act in [None, "silu_gate"]:
        for M, N, K in sizes:
            if N < gs:
                continue
            for seed in range(n_seeds):
                torch.manual_seed(seed * 100)
                A = torch.randn(M, K, dtype=torch.float16, device="cuda")
                B = torch.randn(N, K, dtype=torch.float16, device="cuda")
                R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")

                C = cute_oft.forward(A, B, R, gs, rs, backend=backend, activation=act)
                C_ref = cute_oft.forward(A, B, R, gs, rs, backend="pytorch", activation=act)

                ok, rel = check_close("C", C, C_ref)
                label = f"fwd {'gated' if act else 'plain':5s} M={M:4d} seed={seed}"
                if ok:
                    passed += 1
                    if verbose:
                        print(f"  PASS {label} rel={rel:.6f}")
                else:
                    failed += 1
                    print(f"  FAIL {label} rel={rel:.6f}")
    return passed, failed


def test_backward_dadr(backend, gs, rs, sizes, n_seeds=5, verbose=False):
    passed, failed = 0, 0
    for act in [None, "silu_gate"]:
        for M, N, K in sizes:
            if N < gs:
                continue
            for seed in range(n_seeds):
                torch.manual_seed(seed * 100)
                A = torch.randn(M, K, dtype=torch.float16, device="cuda")
                B = torch.randn(N, K, dtype=torch.float16, device="cuda")
                R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
                dC = torch.randn(M, N, dtype=torch.float16, device="cuda")

                dA, dR, _ = cute_oft.backward(dC, A, B, R, gs, rs, backend=backend, activation=act)
                dA_ref, dR_ref, _ = cute_oft.backward(dC, A, B, R, gs, rs, backend="pytorch", activation=act)

                ok_a, rel_a = check_close("dA", dA, dA_ref)
                ok_r, rel_r = check_close("dR", dR, dR_ref)
                ok = ok_a and ok_r
                label = f"dadr {'gated' if act else 'plain':5s} M={M:4d} seed={seed}"
                if ok:
                    passed += 1
                    if verbose:
                        print(f"  PASS {label} dA={rel_a:.6f} dR={rel_r:.6f}")
                else:
                    failed += 1
                    print(f"  FAIL {label} dA={rel_a:.6f} dR={rel_r:.6f}")
    return passed, failed


def test_backward_db(backend, gs, rs, sizes, n_seeds=5, verbose=False):
    passed, failed = 0, 0
    for M, N, K in sizes:
        if N < gs:
            continue
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100)
            A = torch.randn(M, K, dtype=torch.float16, device="cuda")
            B = torch.randn(N, K, dtype=torch.float16, device="cuda")
            R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
            dC = torch.randn(M, N, dtype=torch.float16, device="cuda")

            _, _, dB = cute_oft.backward(dC, A, B, R, gs, rs, backend=backend, activation="silu_gate")
            _, _, dB_ref = cute_oft.backward(dC, A, B, R, gs, rs, backend="pytorch", activation="silu_gate")

            ok, rel = check_close("dB", dB, dB_ref)
            label = f"db   gated M={M:4d} seed={seed}"
            if ok:
                passed += 1
                if verbose:
                    print(f"  PASS {label} rel={rel:.6f}")
            else:
                failed += 1
                print(f"  FAIL {label} rel={rel:.6f}")
    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="OFT kernel correctness tests")
    parser.add_argument("--backend", default="cute", choices=["cute", "cublas"])
    parser.add_argument("--kernel", default="all", choices=["all", "fwd", "bwd_dadr", "bwd_db"])
    parser.add_argument("--quick", action="store_true", help="small sizes only")
    parser.add_argument("--seeds", type=int, default=5, help="number of random seeds per config")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--gs", type=int, default=256, help="group size")
    parser.add_argument("--rs", type=int, default=8, help="reconn size")
    args = parser.parse_args()

    gs, rs = args.gs, args.rs

    if args.quick:
        sizes = [(256, 256, 256), (512, 512, 512)]
    else:
        sizes = [
            (256, 256, 256),
            (512, 512, 512),
            (512, 1024, 768),
            (1024, 1024, 1024),
            (2048, 1024, 1024),
        ]

    total_passed, total_failed = 0, 0

    if args.kernel in ("all", "fwd"):
        print(f"=== Forward ({args.backend}) ===")
        p, f = test_forward(args.backend, gs, rs, sizes, args.seeds, args.verbose)
        total_passed += p
        total_failed += f
        print(f"Forward: {p} passed, {f} failed\n")

    if args.kernel in ("all", "bwd_dadr"):
        print(f"=== Backward dAdR ({args.backend}) ===")
        p, f = test_backward_dadr(args.backend, gs, rs, sizes, args.seeds, args.verbose)
        total_passed += p
        total_failed += f
        print(f"Backward dAdR: {p} passed, {f} failed\n")

    if args.kernel in ("all", "bwd_db"):
        print(f"=== Backward dB ({args.backend}) ===")
        p, f = test_backward_db(args.backend, gs, rs, sizes, args.seeds, args.verbose)
        total_passed += p
        total_failed += f
        print(f"Backward dB: {p} passed, {f} failed\n")

    print(f"{'='*40}")
    print(f"Total: {total_passed} passed, {total_failed} failed")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
