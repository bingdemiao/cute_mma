"""End-to-end cute-vs-cublas validation + speed for PrismLinear, on a real GPT.

Reuses the nanoGPT model (``integration/model.py``, copied from ~/nanogpt) so
the cute backend is exercised exactly as it is in training: gated PrismLinear
MLP up-projection, GroupNorm after, bf16 throughout, no autocast (per
TRAINING_GATED_PRISM.md §9 and nanogpt/train.py).

cublas is the trusted reference. We run the SAME model (same init, same data)
once with ``prism_backend="cublas"`` and once with ``"cute"`` and check:

  Phase 1 — strict step-0 parity (eval, dropout off):
      logits, loss, and every parameter gradient must match cublas within a
      bf16 tolerance. This is full-model correctness of the cute kernel,
      isolated to the PrismMLP.up call (everything else is identical nn ops).

  Phase 2 — short overfit run (train):
      both backends train the same fixed batch for N steps from identical init;
      assert no NaN/Inf and that the loss decreases. The cute and cublas loss
      curves are reported side by side (they drift apart slowly due to bf16
      rounding in the updates — that is expected; we only require both to
      descend and stay finite).

  Phase 3 — speed:
      per-step forward and forward+backward wall time for each backend.

This is a GPU script: run inside the container (oft.sqfs / nanogpt EDF) on
GH200 for a smoke, then on the A100/Ault target for the authoritative numbers.

Usage (inside the container venv)::

    CUTE_PRISM_COMPILE_WORKERS=2 python integration/compare_backends.py
    python integration/compare_backends.py --internal_bias --dropout 0.1
    python integration/compare_backends.py --input_shuffle --reconn_sz 16
    python integration/compare_backends.py --steps 300 --speed-iters 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig  # noqa: E402


def build_config(args) -> GPTConfig:
    return GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,                      # outer (residual) dropout off for parity
        bias=True,
        mlp_type="prism",
        mlp_expansion=args.mlp_expansion,
        group_size=args.group_size,
        reconn_sz=args.reconn_sz,
        prism_backend="cublas",           # overridden per build
        prism_activation="silu_gate",
        prism_internal_bias=args.internal_bias,
        prism_internal_dropout=args.dropout,
        prism_input_shuffle=args.input_shuffle,
        prism_shuffle_blk_k=args.shuffle_blk_k,
    )


def make_model(cfg: GPTConfig, backend: str, dtype, device, ref_state=None):
    cfg.prism_backend = backend
    torch.manual_seed(1234)               # identical init regardless of backend
    m = GPT(cfg).to(device)
    if ref_state is not None:
        m.load_state_dict(ref_state)      # force bit-identical params to the ref
    if dtype != torch.float32:
        m = m.to(dtype)
    return m


def max_rel_err(val, ref) -> float:
    d = (val.float() - ref.float()).abs().max()
    s = ref.float().abs().max() + 1e-6
    return (d / s).item()


def fixed_batch(cfg, batch_size, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), generator=g)
    Y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), generator=g)
    return X.to(device), Y.to(device)


# ---------------------------------------------------------------------------

def phase1_parity(cfg, dtype, device, batch_size, fwd_tol, grad_tol):
    print("\n=== Phase 1: step-0 parity (cute vs cublas, eval, dropout off) ===")
    X, Y = fixed_batch(cfg, batch_size, device, seed=0)

    cublas = make_model(cfg, "cublas", dtype, device)
    ref_state = cublas.state_dict()
    cute = make_model(cfg, "cute", dtype, device, ref_state=ref_state)
    cublas.eval(); cute.eval()

    def fwd_bwd(m):
        for p in m.parameters():
            p.grad = None
        logits, loss = m(X, Y)
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
                 if p.grad is not None}
        return logits.detach(), loss.detach(), grads

    lo_cb, loss_cb, g_cb = fwd_bwd(cublas)
    lo_ct, loss_ct, g_ct = fwd_bwd(cute)

    rel_logits = max_rel_err(lo_ct, lo_cb)
    rel_loss = abs(loss_ct.item() - loss_cb.item()) / (abs(loss_cb.item()) + 1e-6)
    print(f"  loss   cublas={loss_cb.item():.5f} cute={loss_ct.item():.5f} rel={rel_loss:.4e}")
    print(f"  logits rel_err={rel_logits:.4e} (tol {fwd_tol})")

    worst_name, worst = "", 0.0
    for n in g_cb:
        if n not in g_ct:
            continue
        r = max_rel_err(g_ct[n], g_cb[n])
        if r > worst:
            worst, worst_name = r, n
    print(f"  grads  worst rel_err={worst:.4e} @ {worst_name} (tol {grad_tol})")

    ok = rel_logits < fwd_tol and rel_loss < fwd_tol and worst < grad_tol
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def phase2_overfit(cfg, dtype, device, batch_size, steps, lr):
    print(f"\n=== Phase 2: overfit a fixed batch, {steps} steps each backend ===")
    X, Y = fixed_batch(cfg, batch_size, device, seed=0)
    ref_state = make_model(cfg, "cublas", dtype, device).state_dict()

    curves = {}
    ok = True
    for backend in ("cublas", "cute"):
        m = make_model(cfg, backend, dtype, device, ref_state=ref_state)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95))
        losses = []
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            _, loss = m(X, Y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        finite = all(map(lambda v: v == v and abs(v) != float("inf"), losses))
        descended = losses[-1] < losses[0]
        curves[backend] = losses
        print(f"  {backend:6s} loss {losses[0]:.4f} -> {losses[-1]:.4f}  "
              f"finite={finite} descended={descended}")
        ok = ok and finite and descended

    every = max(1, steps // 8)
    print("  step :   cublas    cute     |Δ|")
    for i in range(0, steps, every):
        a, b = curves["cublas"][i], curves["cute"][i]
        print(f"  {i:5d}: {a:8.4f} {b:8.4f}  {abs(a-b):.4f}")
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def phase3_speed(cfg, dtype, device, batch_size, warmup, iters):
    print(f"\n=== Phase 3: speed ({warmup} warmup + {iters} timed iters) ===")
    X, Y = fixed_batch(cfg, batch_size, device, seed=0)
    ref_state = make_model(cfg, "cublas", dtype, device).state_dict()

    def time_backend(backend):
        m = make_model(cfg, backend, dtype, device, ref_state=ref_state)
        m.train()

        def run(do_bwd):
            for _ in range(warmup):
                for p in m.parameters():
                    p.grad = None
                _, loss = m(X, Y)
                if do_bwd:
                    loss.backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                for p in m.parameters():
                    p.grad = None
                _, loss = m(X, Y)
                if do_bwd:
                    loss.backward()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3  # ms/iter

        return run(False), run(True)

    print(f"  {'backend':8s} {'fwd ms':>10s} {'fwd+bwd ms':>12s}")
    res = {}
    for backend in ("cublas", "cute"):
        fwd_ms, fb_ms = time_backend(backend)
        res[backend] = (fwd_ms, fb_ms)
        print(f"  {backend:8s} {fwd_ms:10.3f} {fb_ms:12.3f}")
    if "cublas" in res and "cute" in res:
        sf = res["cublas"][0] / res["cute"][0]
        sb = res["cublas"][1] / res["cute"][1]
        print(f"  cute speedup vs cublas: fwd {sf:.2f}x  fwd+bwd {sb:.2f}x")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--vocab_size", type=int, default=256)
    ap.add_argument("--mlp_expansion", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=64)
    ap.add_argument("--reconn_sz", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--internal_bias", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0, help="prism internal (post-gate) dropout")
    ap.add_argument("--input_shuffle", action="store_true")
    ap.add_argument("--shuffle_blk_k", type=int, default=64)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--speed-warmup", type=int, default=10)
    ap.add_argument("--speed-iters", type=int, default=30)
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. Run inside the GPU container "
              "(see nanogpt/submit.sh / ~/.edf/nanogpt.toml).")
        sys.exit(2)

    if args.clear_cache:
        import cute_prism
        cute_prism.clear_cache()

    device = "cuda"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    # bf16 has ~8 mantissa bits; full-model accumulation needs looser tol than
    # the unit tests. fp16 is a touch tighter.
    fwd_tol = 0.05 if dtype == torch.bfloat16 else 0.03
    grad_tol = 0.10 if dtype == torch.bfloat16 else 0.05

    cfg = build_config(args)
    print(f"config: n_layer={cfg.n_layer} n_embd={cfg.n_embd} d_ff={cfg.mlp_expansion*cfg.n_embd} "
          f"gs={cfg.group_size} rs={cfg.reconn_sz} dtype={args.dtype}")
    print(f"features: internal_bias={args.internal_bias} dropout={args.dropout} "
          f"input_shuffle={args.input_shuffle}")

    results = []
    results.append(("parity", phase1_parity(cfg, dtype, device, args.batch_size, fwd_tol, grad_tol)))
    results.append(("overfit", phase2_overfit(cfg, dtype, device, args.batch_size, args.steps, args.lr)))
    if not args.skip_speed:
        results.append(("speed", phase3_speed(cfg, dtype, device, args.batch_size,
                                              args.speed_warmup, args.speed_iters)))

    print("\n=== Summary ===")
    failed = 0
    for name, ok in results:
        print(f"  {name:8s} {'PASS' if ok else 'FAIL'}")
        failed += (not ok)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
