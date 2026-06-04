# Known issues

## 1. cute `bwd_dadr` race at `group_size ≥ 256` — RESOLVED

**Status:** fixed (2026-06). Two consume-side barriers added to
`cute_prism_backward_dadr.cu`. Verified by stress (0/200 at gs=256 and gs=512;
0/50 shuffle gs=256 rs=16) and the full bwd parity suite.

### Symptom (before fix)
Flaky, **nondeterministic** wrong `dA`/`dR`/`d_internal_bias` at `group_size ≥
256` (same inputs → different output run-to-run; rate 1–65%, occasional
`dR=inf`/`dIB≈68`). `dB`, the forward, and `group_size ≤ 128` were unaffected.
Training (gs=64) was never affected.

### Root cause
The kernel is warp-specialized with software-pipelined shared-memory buffers.
Two of them were missing a **consume-side barrier** — i.e. a slot could be
overwritten by the next async load before all warps finished reading it:
1. **Producer `dC`/`B` pipeline** (`sdC`/`sB`): no barrier between the `ldmatrix`
   read and the next iteration's `cp.async` overwrite of the same slot.
2. **Consumer per-group `sA` reload** (shuffle mode only): no barrier between the
   group's `sA` reads and the next group's in-place `sA` reload.
The hazard only fires once the inner loop reuses a slot enough times, i.e.
`tiles_per_group = group_size/32 ≥ 8` (gs≥256). At gs≤128 (2–4 iters) the timing
almost always worked out, which is why it looked clean there.

### Fix
- Producer: `bar.sync 14` after `gemm` (consume-side barrier on `sdC`/`sB`).
- Consumer: `bar.sync 15` at the end of the per-group loop (consume-side barrier
  on `sA`/`sR` before the next group's reload).
Cost: ~8% on the dadr backward (extra per-iteration producer barrier).

### How it was found (method, for reference)
References (cublas≈pytorch) agreed → cute-only bug. memcheck+initcheck clean →
shared-memory hazard. racecheck impractical (≫100× slowdown on this kernel). A
group-id **sequence-tag detector** proved the producer↔consumer dH *handshake*
was correct (no stale read, no lapping), redirecting suspicion to the *pipeline
buffers*. A `(bP_dc_b,bP_dh)` depth sweep at **≥100 reps** (20 reps gave false
0/20!) showed depth only modulates the rate → a missing barrier, not a config
issue. The consume-side barrier confirmed it.

### Regression test (keep)
```python
import torch, cute_prism
re=lambda a,b:((a-b).abs()/(b.abs().max()+1e-6)).max().item()
def stress(gs, n=200):
    torch.manual_seed(0); M,N,K,rs=1024,1024,1024,8
    f=lambda *s: torch.randn(*s,dtype=torch.float16,device='cuda')
    A,B,R,bias,dC=f(M,K),f(N,K),f(N//gs*rs,K),f(N//gs,K)*0.1,f(M,N)
    dab,drb,_,_=cute_prism.backward(dC,A,B,R,gs,rs,backend='cublas',activation='silu_gate',internal_bias=bias)
    bad=sum(max(re(a,b) for a,b in zip(
        cute_prism.backward(dC,A,B,R,gs,rs,backend='cute',activation='silu_gate',internal_bias=bias)[:2],(dab,drb)))>0.02
        for _ in range(n))
    print(f"gs={gs}: {bad}/{n}")
stress(256); stress(512)   # must be 0/200 each
```

---

## 2. cute `input_shuffle` requires `group_size ≥ bN` (OPEN, guarded)

`input_shuffle` on the cute backend is only correct when `group_size ≥ bN` (the
forward N-tile, default 128). At `gs < bN` a CTA spans multiple output groups but
the shuffled-A buffer is selected by a single per-CTA stride, so all but the
first group read the wrong shuffled A (~100% forward error). **Guarded**:
`cute_prism.forward` raises for `shuffle_masks` + `gs < bN`. Use `gs ≥ 128`, or
the cublas backend, for input_shuffle at small group_size. Proper fix (per-group
A selection inside the CTA) is future work.

## 3. cute backward partial-M tiles give `inf` (OPEN, minor)

When `M` is **not a multiple of `bM` (=64)**, the backward `dR`/`d_internal_bias`
reduction reads unpredicated partial-tile rows → `inf`/garbage (forward is fine).
Real training always has `M = batch×block` aligned, so this is an edge case. Fix
= predicate the partial-M rows in the dR/dIB accumulation; workaround = use
`M % 64 == 0`.
