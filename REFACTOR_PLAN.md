# PrismLinear kernel consolidation

## Goal

Collapse the two cublas kernels (`cublas_prism_torch.cu` and
`shuffle_prism_torch.cu`) into **one** unified kernel whose behavior is
controlled by four orthogonal flags, so every combination works in both the
`cublas` and `pytorch` backends:

| flag             | values              | effect                                                        |
|------------------|---------------------|---------------------------------------------------------------|
| `activation`     | `None` / `silu_gate`| gated mode with `H = A * SiLU(AR + bias)`                     |
| `internal_bias`  | off / on            | adds per-group bias before the SiLU gate                      |
| `dropout`        | `0` / `p>0`         | dropout on `H` with seed-based backward replay                |
| `shuffle`        | off / on            | per-group gather of `A` into a permuted layout before `AR`    |

Today, `shuffle` lives in a separate kernel that supports none of the other
three features, and the dispatcher in `PrismLinear.forward` silently drops
`internal_bias`/`dropout` when `input_shuffle=True`. After this refactor all
16 combinations will execute through a single code path.

## Design

### Pipeline layout

Producer/consumer split, with the producer owning whichever side of the
pipeline boundary does the expensive bulk work. The **only** double-buffered
tensor is the hand-off between producer and consumer; every other intermediate
is single-buffered because it never leaves its own stream.

1. **Forward**
   - Producer (`stream_p`): `[gather A→A_perm if shuffle] → batched AR-GEMM → (silu + bias + dropout if gated) → H_perm`
   - Consumer (`stream`):   dense `C_g = H_perm @ B_g^T`
   - Double-buffer: `H_perm[PIPE]`. Single-buffer: `A_perm`, `AR_perm` (same
     allocation, actually — the gate kernel writes H into the AR buffer).

2. **dB backward**
   - Producer: identical recompute of `H_perm` (gather→AR→gate→dropout).
   - Consumer: dense `dB_g = H_perm^T @ dC_g`.
   - Same buffering as forward.

3. **dA/dR backward**
   - Producer: dense `dH_perm = dC_g @ B_g`   ← the expensive GEMM.
   - Consumer: `[gather A→A_perm if shuffle] → recompute AR_perm → gate-backward fused kernel (bias add, dropout replay, dA_gate_perm, dS_perm) → dR_g = A_perm^T @ dS_perm (batched) → dA_gemm_perm = dS_perm @ R_g (batched) → dA_perm = dA_gate_perm + dA_gemm_perm → [unpermute-add into dA if shuffle, else beta=1 accumulate directly into dA] → d_internal_bias[g] = dS_perm.sum(0)`
   - Double-buffer: `dH_perm[PIPE]`. Single-buffer: `A_perm`, `AR_perm`,
     `dS_perm`, `dA_perm`.

### Why no atomics in dA

The existing shuffle kernel scatters `dS_perm` into `dA` with
`atomicAdd` on an fp32 accumulator. That's unnecessary once we do the
unpermute as a **gather** driven by the inverse permutation: for each output
`(row, nat_col)`, read `dA_perm[row, perm_g_inv[nat_col]]` and accumulate.
Inside one kernel launch each output element is written exactly once —
no intra-kernel conflict. Across groups the kernels serialize on the stream,
so the `+=` is naturally race-free. The fp32 accumulator goes away; `dA`
can live directly in the activation dtype.

We need `perm_g_inv` per group. Either (a) precompute it at load time
alongside `seg_pairs` (cheap, `(n_groups, K)` int32), or (b) derive it from
`seg_pairs[g]` on the fly inside the unpermute kernel. (a) is simpler —
do that.

### Bias and dropout are permutation-unaware

The shuffle only exists to diversify which pairs of input segments each
R-block sees — it prevents R from collapsing. Every downstream element-wise
op (`bias`, `silu`, `dropout`) lives in whatever column order `AR_perm`
happens to be in for that group. The bias parameter `(n_groups, K)` is
*not* translated into "natural" space; it's added elementwise to `AR_perm`
with the same `bias[g, col]` indexing the non-shuffle kernel uses. The
model just learns the bias in its per-group permuted coordinate frame.
Dropout is pure elementwise on `H_perm` and cares even less.

Practical upshot: `silu_gate_bias_dropout_kernel` and
`silu_gate_backward_fused_kernel` stay **exactly as they are** and need no
shuffle-specific codepath. Only the GEMM surroundings (gather on the input,
unpermute on the dA output) differ.

### dA accumulator dtype

Non-shuffle currently uses a `beta=1` batched GEMM to accumulate
`dA += dS_g @ R_g` directly into the dtype `dA` buffer. Keep that path
unchanged. When `shuffle=true`, we instead run the gather-add unpermute
kernel described above, also into the dtype buffer. One extra branch inside
the per-group loop — no separate fp32 allocation.

## File-level changes

### `torch_ext/cublas_prism_torch.cu` (edit — grows)

- `cublas_prism_ar_impl`: add optional `const int64_t* seg_pairs_ptr`.
  - Allocate a **single-buffer** `A_perm[m,k]` when `seg_pairs_ptr != nullptr`.
  - Inside `produce(g, p)`, if shuffle: launch `gather_segments_kernel` on
    `stream_p` before the AR batched GEMM, then use `A_perm` in place of
    `A_ptr` for both the GEMM and the downstream gate kernel.
  - Downstream `silu_gate_bias_dropout_kernel` unchanged.
- `cublas_backward_dA_dR_impl`: add `seg_pairs_ptr` + optional `perm_inv_ptr`.
  - Keep producer on `stream_p` doing the dense `dH = dC_g @ B_g` GEMM
    (`dH_perm[PIPE]` is the only double-buffer).
  - On the consumer stream, if shuffle: gather `A→A_perm`, recompute AR_perm,
    run the existing `silu_gate_backward_fused_kernel`, batched
    `dR_g = A_perm^T @ dS_perm`, batched `dA_perm = dS_perm @ R_g`, then call
    a new `unpermute_add_kernel` that does `dA[r,c] += dA_perm[r, perm_inv[c]]`.
  - If not shuffle: current `beta=1` GEMM path into `dA` (unchanged modulo
    moving it onto consumer stream per the new pipeline layout).
- `cublas_backward_dB_impl`: add `seg_pairs_ptr`. Same gather-before-AR
  treatment as forward. Single-buffer A_perm/AR_perm, double-buffer H_perm.
- New kernels:
  - `gather_segments_kernel<T>` (move from `shuffle_prism_torch.cu`).
  - `unpermute_add_kernel<T>`: reads `dA_perm[row, perm_inv[col]]` and
    accumulates into `dA[row, col]` — **not atomic**.
- Public entry points (`cublas_prism_forward`, `cublas_backward_dA_dR`,
  `cublas_backward_dB`) gain one `c10::optional<torch::Tensor> seg_pairs`
  argument and one `c10::optional<torch::Tensor> seg_pairs_inv` (or compute
  inverse internally once per call and cache it).

### `torch_ext/cublas_prism_torch_bind.cpp` (edit)

Thread the new `seg_pairs` / `seg_pairs_inv` args through the pybind.

### `torch_ext/shuffle_prism_torch.cu`, `torch_ext/shuffle_prism_bind.cpp` (delete)

Gone entirely once the unified kernel compiles and passes tests.

### `src/cute_prism/__init__.py` (edit)

`forward()` and `backward()` gain `seg_pairs: Tensor | None = None` (and, if
we go with precomputed inverse, `seg_pairs_inv`). Pass through to the
`cublas` and `pytorch` backends. No change to `cute` backend for now.

### `src/cute_prism/_pytorch_backend.py` (edit)

Mirror the unified shape: `pytorch_forward` / `pytorch_backward` accept
`seg_pairs` and do the gather / unpermute-add in torch ops. Element-wise
bias/dropout code unchanged. Use `torch.gather`/`index_select` for the
per-group permutation.

### `src/cute_prism/_module.py` (edit — shrinks)

- Delete `_get_shuffle_module`, `_shuffle_module`, `_shuffle_lock` (no more
  separate JIT build).
- Delete `_ShufflePrismFunction` — the single autograd wrapper handles both
  shuffle and non-shuffle.
- Rename/refactor `_PrismWithBiasDropoutFn` to `_PrismFn` (the only wrapper)
  and thread `seg_pairs` through `ctx`.
- `PrismLinear.forward`: one dispatch path. No `_forward_shuffle` /
  `_forward_shuffle_pytorch`.
- `PrismLinear.__init__`:
  - When `input_shuffle=True`, precompute `seg_pairs_inv` next to
    `seg_pairs` and register both as buffers.
  - Drop the `shuffle_blk_k > 256` / `backend == "cute"` guard-rail
    restrictions that only existed for the old kernel. (Keep `reconn_sz=16`
    if that's still an alignment requirement of `gather_segments_kernel`.)
- `_forward_shuffle` / `_forward_shuffle_pytorch` / the stopgap dispatcher
  fix from the current WIP diff — all gone.

## Implementation order (suggested)

1. **pytorch backend first** — cheapest, fully unit-testable on CPU/GPU
   without touching CUDA code. Land `_pytorch_backend.py` changes plus the
   autograd wrapper refactor in `_module.py` and the `forward`/`backward`
   signature additions in `__init__.py`. All 16 combinations should pass a
   numerical equivalence test against the current shuffle kernel + the
   current non-shuffle path.
2. **cublas forward** — add `seg_pairs` to `cublas_prism_ar_impl` +
   `gather_segments_kernel`. Test forward only against pytorch reference.
3. **cublas dB backward** — same gather treatment.
4. **cublas dA/dR backward** — add `unpermute_add_kernel` and the shuffle
   branch. Test `(shuffle × gated × bias × dropout)` against pytorch.
5. **Delete** `shuffle_prism_torch.cu`, `shuffle_prism_bind.cpp`, and all
   shuffle-specific plumbing in `_module.py` / `__init__.py`.

## Tests

Write `test_prism_combos.py` that enumerates the 16 combinations
(`{gated, bias, dropout, shuffle}`) on a small shape (e.g. `M=64, K=256,
N=256, group_size=64, reconn_sz=16`) and for each combo:

- Compares `cublas` forward output against `pytorch` forward output (rtol 1e-2
  for bf16, tighter for fp32 if we add fp32 support).
- Compares `cublas` `dA`, `dR`, `dB`, `d_internal_bias` against `pytorch`.
- Checks `dropout` reproducibility via seed replay (forward twice with the
  same seed → identical masks).

Also add a gain sanity check (`test_prism_gain.py` already exists in
`/users/lshuhao/oft`) to confirm `{shuffle=True, bias=True, dropout=0.1}`
still lands around 1.0 fwd/bwd gain through `PrismLinear → GroupNorm`.

## Current WIP state (what's in this commit)

- `src/cute_prism/_module.py` has a stop-gap edit in `_forward_shuffle`
  that routes shuffle+bias/dropout through the pytorch fallback. **This
  will be deleted outright** in step 5 — it's in the diff only so the
  refactor branch starts from a consistent place.
- The `# option b` and `# adding inner dropout part` comments inside
  `_forward_shuffle_pytorch` are dead code markers from a previous
  exploration; also going away in step 5.
- No changes to `torch_ext/` yet; CUDA work starts in step 2.
