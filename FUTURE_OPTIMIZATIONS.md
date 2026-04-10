# Future optimizations — cublas backend

Fusion opportunities identified but **not** implemented yet, ranked by
effort/impact. See the discussion around the cublas backend refactor for
the original ranking (1–4 were completed in the previous pass).

## 5. cuBLASLt epilogue fusion for the AR GEMM + bias add

Forward gated+bias today:

```
AR = A @ R^T        (cuBLAS gemm_strided_batched_ex)
S  = AR + bias      (elementwise, inside silu_gate_bias_dropout_kernel)
H  = A * silu(S)    (elementwise, same kernel)
```

Switch the AR GEMM to `cublasLtMatmul` with `CUBLASLT_EPILOGUE_BIAS` so the
`AR + bias` add happens inside the GEMM epilogue. The elementwise kernel
then only needs to do `H = A * silu(S)` (plus optional dropout).

**Why we didn't do it:**
- Requires replacing `cublasHandle_t` + `cublasGemmEx` / `GemmStridedBatchedEx`
  with `cublasLtHandle_t` + `cublasLtMatmul`, matmul descriptors, preferences,
  workspace, heuristics — touches every GEMM call site and adds a
  workspace-management layer.
- The `silu_gate` activation has no cuBLASLt epilogue, so we still need a
  separate element-wise kernel — the savings are limited to one read of AR
  and one write of S, not a whole kernel.
- Ballpark win: 5–10% on the gated-forward critical path; only worth it
  if profiling shows AR+bias is actually hot.

## 6. CUTLASS epilogue fusion: AR GEMM + silu + gate into one kernel

Fully fuse the producer side of the forward:

```
AR = A_perm @ R^T             (batched, block-diagonal)
H  = A_perm * silu(AR + bias) (with optional dropout)
```

into a single CUTLASS GEMM with a custom epilogue that runs
`a * silu(ar + bias)` on the MMA accumulator while it's still in registers.
Then write `H` directly to global memory — `AR` never hits DRAM.

**What this buys us:**
- Removes a full `(M, K)` write + read pass per group (the `AR` buffer is
  gone).
- This is essentially "half of what the cute backend already does" — a
  CUTLASS-based cublas forward would approach cute-backend performance in
  the gated/shuffle regime (which the cute backend does *not* currently
  support).

**Why we didn't do it:**
- Big rewrite: requires a CUTLASS dependency in `torch_ext/`, a custom
  epilogue functor (bias add + silu + gate multiplier + optional dropout),
  and plumbing for the batched block-diagonal shape.
- Same dropout-seed input needs to be threaded through the epilogue.
- Only worth it after (1)–(4) land and benchmarking shows AR materialization
  is the dominant cost.

## Things considered and rejected

- **Gather → AR GEMM fusion.** cuBLAS can't consume indexed/strided inputs
  and the gather is not on the critical path relative to the GEMM. Would
  require a CUTLASS input iterator or a custom kernel; not worth it.
- **`dR_g = A^T @ dS` + silu-backward epilogue.** silu-backward lives on
  the opposite operand, so there's no natural epilogue hook.
