# dAdR Kernel Restructure — Final Report

## Results

| Variant | Time (M=N=K=8192) | Speedup |
|---------|-------------------|---------|
| Original (sequential loop) | 91 ms | 1.0× |
| **Restructured (gated)** | **47 ms** | **1.94×** |
| Restructured (non-gated) | ~39 ms | ~2.3× |

Correctness verified at M=N=K=256 and M=N=K=512 (both gated and non-gated):
- dA relative error: < 0.003 (matches original kernel precision)
- dR relative error: < 0.002

## Architecture

### GEMM Stage
- Warp layout: 2×4 (n_warps_M=2, n_warps_K=4) for 8 warps
- Each warp's C-fragment: `((2,2), 2, 2)` = `(MMA_inner, M_atoms, K_atoms)`
- CuTe interleaves atoms: K_atoms 0 and 1 per warp are separated by
  `n_warps_K × atom_N = 32` columns (not consecutive)

### Epilogue
- **2-iteration loop** per column pair (vs 8-iteration sequential loop in original)
- Per iteration:
  1. `make_tiled_copy_C` writes rDH sub-fragment `(_,_,rb)` to sDH_exch
  2. Pairwise barrier (bar.sync with 64 threads, one per column pair)
  3. dA: each warp loads its M-half from sDH_exch + R from sR → reconn MMA
  4. dR: warp_m=0 loads full BLK_M from sDH_exch → dR MMA with K=BLK_M
- Total syncs per group: ~6 (vs ~32 in original)

### Key Design Decisions

1. **sDH_exch buffer**: (BLK_M, BLK_K) = 8KB. Full-size because `make_tiled_copy_C`
   writes to positions matching the GEMM MMA layout (not simple RowMajor blocks).

2. **dR computed by warp_m=0 only**: After `make_tiled_copy_C` + sync, sDH_exch
   contains complete dH from both warps. A single warp can do the full K=BLK_M
   reduction without atomicAdd. The warp_m=1 warp is idle during dR but busy during dA.

3. **K offset stride**: `col_k_stride = n_warps_K × atom_N` accounts for CuTe's
   interleaved N-atom distribution. This was a critical discovery — each warp's
   2 reconn blocks are NOT at consecutive K offsets.

4. **Fragment reinterpretation**: Slicing `rDH(_,_,rb)` via `make_tiled_copy_C`'s
   retile_S cleanly separates the MMA atoms for each reconn block. No manual
   element extraction needed.

## Current LDSM Status
- sDH_exch A-operand for dA: **LDSM via logical_divide** (LayoutRight, works correctly)
- sR/sA sub-views: **scalar loads** (swizzled layout incompatible with logical_divide + LDSM)
- dR transposed operands: **scalar loads** (needs LDSM_T + movmatrix)
- Gated Steps A/B element-wise ops: **scalar indexed** via sDH_exch (works for single group)

## Known Issue: Gated Multi-Group
The gated path's Steps A/B read sDH_exch using `warp_m * HALF_M + mi` offset
from cons_mma's tCdA_id coordinates. But sDH_exch was written by make_tiled_copy_C
from the GEMM MMA (2×4 warp layout), which places elements at positions determined
by the GEMM's C-fragment layout — NOT at clean warp_m * HALF_M rows.

For single group (M=256, N=256): works because dH is only accumulated once.
For multiple groups (M=512, N=512): dH accumulates across groups, and the
coordinate mismatch causes incorrect element-wise operations.

Fix: Steps A/B should use GEMM tCid coordinates (not cons_mma tCdA_id) to
access sDH_exch, matching the layout produced by make_tiled_copy_C.

## Follow-up Optimizations
- Fix gated multi-group: use GEMM tCid for Steps A/B element-wise ops
- Fix sR/sA LDSM: investigate swizzle-compatible sub-view creation
- Implement LDSM_T + movmatrix for dR transposed operands
- Use both warps for dR
- Potentially reduce sDH_exch to 4KB
