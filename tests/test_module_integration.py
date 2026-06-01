"""Integration tests for the PrismLinear nn.Module path (backend-agnostic glue).

These tests validate the *Python wiring* around the kernels — the part that is
shared by every backend (cute / cublas / pytorch) and is currently untested by
the per-feature parity tests (which all call ``cute_prism.forward/backward``
directly, never the module). They run on **CPU with the pytorch backend in
fp64**, so they need no GPU and no cute build, yet they exercise exactly the
glue the cute backend will rely on once it runs on the A100:

  1. gradcheck the two autograd wirings directly (the true-gradient paths):
       * ``_prism_forward_op``  — custom-op autograd (plain & gated, no features)
       * ``_PrismFn``           — feature path (shuffle / internal_bias)
     This catches a mis-ordered gradient tuple (e.g. d_internal_bias landing on
     the wrong input), which would pass every existing parity test.
  2. gradcheck the full PrismLinear module end-to-end (reshape + bias + Cayley R
     build), restricted to *plain* mode where the deliberate ``_BwdScaleFn``
     gain correction is not applied (gated mode rescales the backward on
     purpose, so its module-level backward is intentionally not the true
     gradient and cannot be gradchecked).
  3. eval(): dropout must become identity and ``training=False`` must flow
     through; train() must randomize.
  4. state_dict round-trip: the ``_internal_bias`` parameter and the
     ``_shuffle_masks`` buffer must survive save/load.

Usage::

    uv run python tests/test_module_integration.py        # standalone
    uv run pytest tests/test_module_integration.py         # via pytest

All tests are CPU-only; they do not import or build the cute kernels.
"""

from __future__ import annotations

import sys

import torch

import cute_prism
from cute_prism._module import (
    PrismLinear,
    _PrismFn,
    _prism_forward_op,
    _build_shuffle_masks,
)

DEV = "cpu"
DT = torch.float64
BACKEND = "pytorch"


class _Skip(Exception):
    """Raised to skip a test for an environment/toolchain reason (not a defect)."""


def _skip(msg: str):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise _Skip(msg)


def _is_toolchain_compile_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}"
    return any(k in s for k in (
        "CppCompileError", "C++ compile error", "BackendCompilerFailed",
        "requested alignment is not an integer constant",
    ))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk_inputs(M, K, N, gs, rs, *, bias=False, shuffle=False, blk_k=64, seed=0):
    """Raw (A, B, R[, internal_bias][, shuffle_masks]) leaf tensors in fp64."""
    torch.manual_seed(seed)
    n_groups = N // gs
    A = torch.randn(M, K, device=DEV, dtype=DT, requires_grad=True)
    B = torch.randn(N, K, device=DEV, dtype=DT, requires_grad=True)
    R = torch.randn(n_groups * rs, K, device=DEV, dtype=DT, requires_grad=True)
    ib = None
    if bias:
        ib = (torch.randn(n_groups, K, device=DEV, dtype=DT) * 0.1).requires_grad_()
    masks = None
    if shuffle:
        masks, _ = _build_shuffle_masks(K, N, gs, rs, blk_k)
        masks = masks.to(DEV)
    return A, B, R, ib, masks


# ---------------------------------------------------------------------------
# 1. gradcheck the autograd wirings directly
# ---------------------------------------------------------------------------

def test_gradcheck_custom_op():
    """_prism_forward_op autograd (custom-op path: plain & gated, no features).

    Note the deliberate asymmetry of the standard/finetuning convention: in
    plain mode (activation=None) B is *frozen*, and the op returns a **zero**
    gradient for B ([_module.py] _prism_backward_op: ``if dB is None: dB =
    zeros_like(B)``). So we:
      * gated  — gradcheck all of (A, B, R) [all grads are real];
      * plain  — gradcheck (A, R) with B detached [A/R grads are real],
                 and separately assert the op's dB is ~0 to pin the convention.
    """
    M, K, N, gs, rs = 8, 64, 64, 32, 16

    # gated: every gradient is real.
    A, B, R, _, _ = _mk_inputs(M, K, N, gs, rs, seed=1)
    f_gated = lambda a, b, r: _prism_forward_op(a, b, r, gs, rs, BACKEND, "silu_gate", False, False)
    assert torch.autograd.gradcheck(f_gated, (A, B, R), eps=1e-6, atol=1e-5, rtol=1e-3), \
        "custom-op gradcheck failed (gated)"

    # plain: B frozen -> exclude it from gradcheck (detach), check A & R.
    A2, B2, R2, _, _ = _mk_inputs(M, K, N, gs, rs, seed=11)
    B2 = B2.detach()  # B not differentiated in plain mode
    f_plain = lambda a, r: _prism_forward_op(a, B2, r, gs, rs, BACKEND, "", False, False)
    assert torch.autograd.gradcheck(f_plain, (A2, R2), eps=1e-6, atol=1e-5, rtol=1e-3), \
        "custom-op gradcheck failed (plain, A/R)"

    # Document the frozen-B convention: plain-mode dB is exactly zero.
    from cute_prism._module import _prism_backward_op
    A3, B3, R3, _, _ = _mk_inputs(M, K, N, gs, rs, seed=12)
    dC = torch.randn(M, N, device=DEV, dtype=DT)
    _, _, dB_plain = _prism_backward_op(
        dC, A3.detach(), B3.detach(), R3.detach(), gs, rs, BACKEND, "", False, False)
    assert torch.count_nonzero(dB_plain) == 0, \
        "plain-mode custom op no longer returns zero dB — frozen-B convention changed"


def test_gradcheck_prismfn_features():
    """_PrismFn autograd (feature path), each feature combination, dropout off.

    dropout_p=0 keeps the function deterministic (gradcheck calls it many
    times); the dropout *gradient* is covered separately by the existing
    test_pytorch_backend.py seed-replay test.
    """
    M, K, N, gs, rs, blk_k = 8, 128, 64, 32, 16, 64
    for shuffle in (False, True):
        for bias in (False, True):
            A, B, R, ib, masks = _mk_inputs(
                M, K, N, gs, rs, bias=bias, shuffle=shuffle, blk_k=blk_k, seed=2)
            inputs = (
                A, B, R, gs, rs, BACKEND, "silu_gate", False, False,
                ib, 0.0, False, masks,
            )
            ok = torch.autograd.gradcheck(
                _PrismFn.apply, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)
            assert ok, f"_PrismFn gradcheck failed (shuffle={shuffle}, bias={bias})"


# ---------------------------------------------------------------------------
# 2. gradcheck the full module (plain mode only; _bwd_scale == 1)
# ---------------------------------------------------------------------------

def test_gradcheck_module_plain():
    """End-to-end module gradcheck in plain mode (reshape + bias + Cayley R).

    Plain (finetuning) mode builds R via the Cayley transform and applies no
    backward gain correction, so the module backward IS the true gradient.
    Covers both the custom-op path (no shuffle) and the _PrismFn path (shuffle).
    """
    from torch.func import functional_call

    M, K, N, gs, rs = 8, 64, 64, 32, 16
    for shuffle in (False, True):
        torch.manual_seed(3)
        m = PrismLinear(
            K, N, group_size=gs, reconn_sz=rs, bias=True,
            activation=None, backend=BACKEND,
            input_shuffle=shuffle, shuffle_blk_k=32,
        ).to(DEV).to(DT)
        assert m._bwd_scale == 1.0  # plain mode: no deliberate rescale

        x = torch.randn(M, K, device=DEV, dtype=DT, requires_grad=True)
        # Only differentiate trainable params (weight B is frozen in plain mode).
        names = [n for n, p in m.named_parameters() if p.requires_grad]
        params = [m.get_parameter(n).detach().clone().requires_grad_() for n in names]

        def f(x_, *pvals):
            pd = dict(zip(names, pvals))
            return functional_call(m, pd, (x_,))

        ok = torch.autograd.gradcheck(f, (x, *params), eps=1e-6, atol=1e-5, rtol=1e-3)
        assert ok, f"module plain gradcheck failed (shuffle={shuffle})"


# ---------------------------------------------------------------------------
# 3. eval() / train() dropout plumbing
# ---------------------------------------------------------------------------

def test_eval_disables_dropout():
    M, K, N, gs, rs = 16, 64, 64, 32, 16
    torch.manual_seed(4)
    m = PrismLinear(
        K, N, group_size=gs, reconn_sz=rs, bias=True,
        activation="silu_gate", dropout=0.5, backend=BACKEND,
    ).to(DEV).to(DT)

    x = torch.randn(M, K, device=DEV, dtype=DT)

    # eval(): deterministic across calls (dropout off).
    m.eval()
    y1, y2 = m(x), m(x)
    assert torch.allclose(y1, y2), "eval() output is not deterministic — dropout still active"

    # eval() output must equal the same weights with dropout structurally off.
    ref = PrismLinear(
        K, N, group_size=gs, reconn_sz=rs, bias=True,
        activation="silu_gate", dropout=0.0, backend=BACKEND,
    ).to(DEV).to(DT)
    ref.load_state_dict(m.state_dict())
    ref.eval()
    assert torch.allclose(y1, ref(x), atol=1e-8), \
        "eval() output differs from the dropout-disabled forward"

    # train(): stochastic across calls (dropout active).
    m.train()
    y3, y4 = m(x), m(x)
    assert not torch.allclose(y3, y4), \
        "train() output is deterministic — dropout did not engage"


# ---------------------------------------------------------------------------
# 4. state_dict round-trip (internal_bias param + shuffle_masks buffer)
# ---------------------------------------------------------------------------

def test_state_dict_roundtrip():
    M, K, N, gs, rs = 8, 128, 64, 32, 16
    torch.manual_seed(5)
    cfg = dict(
        group_size=gs, reconn_sz=rs, bias=True, activation="silu_gate",
        internal_bias=True, dropout=0.3, input_shuffle=True,
        shuffle_blk_k=64, backend=BACKEND,
    )
    m1 = PrismLinear(K, N, **cfg).to(DEV).to(DT)

    sd = m1.state_dict()
    assert "_internal_bias" in sd, "internal_bias missing from state_dict"
    assert "_shuffle_masks" in sd, "shuffle_masks buffer missing from state_dict"

    m2 = PrismLinear(K, N, **cfg).to(DEV).to(DT)
    m2.load_state_dict(sd)

    assert torch.equal(m1._internal_bias, m2._internal_bias)
    assert torch.equal(m1._shuffle_masks, m2._shuffle_masks)

    x = torch.randn(M, K, device=DEV, dtype=DT)
    m1.eval(); m2.eval()
    assert torch.allclose(m1(x), m2(x), atol=1e-8), \
        "forward differs after state_dict round-trip"


# ---------------------------------------------------------------------------
# 5. torch.compile compatibility
# ---------------------------------------------------------------------------

def test_compile_plain_fullgraph():
    """The custom-op path must compile with no graph breaks (fullgraph=True).

    The docstring claims the Prism kernel call is opaque to torch.compile.
    A graph break under fullgraph=True raises, failing this test.
    """
    M, K, N, gs, rs = 8, 64, 64, 32, 16
    torch.manual_seed(6)
    m = PrismLinear(
        K, N, group_size=gs, reconn_sz=rs, bias=True,
        activation=None, backend=BACKEND,
    ).to(DEV).to(DT)
    x = torch.randn(M, K, device=DEV, dtype=DT)

    eager = m(x)
    try:
        cm = torch.compile(m, fullgraph=True)
        compiled = cm(x)
    except Exception as e:  # noqa: BLE001
        if _is_toolchain_compile_error(e):
            _skip(f"inductor CPU backend toolchain bug (not a PrismLinear defect): {e}")
        raise AssertionError(f"fullgraph compile of the custom-op path failed: {e}")
    assert torch.allclose(eager, compiled, atol=1e-8), \
        "compiled output differs from eager (plain/custom-op path)"


def test_compile_features_numerics():
    """Feature path (gated + internal_bias + shuffle) compiles and matches eager
    in both forward and backward. Graph breaks are allowed here (the autograd
    Function path may break); only numerical agreement is asserted."""
    M, K, N, gs, rs = 8, 128, 64, 32, 16
    torch.manual_seed(7)
    cfg = dict(
        group_size=gs, reconn_sz=rs, bias=True, activation="silu_gate",
        internal_bias=True, input_shuffle=True, shuffle_blk_k=64, backend=BACKEND,
    )
    m = PrismLinear(K, N, **cfg).to(DEV).to(DT)
    m.eval()  # disable dropout for determinism (dropout=0 here anyway)

    x = torch.randn(M, K, device=DEV, dtype=DT, requires_grad=True)
    y_eager = m(x)
    y_eager.sum().backward()
    g_eager = x.grad.clone()

    x2 = x.detach().clone().requires_grad_()
    try:
        cm = torch.compile(m)
        y_comp = cm(x2)
        y_comp.sum().backward()
    except Exception as e:  # noqa: BLE001
        if _is_toolchain_compile_error(e):
            _skip(f"inductor CPU backend toolchain bug (not a PrismLinear defect): {e}")
        raise AssertionError(f"compile of the feature path failed: {e}")

    assert torch.allclose(y_eager, y_comp, atol=1e-7), "compiled forward != eager"
    assert torch.allclose(g_eager, x2.grad, atol=1e-7), "compiled grad != eager"


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("gradcheck custom-op autograd", test_gradcheck_custom_op),
        ("gradcheck _PrismFn features", test_gradcheck_prismfn_features),
        ("gradcheck module (plain)", test_gradcheck_module_plain),
        ("eval disables dropout", test_eval_disables_dropout),
        ("state_dict round-trip", test_state_dict_roundtrip),
        ("compile plain fullgraph", test_compile_plain_fullgraph),
        ("compile features numerics", test_compile_features_numerics),
    ]
    n_pass = n_fail = n_skip = 0
    for name, fn in tests:
        try:
            fn()
        except _Skip as e:
            n_skip += 1
            print(f"  SKIP {name}: {e}")
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0]}")
        else:
            n_pass += 1
            print(f"  PASS {name}")
    print(f"Total: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
