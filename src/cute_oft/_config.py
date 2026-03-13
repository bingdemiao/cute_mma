"""Computation parameter configuration for OFT kernels.

Manages the performance-related parameters (tile sizes, pipeline depths, warp
layouts) that are baked into each compiled kernel variant. Currently uses
sensible defaults; the architecture supports future autotuning by varying
these parameters.
"""

from __future__ import annotations

import dataclasses
import hashlib


@dataclasses.dataclass(frozen=True)
class CompParams:
    """Performance parameters for the OFT kernel.

    These correspond to CurrCompParams in oft_config.hpp.
    """

    bM: int = 128
    bN: int = 128
    bK: int = 32
    c_width: int = 16
    bP_a_r: int = 2
    bP_ar: int = 2
    bP_b: int = 2
    # Warp layout shapes: tuples of ints.
    # (2,) means Layout<Shape<Int<2>>>
    # (2, 2) means Layout<Shape<Int<2>, Int<2>>>
    warp_layout_ar: tuple[int, ...] = (2,)
    warp_layout_arb: tuple[int, ...] = (2, 2)

    def to_header(self) -> str:
        """Generate the C++ struct definition for CurrCompParams."""
        return f"""\
    struct CurrCompParams {{
        static const unsigned int bM = {self.bM};
        static const unsigned int bN = {self.bN};
        static const unsigned int bK = {self.bK};
        static const unsigned int c_width = {self.c_width};
        static const unsigned int bP_a_r = {self.bP_a_r};
        static const unsigned int bP_ar = {self.bP_ar};
        static const unsigned int bP_b = {self.bP_b};
        using warp_layout_ar = {_layout_type(self.warp_layout_ar)};
        using warp_layout_arb = {_layout_type(self.warp_layout_arb)};
    }};"""

    def cache_key(self) -> str:
        """Short hash identifying this parameter combination."""
        content = self.to_header()
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CompParams:
        """Deserialize from a dictionary."""
        d = dict(d)
        # Convert lists back to tuples for warp layouts
        for key in ("warp_layout_ar", "warp_layout_arb"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        return cls(**d)

    @classmethod
    def safe_defaults(cls) -> CompParams:
        """Conservative defaults that compile for all valid shapes."""
        return cls(
            bM=128, bN=128, bK=16, c_width=8,
            bP_a_r=2, bP_ar=2, bP_b=2,
            warp_layout_ar=(2,), warp_layout_arb=(2,),
        )


def _layout_type(shape: tuple[int, ...]) -> str:
    """Convert a tuple of ints to a CuTe Layout type string."""
    if len(shape) == 1:
        return f"cute::Layout<cute::Shape<cute::Int<{shape[0]}>>>"
    inner = ", ".join(f"cute::Int<{s}>" for s in shape)
    return f"cute::Layout<cute::Shape<{inner}>>"


@dataclasses.dataclass(frozen=True)
class BwdDBCompParams:
    """Performance parameters for the backward dB kernel (producer-consumer)."""

    bM: int = 32           # M tile size (reduction dim, iterated)
    bK: int = 32           # K tile size (output dim)
    bP_a: int = 2          # Pipeline depth: A async loads (producer)
    bP_ar: int = 2         # Pipeline depth: AR producer→consumer
    bP_dc: int = 2         # Pipeline depth: dC async loads (consumer)
    warp_layout_ar: tuple[int, ...] = (2,)    # AR producer warps
    warp_layout_arb: tuple[int, ...] = (2, 2)  # heavy consumer warps

    def to_header(self) -> str:
        """Generate the C++ struct definition."""
        return f"""\
    struct BwdDBParams {{
        static const unsigned int bM = {self.bM};
        static const unsigned int bK = {self.bK};
        static const unsigned int bP_a = {self.bP_a};
        static const unsigned int bP_ar = {self.bP_ar};
        static const unsigned int bP_dc = {self.bP_dc};
        using warp_layout_ar = {_layout_type(self.warp_layout_ar)};
        using warp_layout_arb = {_layout_type(self.warp_layout_arb)};
    }};"""

    def cache_key(self) -> str:
        content = self.to_header()
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BwdDBCompParams:
        """Deserialize from a dictionary."""
        d = dict(d)
        for key in ("warp_layout_ar", "warp_layout_arb"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        return cls(**d)

    @classmethod
    def safe_defaults(cls) -> BwdDBCompParams:
        """Conservative defaults that compile for all valid shapes."""
        return cls(
            bM=32, bK=32,
            bP_a=2, bP_ar=2, bP_dc=2,
            warp_layout_ar=(2,), warp_layout_arb=(2,),
        )


@dataclasses.dataclass(frozen=True)
class BwdDAdRCompParams:
    """Performance parameters for the backward fused dA+dR kernel (producer-consumer).

    Producer: loads dC + B, computes dAR = dC @ B^T via MMA (heavy, gs reduction).
    Consumer: loads A + R, computes AR via MMA, then dA/dR via MMA (light).
    """

    bM: int = 32           # M tile size
    n_buf_slots: int = 16  # Number of dR buffer slots (controls atomic contention)
    bP_dc_b: int = 2       # Pipeline depth: dC+B async loads (producer)
    bP_dar: int = 2        # Pipeline depth: dAR producer→consumer
    bP_a_r: int = 2        # Pipeline depth: A+R async loads (consumer)
    # Heavy GEMM (gs reduction, dC@B^T producer)
    warp_layout_arb: tuple[int, ...] = (2,)    # 2 warps
    # Light MMA ops (rs reduction, A@R^T consumer)
    warp_layout_ar: tuple[int, ...] = (2,)    # 2 warps

    def to_header(self) -> str:
        """Generate the C++ struct definition."""
        return f"""\
    struct BwdDAdRParams {{
        static const unsigned int bM = {self.bM};
        static const unsigned int n_buf_slots = {self.n_buf_slots};
        static const unsigned int bP_dc_b = {self.bP_dc_b};
        static const unsigned int bP_dar = {self.bP_dar};
        static const unsigned int bP_a_r = {self.bP_a_r};
        using warp_layout_arb = {_layout_type(self.warp_layout_arb)};
        using warp_layout_ar = {_layout_type(self.warp_layout_ar)};
    }};"""

    def cache_key(self) -> str:
        content = self.to_header()
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BwdDAdRCompParams:
        """Deserialize from a dictionary."""
        d = dict(d)
        for key in ("warp_layout_arb", "warp_layout_ar"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        return cls(**d)

    @classmethod
    def safe_defaults(cls) -> BwdDAdRCompParams:
        """Conservative defaults that compile for all valid shapes."""
        return cls(
            bM=32, n_buf_slots=8,
            bP_dc_b=2, bP_dar=2, bP_a_r=2,
            warp_layout_arb=(2,), warp_layout_ar=(2,),
        )
