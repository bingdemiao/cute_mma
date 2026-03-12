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
    warp_layout1: tuple[int, ...] = (2,)
    warp_layout2: tuple[int, ...] = (2, 2)

    def to_header(self) -> str:
        """Generate the oft_config.hpp content."""

        def _layout_type(shape: tuple[int, ...]) -> str:
            if len(shape) == 1:
                return f"cute::Layout<cute::Shape<cute::Int<{shape[0]}>>>"
            inner = ", ".join(f"cute::Int<{s}>" for s in shape)
            return f"cute::Layout<cute::Shape<{inner}>>"

        return f"""\
#pragma once
#include <cute/tensor.hpp>
namespace cute {{

    struct CurrCompParams {{
        static const unsigned int bM = {self.bM};
        static const unsigned int bN = {self.bN};
        static const unsigned int bK = {self.bK};
        static const unsigned int c_width = {self.c_width};
        static const unsigned int bP_a_r = {self.bP_a_r};
        static const unsigned int bP_ar = {self.bP_ar};
        static const unsigned int bP_b = {self.bP_b};
        using warp_layout1 = {_layout_type(self.warp_layout1)};
        using warp_layout2 = {_layout_type(self.warp_layout2)};
    }};

}}
"""

    def cache_key(self) -> str:
        """Short hash identifying this parameter combination."""
        content = self.to_header()
        return hashlib.sha256(content.encode()).hexdigest()[:12]
