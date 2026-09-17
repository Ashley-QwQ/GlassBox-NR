"""D36-only adapter for the six ordinary decoder blocks.

The arithmetic is reused from the already frozen D22 candidate, but the
accepted block/function/origin/grid domain is deliberately finite.  This file
does not add transition blocks or claim arbitrary block support.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
ROOT = ROUND.parents[1]
E = ROOT / "experiments"

for rel in (
    "glassbox_nr/kernels/encoder_windows",
):
    path = ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import d17_backend as D17  # noqa: E402
import d22_backend as D22  # noqa: E402


class D36Error(D22.Unsupported):
    pass


SPECS = {
    57: dict(head_count=4, side=40, origin_xy=(0, -4), grid_xy=(5, 6),
            function="cc_tinlayout_fused_swin_4h_128_4_chained_fp8", launch=142,
            layout="TIN128"),
    58: dict(head_count=4, side=40, origin_xy=(0, 0), grid_xy=(5, 5),
            function="cc_tinlayout_fused_swin_4h_128_4_chained_fp8", launch=143,
            layout="TIN128"),
    59: dict(head_count=4, side=40, origin_xy=(-4, -4), grid_xy=(6, 6),
            function="cc_tinlayout_fused_swin_4h_128_4_chained_fp8", launch=144,
            layout="TIN128"),
    60: dict(head_count=4, side=40, origin_xy=(-4, 0), grid_xy=(6, 5),
            function="cc_tinlayout_fused_swin_4h_128_4_chained_fp8", launch=145,
            layout="TIN128"),
    61: dict(head_count=4, side=40, origin_xy=(0, -4), grid_xy=(5, 6),
            function="cc_tinlayout_fused_swin_4h_128_4_outview_wait_fp8", launch=146,
            layout="TIN128"),
    63: dict(head_count=2, side=80, origin_xy=(-4, -4), grid_xy=(11, 11),
            function="cc_tinlayout_fused_swin_2h_64_2_chained_fp8", launch=148,
            layout="TIN64"),
    64: dict(head_count=2, side=80, origin_xy=(-4, 0), grid_xy=(11, 10),
            function="cc_tinlayout_fused_swin_2h_64_2_chained_fp8", launch=149,
            layout="TIN64"),
    65: dict(head_count=2, side=80, origin_xy=(0, -4), grid_xy=(10, 11),
            function="cc_tinlayout_fused_swin_2h_64_2_outview_wait_fp8", launch=150,
            layout="TIN64"),
}

SINGLE_BLOCKS = (58, 59, 60, 61, 64, 65)
COMPAT_BLOCKS = (57, 63)


def _copy_spec(spec):
    return {k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in spec.items()}


class D36OrdinaryBlock:
    """Finite D36 block wrapper around the D22 arithmetic body."""

    def __init__(self, block: int, info: dict, weights, tables):
        if block not in SPECS:
            raise D36Error(f"D36 does not support B{block}")
        spec = SPECS[block]
        actual = {
            "block": info.get("block"),
            "function": info.get("function"),
            "launch": info.get("launch"),
            "grid_xy": tuple(info.get("grid_xyz", [])[:2]),
            "block_dim": tuple(info.get("block_dim", [])),
            "params_bytes": info.get("params_bytes"),
            "extent_wh": tuple(info.get("extent_wh", [])),
            "origin_xy": tuple(info.get("origin_xy", [])),
            "input_shape": tuple(info.get("input_shape", [])),
            "output_shape": tuple(info.get("output_shape", [])),
            "input_layout": info.get("input_layout"),
            "output_layout": info.get("output_layout"),
            "transition": info.get("transition"),
        }
        expected = {
            "block": block, "function": spec["function"], "launch": spec["launch"],
            "grid_xy": spec["grid_xy"], "block_dim": (32, spec["head_count"], 1),
            "params_bytes": 88, "extent_wh": (spec["side"], spec["side"]),
            "origin_xy": spec["origin_xy"],
            "input_shape": (spec["side"], spec["side"], 32 * spec["head_count"]),
            "output_shape": (spec["side"], spec["side"], 32 * spec["head_count"]),
            "input_layout": "TIN_candidate", "output_layout": "TIN_candidate",
            "transition": False,
        }
        if actual != expected:
            raise D36Error(f"B{block} interface mismatch: actual={actual!r} expected={expected!r}")
        self.block = block
        self.spec = _copy_spec(spec)
        self.info = info
        self.backend = D22.BranchedWindowSm120(
            block, spec["head_count"], spec["side"], spec["origin_xy"], weights, tables,
        )
        self.backend.validate_invocation(
            block_index=block,
            head_count=spec["head_count"],
            origin_xy=spec["origin_xy"],
            grid_xy=spec["grid_xy"],
            shape=(spec["side"], spec["side"], 32 * spec["head_count"]),
        )

    def decode(self, raw: bytes) -> np.ndarray:
        if self.spec["layout"] == "TIN128":
            return D22.tin128_decode(raw)
        return D17.tin64_decode(raw)

    def encode(self, codes: np.ndarray) -> bytes:
        if self.spec["layout"] == "TIN128":
            return D22.tin128_encode(codes)
        return D17.tin64_encode(codes)

    def run_raw(self, raw: bytes) -> tuple[bytes, dict]:
        codes = self.decode(raw)
        result = self.backend.run_codes(codes)
        pred = self.encode(result["Y"])
        expected = self.spec["side"] * self.spec["side"] * (32 * self.spec["head_count"])
        if len(pred) != expected:
            raise D36Error(f"B{self.block} produced {len(pred)} bytes, expected {expected}")
        detail = self.backend.describe()
        detail.update({
            "d36_block": self.block,
            "function": self.spec["function"],
            "launch": self.spec["launch"],
            "layout": self.spec["layout"],
            "grid_xy": list(self.spec["grid_xy"]),
            "origin_xy": list(self.spec["origin_xy"]),
            "canvas_hw": list(result.get("canvas_hw", ())),
            "n_windows": result.get("n_windows"),
            "tail_publication": "ordinary TIN publication; outview_wait is a launch/function variant",
        })
        return pred, detail
