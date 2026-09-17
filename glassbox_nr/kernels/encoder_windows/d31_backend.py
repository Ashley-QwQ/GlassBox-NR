"""D31 candidate: D26 eight-head body plus D25-style half-precision downsample tail.

The body is imported read-only from the D26 isolated copy.  The B22-specific adaptation
is explicit: block 22 uses (heads=8, side=20, origin=(0,-4)); its cropped unpublished
half output is zero-padded at the spatial end to 24x24, row-pair 2x2 pooled, published
to E4M3, then projected by block22.layer0.weight0 [256,512] in eight K=32 steps with
the existing zero-seeded QMMA-chain helper.  The projected E4M3 codes are encoded for
the launch-25 INPVIEW512 consumer.  No gate/MUFU/W27 primitive is changed.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D26 = HERE
D25 = HERE
if str(D26) not in sys.path:
    sys.path.insert(0, str(D26))
if str(D25) not in sys.path:
    sys.path.insert(0, str(D25))
import d26_backend as BODY  # noqa: E402
import d25_backend as TAIL  # noqa: E402

S = BODY.S
Unsupported = S.UnsupportedByBackend


def inpview_encode(codes):
    codes = np.asarray(codes, np.uint8)
    if codes.ndim != 3:
        raise Unsupported("INPVIEW requires HWC")
    h, w, c = codes.shape
    if c % 16:
        raise Unsupported("INPVIEW channel count must be a multiple of 16")
    ch = np.arange(c)
    order = (ch // 16) * 16 + ((ch % 16) // 4) * 2 + ch % 2 + ((ch % 4) // 2) * 8
    return codes[..., order].reshape(h, w, c // 16, 16).transpose(2, 0, 1, 3).copy().tobytes()


def inpview_decode(raw, h, w, c):
    x = np.frombuffer(raw, np.uint8)
    if x.size != h * w * c or c % 16:
        raise Unsupported("INPVIEW buffer length or channels invalid")
    phys = x.reshape(c // 16, h, w, 16).transpose(1, 2, 0, 3).reshape(h, w, c)
    ch = np.arange(c)
    order = (ch // 16) * 16 + ((ch % 16) // 4) * 2 + ch % 2 + ((ch % 4) // 2) * 8
    out = np.empty_like(phys)
    out[..., order] = phys
    if inpview_encode(out) != bytes(raw):
        raise Unsupported("INPVIEW round trip failed")
    return out


def b22_pool_tail(y_half, weight0):
    """Pool unpublished [20,20,256] half output after explicit end padding."""
    y = np.asarray(y_half, np.float16)
    if y.shape != (20, 20, 256):
        raise Unsupported("B22 unpublished half output must be [20,20,256]")
    padded = np.pad(y, ((0, 4), (0, 4), (0, 0)), mode="constant")
    pool, projected = TAIL.pool_tail(padded, weight0, grouping="row_pairs", source="unpublished")
    if pool.shape != (12, 12, 256) or projected.shape != (12, 12, 512):
        raise Unsupported("B22 pool tail produced unexpected shape")
    return pool, projected


class B22Sm120:
    block_index = 22
    head_count = 8
    side = 20
    origin_xy = (0, -4)

    def __init__(self, weights, tables, *, tail_seed="published_qz"):
        self.body = BODY.BranchedWindowSm120N(22, 8, 20, self.origin_xy, weights, tables, tail_seed=tail_seed)
        key = "block22.layer0.weight0"
        if key not in weights:
            raise Unsupported("missing %s" % key)
        w0 = np.array(S._as_numpy(weights[key]), dtype=np.float64, copy=True)
        if w0.shape != (256, 512):
            raise Unsupported("weight0 %s != (256,512)" % (w0.shape,))
        w0.setflags(write=False)
        self.weight0 = w0

    def validate_invocation(self, *, block_index, head_count, origin_xy, grid_xy, shape):
        self.body.validate_invocation(block_index=block_index, head_count=head_count,
                                      origin_xy=origin_xy, grid_xy=grid_xy, shape=shape)
        if tuple(origin_xy) != self.origin_xy or tuple(shape) != (20, 20, 256):
            raise Unsupported("B22 invocation outside frozen domain")

    def run_codes(self, codes):
        raw = np.asarray(codes)
        if raw.dtype != np.uint8 or raw.shape != (20, 20, 256):
            raise Unsupported("B22 requires uint8 [20,20,256] E4M3 input")
        if np.any((raw & 0x7f) == 0x7f):
            raise Unsupported("E4M3 NaN in B22 input")
        body = self.body.run_codes(raw)
        pool_half, pooled = b22_pool_tail(body["Y_half"], self.weight0)
        return dict(body=body, pool_half=pool_half, pooled=pooled,
                    full_physical=BODY.tin256_encode(body["Y"]),
                    pooled_physical=inpview_encode(pooled),
                    padded_half_shape=[24, 24, 256],
                    pool_shape=list(pool_half.shape), pooled_shape=list(pooled.shape))

    def describe(self):
        return {
            "backend": "D31 B22Sm120 (isolated; D26 body + D25 tail)",
            "block_index": 22,
            "head_count": 8,
            "side": 20,
            "origin_xy": [0, -4],
            "body": self.body.describe(),
            "tail": {
                "input": "unpublished half Y_half",
                "pad_spatial_end": [24, 24],
                "pool_grouping": "row_pairs",
                "pool_factor": "2x2 average, float16 intermediate additions",
                "publish_before_weight0": True,
                "weight0_shape": [256, 512],
                "weight0_seed": "RZ",
                "weight0_k_steps": 8,
                "output_layout": "INPVIEW512",
            },
        }
