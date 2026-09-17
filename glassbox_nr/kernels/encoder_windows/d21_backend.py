"""D21 isolated sm_120 candidate for B8 (2h_64_2_ds_wait_fp8).

Body: D19 BranchedWindow2hSm120, unchanged, reached through a subclass that only sets
block index 8, origin (0,-4) and block8 weights (D19's module-level BLOCKS dict is NOT
modified). Tail (new): pool on the cropped Y half, publish, weight0 matmul, publish.
`pool_tail` is generic in channels / K splits so gate G2 can run it at D13's B4 config.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for p in (HERE,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import d19_backend as BW  # noqa: E402
import d17_backend as B5  # noqa: E402

S, M = BW.S, BW.M
Unsupported = S.UnsupportedByBackend
_CH128 = np.arange(128)
_ORDER128 = (_CH128 // 16) * 16 + ((_CH128 % 16) // 4) * 2 + _CH128 % 2 + ((_CH128 % 4) // 2) * 8


def inpview128_encode(codes):
    codes = np.asarray(codes, np.uint8)
    if codes.shape != (40, 40, 128):
        raise Unsupported('pooled layout fixed at 40x40x128')
    return codes[..., _ORDER128].reshape(40, 40, 8, 16).transpose(2, 0, 1, 3).copy().tobytes()


def inpview128_decode(raw):
    x = np.frombuffer(raw, np.uint8)
    if x.size != 204800:
        raise Unsupported('pooled buffer must be 204800 bytes')
    phys = x.reshape(8, 40, 40, 16).transpose(1, 2, 0, 3).reshape(40, 40, 128)
    out = np.empty_like(phys)
    out[..., _ORDER128] = phys
    if inpview128_encode(out) != bytes(raw):
        raise Unsupported('INPVIEW128 round trip failed')
    return out


def pool_half(h, grouping='row_pairs'):
    h = np.asarray(h, np.float16)
    if grouping == 'row_pairs':            # H2 (as D13 B4)
        a = (h[0::2, 0::2].astype(np.float64) + h[0::2, 1::2].astype(np.float64)).astype(np.float16)
        b = (h[1::2, 0::2].astype(np.float64) + h[1::2, 1::2].astype(np.float64)).astype(np.float16)
    elif grouping == 'column_pairs':       # DIAG1 only
        a = (h[0::2, 0::2].astype(np.float64) + h[1::2, 0::2].astype(np.float64)).astype(np.float16)
        b = (h[0::2, 1::2].astype(np.float64) + h[1::2, 1::2].astype(np.float64)).astype(np.float16)
    else:
        raise Unsupported('grouping')
    return ((a.astype(np.float64) + b.astype(np.float64)).astype(np.float16).astype(np.float64) * 0.25).astype(np.float16)


def pool_tail(y_half, weight0, splits, grouping='row_pairs', source='unpublished'):
    """y_half [H,W,C] -> (pool_half [H/2,W/2,C], projected codes [H/2,W/2,N])."""
    y = np.asarray(y_half, np.float16)
    if source == 'published':              # DIAG2 only
        y = S.decode_codes(S.quantise_codes(y)).astype(np.float16)
    pool = pool_half(y, grouping)
    hh, ww, c = pool.shape
    pin = S.decode_codes(S.quantise_codes(pool)).astype(np.float64).reshape(-1, c)
    proj = B5._chain(pin, np.asarray(weight0, np.float64), None, splits).reshape(hh, ww, -1)
    return pool, S.quantise_codes(proj)


class B8Sm120(BW.BranchedWindow2hSm120):
    """D19 body at block 8 / origin (0,-4) plus the ds_wait tail."""

    def __init__(self, weights, tables, *, grouping='row_pairs', source='unpublished'):
        self.block_index = 8
        self.origin_xy = (0, -4)
        self.pad_mode = 'canvas_before_ffn'
        self.tables = tables
        self.grouping, self.source = grouping, source
        self._w = {}
        shapes = dict(B5.SHAPES, weight0=(64, 128))
        for name, shape in shapes.items():
            key = 'block8.layer0.' + name
            if key not in weights:
                raise Unsupported('missing %s' % key)
            arr = np.array(S._as_numpy(weights[key]), dtype=np.float64, copy=True)
            if tuple(arr.shape) != shape:
                raise Unsupported('%s shape %s != %s' % (key, arr.shape, shape))
            arr.setflags(write=False)
            self._w[name] = arr
        if M.uses_fragment_swizzle(8, 2):
            raise Unsupported('model says fragment bias order for block 8')

    def run_codes(self, codes):
        r = super().run_codes(codes)                  # D19 body, canvas 88 x 80, cropped to 80 x 80
        pool, pooled = pool_tail(r['Y_half'], self._w['weight0'], B5.K64, self.grouping, self.source)
        r.update(full=r['Y'], pooled=pooled, pool_half=pool)
        return r

    def describe(self):
        d = super().describe()
        d.update(backend='D21 B8Sm120 (isolated, unregistered)', grouping=self.grouping, pool_source=self.source,
                 weight0_splits=B5.K64)
        return d
