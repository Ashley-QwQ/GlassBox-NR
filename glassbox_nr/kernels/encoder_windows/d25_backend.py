"""D25 isolated sm_120 candidate for B14 (4h_128_4_ds_wait_fp8).

Body: D22 BranchedWindowSm120 unchanged at (block 14, heads 4, side 40, origin (-4,-4)) — the same
class and configuration family that D24 verified on B10-B13. Tail (new, generic): pool the cropped
UNPUBLISHED Y half with D21's pool_half, publish E4M3, weight0 matmul in 32-channel QMMA steps with
C=RZ, publish. Codec: generic INPVIEW (channel order (ch//16)*16+((ch%16)//4)*2+ch%2+((ch%4)//2)*8,
C/16 planes of 16). The generic tail and codec are gated against D13 (B4, 64 ch) and D21 (B8, 128 ch)
known answers before B14 (256 ch) is predicted.
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
import d22_backend as B9  # noqa: E402
import d21_backend as B8  # noqa: E402
import d17_backend as B5  # noqa: E402

S, M = B9.S, B9.M
Unsupported = S.UnsupportedByBackend


def inpview_order(C):
    ch = np.arange(C)
    return (ch // 16) * 16 + ((ch % 16) // 4) * 2 + ch % 2 + ((ch % 4) // 2) * 8


def inpview_encode(codes):
    codes = np.asarray(codes, np.uint8)
    H, W, C = codes.shape
    if C % 16:
        raise Unsupported('INPVIEW needs C multiple of 16')
    return codes[..., inpview_order(C)].reshape(H, W, C // 16, 16).transpose(2, 0, 1, 3).copy().tobytes()


def inpview_decode(raw, H, W, C):
    x = np.frombuffer(raw, np.uint8)
    if x.size != H * W * C:
        raise Unsupported('INPVIEW buffer %d bytes, expected %d' % (x.size, H * W * C))
    phys = x.reshape(C // 16, H, W, 16).transpose(1, 2, 0, 3).reshape(H, W, C)
    out = np.empty_like(phys)
    out[..., inpview_order(C)] = phys
    if inpview_encode(out) != bytes(raw):
        raise Unsupported('INPVIEW round trip failed')
    return out


def pool_tail(y_half, weight0, grouping='row_pairs', source='unpublished'):
    """y_half [H,W,C] -> (pool half [H/2,W/2,C], projected codes [H/2,W/2,N]); weight0 [C,N]; K=C in 32-channel steps."""
    y = np.asarray(y_half, np.float16)
    if source == 'published':                       # pre-registered diagnostic only
        y = S.decode_codes(S.quantise_codes(y)).astype(np.float16)
    pool = B8.pool_half(y, grouping)
    h, w, C = pool.shape
    W0 = np.asarray(weight0, np.float64)
    if W0.shape[0] != C:
        raise Unsupported('weight0 %s does not take %d channels' % (W0.shape, C))
    pin = S.decode_codes(S.quantise_codes(pool)).astype(np.float64).reshape(-1, C)
    proj = B5._chain(pin, W0, None, B9.splits32(C)).reshape(h, w, W0.shape[1])
    return pool, S.quantise_codes(proj)


class B14Sm120:
    block_index, head_count, side, origin_xy = 14, 4, 40, (-4, -4)

    def __init__(self, weights, tables, *, grouping='row_pairs', source='unpublished'):
        self.body = B9.BranchedWindowSm120(14, 4, 40, (-4, -4), weights, tables)
        key = 'block14.layer0.weight0'
        if key not in weights:
            raise Unsupported('missing %s' % key)
        w0 = np.array(S._as_numpy(weights[key]), dtype=np.float64, copy=True)
        if w0.shape != (128, 256):
            raise Unsupported('weight0 shape %s != (128, 256)' % (w0.shape,))
        w0.setflags(write=False)
        self.weight0, self.grouping, self.source = w0, grouping, source

    def validate_invocation(self, **kw):
        self.body.validate_invocation(**kw)

    def run_codes(self, codes):
        r = self.body.run_codes(codes)                  # canvas 48x48, cropped to 40x40
        pool, pooled = pool_tail(r['Y_half'], self.weight0, self.grouping, self.source)
        r.update(full=r['Y'], pooled=pooled, pool_half=pool)
        return r

    def describe(self):
        d = self.body.describe()
        d.update(backend='D25 B14Sm120 (isolated, unregistered)', tail=dict(pool_grouping=self.grouping, pool_source=self.source,
                                                                           weight0_splits=B9.splits32(128), full_layout='TIN128',
                                                                           pooled_layout='INPVIEW256 20x20x256'))
        return d
