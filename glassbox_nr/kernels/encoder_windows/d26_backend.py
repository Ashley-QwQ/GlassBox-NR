"""D26 isolated sm_120 candidate for the branched_window family, COPIED from D22's BranchedWindowSm120 and
parameterised to 8 heads.  The ONLY change to the arithmetic path is the supported (heads, side) domain:
    D22: heads in (2, 4), side in (80, 40)        D26: (heads, side) in {(2, 80), (4, 40), (8, 20)}
Everything else is D22 verbatim: C = 32*heads; FFN expansion K=C in 32-channel steps; branch projection K=128
per branch; output projection K=32 per head with z carried head 0..heads-1; QKV K=C in 32-channel steps seeded
zero; per-head normalise with logical bias[h] and attn_scale[h]; AV 2 steps; tail K=C in 32-channel steps seeded
half(decode(QZ)*cA). Canvas placement (zero pad participates) and crop as D19/D22.
Gate G0 re-runs this copy at the verified 2-head B5 (vs D17) and 4-head B9 (vs D22 frozen) configurations.

Layout codecs: INPVIEW256 (generic formula, D25 writer-side) and TIN256 (closed form wo_spec.tin_map(20,20,256),
which reproduces the sm_120-verified TIN64 / TIN128 maps; C=256 is a formula extension).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for p in (HERE,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import d22_backend as B9  # noqa: E402
import d25_backend as B14  # noqa: E402
import wo_spec  # noqa: E402

S, M, BW, B5 = B9.S, B9.M, B9.BW, B9.B5
Unsupported = S.UnsupportedByBackend
HEAD_SIDE = {2: 80, 4: 40, 8: 20}
splits32, shapes_for = B9.splits32, B9.shapes_for
_TIN256 = []


def tin256_map():
    if not _TIN256:
        m = np.asarray(wo_spec.tin_map(20, 20, 256), np.int64)
        m.setflags(write=False)
        _TIN256.append(m)
    return _TIN256[0]


def tin256_encode(codes):
    codes = np.asarray(codes, np.uint8)
    if codes.shape != (20, 20, 256):
        raise Unsupported('TIN256 fixed at 20x20x256')
    return codes.reshape(-1)[tin256_map()].copy().tobytes()


def tin256_decode(raw):
    x = np.frombuffer(raw, np.uint8)
    if x.size != 102400:
        raise Unsupported('TIN256 buffer must be 102400 bytes')
    out = np.zeros(102400, np.uint8)
    out[tin256_map()] = x
    return out.reshape(20, 20, 256)


def inpview256_encode(codes):
    codes = np.asarray(codes, np.uint8)
    if codes.shape != (20, 20, 256):
        raise Unsupported('INPVIEW256 fixed at 20x20x256')
    return B14.inpview_encode(codes)


def inpview256_decode(raw):
    return B14.inpview_decode(raw, 20, 20, 256)


class BranchedWindowSm120N:
    architecture = S.TARGET_ARCHITECTURE
    accumulator_model = S.ACCUMULATOR_MODEL
    window_size = 8

    def __init__(self, block_index, heads, side, origin_xy, weights, tables, *, tail_seed='published_qz'):
        if HEAD_SIDE.get(heads) != side:
            raise Unsupported('D26 candidate covers (heads, side) in %s only' % sorted(HEAD_SIDE.items()))
        if tail_seed not in ('published_qz', 'unquantised_z'):
            raise Unsupported('tail_seed')
        self.block_index, self.head_count, self.side, self.origin_xy = block_index, heads, side, tuple(origin_xy)
        self.C = 32 * heads
        self.tables, self.tail_seed = tables, tail_seed
        BW.canvas_grid(self.origin_xy, side)
        prefix = 'block%d.layer0.' % block_index
        self._w = {}
        for name, shape in shapes_for(heads).items():
            key = prefix + name
            if key not in weights:
                raise Unsupported('missing %s' % key)
            arr = np.array(S._as_numpy(weights[key]), dtype=np.float64, copy=True)
            if tuple(arr.shape) != shape:
                raise Unsupported('%s shape %s != %s' % (key, arr.shape, shape))
            arr.setflags(write=False)
            self._w[name] = arr
        if M.uses_fragment_swizzle(block_index, heads):
            raise Unsupported('model says fragment bias order for (%d, %d heads); D26 froze logical' % (block_index, heads))

    def validate_invocation(self, *, block_index, head_count, origin_xy, grid_xy, shape):
        gx, gy = BW.canvas_grid(self.origin_xy, self.side)
        want = (self.block_index, self.head_count, self.origin_xy, (gx, gy), (self.side, self.side, self.C))
        got = (block_index, head_count, tuple(origin_xy), tuple(grid_xy), tuple(shape))
        if got != want:
            raise Unsupported('refusing %s; verified-for %s' % (got, want))

    def _ffn(self, x2d):
        w, C, H = self._w, self.C, self.head_count
        z = S._half(x2d * w['ffn_cos_skip'])
        for h in range(H):
            w_exp = w['ffn_expand_weight'][h].transpose(1, 2, 0, 3).reshape(C, 128)
            e = B5._chain(x2d, w_exp, None, splits32(C))
            g = S.decode_codes(S.feed_forward_gate(e)).astype(np.float64).reshape(-1, 128)
            s = B5._chain(g, w['ffn_branch_projection_weight'][h].reshape(128, 32), None, B5.K128)
            u = S.decode_codes(S.quantise_codes(s)).astype(np.float64)
            z = S.qmma_rowwise(u, np.ascontiguousarray(w['ffn_output_projection_weight'][32 * h:32 * h + 32]), z)
        return z

    def run_codes(self, codes):
        C, side = self.C, self.side
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != (side, side, C):
            raise Unsupported('uint8 [%d,%d,%d] E4M3 codes required' % (side, side, C))
        if np.any((codes & 0x7F) == 0x7F):
            raise Unsupported('E4M3 NaN codes in input')
        w = self._w
        ox, oy = self.origin_xy
        gx, gy = BW.canvas_grid(self.origin_xy, side)
        Hc, Wc = gy * 8, gx * 8
        rows, cols = slice(-oy, -oy + side), slice(-ox, -ox + side)
        canvas = np.zeros((Hc, Wc, C), np.float64)
        canvas[rows, cols] = S.decode_codes(codes).astype(np.float64)
        z = self._ffn(canvas.reshape(-1, C))
        qz = S.quantise_codes(z)
        xq = S.decode_codes(qz).astype(np.float64)
        qkv = B5._chain(xq, w['qkv_weight'], None, splits32(C))
        heads = []
        for h in range(self.head_count):
            c = slice(32 * h, 32 * h + 32)
            q, _ = S.normalise(qkv[:, 0:C][:, c].reshape(Hc, Wc, 32), self.tables, float(w['attn_scale'][h]))
            k, _ = S.normalise(qkv[:, C:2 * C][:, c].reshape(Hc, Wc, 32), self.tables)
            v = qkv[:, 2 * C:3 * C][:, c].reshape(Hc, Wc, 32)
            parts = []
            for t in (q, k, v):
                pub = S.decode_codes(S.quantise_codes(t))
                parts.append(np.asarray(M.partition_windows(
                    torch.from_numpy(np.ascontiguousarray(pub, np.float32)).unsqueeze(0), 8).numpy(), np.float64))
            qw, kw, vw = parts
            score = S.qmma_batched(qw, np.ascontiguousarray(np.transpose(kw, (0, 2, 1))),
                                   np.broadcast_to(w['attn_bias'][h].astype(np.float16), (qw.shape[0], 64, 64)))
            pw = S.decode_codes(S.quantise_codes(S.published_attention_weights(score, self.tables))).astype(np.float64)
            acc = None
            for keys in (slice(0, 32), slice(32, 64)):
                acc = S.qmma_batched(np.ascontiguousarray(pw[:, :, keys]), np.ascontiguousarray(vw[:, keys, :]), acc)
            av = M.e4m3_round_trip(torch.from_numpy(np.ascontiguousarray(acc, np.float32)))
            heads.append(M.reverse_windows(av, batch_count=1, height=Hc, width=Wc, window_size=8)[0].numpy())
        av = np.concatenate(heads, axis=-1).astype(np.float64).reshape(-1, C)
        zfac = xq if self.tail_seed == 'published_qz' else z.astype(np.float64)
        seed = S._half(zfac * w['attn_cos_skip'])
        y = B5._chain(av, w['projection_weight'], seed, splits32(C)).reshape(Hc, Wc, C)
        crop = (rows, cols)
        return dict(Y=S.quantise_codes(y[crop]), QZ=qz.reshape(Hc, Wc, C)[crop], AV=S.quantise_codes(av.reshape(Hc, Wc, C)[crop]),
                    Y_half=y[crop], canvas_hw=(Hc, Wc), n_windows=int(gx * gy))

    def describe(self):
        return dict(backend='D26 BranchedWindowSm120N (isolated copy of D22, unregistered)', supported_heads_side=HEAD_SIDE,
                    block_index=self.block_index, head_count=self.head_count, side=self.side, origin_xy=self.origin_xy,
                    tail_seed=self.tail_seed,
                    k_splits=dict(expansion=splits32(self.C), branch=B5.K128, output_projection='K32 per head, z carried',
                                  qkv=splits32(self.C), av=B5.K64, tail=splits32(self.C)))
