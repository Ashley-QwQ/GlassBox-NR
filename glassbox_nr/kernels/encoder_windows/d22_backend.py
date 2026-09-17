"""D22 isolated sm_120 candidate for the branched_window family, parameterised by head count,
side and window origin (B5/B6/B7 at 2 heads x 80, B9 at 4 heads x 40).

All numeric primitives: mlxdlss.sm120_b1 (unchanged) through d17_backend helpers (_chain, gate,
norm, weights, publish). Channel-width dependent pieces are the ONLY generalisations:
  C = 32*heads; FFN expansion K=C split into 32-channel steps; branch projection K=128 per branch;
  output projection K=32 per head with z carried head 0..heads-1; QKV K=C split into 32-channel
  steps; per-head attention on 32 dims with logical bias[h]; AV 2 steps; tail K=C 32-channel steps
  seeded half(decode(QZ)*cA). Canvas placement/crop as D19.
Gate G0 runs this class at the B5 (heads 2, origin 0) and B6 (heads 2, origin -4,-4) configurations
against D17 / D19. Layout codecs: INPVIEW128 (D21) and TIN128 (recorded sm75 map).
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
import d17_backend as B5  # noqa: E402
import d19_backend as BW  # noqa: E402
import d21_backend as B8  # noqa: E402

S, M = B5.S, B5.M
Unsupported = S.UnsupportedByBackend
TIN128_MAP = np.load(ROOT / 'data/maps/TIN128-physical-to-logical.npy')


def tin128_encode(codes):
    codes = np.asarray(codes, np.uint8)
    if codes.shape != (40, 40, 128):
        raise Unsupported('TIN128 fixed at 40x40x128')
    return codes.reshape(-1)[TIN128_MAP].copy().tobytes()


def tin128_decode(raw):
    x = np.frombuffer(raw, np.uint8)
    if x.size != 204800:
        raise Unsupported('TIN128 buffer must be 204800 bytes')
    out = np.zeros(204800, np.uint8)
    out[TIN128_MAP] = x
    return out.reshape(40, 40, 128)


inpview128_encode, inpview128_decode = B8.inpview128_encode, B8.inpview128_decode


def splits32(n):
    return [(i, i + 32) for i in range(0, n, 32)]


def shapes_for(heads):
    C = 32 * heads
    return {'ffn_cos_skip': (C,), 'ffn_expand_weight': (heads, 4, heads, 32, 32),
            'ffn_branch_projection_weight': (heads, 4, 32, 32), 'ffn_output_projection_weight': (C, C),
            'qkv_weight': (C, 3 * C), 'attn_bias': (heads, 64, 64), 'attn_scale': (heads,),
            'projection_weight': (C, C), 'attn_cos_skip': (C,)}


class BranchedWindowSm120:
    architecture = S.TARGET_ARCHITECTURE
    accumulator_model = S.ACCUMULATOR_MODEL
    window_size = 8

    def __init__(self, block_index, heads, side, origin_xy, weights, tables, *, tail_seed='published_qz'):
        if heads not in (2, 4) or side not in (80, 40):
            raise Unsupported('D22 candidate covers heads 2 (side 80) and 4 (side 40) only')
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
            raise Unsupported('model says fragment bias order for (%d, %d heads); D22 froze logical' % (block_index, heads))

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
        return dict(backend='D22 BranchedWindowSm120 (isolated, unregistered)', block_index=self.block_index,
                    head_count=self.head_count, side=self.side, origin_xy=self.origin_xy, tail_seed=self.tail_seed,
                    k_splits=dict(expansion=splits32(self.C), branch=B5.K128, output_projection='K32 per head, z carried',
                                  qkv=splits32(self.C), av=B5.K64, tail=splits32(self.C)))
