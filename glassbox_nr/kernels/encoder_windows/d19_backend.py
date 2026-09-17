"""D19 isolated sm_120 candidate for the 2-head branched_window blocks B5/B6/B7.

Arithmetic and helpers are imported unchanged from D17 (d17_backend) and the core.
The only new thing is the window canvas (chain-and-hypotheses.json H2): X is placed
on a zero-filled (gy*8, gx*8) canvas derived from window_origin, the whole body runs
on the canvas, and Y is cropped back. At origin (0,0) the canvas is the buffer itself,
which is what gate G0 compares against D17's B5Sm120.

Substantive differences from d17_backend.B5Sm120.run_codes:
  1. canvas placement / crop (origin-derived), 2. canvas extents in window partition
  and reverse, 3. block index and weight prefix are parameters. Stage calls unchanged.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
D17S = Path(__file__).resolve().parent
if str(D17S) not in sys.path:
    sys.path.insert(0, str(D17S))
import d17_backend as B5  # noqa: E402

S, M = B5.S, B5.M
Unsupported = S.UnsupportedByBackend
BLOCKS = {5: (0, 0), 6: (-4, -4), 7: (-4, 0)}      # origin_xy, as read from params word5


def canvas_grid(origin_xy, side=80):
    ox, oy = origin_xy
    if ox not in (0, -4) or oy not in (0, -4):
        raise Unsupported('origin %s outside {0,-4}^2' % (origin_xy,))
    return -(-(side - ox) // 8), -(-(side - oy) // 8)      # (gx, gy)


class BranchedWindow2hSm120:
    architecture = S.TARGET_ARCHITECTURE
    accumulator_model = S.ACCUMULATOR_MODEL
    head_count = 2
    window_size = 8

    def __init__(self, block_index, weights, tables, *, pad_mode='canvas_before_ffn'):
        if block_index not in BLOCKS:
            raise Unsupported('D19 candidate covers blocks 5, 6, 7 only')
        if pad_mode not in ('canvas_before_ffn', 'pad_after_ffn'):
            raise Unsupported('pad_mode')
        self.block_index = block_index
        self.origin_xy = BLOCKS[block_index]
        self.pad_mode = pad_mode
        self.tables = tables
        prefix = 'block%d.layer0.' % block_index
        self._w = {}
        for name, shape in B5.SHAPES.items():
            key = prefix + name
            if key not in weights:
                raise Unsupported('missing %s' % key)
            arr = np.array(S._as_numpy(weights[key]), dtype=np.float64, copy=True)
            if tuple(arr.shape) != shape:
                raise Unsupported('%s shape %s != %s' % (key, arr.shape, shape))
            arr.setflags(write=False)
            self._w[name] = arr
        if M.uses_fragment_swizzle(block_index, 2):
            raise Unsupported('model says fragment bias order for this block; D19 froze logical')

    def validate_invocation(self, *, block_index, head_count, window_size, origin_xy, grid_xy, shape):
        gx, gy = canvas_grid(self.origin_xy)
        want = (self.block_index, 2, 8, tuple(self.origin_xy), (gx, gy), (80, 80, 64))
        got = (block_index, head_count, window_size, tuple(origin_xy), tuple(grid_xy), tuple(shape))
        if got != want:
            raise Unsupported('refusing %s; verified-for %s' % (got, want))

    def _ffn(self, x2d):
        w = self._w
        z = S._half(x2d * w['ffn_cos_skip'])
        for h in range(2):
            w_exp = w['ffn_expand_weight'][h].transpose(1, 2, 0, 3).reshape(64, 128)
            e = B5._chain(x2d, w_exp, None, B5.K64)
            g = S.decode_codes(S.feed_forward_gate(e)).astype(np.float64).reshape(-1, 128)
            s = B5._chain(g, w['ffn_branch_projection_weight'][h].reshape(128, 32), None, B5.K128)
            u = S.decode_codes(S.quantise_codes(s)).astype(np.float64)
            z = S.qmma_rowwise(u, np.ascontiguousarray(w['ffn_output_projection_weight'][32 * h:32 * h + 32]), z)
        return z

    def run_codes(self, codes):
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != (80, 80, 64):
            raise Unsupported('uint8 [80,80,64] E4M3 codes required')
        if np.any((codes & 0x7F) == 0x7F):
            raise Unsupported('E4M3 NaN codes in input')
        w = self._w
        ox, oy = self.origin_xy
        gx, gy = canvas_grid(self.origin_xy)
        H, W = gy * 8, gx * 8
        rows, cols = slice(-oy, -oy + 80), slice(-ox, -ox + 80)
        x = S.decode_codes(codes).astype(np.float64)
        if self.pad_mode == 'canvas_before_ffn':
            canvas = np.zeros((H, W, 64), np.float64)
            canvas[rows, cols] = x
            z = self._ffn(canvas.reshape(-1, 64))
        else:                                      # pre-registered diagnostic DIAG2 only
            z80 = self._ffn(x.reshape(-1, 64)).reshape(80, 80, 64)
            zc = np.zeros((H, W, 64), np.float16)
            zc[rows, cols] = z80
            z = zc.reshape(-1, 64)
        qz = S.quantise_codes(z)
        xq = S.decode_codes(qz).astype(np.float64)
        qkv = B5._chain(xq, w['qkv_weight'], None, B5.K64)
        heads = []
        for h in range(2):
            c = slice(32 * h, 32 * h + 32)
            q, _ = S.normalise(qkv[:, 0:64][:, c].reshape(H, W, 32), self.tables, float(w['attn_scale'][h]))
            k, _ = S.normalise(qkv[:, 64:128][:, c].reshape(H, W, 32), self.tables)
            v = qkv[:, 128:192][:, c].reshape(H, W, 32)
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
            heads.append(M.reverse_windows(av, batch_count=1, height=H, width=W, window_size=8)[0].numpy())
        av = np.concatenate(heads, axis=-1).astype(np.float64).reshape(-1, 64)
        seed = S._half(xq * w['attn_cos_skip'])
        y = B5._chain(av, w['projection_weight'], seed, B5.K64).reshape(H, W, 64)
        crop = (rows, cols)
        return dict(Y=S.quantise_codes(y[crop]), QZ=qz.reshape(H, W, 64)[crop], AV=S.quantise_codes(av.reshape(H, W, 64)[crop]),
                    Y_half=y[crop], canvas_hw=(H, W), n_windows=int(gx * gy))

    def describe(self):
        return dict(backend='D19 BranchedWindow2hSm120 (isolated, unregistered)', block_index=self.block_index,
                    origin_xy=self.origin_xy, canvas_grid_xy=canvas_grid(self.origin_xy), pad_mode=self.pad_mode,
                    arithmetic='d17_backend helpers + mlxdlss.sm120_b1, unchanged')
