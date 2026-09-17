"""D20 isolated sm_120 implementation for block 0 (cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8), RESET PATH ONLY.

Not registered into mlxdlss; imports numeric primitives from mlxdlss.sm120_b1 unchanged.

Pipeline:
  frontend  per network pixel (x,y) in 320x320, counter 0, temporal branch skipped.
            ch0,ch1,ch2 = half(g3), half(g2), half(g1) (Box-Muller, measured MUFU tables)
            ch3 = 1.0
            ch4..6 = ch7..9 = half(half(half(C) - 0.5) * s), s = half(2*word24.hi) = 0.125,
                              C = colour texel at (x, mirror(y))
            ch10 = half(word22.hi) = 1/128 ; ch11 = half(word22.lo) = 1.0 ; ch12 = half(word21.hi) = 1.0
            ch13 = ch14 = -1 (skin/automask off: word23 < 0) ; ch15 = +0
  adapter   z0[o] = half_RNE( exact sum_i f[i] * A[i,o] ), C = RZ
  publish   X = F2FP.SATFINITE.E4M3(z0)
  FFN       expansion = QMMA(X, weight1) ; gate ; W2 4 x K32 seeded with half(z0 * ffn_cos_skip)
  attention B1 core sequence at extent 320x320, window 8, origin (0,0), 1600 windows
  Y         projection seeded with half(z * attn_cos_skip) ; full-skip = TIN(quantise(Y))
  pooled    even=(Y[2i,2j]+Y[2i,2j+1]) odd=(Y[2i+1,2j]+Y[2i+1,2j+1]) (half sums), half(even+odd)*0.25,
            published E4M3, INPVIEW 160x160
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from glassbox_nr.kernels.core import sm120_b1 as S  # noqa: E402
from third_party.mlxdlss import model as M  # noqa: E402

FRONTEND = Path(__file__).resolve().parent  # noise_reference.py / mufu_tables.py now live alongside this file
Unsupported = S.UnsupportedByBackend
NET = 320
CW, CH = 320, 240
SHAPES = {'attn_bias': (1, 64, 64), 'attn_cos_skip': (32,), 'attn_scale': (1,), 'ffn_cos_skip': (32,),
          'input_adapter_weight': (16, 32), 'projection_weight': (32, 32), 'qkv_weight': (32, 96),
          'weight1': (32, 128), 'weight2': (128, 32)}


def _load_frontend_modules():
    import importlib.util
    mods = {}
    for name in ('noise_reference', 'mufu_tables'):
        spec = importlib.util.spec_from_file_location('d20_' + name, FRONTEND / f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


# ------------------------------------------------------------------------------------------------ frontend
def noise_channels(frame_word: int = 0):
    """[320,320,3] float16: ch0 = g3 (r2*cos a2), ch1 = g2 (r2*sin a2), ch2 = g1 (r1*cos a1).
    Generates Box-Muller Gaussian noise channels packed into 16-bit half words."""
    mods = _load_frontend_modules()
    ys, xs = np.meshgrid(np.arange(NET, dtype=np.uint32), np.arange(NET, dtype=np.uint32), indexing='ij')
    g1, g2, g3 = mods['noise_reference'].gaussians(xs.reshape(-1), ys.reshape(-1), frame_word, mufu=mods['mufu_tables'].mufu)
    return np.stack([g3, g2, g1], axis=-1).reshape(NET, NET, 3).astype(np.float16)


def mirror_index(i, extent):
    """Mirror index at boundary: i >= extent -> 2*extent - 2 - i."""
    i = np.asarray(i, dtype=np.int64)
    return np.where(i >= extent, 2 * extent - 2 - i, i)


def colour_texel_indices():
    """Point sampler (entry generation filter 0, address 3) at the literal coordinate chain; returns the texel
    index actually addressed, asserting it is the mirrored integer position (no rounding across a texel edge)."""
    rcp_w, rcp_h = np.float32(1.0 / CW), np.float32(1.0 / CH)     # MUFU.RCP(320)/(240) modelled; +/- ulp checked by gate
    xs = mirror_index(np.arange(NET), CW)
    ys = mirror_index(np.arange(NET), CH)
    u = ((((xs.astype(np.float32) + np.float32(0.5)) * rcp_w) * np.float32(CW)) + np.float32(0)) * rcp_w
    v = ((((ys.astype(np.float32) + np.float32(0.5)) * rcp_h) * np.float32(CH)) + np.float32(0)) * rcp_h
    tx = np.clip(np.floor(u * np.float32(CW)).astype(np.int64), 0, CW - 1)
    ty = np.clip(np.floor(v * np.float32(CH)).astype(np.int64), 0, CH - 1)
    if not (np.array_equal(tx, xs) and np.array_equal(ty, ys)):
        raise Unsupported('colour coordinate chain does not land on the mirrored texel centre')
    return ty, tx


def colour_scale(params: bytes) -> np.float16:
    """Extract color scale factor from pre_block params word24: half(2*word24.hi)."""
    w24 = struct.unpack('<2f', struct.pack('<Q', struct.unpack('<33Q', params)[24]))
    return np.float16(np.float32(w24[1]) + np.float32(w24[1]))


def colour_channels(colour_bytes: bytes, params: bytes):
    """[320,320,3] float16 of half(half(half(C)-0.5) * scale) for R,G,B, scale = half(2*word24.hi) (0.125 here).
    Applies centering and scaling to mirrored texel inputs."""
    if len(colour_bytes) != CW * CH * 8:
        raise Unsupported('colour buffer must be 320x240 RGBA16F')
    scale = colour_scale(params)
    tex = np.frombuffer(colour_bytes, '<f2').reshape(CH, CW, 4)
    ty, tx = colour_texel_indices()
    c = tex[ty][:, tx][..., :3].astype(np.float32)               # TEX float32 lanes
    h = c.astype(np.float16)                                       # PACK_AB half
    centred = (h.astype(np.float64) - 0.5).astype(np.float16)     # HADD2 -0.5
    return (centred.astype(np.float64) * np.float64(scale)).astype(np.float16)  # HMUL2 x scale


def constant_channels(params: bytes):
    """ch10..ch15 from the captured pre_block params (word k = c[0x0][0x380+8k]); reset path, depth/mask unbound."""
    w = struct.unpack('<33Q', params)
    f32 = lambda q: struct.unpack('<2f', struct.pack('<Q', q))
    if not (w[1] == 0 and w[2] == 0 and w[3] == 0 and w[4] == 0):
        raise Unsupported('reset path with null history/MV/depth/mask handles only')
    w21, w22, w23 = f32(w[21]), f32(w[22]), f32(w[23])
    if not (min(w23) < 0):
        raise Unsupported('only the P5-false path (word23 min < 0) is modelled')
    ch10 = np.float16(w22[1])     # constant channel 10: word22.hi
    ch11 = np.float16(w22[0])     # constant channel 11: word22.lo
    ch12 = np.float16(w21[1])     # constant channel 12: word21.hi
    ch13 = np.float16(-1.0)       # constant channel 13: -1.0
    ch14 = np.float16(-1.0)       # constant channel 14: -1.0
    ch15 = np.float16(0.0)        # constant channel 15: +0.0
    return np.array([ch10, ch11, ch12, ch13, ch14, ch15], dtype=np.float16), dict(word21=w21, word22=w22, word23=w23)


def features(colour_bytes: bytes, params: bytes):
    w = struct.unpack('<33Q', params)
    counter = w[25] & 0xffffffff
    f = np.zeros((NET, NET, 16), np.float16)
    f[..., 0:3] = noise_channels(counter)
    f[..., 3] = np.float16(1.0)
    col = colour_channels(colour_bytes, params)
    f[..., 4:7] = col
    f[..., 7:10] = col
    consts, info = constant_channels(params)
    f[..., 10:16] = consts
    return f, dict(counter=counter, **info)


# ------------------------------------------------------------------------------------------------ adapter
def hmma_full_rne(feat: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """z[o] = half_RNE(exact sum_i f[i]*W[i,o]), no seed; zero sums -> +0.  [N,16] x [16,32] -> [N,32] float16."""
    f = np.asarray(feat, np.float16).astype(np.float64)
    wgt = np.asarray(weight, np.float64)
    if not np.array_equal(wgt.astype(np.float16).astype(np.float64), wgt):
        raise Unsupported('adapter weight not exactly binary16')
    terms = f[:, :, None] * wgt[None, :, :]                        # exact: 11-bit x 11-bit significands
    naive = terms.sum(axis=1)
    out = naive.astype(np.float16)
    # exactness guard: re-do with math.fsum wherever float64 summation could have moved across a half boundary
    scale = np.abs(terms).max(axis=1)
    err_bound = 32.0 * np.finfo(np.float64).eps * np.maximum(scale, np.abs(naive))
    lo = np.nextafter(naive - err_bound, -np.inf).astype(np.float16)
    hi = np.nextafter(naive + err_bound, np.inf).astype(np.float16)
    risky = (lo != hi) | (lo.view(np.uint16) != hi.view(np.uint16))
    idx = np.argwhere(risky)
    for n, o in idx:
        t = terms[n, :, o].tolist()
        s = math.fsum(t)
        h = np.float16(s)
        # s is the correctly rounded float64 of the exact sum; resolve a float64 tie onto a half midpoint exactly
        hf = float(h)
        nb = [float(np.nextafter(np.float16(hf), np.float16(-np.inf))), float(np.nextafter(np.float16(hf), np.float16(np.inf)))]
        for other in nb:
            mid = (hf + other) / 2.0
            if s == mid:
                resid = math.fsum(t + [-mid])
                h = np.float16(other) if (resid != 0 and (other > hf) == (resid > 0)) else h
        out[n, o] = h
    out[out == 0] = np.float16(0.0)                                   # no contribution / cancellation -> +0
    if not np.isfinite(out).all():
        raise Unsupported('adapter overflow: accumulator behaviour outside tested domain')
    return out, int(risky.sum())


# ------------------------------------------------------------------------------------------------ layer + pool
class B0ResetCandidate:
    def __init__(self, weights, tables: S.Sm120Tables, *, seed_source='adapter_half', pool_source='y_half'):
        assert seed_source in ('adapter_half', 'published_codes')
        assert pool_source in ('y_half', 'y_codes')
        self.seed_source, self.pool_source, self.tables = seed_source, pool_source, tables
        self._w = {}
        for name, shape in SHAPES.items():
            arr = np.array(S._as_numpy(weights['block0.layer0.' + name]), dtype=np.float64, copy=True)
            if tuple(arr.shape) != shape:
                raise Unsupported(f'{name} shape {arr.shape} != {shape}')
            arr.setflags(write=False)
            self._w[name] = arr
        bias = torch.from_numpy(np.ascontiguousarray(self._w['attn_bias'], dtype=np.float32))
        laid = M.recover_attention_bias_layout(bias).reshape(64, 64).numpy().astype(np.float16)
        laid.setflags(write=False)
        self._bias_qk = laid

    def layer(self, x_codes: np.ndarray, seed_half: np.ndarray, extent: int):
        """B1 core stage sequence at extent x extent (window 8, origin 0); seed_half is the W2 accumulator seed."""
        w, H, W = self._w, extent, extent
        x2d = S.decode_codes(x_codes).astype(np.float64).reshape(-1, 32)
        expansion = S.qmma_rowwise(x2d, w['weight1'], None)
        gated = S.decode_codes(S.feed_forward_gate(expansion)).astype(np.float64).reshape(-1, 128)
        acc = np.asarray(seed_half, np.float16).reshape(-1, 32)
        for step in range(4):
            rows = slice(32 * step, 32 * (step + 1))
            acc = S.qmma_rowwise(gated[:, rows], w['weight2'][rows, :], acc)
        z = acc.reshape(H, W, 32)
        published_x = S.decode_codes(S.quantise_codes(z)).astype(np.float64)
        qkv = S.qmma_rowwise(published_x.reshape(-1, 32), w['qkv_weight'], None)
        shape = (H, W, 32)
        q, _ = S.normalise(qkv[..., 0:32].reshape(shape), self.tables, float(w['attn_scale'].ravel()[0]))
        k, _ = S.normalise(qkv[..., 32:64].reshape(shape), self.tables)
        v = qkv[..., 64:96].reshape(shape)
        parts = []
        for tensor in (q, k, v):
            published = S.decode_codes(S.quantise_codes(tensor))
            windows = M.partition_windows(torch.from_numpy(np.ascontiguousarray(published, np.float32)).unsqueeze(0), 8)
            parts.append(np.asarray(windows.numpy(), dtype=np.float64))
        qw, kw, vw = parts
        score = S.qmma_batched(qw, np.ascontiguousarray(np.transpose(kw, (0, 2, 1))),
                               np.broadcast_to(self._bias_qk, (qw.shape[0], 64, 64)))
        pw = S.decode_codes(S.quantise_codes(S.published_attention_weights(score, self.tables))).astype(np.float64)
        av = None
        for keys in (slice(0, 32), slice(32, 64)):
            av = S.qmma_batched(np.ascontiguousarray(pw[:, :, keys]), np.ascontiguousarray(vw[:, keys, :]), av)
        av_pub = M.e4m3_round_trip(torch.from_numpy(np.ascontiguousarray(av, np.float32)))
        avr = M.reverse_windows(av_pub, batch_count=1, height=H, width=W, window_size=8)[0].numpy()
        proj_seed = S._half(z.reshape(-1, 32).astype(np.float64) * w['attn_cos_skip'])
        y = S.qmma_rowwise(np.ascontiguousarray(avr, np.float64).reshape(-1, 32), w['projection_weight'], proj_seed)
        return y.reshape(H, W, 32), S.quantise_codes(avr), dict(n_windows=int(qw.shape[0]))

    def run(self, colour_bytes: bytes, params: bytes, *, stages=False):
        feat, finfo = features(colour_bytes, params)
        z0, n_fsum = hmma_full_rne(feat.reshape(-1, 16), self._w['input_adapter_weight'])
        z0 = z0.reshape(NET, NET, 32)
        x_codes = S.quantise_codes(z0)
        if self.seed_source == 'adapter_half':
            seed = S._half(z0.reshape(-1, 32).astype(np.float64) * self._w['ffn_cos_skip'])
        else:
            seed = S._half(S.decode_codes(x_codes).astype(np.float64).reshape(-1, 32) * self._w['ffn_cos_skip'])
        y, av_codes, linfo = self.layer(x_codes, seed, NET)
        y_codes = S.quantise_codes(y)
        src = y.astype(np.float16) if self.pool_source == 'y_half' else S.decode_codes(y_codes).astype(np.float16)
        even = (src[0::2, 0::2].astype(np.float64) + src[0::2, 1::2].astype(np.float64)).astype(np.float16)
        odd = (src[1::2, 0::2].astype(np.float64) + src[1::2, 1::2].astype(np.float64)).astype(np.float16)
        pool = ((even.astype(np.float64) + odd.astype(np.float64)).astype(np.float16).astype(np.float64) * 0.25).astype(np.float16)
        pooled_codes = S.quantise_codes(pool)
        out = dict(full_tin=S.tin_encode_codes(y_codes), pooled_inpview=S.inpview_encode_codes(pooled_codes),
                   info=dict(frontend=finfo, adapter_fsum_fallbacks=n_fsum, seed_source=self.seed_source,
                             pool_source=self.pool_source, **linfo))
        if stages:
            out['stages'] = dict(features=feat, z0=z0, x_codes=x_codes, y_half=y, pool_half=pool)
        return out
