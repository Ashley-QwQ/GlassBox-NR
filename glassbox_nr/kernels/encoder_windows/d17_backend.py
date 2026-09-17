"""D17 isolated sm_120 candidate for B5 (2 heads, 64 channels, 80x80, origin 0).

Not registered, not merged. All numeric primitives are imported unchanged from
mlxdlss.sm120_b1 / mlxdlss.model. What this file owns is B5's structure
(chain-and-hypotheses.json H2-H5): per-head FFN loop with a carried z
accumulator, branch projection, 2-head attention with logical bias, tail with
published-QZ seed, INPVIEW64 / TIN64 layouts.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from glassbox_nr.kernels.core import sm120_b1 as S  # noqa: E402
from third_party.mlxdlss import model as M  # noqa: E402

Unsupported = S.UnsupportedByBackend
_TIN64 = np.load(ROOT / 'data/maps/TIN64-physical-to-logical.npy')
_CH = np.arange(64)
_ORDER64 = (_CH // 16) * 16 + ((_CH % 16) // 4) * 2 + _CH % 2 + ((_CH % 4) // 2) * 8

SHAPES = {
    'ffn_cos_skip': (64,), 'ffn_expand_weight': (2, 4, 2, 32, 32), 'ffn_branch_projection_weight': (2, 4, 32, 32),
    'ffn_output_projection_weight': (64, 64), 'qkv_weight': (64, 192), 'attn_bias': (2, 64, 64),
    'attn_scale': (2,), 'projection_weight': (64, 64), 'attn_cos_skip': (64,),
}


def inpview64_decode(raw: bytes) -> np.ndarray:
    """From parity-b5-one-group-v1/b5_layouts.inpview_decode (round trip asserted)."""
    x = np.frombuffer(raw, np.uint8)
    if x.size != 409600:
        raise Unsupported('INPVIEW64 must be 409600 bytes')
    phys = x.reshape(4, 80, 80, 16).transpose(1, 2, 0, 3).reshape(80, 80, 64)
    logical = np.empty_like(phys)
    logical[..., _ORDER64] = phys
    if inpview64_encode(logical) != bytes(raw):
        raise Unsupported('INPVIEW64 round trip failed')
    return logical


def inpview64_encode(codes: np.ndarray) -> bytes:
    return np.asarray(codes, np.uint8)[..., _ORDER64].reshape(80, 80, 4, 16).transpose(2, 0, 1, 3).copy().tobytes()


def tin64_decode(raw: bytes) -> np.ndarray:
    logical = np.zeros(80 * 80 * 64, np.uint8)
    logical[_TIN64] = np.frombuffer(raw, np.uint8)
    return logical.reshape(80, 80, 64)


def tin64_encode(codes: np.ndarray) -> bytes:
    return np.asarray(codes, np.uint8).reshape(-1)[_TIN64].copy().tobytes()


def _chain(a2d, w, seed, splits):
    """Sequential W27_trunc QMMA steps over contiguous K ranges of the A columns."""
    acc = seed
    for lo, hi in splits:
        acc = S.qmma_rowwise(np.ascontiguousarray(a2d[:, lo:hi]), np.ascontiguousarray(w[lo:hi, :]), acc)
    return acc


K64 = [(0, 32), (32, 64)]
K128 = [(0, 32), (32, 64), (64, 96), (96, 128)]


class B5Sm120:
    architecture = S.TARGET_ARCHITECTURE
    accumulator_model = S.ACCUMULATOR_MODEL
    block_index = 5
    head_count = 2
    window_size = 8
    origin_xy = (0, 0)

    def __init__(self, weights, tables: S.Sm120Tables, *, tail_seed='published_qz'):
        if tail_seed not in ('published_qz', 'unquantised_z'):
            raise Unsupported('tail_seed must be published_qz (frozen) or unquantised_z (pre-registered diagnostic)')
        self.tail_seed = tail_seed
        self.tables = tables
        self._w = {}
        for name, shape in SHAPES.items():
            key = 'block5.layer0.' + name
            if key not in weights:
                raise Unsupported('missing %s' % key)
            arr = np.array(S._as_numpy(weights[key]), dtype=np.float64, copy=True)
            if tuple(arr.shape) != shape:
                raise Unsupported('%s shape %s != %s' % (key, arr.shape, shape))
            arr.setflags(write=False)
            self._w[name] = arr
        if M.uses_fragment_swizzle(5, 2):
            raise Unsupported('model says 2-head blocks use fragment bias order; D17 froze logical order')

    def validate_invocation(self, *, block_index, head_count, window_size, origin_xy, shape):
        want = (5, 2, 8, (0, 0), (80, 80, 64))
        got = (block_index, head_count, window_size, tuple(origin_xy), tuple(shape))
        if got != want:
            raise Unsupported('B5Sm120 refuses %s; verified-for %s' % (got, want))

    def run_codes(self, codes: np.ndarray):
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != (80, 80, 64):
            raise Unsupported('uint8 [80,80,64] E4M3 codes required')
        if np.any((codes & 0x7F) == 0x7F):
            raise Unsupported('E4M3 NaN codes in input')
        w = self._w
        x2d = S.decode_codes(codes).astype(np.float64).reshape(-1, 64)
        # ---- FFN loop, h = 0, 1 (H2)
        z = S._half(x2d * w['ffn_cos_skip'])
        for h in range(2):
            w_exp = w['ffn_expand_weight'][h].transpose(1, 2, 0, 3).reshape(64, 128)
            e = _chain(x2d, w_exp, None, K64)
            g = S.decode_codes(S.feed_forward_gate(e)).astype(np.float64).reshape(-1, 128)
            s = _chain(g, w['ffn_branch_projection_weight'][h].reshape(128, 32), None, K128)
            u = S.decode_codes(S.quantise_codes(s)).astype(np.float64)
            z = S.qmma_rowwise(u, np.ascontiguousarray(w['ffn_output_projection_weight'][32 * h:32 * h + 32]), z)
        qz = S.quantise_codes(z)
        xq = S.decode_codes(qz).astype(np.float64)
        # ---- attention (H3)
        qkv = _chain(xq, w['qkv_weight'], None, K64)
        heads = []
        for h in range(2):
            cols = slice(32 * h, 32 * h + 32)
            q, _ = S.normalise(qkv[:, 0:64][:, cols].reshape(80, 80, 32), self.tables, float(w['attn_scale'][h]))
            k, _ = S.normalise(qkv[:, 64:128][:, cols].reshape(80, 80, 32), self.tables)
            v = qkv[:, 128:192][:, cols].reshape(80, 80, 32)
            parts = []
            for t in (q, k, v):
                pub = S.decode_codes(S.quantise_codes(t))
                parts.append(np.asarray(M.partition_windows(
                    torch.from_numpy(np.ascontiguousarray(pub, np.float32)).unsqueeze(0), 8).numpy(), np.float64))
            qw, kw, vw = parts
            bias = w['attn_bias'][h].astype(np.float16)
            score = S.qmma_batched(qw, np.ascontiguousarray(np.transpose(kw, (0, 2, 1))),
                                   np.broadcast_to(bias, (qw.shape[0], 64, 64)))
            pw = S.decode_codes(S.quantise_codes(S.published_attention_weights(score, self.tables))).astype(np.float64)
            acc = None
            for keys in (slice(0, 32), slice(32, 64)):
                acc = S.qmma_batched(np.ascontiguousarray(pw[:, :, keys]), np.ascontiguousarray(vw[:, keys, :]), acc)
            av = M.e4m3_round_trip(torch.from_numpy(np.ascontiguousarray(acc, np.float32)))
            heads.append(M.reverse_windows(av, batch_count=1, height=80, width=80, window_size=8)[0].numpy())
        av = np.concatenate(heads, axis=-1).astype(np.float64).reshape(-1, 64)
        # ---- tail (H4)
        zfac = xq if self.tail_seed == 'published_qz' else z.astype(np.float64)
        seed = S._half(zfac * w['attn_cos_skip'])
        y = _chain(av, w['projection_weight'], seed, K64).reshape(80, 80, 64)
        return dict(Y=S.quantise_codes(y), QZ=qz.reshape(80, 80, 64), AV=S.quantise_codes(av.reshape(80, 80, 64)),
                    Y_half=y, z_half=z.reshape(80, 80, 64))

    def describe(self):
        return dict(backend='D17 B5Sm120 (isolated, unregistered)', tail_seed=self.tail_seed,
                    accumulator_model=self.accumulator_model, core=S.__file__, model=M.__file__,
                    k_splits=dict(K64=K64, K128=K128))
