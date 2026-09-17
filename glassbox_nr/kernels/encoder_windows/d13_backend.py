"""D13 isolated sm_120 candidate for the single_window blocks B1-B4.

NOT registered into mlxdlss; not a change to the core. Imports every numeric
primitive from mlxdlss.sm120_b1 unchanged. The only things this file owns are
the facts that differ between blocks (chain-and-hypotheses.json H2-H6):

  SUBSTANTIVE DIFFERENCES FROM SM120B1Reference._ffn/_attention/run_codes
  1. the input is zero-padded (+0) on the axes whose origin is -4, and every
     stage runs over the padded extent (H3);
  2. window partition / reverse use the padded extent instead of 160x160;
  3. the Y half is cropped back to 160x160 before publication;
  4. block 4 adds the pooled tail (H6).
  The stage call sequence (qmma_rowwise / feed_forward_gate / 4-step W2 /
  normalise / quantise / partition / qmma_batched score with bias seed /
  published_attention_weights / 2-step AV / e4m3_round_trip / projection seed)
  is the core's, copied line for line; gate G2 checks the block-1 configuration
  against the core backend byte for byte.
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

# origin_xy as read from the C capture params (x first); pad is (rows, cols)
BLOCKS = {
    1: dict(origin_xy=(0, 0), pad_rows=0, pad_cols=0, pooled=False),
    2: dict(origin_xy=(-4, -4), pad_rows=4, pad_cols=4, pooled=False),
    3: dict(origin_xy=(-4, 0), pad_rows=0, pad_cols=4, pooled=False),
    4: dict(origin_xy=(0, -4), pad_rows=4, pad_cols=0, pooled=True),
}
SHAPES = dict(S.SUPPORTED["weight_shapes"])


def pooled_encode(codes: np.ndarray) -> bytes:
    """80x80x64 logical codes -> physical planes. From parity-b4-pre-color/run_b4_teacher_forced.py."""
    if codes.shape != (80, 80, 64) or codes.dtype != np.uint8:
        raise Unsupported("pooled layout is fixed 80x80x64 uint8")
    ch = np.arange(64)
    order = (ch // 16) * 16 + ((ch % 16) // 4) * 2 + ch % 2 + ((ch % 4) // 2) * 8
    return codes[..., order].reshape(80, 80, 4, 16).transpose(2, 0, 1, 3).copy().tobytes()


def pooled_decode(raw: bytes) -> np.ndarray:
    x = np.frombuffer(raw, np.uint8).reshape(4, 80, 80, 16).transpose(1, 2, 0, 3).reshape(80, 80, 64)
    ch = np.arange(64)
    order = (ch // 16) * 16 + ((ch % 16) // 4) * 2 + ch % 2 + ((ch % 4) // 2) * 8
    out = np.empty_like(x)
    out[..., order] = x
    return out


class SingleWindowSm120:
    architecture = S.TARGET_ARCHITECTURE
    accumulator_model = S.ACCUMULATOR_MODEL
    head_count = 1
    window_size = 8

    def __init__(self, block_index: int, weights, tables: S.Sm120Tables):
        if block_index not in BLOCKS:
            raise Unsupported("D13 candidate covers blocks 1-4 only, not %r" % (block_index,))
        self.block_index = block_index
        self.spec = BLOCKS[block_index]
        self.tables = tables
        prefix = "block%d.layer0." % block_index
        shapes = dict(SHAPES)
        if self.spec["pooled"]:
            shapes["weight0"] = (32, 64)
        self._w = {}
        for name, shape in shapes.items():
            key = prefix + name
            if key not in weights:
                raise Unsupported("missing logical weight %r" % key)
            arr = np.array(S._as_numpy(weights[key]), dtype=np.float64, copy=True)
            if tuple(arr.shape) != shape:
                raise Unsupported("%s has shape %s, expected %s" % (key, arr.shape, shape))
            arr.setflags(write=False)
            self._w[name] = arr
        bias = torch.from_numpy(np.ascontiguousarray(self._w["attn_bias"], dtype=np.float32))
        laid = M.recover_attention_bias_layout(bias).reshape(64, 64).numpy().astype(np.float16)
        laid.setflags(write=False)
        self._bias_qk = laid

    def validate_invocation(self, *, block_index, head_count, window_size, origin_xy, shape):
        want = (self.block_index, 1, 8, tuple(self.spec["origin_xy"]), (160, 160, 32))
        got = (block_index, head_count, window_size, tuple(origin_xy), tuple(shape))
        if got != want:
            raise Unsupported("D13 candidate refuses %s; verified-for %s" % (got, want))

    def _body(self, codes: np.ndarray):
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != (160, 160, 32):
            raise Unsupported("uint8 [160,160,32] E4M3 codes required")
        if np.any((codes & 0x7F) == 0x7F):
            raise Unsupported("E4M3 NaN codes in input")
        pr, pc = self.spec["pad_rows"], self.spec["pad_cols"]
        x = S.decode_codes(codes).astype(np.float64)
        x = np.pad(x, ((pr, pr), (pc, pc), (0, 0)))           # +0 pad (H3)
        H, W = x.shape[:2]
        x2d = x.reshape(-1, 32)
        w = self._w
        # ---- FFN (core _ffn)
        expansion = S.qmma_rowwise(x2d, w["weight1"], None)
        gate_codes = S.feed_forward_gate(expansion)
        gated = S.decode_codes(gate_codes).astype(np.float64).reshape(-1, 128)
        acc = S._half(x2d * w["ffn_cos_skip"])
        for step in range(4):
            rows = slice(32 * step, 32 * (step + 1))
            acc = S.qmma_rowwise(gated[:, rows], w["weight2"][rows, :], acc)
        z = acc.reshape(H, W, 32)
        # ---- attention (core _attention, extent H x W)
        published_x = S.decode_codes(S.quantise_codes(z)).astype(np.float64)
        qkv = S.qmma_rowwise(published_x.reshape(-1, 32), w["qkv_weight"], None)
        shape = (H, W, 32)
        q, _ = S.normalise(qkv[..., 0:32].reshape(shape), self.tables, float(w["attn_scale"].ravel()[0]))
        k, _ = S.normalise(qkv[..., 32:64].reshape(shape), self.tables)
        v = qkv[..., 64:96].reshape(shape)
        parts = []
        for tensor in (q, k, v):
            published = S.decode_codes(S.quantise_codes(tensor))
            windows = M.partition_windows(
                torch.from_numpy(np.ascontiguousarray(published, np.float32)).unsqueeze(0), 8)
            parts.append(np.asarray(windows.numpy(), dtype=np.float64))
        qw, kw, vw = parts
        score = S.qmma_batched(qw, np.ascontiguousarray(np.transpose(kw, (0, 2, 1))),
                               np.broadcast_to(self._bias_qk, (qw.shape[0], 64, 64)))
        weights = S.published_attention_weights(score, self.tables)
        pw = S.decode_codes(S.quantise_codes(weights)).astype(np.float64)
        acc = None
        for keys in (slice(0, 32), slice(32, 64)):
            acc = S.qmma_batched(np.ascontiguousarray(pw[:, :, keys]),
                                 np.ascontiguousarray(vw[:, keys, :]), acc)
        av_published = M.e4m3_round_trip(torch.from_numpy(np.ascontiguousarray(acc, np.float32)))
        av = M.reverse_windows(av_published, batch_count=1, height=H, width=W, window_size=8)[0].numpy()
        # ---- projection (core run_codes)
        proj_seed = S._half(z.reshape(-1, 32).astype(np.float64) * w["attn_cos_skip"])
        y = S.qmma_rowwise(np.ascontiguousarray(av, np.float64).reshape(-1, 32),
                           w["projection_weight"], proj_seed).reshape(H, W, 32)
        crop = (slice(pr, pr + 160), slice(pc, pc + 160))
        return y[crop], S.quantise_codes(av[crop]), dict(n_windows=int(qw.shape[0]), extent=[H, W])

    def run_codes(self, codes: np.ndarray):
        """-> dict(Y=codes [160,160,32], AV=codes, pooled=codes [80,80,64] for block 4, Y_half, info)"""
        y_half, av_codes, info = self._body(codes)
        out = dict(Y=S.quantise_codes(y_half), AV=av_codes, Y_half=y_half, info=info)
        if self.spec["pooled"]:
            h = y_half.astype(np.float16)
            even = (h[0::2, 0::2].astype(np.float64) + h[0::2, 1::2].astype(np.float64)).astype(np.float16)
            odd = (h[1::2, 0::2].astype(np.float64) + h[1::2, 1::2].astype(np.float64)).astype(np.float16)
            pool = ((even.astype(np.float64) + odd.astype(np.float64)).astype(np.float16)
                    .astype(np.float64) * 0.25).astype(np.float16)
            pin = S.decode_codes(S.quantise_codes(pool)).astype(np.float64)
            projected = S.qmma_rowwise(pin.reshape(-1, 32), self._w["weight0"], None).reshape(80, 80, 64)
            out.update(pooled=S.quantise_codes(projected), pool_half=pool)
        return out

    def describe(self):
        return dict(backend="D13 SingleWindowSm120 (isolated, unregistered)", block_index=self.block_index,
                    spec=dict(self.spec), accumulator_model=self.accumulator_model,
                    core_module=S.__file__, model_module=M.__file__)
