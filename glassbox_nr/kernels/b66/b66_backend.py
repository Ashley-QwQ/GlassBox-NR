"""Isolated B66 sm_120 candidate (upsample transition, 1 head, 80x80x64 + skip 160x160x32 -> 160x160x32).

Nothing in mlxdlss or any earlier experiment is edited.  Every numeric primitive is imported unchanged:
  * mlxdlss.sm120_b1          : decode_codes, quantise_codes, _half, tin_decode_codes / tin_encode_codes (TIN32)
  * D17 d17_backend._chain    : sequential W27_trunc QMMA steps over contiguous K ranges (seed None = RZ)
  * D13 d13_backend.SingleWindowSm120 : 1-head window block (validated B1-B4 on sm_120; reused by D33 R02 for B67)
What this file owns is only the B66 stage order, each item tied to static evidence (static/merge-trace-B66.json,
static/upsample-check-B66-img-r1.json, funcdiff-1h-B67chained-vs-B66upsample.json):
  1. input X: B65 publication PLANAR16 (Opus outview H1, two-sample MATCH) decoded with the frozen map
  2. projection: _chain(X[6400, 64], weight0[64, 32], seed None, splits [(0,32), (32,64)])
       evidence: the first uniform loop runs 2 iterations with 4 QMMA each and zero seeds (CS2R) before the loop
  3. spatial: nearest x2 of the projected half values, crop to 160x160
       evidence: input reads confined to 80x80 ((79,79) max), outputs to 160x160; kernel indexes the source with >>1;
       B66 has no interpolation weights (only weight0 and sin), so a learned bilinear upsample is not available
  4. merge: merged = _half(decode(skip) * sin + projected)   -- one fused multiply-add, one rounding
       evidence: 32 HFMA2 Rd, a=F2FP.F16.E4M3.UNPACK_B(skip lane), b=LDG.E weight (+0x2860), c=projection QMMA D
  5. quantise: quantise_codes(merged)  (F2FP.SATFINITE...MERGE_C directly after the HFMA2 sites)
  6. window: D13 SingleWindowSm120 bound process-locally as block 66, origin (0, 0), no pad (grid 20x20, word4 = 0)
  7. output: TIN32 encode (B66 stores 16-byte lanes, every byte once; B67 reads TIN32)
Skip: B4 full exit (launch 6 word1 == B66 word10, D13 two-sample MATCH), TIN32, unchanged from launch 6 to 151.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
for rel in ("glassbox_nr/kernels/encoder_windows",):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)
import d13_backend as D13  # noqa: E402
import d17_backend as D17  # noqa: E402

S = D13.S
Unsupported = S.UnsupportedByBackend
B66_SPEC = dict(origin_xy=(0, 0), pad_rows=0, pad_cols=0, pooled=False)
SPLITS = [(0, 32), (32, 64)]


class B66Candidate:
    def __init__(self, weights, tables, planar16_map: np.ndarray):
        if 66 not in D13.BLOCKS:
            D13.BLOCKS[66] = B66_SPEC.copy()
        elif D13.BLOCKS[66] != B66_SPEC:
            raise Unsupported("D13 block table drifted before B66 binding")
        self.window = D13.SingleWindowSm120(66, weights, tables)
        self.window.validate_invocation(block_index=66, head_count=1, window_size=8, origin_xy=(0, 0), shape=(160, 160, 32))
        w0 = np.array(S._as_numpy(weights["block66.layer0.weight0"]), dtype=np.float64, copy=True)
        sn = np.array(S._as_numpy(weights["block66.layer0.sin"]), dtype=np.float64, copy=True)
        if w0.shape != (64, 32) or sn.shape != (32,):
            raise Unsupported("block66 weight0/sin shapes %s %s" % (w0.shape, sn.shape))
        self.weight0, self.sin = w0, sn
        mp = np.asarray(planar16_map, np.int64)
        if mp.size != 80 * 80 * 64 or not np.array_equal(np.sort(mp), np.arange(mp.size)):
            raise Unsupported("PLANAR16 map is not a bijection of 80x80x64")
        self.planar16 = mp

    def decode_input(self, raw: bytes) -> np.ndarray:
        x = np.frombuffer(raw, np.uint8)
        if x.size != 409600:
            raise Unsupported("B66 main input must be 409600 bytes")
        logical = np.zeros(409600, np.uint8)
        logical[self.planar16] = x
        return logical.reshape(80, 80, 64)

    @staticmethod
    def decode_skip(raw: bytes) -> np.ndarray:
        codes = S.tin_decode_codes(raw, 160, 160)
        if codes.shape != (160, 160, 32):
            raise Unsupported("B66 skip must be TIN32 160x160x32")
        return codes

    def merged_codes(self, x_codes: np.ndarray, skip_codes: np.ndarray):
        if np.any((x_codes & 0x7F) == 0x7F) or np.any((skip_codes & 0x7F) == 0x7F):
            raise Unsupported("E4M3 NaN codes in input or skip")
        x = S.decode_codes(x_codes).astype(np.float64).reshape(-1, 64)
        proj = D17._chain(x, self.weight0, None, SPLITS).reshape(80, 80, 32)          # float16
        up = np.repeat(np.repeat(proj, 2, axis=0), 2, axis=1)[:160, :160, :]
        sk = S.decode_codes(skip_codes).astype(np.float64)
        merged_half = S._half(sk * self.sin + up.astype(np.float64))
        return S.quantise_codes(merged_half), dict(projected=proj, merged_half=merged_half)

    def run_raw(self, x_raw: bytes, skip_raw: bytes):
        mc, stages = self.merged_codes(self.decode_input(x_raw), self.decode_skip(skip_raw))
        res = self.window.run_codes(mc)
        y = np.asarray(res["Y"], np.uint8)
        if y.shape != (160, 160, 32):
            raise Unsupported("window output shape %s" % (y.shape,))
        return S.tin_encode_codes(y), dict(merged_codes=mc, **stages)

    def describe(self):
        return dict(name="b66_backend.B66Candidate", hypothesis="H1_FUSED_MERGE_NEAREST",
                    arithmetic_sources=[str((ROOT / "glassbox_nr/kernels/encoder_windows/d13_backend.py").resolve()),
                                        str((ROOT / "glassbox_nr/kernels/encoder_windows/d17_backend.py").resolve())],
                    projection_splits=SPLITS, merge="_half(decode(skip) * sin + nearest2(projected))", window_spec=B66_SPEC,
                    input_codec="PLANAR16 S80 C64 map", skip_codec="TIN32 (sm120_b1)", output_codec="TIN32 (sm120_b1)")
