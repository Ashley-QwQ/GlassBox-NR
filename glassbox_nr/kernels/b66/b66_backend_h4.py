"""B66 candidate H4_FFN_SEED_UNQUANTISED_MERGE (correction 1 of at most 2; static evidence, no target read).

Evidence (static/cos-skip-trace.json, FAILURE_LEDGER_ZH.md F2 check 4):
  validated B67: the 32 ffn_cos_skip operations take decode(input codes) -> sm120_b1 / D13 seed
  B66          : the 32 ffn_cos_skip operations take the merge-site half directly
                 -> the unquantised merged half
Everything else is H1 unchanged (b66_backend.B66Candidate): PLANAR16 input, projection _chain, nearest x2, fused
merge, quantise_codes(merged) feeding the FFN expansion QMMA (a QMMA takes E4M3 codes), D13 window stages, TIN32.
The window body below is d13_backend.SingleWindowSm120._body copied line for line; the single change is the FFN
seed operand (marked H4).  equivalence_gate(): with seed = decode(codes) it must equal the D13 body byte for byte.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import numpy as np
import torch

import b66_backend as B

D13 = B.D13
S = B.S
M = D13.M
Unsupported = B.Unsupported


class B66CandidateH4(B.B66Candidate):
    hypothesis = "H4_FFN_SEED_UNQUANTISED_MERGE"

    def body(self, codes: np.ndarray, seed_x: np.ndarray):
        win = self.window
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != (160, 160, 32):
            raise Unsupported("uint8 [160,160,32] E4M3 codes required")
        if np.any((codes & 0x7F) == 0x7F):
            raise Unsupported("E4M3 NaN codes in input")
        seed_x = np.asarray(seed_x, dtype=np.float64)
        if seed_x.shape != (160, 160, 32):
            raise Unsupported("FFN seed operand must be [160,160,32]")
        pr, pc = win.spec["pad_rows"], win.spec["pad_cols"]
        x = S.decode_codes(codes).astype(np.float64)
        x = np.pad(x, ((pr, pr), (pc, pc), (0, 0)))           # +0 pad (H3)
        sx = np.pad(seed_x, ((pr, pr), (pc, pc), (0, 0)))     # H4: same padding for the seed operand
        H, W = x.shape[:2]
        x2d = x.reshape(-1, 32)
        w = win._w
        # ---- FFN (core _ffn)
        expansion = S.qmma_rowwise(x2d, w["weight1"], None)
        gate_codes = S.feed_forward_gate(expansion)
        gated = S.decode_codes(gate_codes).astype(np.float64).reshape(-1, 128)
        acc = S._half(sx.reshape(-1, 32) * w["ffn_cos_skip"])  # H4: seed operand is the merge half, not decode(codes)
        for step in range(4):
            rows = slice(32 * step, 32 * (step + 1))
            acc = S.qmma_rowwise(gated[:, rows], w["weight2"][rows, :], acc)
        z = acc.reshape(H, W, 32)
        # ---- attention (core _attention, extent H x W)
        published_x = S.decode_codes(S.quantise_codes(z)).astype(np.float64)
        qkv = S.qmma_rowwise(published_x.reshape(-1, 32), w["qkv_weight"], None)
        shape = (H, W, 32)
        q, _ = S.normalise(qkv[..., 0:32].reshape(shape), win.tables, float(w["attn_scale"].ravel()[0]))
        k, _ = S.normalise(qkv[..., 32:64].reshape(shape), win.tables)
        v = qkv[..., 64:96].reshape(shape)
        parts = []
        for tensor in (q, k, v):
            published = S.decode_codes(S.quantise_codes(tensor))
            windows = M.partition_windows(
                torch.from_numpy(np.ascontiguousarray(published, np.float32)).unsqueeze(0), 8)
            parts.append(np.asarray(windows.numpy(), dtype=np.float64))
        qw, kw, vw = parts
        score = S.qmma_batched(qw, np.ascontiguousarray(np.transpose(kw, (0, 2, 1))),
                               np.broadcast_to(win._bias_qk, (qw.shape[0], 64, 64)))
        weights = S.published_attention_weights(score, win.tables)
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

    def equivalence_gate(self, codes: np.ndarray) -> dict:
        """seed = decode(codes) must reproduce the unchanged D13 body byte for byte"""
        y1, av1, _ = self.body(codes, S.decode_codes(codes).astype(np.float64))
        y0, av0, _ = self.window._body(codes)
        return dict(y_half_equal=np.asarray(y1).tobytes() == np.asarray(y0).tobytes(),
                    av_codes_equal=np.asarray(av1).tobytes() == np.asarray(av0).tobytes())

    def run_raw(self, x_raw: bytes, skip_raw: bytes):
        mc, stages = self.merged_codes(self.decode_input(x_raw), self.decode_skip(skip_raw))
        merged_half = np.asarray(stages["merged_half"])
        y_half, _av, _info = self.body(mc, merged_half.astype(np.float64))
        y = np.asarray(S.quantise_codes(y_half), np.uint8)
        if y.shape != (160, 160, 32):
            raise Unsupported("window output shape %s" % (y.shape,))
        return S.tin_encode_codes(y), dict(merged_codes=mc, **stages)

    def describe(self):
        d = super().describe()
        d.update(name="b66_backend_h4.B66CandidateH4", hypothesis=self.hypothesis,
                 correction="FFN seed = _half(merged_half * ffn_cos_skip); expansion keeps quantise(merged) codes",
                 evidence="static/cos-skip-trace.json (B66 ffn_cos_skip HMUL2 <- merge HFMA2; B67 <- decode(input))",
                 body_source="d13_backend.SingleWindowSm120._body copied line for line, one marked change")
        return d
