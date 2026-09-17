"""Parameterised upsample-transition backend: B66 (1 head) / B62 (2 heads) / B56 (4 heads) / B48 (8 heads).  T0 artefact.

B47/B48 round edit (copied from the audited B56/B62 round file sha add4fc35...; diff in static/tr-pipeline-diff.json):
  * heads 8 accepted: channels 256, out_hw 20, in_planes 32, in_hw 12 (B48 main input is 12x12x512, not out_hw/2);
    window body = d26_backend.BranchedWindowSm120N (the D22 class parameterised to 8 heads; B49/B55 two-sample lineage),
    driven by the SAME copied run_codes/_ffn body below (attribute names identical; D26.B5 is D17 and D26.BW is D22.BW, asserted);
  * in_hw is explicit in VERIFIED (equal to out_hw // 2 for heads 1/2/4, so their arithmetic is unchanged);
  * TIN256 codec from d26_backend.
  Card T0 rule applies: equivalence gate (G1 for heads 1/2/4/8) + B66 regression + B62/B56 regressions before any B48 number.

Parameterised, not forked, from experiments/parity-astra-opus-b66-upsample-chain-20260913/scripts/b66_backend{,_h4}.py,
which hard-coded (160,160,32), 32 lanes and 1 head.  Parameters:
  block      logical weight prefix block<N>.layer0.*
  heads      1 | 2 | 4           channels = 32*heads
  out_hw     160 | 80 | 40       output side; main input side = out_hw/2 with 2*channels channels
  in_planes  4 | 8 | 16          PLANAR16 planes of the main input (= 2*channels/16)
  origin_xy  window origin (x, y) read from the launch params; each axis 0 or -4
  ffn_seed   "merged_half"   (H4 form: FFN tail seed = _half(merged_half * ffn_cos_skip))
             "decoded_codes" (H1 form: seed = _half(decode(quantise(merged)) * ffn_cos_skip))
  tail_seed  branched bodies only: "published_qz" (D22 default) | "unquantised_z"; the 1-head body keeps D13's z seed
Only the three verified (heads, channels, out_hw, in_planes) combinations are accepted.

Stage order (the B66 acceptance model H4, generalised; every numeric primitive imported unchanged):
  1. PLANAR16 decode of the main input with the frozen outview map (S80-C64 / S40-C128 / S20-C256)
  2. projection  D17._chain(X[(out/2)^2, 2C], weight0[2C, C], seed None, 32-channel K splits)
  3. nearest x2 of the projected half values, crop to out_hw
  4. merged_half = _half(decode(skip) * sin + up)
  5. merged codes = quantise_codes(merged_half)
  6. window body on the merged codes with an explicit FFN seed operand:
       heads 1   : copy of d13_backend.SingleWindowSm120._body (the B66 H4 copy; B66/B67 two-sample MATCH lineage)
       heads 2/4 : copy of d22_backend.BranchedWindowSm120.run_codes/_ffn (B63 2h and B57 4h compat MATCH lineage)
     each copy differs from its source ONLY in the FFN seed operand (marked SEED); gate_t0_equivalence.py proves that
     with seed = decode(codes) the copy equals the unchanged source byte for byte.
  7. TIN<C> encode of Y (TIN32 sm120_b1, TIN64 D17 map, TIN128 D22 map); skip decoded with the same codec.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
for rel in ("glassbox_nr/kernels/encoder_windows",):     # B47/B48 round: D26 8-head body
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)
import d13_backend as D13  # noqa: E402
import d17_backend as D17  # noqa: E402
import d22_backend as D22  # noqa: E402
import d26_backend as D26  # noqa: E402

S = D13.S
M = D13.M
Unsupported = S.UnsupportedByBackend
if D17.S is not S or D22.S is not S or D26.S is not S:
    raise Unsupported("sm120_b1 module identity differs between D13/D17/D22/D26")
if D26.B5 is not D17 or D26.BW is not D22.BW:
    raise Unsupported("D26 does not share D17 _chain / D22 canvas module objects")

# B47/B48 round: in_hw is explicit (main-input side); heads 8 = B48 (12x12x512 -> 20x20x256), body D26 BranchedWindowSm120N
VERIFIED = {1: dict(channels=32, out_hw=160, in_planes=4, in_hw=80), 2: dict(channels=64, out_hw=80, in_planes=8, in_hw=40),
            4: dict(channels=128, out_hw=40, in_planes=16, in_hw=20), 8: dict(channels=256, out_hw=20, in_planes=32, in_hw=12)}
FFN_SEEDS = ("merged_half", "decoded_codes")


def splits32(n):
    return [(i, i + 32) for i in range(0, n, 32)]


def tin_decode(raw: bytes, channels: int, side: int) -> np.ndarray:
    if len(raw) != side * side * channels:
        raise Unsupported("TIN%d buffer must be %d bytes" % (channels, side * side * channels))
    if channels == 32:
        codes = S.tin_decode_codes(raw, side, side)
    elif channels == 64:
        codes = D17.tin64_decode(raw)
    elif channels == 128:
        codes = D22.tin128_decode(raw)
    else:
        codes = D26.tin256_decode(raw)                       # B47/B48 round: TIN256 (D26 codec, B49/B55 lineage)
    if codes.shape != (side, side, channels):
        raise Unsupported("TIN%d decode shape %s" % (channels, codes.shape))
    return codes


def tin_encode(codes: np.ndarray, channels: int, side: int) -> bytes:
    codes = np.asarray(codes, np.uint8)
    if codes.shape != (side, side, channels):
        raise Unsupported("TIN%d encode requires [%d,%d,%d]" % (channels, side, side, channels))
    if channels == 32:
        return S.tin_encode_codes(codes)
    if channels == 64:
        return D17.tin64_encode(codes)
    if channels == 128:
        return D22.tin128_encode(codes)
    return D26.tin256_encode(codes)                          # B47/B48 round


class TransitionCandidate:
    def __init__(self, *, block, heads, channels, out_hw, in_planes, origin_xy, weights, tables, planar16_map,
                 ffn_seed="merged_half", tail_seed="published_qz", in_hw=None):
        want = VERIFIED.get(heads)
        if want is None or dict(channels=channels, out_hw=out_hw, in_planes=in_planes) != {k: want[k] for k in ("channels", "out_hw", "in_planes")}:
            raise Unsupported("unverified transition configuration heads=%r channels=%r out_hw=%r in_planes=%r"
                              % (heads, channels, out_hw, in_planes))
        if in_hw is not None and in_hw != want["in_hw"]:           # B47/B48 round: optional, must equal the verified main-input side
            raise Unsupported("in_hw %r differs from the verified %r for heads %r" % (in_hw, want["in_hw"], heads))
        if ffn_seed not in FFN_SEEDS:
            raise Unsupported("ffn_seed %r" % (ffn_seed,))
        if tail_seed not in ("published_qz", "unquantised_z"):
            raise Unsupported("tail_seed %r" % (tail_seed,))
        ox, oy = (int(v) for v in origin_xy)
        if ox not in (0, -4) or oy not in (0, -4):
            raise Unsupported("origin %s outside {0,-4}^2" % (origin_xy,))
        self.block, self.heads, self.C, self.out_hw, self.in_planes = int(block), heads, channels, out_hw, in_planes
        self.in_hw, self.in_C = want["in_hw"], 2 * channels      # B47/B48 round: explicit (was out_hw // 2; equal for heads 1/2/4)
        self.origin_xy, self.ffn_seed, self.tables = (ox, oy), ffn_seed, tables
        self.tail_seed = tail_seed if heads > 1 else "d13_unquantised_z"
        if heads == 1:
            spec = dict(origin_xy=(ox, oy), pad_rows=4 if oy else 0, pad_cols=4 if ox else 0, pooled=False)
            if self.block not in D13.BLOCKS:
                D13.BLOCKS[self.block] = spec.copy()
            elif D13.BLOCKS[self.block] != spec:
                raise Unsupported("D13 block table entry for %d differs from %s" % (self.block, spec))
            self.window = D13.SingleWindowSm120(self.block, weights, tables)
            self.window.validate_invocation(block_index=self.block, head_count=1, window_size=8, origin_xy=(ox, oy),
                                            shape=(160, 160, 32))
        elif heads == 8:
            # B47/B48 round: the D26 8-head body (B49/B55 two-sample lineage); same attribute / method names as D22's class
            self.window = D26.BranchedWindowSm120N(self.block, heads, out_hw, (ox, oy), weights, tables, tail_seed=tail_seed)
            gx, gy = D22.BW.canvas_grid((ox, oy), out_hw)
            self.window.validate_invocation(block_index=self.block, head_count=heads, origin_xy=(ox, oy), grid_xy=(gx, gy),
                                            shape=(out_hw, out_hw, channels))
        else:
            self.window = D22.BranchedWindowSm120(self.block, heads, out_hw, (ox, oy), weights, tables, tail_seed=tail_seed)
            gx, gy = D22.BW.canvas_grid((ox, oy), out_hw)
            self.window.validate_invocation(block_index=self.block, head_count=heads, origin_xy=(ox, oy), grid_xy=(gx, gy),
                                            shape=(out_hw, out_hw, channels))
        prefix = "block%d.layer0." % self.block
        w0 = np.array(S._as_numpy(weights[prefix + "weight0"]), dtype=np.float64, copy=True)
        sn = np.array(S._as_numpy(weights[prefix + "sin"]), dtype=np.float64, copy=True)
        if w0.shape != (self.in_C, channels) or sn.shape != (channels,):
            raise Unsupported("%sweight0/sin shapes %s %s" % (prefix, w0.shape, sn.shape))
        self.weight0, self.sin = w0, sn
        n = self.in_hw * self.in_hw * self.in_C
        mp = np.asarray(planar16_map, np.int64)
        if mp.size != n or not np.array_equal(np.sort(mp), np.arange(n)):
            raise Unsupported("PLANAR16 map is not a bijection of %dx%dx%d" % (self.in_hw, self.in_hw, self.in_C))
        self.planar16 = mp

    # ------------------------------------------------------------------ transition stages
    def decode_input(self, raw: bytes) -> np.ndarray:
        n = self.in_hw * self.in_hw * self.in_C
        x = np.frombuffer(raw, np.uint8)
        if x.size != n:
            raise Unsupported("main input must be %d bytes" % n)
        logical = np.zeros(n, np.uint8)
        logical[self.planar16] = x
        return logical.reshape(self.in_hw, self.in_hw, self.in_C)

    def decode_skip(self, raw: bytes) -> np.ndarray:
        return tin_decode(raw, self.C, self.out_hw)

    def merged(self, x_codes: np.ndarray, skip_codes: np.ndarray):
        if np.any((x_codes & 0x7F) == 0x7F) or np.any((skip_codes & 0x7F) == 0x7F):
            raise Unsupported("E4M3 NaN codes in input or skip")
        x = S.decode_codes(x_codes).astype(np.float64).reshape(-1, self.in_C)
        proj = D17._chain(x, self.weight0, None, splits32(self.in_C)).reshape(self.in_hw, self.in_hw, self.C)
        up = np.repeat(np.repeat(proj, 2, axis=0), 2, axis=1)[:self.out_hw, :self.out_hw, :]
        sk = S.decode_codes(skip_codes).astype(np.float64)
        merged_half = S._half(sk * self.sin + up.astype(np.float64))
        return S.quantise_codes(merged_half), dict(projected=proj, merged_half=merged_half)

    def seed_operand(self, merged_codes, merged_half):
        if self.ffn_seed == "merged_half":
            return np.asarray(merged_half).astype(np.float64)
        return S.decode_codes(merged_codes).astype(np.float64)

    # ------------------------------------------------------------------ window bodies
    def body(self, codes: np.ndarray, seed_x: np.ndarray):
        return self._body1(codes, seed_x) if self.heads == 1 else self._body_branched(codes, seed_x)

    def _body1(self, codes, seed_x):
        """d13_backend.SingleWindowSm120._body, copied line for line; SEED marks the only change."""
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
        sx = np.pad(seed_x, ((pr, pr), (pc, pc), (0, 0)))     # SEED: same padding for the seed operand
        H, W = x.shape[:2]
        x2d = x.reshape(-1, 32)
        w = win._w
        # ---- FFN (core _ffn)
        expansion = S.qmma_rowwise(x2d, w["weight1"], None)
        gate_codes = S.feed_forward_gate(expansion)
        gated = S.decode_codes(gate_codes).astype(np.float64).reshape(-1, 128)
        acc = S._half(sx.reshape(-1, 32) * w["ffn_cos_skip"])  # SEED: explicit seed operand instead of x2d
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
        return y[crop], dict(AV=S.quantise_codes(av[crop]), n_windows=int(qw.shape[0]), extent=[H, W])

    def _ffn_branched(self, x2d, sx2d):
        """d22_backend.BranchedWindowSm120._ffn, copied line for line; SEED marks the only change."""
        obj = self.window
        w, C, H = obj._w, obj.C, obj.head_count
        z = S._half(sx2d * w['ffn_cos_skip'])                  # SEED: explicit seed operand instead of x2d
        for h in range(H):
            w_exp = w['ffn_expand_weight'][h].transpose(1, 2, 0, 3).reshape(C, 128)
            e = D17._chain(x2d, w_exp, None, D22.splits32(C))
            g = S.decode_codes(S.feed_forward_gate(e)).astype(np.float64).reshape(-1, 128)
            s = D17._chain(g, w['ffn_branch_projection_weight'][h].reshape(128, 32), None, D17.K128)
            u = S.decode_codes(S.quantise_codes(s)).astype(np.float64)
            z = S.qmma_rowwise(u, np.ascontiguousarray(w['ffn_output_projection_weight'][32 * h:32 * h + 32]), z)
        return z

    def _body_branched(self, codes, seed_x):
        """d22_backend.BranchedWindowSm120.run_codes, copied line for line; SEED marks the only changes."""
        obj = self.window
        C, side = obj.C, obj.side
        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != (side, side, C):
            raise Unsupported('uint8 [%d,%d,%d] E4M3 codes required' % (side, side, C))
        if np.any((codes & 0x7F) == 0x7F):
            raise Unsupported('E4M3 NaN codes in input')
        seed_x = np.asarray(seed_x, dtype=np.float64)
        if seed_x.shape != (side, side, C):
            raise Unsupported('FFN seed operand must be [%d,%d,%d]' % (side, side, C))
        w = obj._w
        ox, oy = obj.origin_xy
        gx, gy = D22.BW.canvas_grid(obj.origin_xy, side)
        Hc, Wc = gy * 8, gx * 8
        rows, cols = slice(-oy, -oy + side), slice(-ox, -ox + side)
        canvas = np.zeros((Hc, Wc, C), np.float64)
        canvas[rows, cols] = S.decode_codes(codes).astype(np.float64)
        scanvas = np.zeros((Hc, Wc, C), np.float64)           # SEED: seed operand placed on the same canvas
        scanvas[rows, cols] = seed_x
        z = self._ffn_branched(canvas.reshape(-1, C), scanvas.reshape(-1, C))   # SEED
        qz = S.quantise_codes(z)
        xq = S.decode_codes(qz).astype(np.float64)
        qkv = D17._chain(xq, w['qkv_weight'], None, D22.splits32(C))
        heads = []
        for h in range(obj.head_count):
            c = slice(32 * h, 32 * h + 32)
            q, _ = S.normalise(qkv[:, 0:C][:, c].reshape(Hc, Wc, 32), obj.tables, float(w['attn_scale'][h]))
            k, _ = S.normalise(qkv[:, C:2 * C][:, c].reshape(Hc, Wc, 32), obj.tables)
            v = qkv[:, 2 * C:3 * C][:, c].reshape(Hc, Wc, 32)
            parts = []
            for t in (q, k, v):
                pub = S.decode_codes(S.quantise_codes(t))
                parts.append(np.asarray(M.partition_windows(
                    torch.from_numpy(np.ascontiguousarray(pub, np.float32)).unsqueeze(0), 8).numpy(), np.float64))
            qw, kw, vw = parts
            score = S.qmma_batched(qw, np.ascontiguousarray(np.transpose(kw, (0, 2, 1))),
                                   np.broadcast_to(w['attn_bias'][h].astype(np.float16), (qw.shape[0], 64, 64)))
            pw = S.decode_codes(S.quantise_codes(S.published_attention_weights(score, obj.tables))).astype(np.float64)
            acc = None
            for keys in (slice(0, 32), slice(32, 64)):
                acc = S.qmma_batched(np.ascontiguousarray(pw[:, :, keys]), np.ascontiguousarray(vw[:, keys, :]), acc)
            av = M.e4m3_round_trip(torch.from_numpy(np.ascontiguousarray(acc, np.float32)))
            heads.append(M.reverse_windows(av, batch_count=1, height=Hc, width=Wc, window_size=8)[0].numpy())
        av = np.concatenate(heads, axis=-1).astype(np.float64).reshape(-1, C)
        zfac = xq if obj.tail_seed == 'published_qz' else z.astype(np.float64)
        seed = S._half(zfac * w['attn_cos_skip'])
        y = D17._chain(av, w['projection_weight'], seed, D22.splits32(C)).reshape(Hc, Wc, C)
        crop = (rows, cols)
        return y[crop], dict(AV=S.quantise_codes(av.reshape(Hc, Wc, C)[crop]), QZ=qz.reshape(Hc, Wc, C)[crop],
                             canvas_hw=(Hc, Wc), n_windows=int(gx * gy))

    def source_body(self, codes):
        """the unchanged source body (for the equivalence gate): -> Y half"""
        if self.heads == 1:
            return self.window._body(codes)[0]
        return self.window.run_codes(codes)["Y_half"]

    # ------------------------------------------------------------------ entry
    def run_raw(self, x_raw: bytes, skip_raw: bytes):
        mc, stages = self.merged(self.decode_input(x_raw), self.decode_skip(skip_raw))
        y_half, info = self.body(mc, self.seed_operand(mc, stages["merged_half"]))
        y = np.asarray(S.quantise_codes(y_half), np.uint8)
        if y.shape != (self.out_hw, self.out_hw, self.C):
            raise Unsupported("window output shape %s" % (y.shape,))
        return tin_encode(y, self.C, self.out_hw), dict(merged_codes=mc, **stages)

    def describe(self):
        prefix = "block%d.layer0." % self.block
        body_tensors = (["weight1", "weight2"] if self.heads == 1 else
                        ["ffn_expand_weight", "ffn_branch_projection_weight", "ffn_output_projection_weight"])
        return dict(name="transition_backend.TransitionCandidate", block=self.block, heads=self.heads, channels=self.C,
                    out_hw=self.out_hw, in_hw=self.in_hw, in_channels=self.in_C, in_planes=self.in_planes,
                    origin_xy=list(self.origin_xy), ffn_seed=self.ffn_seed, tail_seed=self.tail_seed,
                    projection_splits=splits32(self.in_C), merge="_half(decode(skip) * sin + nearest2(projected))",
                    weight_tensors_read=[prefix + t for t in ["weight0", "sin"] + body_tensors +
                                         ["ffn_cos_skip", "qkv_weight", "attn_bias", "attn_scale", "projection_weight", "attn_cos_skip"]],
                    body_source="d13_backend.SingleWindowSm120._body" if self.heads == 1 else "d22_backend.BranchedWindowSm120.run_codes/_ffn",
                    codec_in="PLANAR16 S%d-C%d" % (self.in_hw, self.in_C), codec_skip_out="TIN%d" % self.C,
                    arithmetic_sources=[str(Path(D13.__file__).resolve()), str(Path(D17.__file__).resolve()), str(Path(D22.__file__).resolve())])
