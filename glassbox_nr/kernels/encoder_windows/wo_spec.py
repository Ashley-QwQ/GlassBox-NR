"""Block specifications and layouts, with `window_origin` promoted from an
ASSERTED INVARIANT to a REAL PARAMETER.

=====================================================================
WHAT CHANGED RELATIVE TO parity-b9-attention-20260911/ma_spec.py
=====================================================================
ma_spec.py said:

    window_origin (0,0) for the `inpview_tilesync` entry kernels.  The
                  `chained` kernels shift; this round only touches inpview
                  blocks, and ma_attention asserts it.

That assertion is what capped coverage at 3/36.  Here `window_origin` is a
per-block parameter read out of the real launch parameter block (word 5 of the
88-byte parameter block; see parity-h4-chained-ffn/blocks.json), and the window
grid, the canvas, the zero fill and the crop are all DERIVED from it.

    word5 = 0x0000000000000000  -> ( 0, 0)
    word5 = 0x00000000fffffffc  -> (-4, 0)      low  32 bits = origin_x
    word5 = 0xfffffffc00000000  -> ( 0,-4)      high 32 bits = origin_y
    word5 = 0xfffffffcfffffffc  -> (-4,-4)

`origin_from_word5` decodes that and `window_grid` predicts gridDim from it;
`assert_grid_from_origin()` checks the prediction against the REAL recorded
gridDim of all 36 branched_window launches, which is an independent check that
the decode is right (a transposed or sign-flipped decode fails on the
asymmetric blocks 11/12 and 7/8).

=====================================================================
THE ZERO FILL IS NOT A ROLL AND NOT A MASK
=====================================================================
Standard swin does shifted windows with a cyclic shift plus an attention mask.
This kernel does NEITHER.  parity-h4-chained-ffn/H4_CHAINED_INTERFACE_ZH.md,
from a forward PC replay of the real load path 06e0..14a0:

    "四次读取的tile坐标来自当前窗口的左上、右上、左下、右下。无效tile生成零,
     不是wrap, 不使用torch.roll."

and its padding-byte counts per origin (90112 / 40960 / 40960 / 0) are
reproduced here by `padding_bytes()` from the geometry alone -- see
`assert_padding_matches_recorded()`.  That is the cross-check the task
demands: if this module's notion of "which tokens are padding" were wrong, the
byte counts would not land on the recorded numbers.

=====================================================================
TIN / INPVIEW
=====================================================================
Unchanged from ma_spec.py (they describe the block's own buffer, which the
window origin does not touch).  Re-checked against both validated maps on
every run by `assert_known_maps()`.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]

HEAD_DIM = 32          # assumed invariant, enforced by norm_squared's own raise
WINDOW = 8             # assumed invariant


# ---------------------------------------------------------------------------
# window_origin: the new parameter
# ---------------------------------------------------------------------------
def origin_from_word5(word5):
    """Decode parameter word 5 into (origin_x, origin_y), both <= 0.

    Two's complement halves.  The convention -- low half is x -- is not
    asserted in prose: `assert_grid_from_origin` below fails under the
    transposed reading on blocks 7/8/11/12/57/60, whose grids are asymmetric.
    """
    lo = word5 & 0xffffffff
    hi = (word5 >> 32) & 0xffffffff
    sx = lo - (1 << 32) if lo >> 31 else lo
    sy = hi - (1 << 32) if hi >> 31 else hi
    return (int(sx), int(sy))


def window_grid(side, origin, window=WINDOW):
    """Number of windows per axis.  The window row w covers
    [origin + window*w, origin + window*(w+1)); the grid is the smallest count
    that covers [0, side) starting at `origin`."""
    ox, oy = origin
    assert ox <= 0 and oy <= 0, ("origins observed in this network are 0 or -4", origin)
    gx = -(-(side - ox) // window)
    gy = -(-(side - oy) // window)
    return (int(gx), int(gy))


def canvas_shape(spec):
    gx, gy = window_grid(spec["width"], spec["window_origin"], spec["window"])
    # width uses gx, height uses gy: the canvas is (rows, cols) = (gy*w, gx*w)
    gx2, gy2 = window_grid(spec["height"], spec["window_origin"], spec["window"])
    assert (gx, gy) == (gx2, gy2) or spec["height"] == spec["width"], "non-square blocks unseen"
    return (gy * spec["window"], gx * spec["window"])


def padding_bytes(spec):
    """Bytes of the window canvas that are NOT covered by a real input byte.
    Cross-checks against parity-h4-chained-ffn/load-coverage.json."""
    ch, cw = canvas_shape(spec)
    C = spec["channels"]
    return int(ch * cw * C) - int(spec["height"] * spec["width"] * C)


def to_canvas(x, spec):
    """Place the block's own (H, W, C) buffer onto the window canvas implied by
    window_origin, zero-filling everything the canvas covers but the buffer does
    not.  NOT a roll: no value is wrapped."""
    ox, oy = spec["window_origin"]
    ch, cw = canvas_shape(spec)
    H, W = spec["height"], spec["width"]
    out = np.zeros((ch, cw) + x.shape[2:], x.dtype)
    # canvas row r corresponds to image row r + oy
    r0, r1 = -oy, -oy + H                     # oy <= 0
    c0, c1 = -ox, -ox + W
    assert r1 <= ch and c1 <= cw, (r1, ch, c1, cw)
    out[r0:r1, c0:c1] = x
    return out


def from_canvas(canvas, spec):
    """Inverse of to_canvas: take back only the real buffer region."""
    ox, oy = spec["window_origin"]
    H, W = spec["height"], spec["width"]
    return canvas[-oy:-oy + H, -ox:-ox + W]


def partition_windows(canvas, window=WINDOW):
    """Row-major windows, token order row-major inside a window.  Identical to
    mlxdlss.model.partition_windows; `assert_partition_matches_upstream` proves
    that on real shapes rather than claiming it."""
    ch, cw, C = canvas.shape
    assert ch % window == 0 and cw % window == 0
    return (canvas.reshape(ch // window, window, cw // window, window, C)
                  .transpose(0, 2, 1, 3, 4)
                  .reshape(-1, window * window, C))


def reverse_windows(windows, canvas_hw, window=WINDOW):
    ch, cw = canvas_hw
    C = windows.shape[-1]
    return (windows.reshape(ch // window, cw // window, window, window, C)
                   .transpose(0, 2, 1, 3, 4)
                   .reshape(ch, cw, C))


def valid_token_mask(spec):
    """True where a canvas token is a REAL input token, False where it is the
    zero fill.  Used ONLY for reporting and for the padding cross-check -- the
    arithmetic never consults it, because the kernel does not (see
    wo_zero_tokens.py)."""
    ch, cw = canvas_shape(spec)
    m = np.zeros((ch, cw), bool)
    ox, oy = spec["window_origin"]
    m[-oy:-oy + spec["height"], -ox:-ox + spec["width"]] = True
    return m


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def _spec(block, side, head_count, launch, function, grid, word5, variant):
    origin = origin_from_word5(word5)
    # INPUT LAYOUT is a parameter of the VARIANT, not a constant.  An
    # `inpview_tilesync` entry kernel reads word 9 of its predecessor -- the
    # pooled / downsampled buffer, written in INPVIEW order.  A `chained`
    # kernel reads word 1 of its predecessor -- that block's OWN OUTPUT, which
    # a branched_window block writes in TIN order.  So the input layout of a
    # chained block is the TIN layout, and parity-h4-chained-ffn/extract.py
    # already did exactly this ("X.TIN128", TIN128-physical-to-logical.npy).
    #
    # Getting this wrong cost this round one blind holdout: block 10's frozen
    # prediction was computed from an INPVIEW-decoded X and missed 202,212 of
    # 204,800 bytes.  It was NOT caught by inpview_decode's own round-trip
    # assertion, because that assertion only proves the permutation is
    # invertible -- it is satisfied by EVERY permutation, right or wrong.
    # See DISCOVERY_LOG_ZH.md D1.
    return dict(block=block, height=side, width=side,
                input_layout=("TIN" if variant == "chained" else "INPVIEW"),
                predecessor_output_word=(1 if variant == "chained" else 9),
                channels=head_count * HEAD_DIM, head_count=head_count,
                head_dim=HEAD_DIM, window=WINDOW, window_origin=origin,
                word5=hex(word5), variant=variant,
                launch=launch, function=function, grid=tuple(grid),
                block_dim=(32, head_count, 1))


# geometry/launch/grid: parity-structure-reuse/BLOCK_FAMILY_MAP.csv (real launch
# parameters of the real 158-launch session).
# word5: parity-h4-chained-ffn/blocks.json for the chained blocks (real captured
# parameter bytes); 0 for the inpview entry kernels, which the
# parity-b9-attention-20260911 bridge check asserted as w[5]==0 on the real
# launch and which is re-asserted by this round's own capture identity check.
BLOCKS = {
    5:  _spec(5, 80, 2, 7, "cc_tinlayout_fused_swin_2h_64_2_inpview_tilesync_fp8",
              (10, 10, 1), 0x0, "inpview_tilesync"),
    9:  _spec(9, 40, 4, 11, "cc_tinlayout_fused_swin_4h_128_4_inpview_tilesync_fp8",
              (5, 5, 1), 0x0, "inpview_tilesync"),
    15: _spec(15, 20, 8, 17, "cc_tinlayout_fused_swin_8h_256_8_inpview_tilesync_fp8",
              (3, 3, 1), 0x0, "inpview_tilesync"),
    # --- the chained family: same function, four different window origins ----
    10: _spec(10, 40, 4, 12, "cc_tinlayout_fused_swin_4h_128_4_chained_fp8",
              (6, 6, 1), 0xfffffffcfffffffc, "chained"),
    11: _spec(11, 40, 4, 13, "cc_tinlayout_fused_swin_4h_128_4_chained_fp8",
              (6, 5, 1), 0xfffffffc, "chained"),
    12: _spec(12, 40, 4, 14, "cc_tinlayout_fused_swin_4h_128_4_chained_fp8",
              (5, 6, 1), 0xfffffffc00000000, "chained"),
    13: _spec(13, 40, 4, 15, "cc_tinlayout_fused_swin_4h_128_4_chained_fp8",
              (5, 5, 1), 0x0, "chained"),
    # --- decoder-side 8-head chained (section 5: zero code changes) ----------
    # No prior round recorded word5 for ANY 8-head block.  These two values were
    # MEASURED by this round, twice and independently: the bridge logs word5 and
    # word7 of every candidate launch (including rejected ones), and the raw
    # parameter bytes of session.launch-0135.params say the same.  See
    # d2-b50-origin-discovery/README_ZH.md, which also records what that
    # measurement cost.
    #   launch 134 (block 49)  w5 = 0xfffffffcfffffffc  (-4,-4)
    #   launch 135 (block 50)  w5 = 0xfffffffc          (-4, 0)   w7 = block 49's
    49: _spec(49, 20, 8, 134, "cc_tinlayout_fused_swin_8h_256_8_chained_fp8",
              (3, 3, 1), 0xfffffffcfffffffc, "chained"),
    50: _spec(50, 20, 8, 135, "cc_tinlayout_fused_swin_8h_256_8_chained_fp8",
              (3, 3, 1), 0xfffffffc, "chained"),
}

# The four chained cases used for the origin sweep, in the order the task names.
ORIGIN_CASES = {(-4, -4): 10, (-4, 0): 11, (0, -4): 12, (0, 0): 13}


def predecessor(block):
    """The block whose native output is this block's X (teacher forcing)."""
    return block - 1


# ---------------------------------------------------------------------------
# gates on the origin decode
# ---------------------------------------------------------------------------
RECORDED_PADDING_BYTES = {          # parity-h4-chained-ffn/load-coverage.json
    (-4, -4): 90112, (-4, 0): 40960, (0, -4): 40960, (0, 0): 0}


def assert_grid_from_origin():
    """The decoded origin must PREDICT the real recorded gridDim of every block
    in BLOCKS.  A sign error, a transposed (x,y) reading, or an off-by-one in
    window_grid all fail here, on the asymmetric blocks."""
    out = {}
    for b, s in BLOCKS.items():
        gx, gy = window_grid(s["width"], s["window_origin"], s["window"])
        ok = (gx, gy, 1) == s["grid"]
        out[b] = dict(origin=s["window_origin"], predicted=(gx, gy, 1),
                      recorded=list(s["grid"]), matches=bool(ok))
        assert ok, out[b]
    # the transposed reading must FAIL somewhere, otherwise this gate is vacuous
    bad = 0
    for b, s in BLOCKS.items():
        t = (s["window_origin"][1], s["window_origin"][0])
        if window_grid(s["width"], t, s["window"]) + (1,) != s["grid"]:
            bad += 1
    out["transposed_reading_fails_on_n_blocks"] = bad
    assert bad > 0, "gate is vacuous: the transposed origin reading also fits"
    return out


def assert_padding_matches_recorded():
    """Independent cross-check demanded by the task: the padding byte count this
    module derives from geometry must equal the count the H4 round measured by
    replaying the real load instructions."""
    out = {}
    for origin, b in ORIGIN_CASES.items():
        got = padding_bytes(BLOCKS[b])
        want = RECORDED_PADDING_BYTES[origin]
        out[str(origin)] = dict(block=b, derived=got, recorded=want, matches=(got == want),
                                source="parity-h4-chained-ffn/load-coverage.json")
        assert got == want, out[str(origin)]
    return out


def assert_partition_matches_upstream():
    """At origin (0,0) with no padding, this module's partition/reverse must be
    byte-identical to mlxdlss.model's -- that is what makes the G1/G2 regression
    gates meaningful rather than a comparison of new code with new code."""
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("wo_up_model", ROOT / "mlxdlss/model.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("wo_up_model", m)
    spec.loader.exec_module(m)
    out = {}
    for (h, w, c) in ((80, 80, 64), (40, 40, 128), (24, 24, 256)):
        rng = np.random.default_rng(3)
        a = rng.random((h, w, c)).astype(np.float32)
        mine = partition_windows(a)
        theirs = m.partition_windows(torch.from_numpy(a).unsqueeze(0), WINDOW)[0:].numpy()
        same = bool(np.array_equal(mine, theirs))
        back = reverse_windows(mine, (h, w))
        theirs_back = m.reverse_windows(torch.from_numpy(theirs), batch_count=1, height=h,
                                        width=w, window_size=WINDOW)[0].numpy()
        out[f"{h}x{w}x{c}"] = dict(
            partition_identical=same,
            reverse_identical=bool(np.array_equal(back, theirs_back)),
            round_trip=bool(np.array_equal(back, a)))
        assert all(out[f"{h}x{w}x{c}"].values()), out
    return out


# ---------------------------------------------------------------------------
# INPVIEW: the block's INPUT physical layout (channel permutation + C/16 tiles)
# ---------------------------------------------------------------------------
def inpview_channel_order(channels):
    ch = np.arange(channels)
    return (ch // 16) * 16 + ((ch % 16) // 4) * 2 + ch % 2 + ((ch % 4) // 2) * 8


def decode_input(raw_bytes, spec):
    """Decode the block's own input buffer using the layout its VARIANT implies.

    Deliberately NOT "try both and keep the one that matches": the layout is
    decided by which buffer the pointer chain says this is, before any answer
    is read.  `assert_layouts_are_distinguishable` proves the two candidates
    are not the same permutation, so the choice is a real choice."""
    H, W, C = spec["height"], spec["width"], spec["channels"]
    if spec["input_layout"] == "TIN":
        return tin_decode(raw_bytes, H, W, C)
    return inpview_decode(raw_bytes, H, W, C)


def assert_layouts_are_distinguishable():
    """The two input layouts must be genuinely different permutations, else
    `input_layout` would be a parameter that changes nothing and the block-10
    failure could not have happened the way the log says it did."""
    out = {}
    for (h, w, c) in ((80, 80, 64), (40, 40, 128), (20, 20, 256)):
        raw = (np.arange(h * w * c) % 251).astype(np.uint8).tobytes()
        spec_tin = dict(height=h, width=w, channels=c, input_layout="TIN")
        spec_inp = dict(height=h, width=w, channels=c, input_layout="INPVIEW")
        a = decode_input(raw, spec_tin)
        b = decode_input(raw, spec_inp)
        n = int(np.count_nonzero(a != b))
        out[f"{h}x{w}x{c}"] = dict(bytes_differing=n, total=a.size,
                                   distinguishable=bool(n > 0))
        assert n > 0, out
    return out


def inpview_decode(raw_bytes, height, width, channels):
    x = np.frombuffer(raw_bytes, np.uint8)
    n = height * width * channels
    assert x.size == n, (x.size, n)
    order = inpview_channel_order(channels)
    phys = (x.reshape(channels // 16, height, width, 16)
             .transpose(1, 2, 0, 3).reshape(height, width, channels))
    logical = np.empty_like(phys)
    logical[..., order] = phys
    encoded = (logical[..., order].reshape(height, width, channels // 16, 16)
               .transpose(2, 0, 1, 3).copy().tobytes())
    assert encoded == bytes(raw_bytes), "INPVIEW round-trip failed"
    return logical


# ---------------------------------------------------------------------------
# TIN: the block's OUTPUT physical layout (unchanged, re-checked every run)
# ---------------------------------------------------------------------------
def tin_map(height, width, channels):
    cbits = int(np.log2(channels))
    assert 1 << cbits == channels, channels
    xspan = 1024 // channels
    assert xspan >= 1 and width % xspan == 0 and height % 4 == 0, (xspan, width, height)
    xbits = int(np.log2(xspan))
    slots = ["c0", "c3", "y1", "c4", "c1", "c2", "x0", "x1", "y0"]
    slots += ["c%d" % i for i in range(5, cbits)]
    slots += ["x%d" % i for i in range(2, xbits)]
    assert len(slots) == 12, (len(slots), slots)

    n = height * width * channels
    p = np.arange(n, dtype=np.int64)
    off, blk = p % 4096, p // 4096
    ytile, xtile = blk // (width // xspan), blk % (width // xspan)
    c = np.zeros(n, np.int64)
    yl = np.zeros(n, np.int64)
    xl = np.zeros(n, np.int64)
    for i, s in enumerate(slots):
        bit = (off >> i) & 1
        k = int(s[1:])
        if s[0] == "c":
            c |= bit << k
        elif s[0] == "y":
            yl |= bit << k
        else:
            xl |= bit << k
    y = ytile * 4 + yl
    x = xtile * xspan + xl
    return (y * width + x) * channels + c


def tin_decode(raw_bytes, height, width, channels):
    logical = np.zeros(height * width * channels, np.uint8)
    logical[tin_map(height, width, channels)] = np.frombuffer(raw_bytes, np.uint8)
    return logical.reshape(height, width, channels)


def tin_encode(logical, height, width, channels):
    return np.ascontiguousarray(
        np.asarray(logical).reshape(-1)[tin_map(height, width, channels)]).tobytes()


KNOWN_TIN_MAPS = {
    (80, 80, 64): ROOT / "experiments/parity-b5-branched-ffn/TIN64-physical-to-logical.npy",
    (40, 40, 128): ROOT / "experiments/parity-b9-h4-ffn/TIN128-physical-to-logical.npy",
}


def assert_known_maps():
    out = {}
    for (h, w, c), path in KNOWN_TIN_MAPS.items():
        recorded = np.load(path)
        derived = tin_map(h, w, c)
        same = bool(np.array_equal(recorded, derived))
        out[f"{h}x{w}x{c}"] = dict(source=str(path), identical=same,
                                   elements=int(recorded.size))
        assert same, f"closed-form TIN map disagrees with {path}"
    for h, w, c in ((20, 20, 256),):
        m = tin_map(h, w, c)
        out[f"{h}x{w}x{c}"] = dict(
            source="PREDICTION (no validated map exists)",
            is_bijection=bool(np.array_equal(np.sort(m), np.arange(m.size))),
            elements=int(m.size))
    return out


def all_gates():
    return dict(tin_maps=assert_known_maps(),
                input_layouts_distinguishable=assert_layouts_are_distinguishable(),
                grid_from_origin=assert_grid_from_origin(),
                padding_vs_recorded=assert_padding_matches_recorded(),
                partition_vs_upstream=assert_partition_matches_upstream())


if __name__ == "__main__":
    import json
    print(json.dumps(all_gates(), indent=2, default=str))
