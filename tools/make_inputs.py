"""Generate the upload-colour input for the two published samples.

    python -B tools/make_inputs.py                     # both samples
    python -B tools/make_inputs.py --sample c1-C-img-full
    python -B tools/make_inputs.py --check             # verify only, write nothing

Both images are procedural: closed-form formulas plus a seeded
numpy.random.default_rng, with no external asset and nothing taken from any
native capture. Each image is converted to the RGBA16F upload buffer the
runtime reads and written to

    anchors/<sample>/inputs/colour.rgba16

which is where tools/run_pipeline.py looks for it. Before writing, the bytes
are checked against the sha256 that the published reference results were
produced from; a mismatch writes nothing and exits non-zero.

  c1-C-img-full      (public name sample-1-reference): checker / gradient /
                     wave scene.
  d23-n2-nat320x240  (public name sample-3-procedural-scene): gradient plus
                     seeded shapes, fine detail and strokes -- made to
                     resemble photographic content, not an actual photograph.

The float32 -> half conversion reproduces, bit for bit, the project's own
capture bridge (self-written code, not vendor code, not published): the
mantissa is truncated rather than rounded, and alpha is a constant 1.0.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHORS = PKG_ROOT / "anchors"

WIDTH, HEIGHT = 320, 240
SEED_SAMPLE_3 = 20260912
SAMPLE_3_VARIANT = 3

EXPECTED_UPLOAD_COLOUR_SHA256 = {
    "c1-C-img-full": "8a1094b3d91d270a23344fccd84af7a492155af7ae6d6658c38ad1e60a4159f0",
    "d23-n2-nat320x240": "6f6c6b497dc612a4f2e9b1a4e87c96523ce859ee13c487df0307c2158e2c2504",
}


def sample_1_reference(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """(height, width, 3) float32 in [0, 1]."""
    y, x = np.indices((height, width))
    u = x / max(width - 1, 1)
    v = y / max(height - 1, 1)
    img = np.stack([
        0.5 + 0.45 * np.sin(x / 7.3) * np.cos(y / 11.7),
        0.15 + 0.7 * ((x // 13 + y // 9) % 2),
        0.1 + 0.8 * (0.65 * u + 0.35 * v),
    ], -1).astype(np.float32)
    return np.ascontiguousarray(img)


def _scene(width, height, dx=0.0, dy=0.0, variant=0, occluder=None, tone=None, seed=SEED_SAMPLE_3):
    rng = np.random.default_rng(seed + variant)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    x = xs - dx
    y = ys - dy
    r = 0.15 + 0.7 * (x / max(width - 1, 1))
    g = 0.2 + 0.6 * (y / max(height - 1, 1))
    b = 0.5 + 0.35 * np.sin(x / 23.0 + y / 31.0)
    img = np.stack([r, g, b], -1)
    for _ in range(9):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        sx, sy = rng.uniform(8, width / 4), rng.uniform(8, height / 4)
        col = rng.uniform(0, 1, 3)
        if rng.uniform() < 0.5:
            mask = (np.abs(x - cx) < sx) & (np.abs(y - cy) < sy)
        else:
            mask = (x - cx) ** 2 + (y - cy) ** 2 < sx ** 2
        img[mask] = col
    fine = 0.08 * np.sin(x * 1.9) * np.cos(y * 2.3)
    img = img + fine[..., None]
    for k in range(5):
        angle = rng.uniform(0, np.pi)
        off = rng.uniform(-width / 2, width / 2)
        d = np.abs(np.cos(angle) * (x - width / 2) + np.sin(angle) * (y - height / 2) - off)
        img[d < 0.8 + 0.3 * k] = [1.0, 1.0, 1.0] if k % 2 else [0.0, 0.0, 0.0]
    if occluder is not None:
        ox, oy, ow, oh = occluder
        mask = (xs >= ox) & (xs < ox + ow) & (ys >= oy) & (ys < oy + oh)
        img[mask] = [0.9, 0.1, 0.1]
    if tone is not None:
        img = np.clip(img, 0, 1) ** tone
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def sample_3_procedural_scene(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """(height, width, 3) float32 in [0, 1]."""
    return _scene(width, height, variant=SAMPLE_3_VARIANT)


def float32_to_half_bits(f: np.ndarray) -> np.ndarray:
    """IEEE754 float32 -> half bit pattern, truncating the mantissa."""
    x = f.astype(np.float32).view(np.uint32)
    s = (x >> 16) & 0x8000
    e = ((x >> 23) & 0xFF).astype(np.int32) - 127 + 15
    m = x & 0x7FFFFF
    out = np.zeros_like(x, dtype=np.uint32)

    small = e <= 0
    very_small = small & (e < -10)
    denorm = small & ~very_small
    normal = (~small) & (e < 31)
    overflow = (~small) & (e >= 31)

    out[very_small] = s[very_small]
    if np.any(denorm):
        ee = e[denorm]
        mm = (m[denorm] | 0x800000) >> (1 - ee)
        out[denorm] = s[denorm] | (mm >> 13)
    out[overflow] = s[overflow] | 0x7C00
    out[normal] = s[normal] | (e[normal].astype(np.uint32) << 10) | (m[normal] >> 13)

    return out.astype(np.uint16)


def rgb_float32_to_upload_rgba16f(rgb: np.ndarray) -> bytes:
    """(h, w, 3) float32 -> row-major RGBA16F bytes (clamp to [0, 1], alpha 1.0)."""
    h, w, c = rgb.shape
    assert c == 3
    clamped = np.clip(rgb, 0.0, 1.0)
    rgba = np.empty((h, w, 4), dtype=np.float32)
    rgba[..., 0:3] = clamped
    rgba[..., 3] = 1.0
    half = float32_to_half_bits(rgba)
    return half.astype("<u2").tobytes()


GENERATORS = {
    "c1-C-img-full": sample_1_reference,
    "d23-n2-nat320x240": sample_3_procedural_scene,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the published samples' upload-colour input")
    ap.add_argument("--sample", default="all", choices=[*GENERATORS, "all"])
    ap.add_argument("--anchors-dir", default=str(DEFAULT_ANCHORS))
    ap.add_argument("--check", action="store_true", help="verify the generated bytes only; write nothing")
    args = ap.parse_args()

    samples = list(GENERATORS) if args.sample == "all" else [args.sample]
    ok = True
    for sample in samples:
        data = rgb_float32_to_upload_rgba16f(GENERATORS[sample]())
        got = hashlib.sha256(data).hexdigest()
        want = EXPECTED_UPLOAD_COLOUR_SHA256[sample]
        if got != want:
            ok = False
            print(f"{sample}: MISMATCH (got {got}, want {want}); nothing written")
            continue
        if args.check:
            print(f"{sample}: MATCH {got} ({len(data)} bytes)")
            continue
        dest = Path(args.anchors_dir) / sample / "inputs" / "colour.rgba16"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"{sample}: MATCH {got} ({len(data)} bytes) -> {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
