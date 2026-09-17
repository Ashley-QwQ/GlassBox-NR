"""CPU reference for block 0's noise generator, with a refusing domain guard.

Scope, stated up front:

  * `uniforms(x, y, frame_word)` is COMPLETE and bit-exact on a CPU.  It is
    32-bit integer arithmetic plus one exact `I2F.U32` and one exact multiply
    by 2**-24; no hardware approximation is involved.

  * `gaussians(...)` is NOT complete.  It needs MUFU.LG2, MUFU.SQRT, MUFU.COS
    and MUFU.SIN over the domains A1 derived, and those tables have not been
    measured (A2 is the GPU step, not run).  Calling it without tables raises
    `OutsideMeasuredDomain` rather than substituting `math.log2` -- a CPU
    libm answer is not what the hardware computes and returning one would be
    a wrong answer delivered confidently.

The domain guard follows `parity-qk-score-resource/norm_trace.py:98-101`:
outside the measured set, raise; never fall back.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True

import math
import struct

import numpy as np

M32 = np.uint32(0xFFFFFFFF)

# Random generation constants
SEED_Y_MUL = np.uint32(0xD8163841)          # multiplier for y coordinate
SEED_X_MUL = np.uint32(0x8DA6B343)          # multiplier for x coordinate
SEED_F_MUL = np.uint32(0x9E3779B9)          # multiplier for frame word
SEED_XOR = np.uint32(0x243F6A88)            # initial XOR constant
PCG_MUL = np.uint32(0x108EF2D9)             # PCG state multiplier
STREAMS = [                                 # (multiplier, addend, PC, role)
    (np.uint32(0xCAA5B80D), np.uint32(0x21DD796B), "0x550", "radius 1 -> LG2 @0x7f0"),
    (np.uint32(0x2C9277B5), np.uint32(0xAC564B05), "0x5a0", "radius 2 -> LG2 @0x840"),
    (np.uint32(0x83232C31), np.uint32(0x3463E0AC), "0x570", "angle 1 -> COS @0x870"),
    (np.uint32(0xFA6DC5F9), np.uint32(0x4712A88E), "0x5b0", "angle 2 -> SIN @0x8e0 / COS @0x910"),
]
TWO_PI = np.float32(6.2831854820251464844)      # 2 * pi
INV_TWO_PI = np.float32(0.15915493667125701904)  # 1 / (2 * pi)
LN2 = np.float32(0.69314718246459960938)         # ln(2)
MINUS_TWO = np.float32(-2.0)                     # -2.0

UNIFORM_MIN_BITS = 0x33800000                    # 2**-24
UNIFORM_MAX_BITS = 0x3F800000                    # 1.0


class OutsideMeasuredDomain(Exception):
    """Raised instead of returning a value that was never measured."""


def _pcg_output(state):
    """((state >> ((state>>28)+4)) ^ state) * 0x108ef2d9"""
    state = state.astype(np.uint32)
    sh = ((state >> np.uint32(28)) + np.uint32(4)) & np.uint32(31)
    return (((state >> sh) ^ state) * PCG_MUL).astype(np.uint32)


def seed(x, y, frame_word):
    """Compute seed from x, y, and frame_word."""
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    f = np.uint32(frame_word)
    s = (y * SEED_Y_MUL) ^ (x * SEED_X_MUL) ^ (f * SEED_F_MUL)
    return (s ^ SEED_XOR).astype(np.uint32)


def uniforms(x, y, frame_word):
    """The four uniforms, in the order the four `FMUL x 2**-24` sites consume
    them: [radius1, radius2, angle1, angle2].  Returns float32."""
    h = _pcg_output(seed(x, y, frame_word))
    h = ((h >> np.uint32(22)) ^ h).astype(np.uint32)
    out = []
    for mul, add, _pc, _role in STREAMS:
        t = _pcg_output((h * mul + add).astype(np.uint32))
        k = (((t >> np.uint32(30)) ^ (t >> np.uint32(8))) + np.uint32(1)).astype(np.uint32)
        out.append((k.astype(np.float64) * 2.0 ** -24).astype(np.float32))
    return out


def fmul_rz(a, b):
    """FMUL.RZ on float32 arrays.  f32*f32 is exact in float64, so round-to-
    nearest then step one ulp toward zero where rounding went the wrong way."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    exact = a.astype(np.float64) * b.astype(np.float64)
    y = np.atleast_1d(exact.astype(np.float32))
    bits = y.view(np.uint32).copy()
    over = np.atleast_1d(np.abs(y.astype(np.float64)) > np.abs(exact))
    bits[over] -= 1
    out = bits.view(np.float32)
    return out.reshape(np.shape(exact)) if np.ndim(exact) else out[0]


def angle_probe_arg(u):
    """Compute unreduced argument for measured trigonometric functions.

    This, NOT the reduced value, is what the measured COS/SIN tables are
    indexed by.  The probe kernel those tables came from contains the same
    reduction constant as the target kernel, so the table covers the reduction
    and the trigonometric evaluation together.
    """
    return (np.asarray(u, dtype=np.float32) * TWO_PI).astype(np.float32)


def angles(u):
    """The reduced argument in revolutions. Pure fp32, exact on a CPU.

    Kept because it is what the calculation sequence actually computes.
    It is NOT the index into the measured tables -- use `angle_probe_arg` for that.
    """
    return fmul_rz(angle_probe_arg(u), INV_TWO_PI)


def assert_uniform_domain(u):
    """A1's derived domain: exactly {k*2**-24 : k = 1..2**24}.

    Refuses rather than clamping.  The point of this function is to fire, so
    `selftest()` proves it fires in every direction.
    """
    u = np.asarray(u, dtype=np.float32)
    bits = u.view(np.uint32)
    bad = (bits < UNIFORM_MIN_BITS) | (bits > UNIFORM_MAX_BITS)
    if np.any(bad):
        i = int(np.argmax(bad))
        raise OutsideMeasuredDomain(
            "uniform 0x%08x (%r) is outside the derived domain [2**-24, 1.0]"
            % (int(bits.reshape(-1)[i]), float(u.reshape(-1)[i])))
    k = u.astype(np.float64) * 2.0 ** 24
    off = k != np.round(k)
    if np.any(off):
        i = int(np.argmax(off))
        raise OutsideMeasuredDomain(
            "uniform %r is not on the 2**-24 grid" % float(u.reshape(-1)[i]))
    return True


def gaussians(x, y, frame_word, mufu=None):
    """The three fp16 noise channels.

    `mufu` must be a callable (variant, float32 array) -> float32 array backed
    by MEASURED tables.  There is no CPU fallback on purpose.
    """
    if mufu is None:
        raise OutsideMeasuredDomain(
            "MUFU.LG2/SQRT/COS/SIN tables for this domain have not been "
            "measured (A2 is the GPU step and has not run); refusing to "
            "substitute a libm approximation")
    u_r1, u_r2, u_a1, u_a2 = uniforms(x, y, frame_word)
    for u in (u_r1, u_r2, u_a1, u_a2):
        assert_uniform_domain(u)
    r = []
    for u in (u_r1, u_r2):
        lg = mufu("LG2", u)
        t = (lg.astype(np.float32) * LN2).astype(np.float32)
        t = (t * MINUS_TWO).astype(np.float32)
        r.append(mufu("SQRT", t))
    # the tables are indexed by the PRE-reduction argument; see angle_probe_arg
    a1, a2 = angle_probe_arg(u_a1), angle_probe_arg(u_a2)
    g1 = (r[0] * mufu("COS", a1)).astype(np.float32)     # 0x900
    g2 = (r[1] * mufu("SIN", a2)).astype(np.float32)     # 0x920
    g3 = (r[1] * mufu("COS", a2)).astype(np.float32)     # 0x940
    return [v.astype(np.float16) for v in (g1, g2, g3)]  # 0x930 / 0x9e0 / 0x950


# ------------------------------------------------------------------ tests

def selftest():
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": str(detail)})
        if not ok:
            raise AssertionError("%s: %s" % (name, detail))

    xs = np.arange(0, 64, dtype=np.uint32)
    ys = np.arange(100, 164, dtype=np.uint32)
    us = uniforms(xs, ys, 0)
    ck("uniforms land in (0, 1]",
       all(np.all((u > 0) & (u <= 1.0)) for u in us))
    for u in us:
        assert_uniform_domain(u)
    ck("derived-domain assertion accepts real uniforms", True)

    # the assertion must actually fire, in four directions
    fired = {}
    for name, val in [("below the grid", np.float32(2.0 ** -25)),
                      ("above 1.0", np.float32(1.0000001)),
                      ("negative", np.float32(-0.5)),
                      # float32 below 0.5 is FINER than the 2**-24 grid, so an
                      # off-grid value has to be taken from there
                      ("off-grid", np.float32(np.nextafter(np.float32(0.25),
                                                           np.float32(1.0))))]:
        try:
            assert_uniform_domain(np.array([val], dtype=np.float32))
        except OutsideMeasuredDomain as e:
            fired[name] = str(e)
    ck("domain assertion fires in all four directions", len(fired) == 4, fired)

    # a batch with one bad element must be rejected, not silently averaged over
    batch = np.concatenate([us[0], np.array([np.float32(2.0)])])
    try:
        assert_uniform_domain(batch)
        ck("a mixed batch is rejected", False, "accepted a batch with one bad value")
    except OutsideMeasuredDomain:
        ck("a mixed batch is rejected", True)

    # gaussians must refuse without measured tables
    try:
        gaussians(np.uint32([1]), np.uint32([1]), 0)
        ck("gaussians refuse without measured tables", False, "returned numbers")
    except OutsideMeasuredDomain:
        ck("gaussians refuse without measured tables", True)

    # angles stay inside the CPU-enumerated interval from A1
    a = angles(us[2])
    ck("angles inside the enumerated interval",
       np.all(a >= np.float32(5.9604641222676946e-08)) and
       np.all(a <= np.float32(0.9999999403953552)),
       (float(a.min()), float(a.max())))

    # FMUL.RZ never rounds away from zero
    rng = np.random.default_rng(20260911)
    t = rng.uniform(1e-6, 1.0, 10000).astype(np.float32)
    rz = fmul_rz(t, INV_TWO_PI)
    exact = t.astype(np.float64) * float(INV_TWO_PI)
    ck("FMUL.RZ never exceeds the exact product",
       np.all(np.abs(rz.astype(np.float64)) <= np.abs(exact)))

    # the frame word actually changes the noise -- if it did not, A3's claim
    # that block 0 depends on a host-supplied word would be untestable
    u0 = uniforms(np.uint32([7]), np.uint32([9]), 0)[0]
    u1 = uniforms(np.uint32([7]), np.uint32([9]), 1)[0]
    ck("the frame word changes the uniforms", float(u0[0]) != float(u1[0]),
       (float(u0[0]), float(u1[0])))

    return {"all_passed": all(c["passed"] for c in checks),
            "count": len(checks), "checks": checks}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=1))
