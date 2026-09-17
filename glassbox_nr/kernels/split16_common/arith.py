"""Split16 round arithmetic shared by calibration and prediction.  No file I/O.

QMMA.16832 slot layout (D18 a1-kmap pairing measured on sm_120; re-checked by D16 r2_a1_layout_check.py; used unchanged
by experiments/parity-astra-opus-b31-l59-20260913/l59_lane.py), per lane l, g = l // 4, t = l % 4, register r at bytes
4r..4r+3 of the lane payload, byte sb little-endian in the word:
    A  reg r byte sb -> (row g + 8*(r%2), k 16*(r//2) + 4t + sb)
    B  reg r byte sb -> (k 16r + 4t + sb,  col g)
    C/D reg r half h -> (row g + 8r,        col 2t + h)          C layout = D layout
Accumulator: mlxdlss.sm120_b1.qmma_batched (W27_trunc), unchanged, with the D27 counterexample-profile refusal copied
from l59_lane.py (rule text below is identical).
f16 primitives: IEEE binary16 round-to-nearest-even of the EXACT result.  mul uses the float64 product (always exact for
two binary16 operands) then numpy's float64->float16 conversion; fma uses float64 when TwoSum proves the sum exact and an
exact rational path otherwise.  Both forms are calibrated on the D23 G1 hardware corpus by calibrate_arith.py.
satfinite E4M3: exact nearest-even onto the finite E4M3 grid, |x| > 448 and +-inf saturate to +-448, the sign of zero is
kept, NaN is refused.  This is the H_SATFINITE_RNE_LO_FIRST definition of experiments/parity-qmma-probe-20260912/
q6_cases_c.py; calibrate_arith.py compares the table with the sm_120 c2 exhaustive hardware measurement.

Domain rule (copied, written before any numeric run in the B31 round): an accumulation whose addends match the profile
shared by ALL eight D27 QMMA counterexamples is refused, not approximated:
    log2 max|addend| >= 12  and  (log2 max|addend| - log2 min|nonzero product|) >= 20  and  cancellation >= 2 bits
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from fractions import Fraction

import numpy as np

from glassbox_nr.kernels.core import sm120_b1 as CORE

LANE = np.arange(32)
G_, T_ = LANE // 4, LANE % 4
DOMAIN = dict(max_log2_ge=12, range_log2_ge=20, cancel_bits_ge=2)


class OutOfDomain(Exception):
    pass


def _check_bijection():
    a = {(int(g + 8 * (r % 2)), int(16 * (r // 2) + 4 * t + sb)) for g, t in zip(G_, T_) for r in range(4) for sb in range(4)}
    b = {(int(16 * r + 4 * t + sb), int(g)) for g, t in zip(G_, T_) for r in range(2) for sb in range(4)}
    d = {(int(g + 8 * r), int(2 * t + h)) for g, t in zip(G_, T_) for r in range(2) for h in range(2)}
    assert len(a) == 512 and len(b) == 256 and len(d) == 128, 'slot layout is not a bijection'


_check_bijection()


# ---------------------------------------------------------------------------------------------------------- layout
def a_matrix(lane16):
    """[..., 32, 16] uint8 lane payloads -> [..., 16, 32] codes A[row, k]."""
    out = np.zeros(lane16.shape[:-2] + (16, 32), np.uint8)
    for r in range(4):
        for sb in range(4):
            out[..., G_ + 8 * (r % 2), 16 * (r // 2) + 4 * T_ + sb] = lane16[..., LANE, 4 * r + sb]
    return out


def b_matrix(lane8):
    """[..., 32, 8] uint8 lane payloads -> [..., 32, 8] codes B[k, col]."""
    out = np.zeros(lane8.shape[:-2] + (32, 8), np.uint8)
    for r in range(2):
        for sb in range(4):
            out[..., 16 * r + 4 * T_ + sb, G_] = lane8[..., LANE, 4 * r + sb]
    return out


def d_to_lanes(d):
    """[..., 16, 8] -> [..., 32, 4] in lane order (reg r, half h) -> index 2r + h."""
    out = np.zeros(d.shape[:-2] + (32, 4), d.dtype)
    for r in range(2):
        for h in range(2):
            out[..., LANE, 2 * r + h] = d[..., G_ + 8 * r, 2 * T_ + h]
    return out


def lanes_to_c(lanes):
    """inverse of d_to_lanes: [..., 32, 4] -> [..., 16, 8]."""
    out = np.zeros(lanes.shape[:-2] + (16, 8), lanes.dtype)
    for r in range(2):
        for h in range(2):
            out[..., G_ + 8 * r, 2 * T_ + h] = lanes[..., LANE, 2 * r + h]
    return out


# ------------------------------------------------------------------------------------------------------------ QMMA
def domain_profile(a_codes, b_codes, c_f16):
    au = CORE._e4m3_units(CORE.decode_codes(a_codes).astype(np.float64), 'A operand')          # units 2**-9
    bu = CORE._e4m3_units(CORE.decode_codes(b_codes).astype(np.float64), 'B operand')
    cu = CORE._f16_units(np.asarray(c_f16, np.float64), 'C seed')                                # units 2**-24
    prod = au[..., :, None, :] * np.swapaxes(bu, -1, -2)[..., None, :, :] * CORE._PROD_SCALE      # units 2**-24
    pa = np.abs(prod)
    top = np.maximum(pa.max(-1), np.abs(cu))
    nz = np.where(pa > 0, pa, np.iinfo(np.int64).max).min(-1)
    exact = prod.sum(-1) + cu
    with np.errstate(divide='ignore', invalid='ignore'):
        l_top = np.where(top > 0, np.floor(np.log2(np.maximum(top, 1))), -np.inf) - 24
        l_min = np.where(pa.max(-1) > 0, np.floor(np.log2(np.where(nz == np.iinfo(np.int64).max, 1, nz))), np.inf) - 24
        l_sum = np.where(exact != 0, np.floor(np.log2(np.maximum(np.abs(exact), 1))), -np.inf) - 24
        rng = l_top - l_min
        cancel = np.where(exact != 0, l_top - l_sum, np.where(top > 0, 99, 0))
    flagged = (l_top >= DOMAIN['max_log2_ge']) & (rng >= DOMAIN['range_log2_ge']) & (cancel >= DOMAIN['cancel_bits_ge'])
    return l_top, rng, cancel, flagged


def qmma_step(a_codes, b_codes, c_f16, stats=None, enforce_domain=True):
    """One QMMA on a batch: a [...,16,32] codes, b [...,32,8] codes, c [...,16,8] f16 -> d [...,16,8] f16."""
    if np.any((np.asarray(a_codes) & 0x7F) == 0x7F) or np.any((np.asarray(b_codes) & 0x7F) == 0x7F):
        raise OutOfDomain('E4M3 NaN code in a QMMA operand')
    c = np.asarray(c_f16, np.float16)
    if not np.isfinite(c).all():
        raise OutOfDomain('non-finite QMMA seed')
    l_top, rng, cancel, flagged = domain_profile(a_codes, b_codes, c)
    if stats is not None:
        stats['qmma'] = stats.get('qmma', 0) + 1
        stats['outputs'] = stats.get('outputs', 0) + int(flagged.size)
        stats['flagged'] = stats.get('flagged', 0) + int(flagged.sum())
        fin = l_top[np.isfinite(l_top)]
        if fin.size:
            stats['max_log2_addend'] = max(stats.get('max_log2_addend', -99), int(fin.max()))
        stats['max_range_log2'] = max(stats.get('max_range_log2', -99), int(np.nan_to_num(rng, neginf=-99, posinf=-99).max()))
    if enforce_domain and flagged.any():
        raise OutOfDomain('%d accumulation(s) match the D27 counterexample profile' % int(flagged.sum()))
    d = CORE.qmma_batched(CORE.decode_codes(a_codes).astype(np.float64), CORE.decode_codes(b_codes).astype(np.float64),
                          c.astype(np.float64))
    if not np.isfinite(d).all():
        raise OutOfDomain('non-finite QMMA result; binary16 overflow is outside the modelled domain')
    return d


# ---------------------------------------------------------------------------------------------- f16 primitives
_TWO = Fraction(2)


def f16_round_fraction(q):
    """Exact IEEE binary16 round-to-nearest-even of a rational; returns np.float16 (sign of a zero result = sign of q)."""
    if q == 0:
        return np.float16(0.0)
    neg = q < 0
    m = -q if neg else q
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if _TWO ** e > m:
        e -= 1
    while _TWO ** (e + 1) <= m:
        e += 1
    qe = max(e, -14) - 10
    scaled = m / (_TWO ** qe)
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
        n += 1
    v = Fraction(n) * (_TWO ** qe)
    if v > 65504:
        return np.float16(-np.inf if neg else np.inf)
    fv = float(v)
    return np.float16(-fv if neg else fv)


def f16_mul(a, b):
    a64, b64 = np.asarray(a, np.float16).astype(np.float64), np.asarray(b, np.float16).astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        return (a64 * b64).astype(np.float16)


def f16_fma(a, b, c):
    """a*b + c rounded once to binary16 (ties to even)."""
    a16, b16, c16 = (np.asarray(x, np.float16) for x in (a, b, c))
    a16, b16, c16 = np.broadcast_arrays(a16, b16, c16)
    a64, b64, c64 = a16.astype(np.float64), b16.astype(np.float64), c16.astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        p = a64 * b64
        s = p + c64
        bb = s - p
        err = (p - (s - bb)) + (c64 - bb)
        out = s.astype(np.float16)
    fix = np.flatnonzero((err != 0) & np.isfinite(s))
    flat = out.reshape(-1)
    for i in fix.tolist():
        q = Fraction(float(a64.flat[i])) * Fraction(float(b64.flat[i])) + Fraction(float(c64.flat[i]))
        flat[i] = f16_round_fraction(q)
    return out


def f16_min(a, b):
    return np.minimum(np.asarray(a, np.float16), np.asarray(b, np.float16))


def f16_max(a, b):
    return np.maximum(np.asarray(a, np.float16), np.asarray(b, np.float16))


def f16_abs(a):
    return np.abs(np.asarray(a, np.float16))


# --------------------------------------------------------------------------------------------- E4M3 -> binary16
def _decode_table():
    t = np.zeros(256, np.float16)
    ok = np.ones(256, bool)
    for c in range(256):
        if (c & 0x7F) == 0x7F:
            ok[c] = False
            continue
        e, mant = (c >> 3) & 15, c & 7
        v = Fraction(mant, 512) if e == 0 else Fraction(8 + mant) * (_TWO ** (e - 10))
        f = float(v)
        assert float(np.float16(f)) == f, 'E4M3 value not exact in binary16'
        t[c] = np.float16(-f if c & 0x80 else f)
    return t, ok


_DEC = [None]


def decode_e4m3_f16(codes):
    """cvt.rn.f16x2.e4m3x2: exact OCP E4M3 decode into binary16 (every finite E4M3 value is exact in binary16; the sm_120
    c1 exhaustive corpus leaves H_OCP_LO_FIRST as the only survivor modulo NaN payload, calibrated in calibrate_arith_c1.py).
    NaN codes are refused."""
    if _DEC[0] is None:
        _DEC[0] = _decode_table()
    t, ok = _DEC[0]
    c = np.asarray(codes, np.uint8)
    if not ok[c].all():
        raise OutOfDomain('E4M3 NaN code into cvt.rn.f16x2.e4m3x2; NaN payload behaviour is not modelled')
    return t[c]


# --------------------------------------------------------------------------------------------- satfinite E4M3
def _e4m3_positive():
    vals = []
    for c in range(1, 0x7F):
        e, mant = (c >> 3) & 15, c & 7
        v = Fraction(mant, 512) if e == 0 else Fraction(8 + mant) * (_TWO ** (e - 10))
        vals.append((v, c))
    assert vals[-1] == (Fraction(448), 0x7E)
    return vals


def build_satfinite_table():
    """int16 table over all 65536 binary16 bit patterns -> E4M3 code (0..255), -1 for NaN inputs."""
    pos = _e4m3_positive()
    pv = [v for v, _ in pos]
    table = np.empty(65536, np.int16)
    halves = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    import bisect
    for bits in range(65536):
        x = float(halves[bits])
        if x != x:
            table[bits] = -1
            continue
        sign = 0x80 if (bits & 0x8000) else 0x00
        if x == 0.0:
            table[bits] = sign
            continue
        if np.isinf(x):
            table[bits] = sign | 0x7E
            continue
        mag = Fraction(abs(x))
        if mag > 448:
            table[bits] = sign | 0x7E
            continue
        i = bisect.bisect_left(pv, mag)
        if pv[i] == mag:
            table[bits] = sign | pos[i][1]
            continue
        lo_v, lo_c = (Fraction(0), 0) if i == 0 else pos[i - 1]
        hi_v, hi_c = pos[i]
        dlo, dhi = mag - lo_v, hi_v - mag
        if dlo < dhi or (dlo == dhi and lo_c % 2 == 0):
            table[bits] = sign | lo_c
        else:
            table[bits] = sign | hi_c
    return table


_SAT = [None]


def satfinite_e4m3(halves_f16):
    if _SAT[0] is None:
        _SAT[0] = build_satfinite_table()
    bits = np.asarray(halves_f16, np.float16).view(np.uint16)
    codes = _SAT[0][bits]
    if np.any(codes < 0):
        raise OutOfDomain('NaN into satfinite E4M3; NaN payload behaviour is not modelled')
    return codes.astype(np.uint8)
