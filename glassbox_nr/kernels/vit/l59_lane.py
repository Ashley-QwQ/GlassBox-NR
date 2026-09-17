"""Launch59 lane arithmetic shared by calibration and prediction.  No file I/O.

QMMA.16832 slot layout (D18 a1-kmap pairing measured on sm_120, re-checked by D16 r2_a1_layout_check.py), per lane l,
g = l // 4, t = l % 4, register r at bytes 4r..4r+3 of the lane payload (H_BRACE_LE), byte sb little-endian in the word:
    A  reg r byte sb -> (row g + 8*(r%2), k 16*(r//2) + 4t + sb)
    B  reg r byte sb -> (k 16r + 4t + sb,  col g)
    C/D reg r half h -> (row g + 8r,        col 2t + h)          C layout = D layout (H_QMMA_C_LAYOUT_EQ_D)
Accumulator: mlxdlss.sm120_b1.qmma_batched (W27_trunc), unchanged.
Gate: mlxdlss.sm120_b1.feed_forward_gate (clamp +-4, two fma, one mul, E4M3 publish), constants re-read from the launch59
PTX by ptx_interp.decode_gate.

Domain rule (written before any launch59 numeric run): an accumulation whose addends match the profile shared by ALL
eight D27 QMMA counterexamples is refused, not approximated:
    log2 max|addend| >= 12  and  (log2 max|addend| - log2 min|nonzero product|) >= 20  and  cancellation >= 2 bits
with addend = exact E4M3 product or the C seed, cancellation = floor(log2 max|addend|) - floor(log2 |exact sum|)
(exact sum 0 counts as full cancellation).  calibrate_components.py reports how this rule behaves on the G2 corpus.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

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
    """[..., 16, 8] halves -> [..., 32, 4] halves in lane order (reg r, half h) -> index 2r + h."""
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


def domain_profile(a_codes, b_codes, c_f16):
    """Per output element: (log2 max addend, log2 range, cancellation bits, flagged) using exact integer units."""
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
    l_top, rng, cancel, flagged = domain_profile(a_codes, b_codes, c_f16)
    if stats is not None:
        stats['qmma'] = stats.get('qmma', 0) + 1
        stats['outputs'] = stats.get('outputs', 0) + int(flagged.size)
        stats['flagged'] = stats.get('flagged', 0) + int(flagged.sum())
        fin = l_top[np.isfinite(l_top)]
        if fin.size:
            stats['max_log2_addend'] = max(stats.get('max_log2_addend', -99), int(fin.max()))
        stats['max_range_log2'] = max(stats.get('max_range_log2', -99), int(np.nan_to_num(rng, neginf=-99, posinf=-99).max()))
        hist = stats.setdefault('log2_addend_hist', {})
        for v, n in zip(*np.unique(np.where(np.isfinite(l_top), l_top, -99).astype(int), return_counts=True)):
            hist[str(int(v))] = hist.get(str(int(v)), 0) + int(n)
    if enforce_domain and flagged.any():
        raise OutOfDomain('%d accumulation(s) match the D27 counterexample profile' % int(flagged.sum()))
    av = CORE.decode_codes(a_codes).astype(np.float64)
    bv = CORE.decode_codes(b_codes).astype(np.float64)
    d = CORE.qmma_batched(av, bv, np.asarray(c_f16, np.float16).astype(np.float64))
    if not np.isfinite(d).all():
        raise OutOfDomain('non-finite QMMA result; binary16 overflow is outside the modelled domain')
    return d


def gate_codes(halves_f16):
    h = np.asarray(halves_f16, np.float16)
    if not np.isfinite(h).all():
        raise OutOfDomain('non-finite gate input')
    return CORE.feed_forward_gate(h)
