"""r02 arithmetic additions for L27 (no file I/O; measured tables are passed in as arrays).  arith.py is unchanged.

  f16_add     exact sum of two binary16 values (always exact in float64), numpy float64 -> float16 round-to-nearest-even
  rsq / rcp   measured sm_120 native float32 result bits for every input half in [0x0410, 0x7c00)
              (mlxdlss/resources/rsq-domain.npz and rcp-domain.npz, sha pinned by mlxdlss.sm120_b1._TABLE_FILES; the same
              RSQ table was used by B31 launch61, H_RSQ_TABLE); any input half outside that domain is refused (OutOfDomain)
  rne16       float32 -> binary16 round-to-nearest-even (numpy); a non-finite result from a finite input is refused
  word ops    exact integer operations on the 32-bit word {half0 (low 16 bits), half1 (high 16 bits)}
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import numpy as np

from arith import OutOfDomain

FIRST, LIMIT = 0x0410, 0x7C00
M32 = 0xFFFFFFFF


def f16_add(a, b):
    a64, b64 = np.asarray(a, np.float16).astype(np.float64), np.asarray(b, np.float16).astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        return (a64 + b64).astype(np.float16)


def table_from_npz_arrays(input_half_bits, native_float_bits):
    """Validate the index layout (input_half_bits == FIRST, FIRST+1, ...) and return float32 results for [FIRST, LIMIT)."""
    ib = np.asarray(input_half_bits).astype(np.int64)
    n = LIMIT - FIRST
    if ib.size < n or not np.array_equal(ib[:n], np.arange(FIRST, LIMIT)):
        raise ValueError('measured table index layout is not FIRST..LIMIT-1')
    return np.ascontiguousarray(np.asarray(native_float_bits).astype('<u4')[:n]).view('<f4')


def table_lookup(halves, table_f32, what):
    bits = np.asarray(halves, np.float16).view(np.uint16).astype(np.int64)
    if (bits < FIRST).any() or (bits >= LIMIT).any():
        raise OutOfDomain('%s input half outside the measured table domain [0x0410, 0x7c00): min %#06x max %#06x'
                          % (what, int(bits.min()), int(bits.max())))
    return table_f32[bits - FIRST]


def rne16(f32):
    f = np.asarray(f32, np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        h = f.astype(np.float16)
    if np.any(~np.isfinite(h) & np.isfinite(f)) or np.any(np.isnan(f)):
        raise OutOfDomain('float32 -> binary16 overflow or NaN')
    return h


def bytes_to_f32(b0, b1, b2, b3):
    w = (np.asarray(b0, np.uint32) | (np.asarray(b1, np.uint32) << 8) | (np.asarray(b2, np.uint32) << 16)
         | (np.asarray(b3, np.uint32) << 24))
    return np.ascontiguousarray(w.astype('<u4')).view('<f4')


def word_op(op, c, h0, h1, k):
    w = np.asarray(h0, np.float16).view(np.uint16).astype(np.uint64) | (np.asarray(h1, np.float16).view(np.uint16).astype(np.uint64) << 16)
    if op == 'shl':
        if not 0 <= c < 32:
            raise ValueError('shift %d' % c)
        w = (w << np.uint64(c)) & np.uint64(M32)
    elif op == 'add':
        w = (w + np.uint64(c & M32)) & np.uint64(M32)
    else:
        raise ValueError('word op %s' % op)
    return ((w >> np.uint64(16 * k)) & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
