"""r02 numeric executor for ptx_interp r02 maps (L27 and later).  No file I/O: the caller passes arrays, bytes and tables.

Everything of mapexec.py (W S X XH E K Q F M atoms, QMMA through arith.qmma_step, stores), plus the r02 node kinds, all
evaluated on [NC, NY, NL] arrays whose last axis is the lane:
  B  (dist, x)         out[lane] = x[lane ^ dist]                                        H_SHFL_BFLY_XOR (launch61)
  SI (site, x)         out[lane] = x[idx], idx = map shfl_idx[c, y, lane, site]          H_SHFL_LANE (B39)
  PS (site, a, b)      out = a where map psel[c, y, lane, site] else b
  MV (k, a, b)         cell = 2 lane + k, src = 8 (cell % 8) + cell // 8; out = (a, b)[src % 2] at lane src // 2
                                                                                         H_MOVM_TRANSPOSE (launch61)
  WI (op, c, a, b, k)  exact integer op on the word a | b << 16 (shl by c, or add c mod 2^32); half k of the result
  HB (lo, hi)          half with bit pattern lo | hi << 8
  F add                binary16 RNE of the exact sum (arith2.f16_add)
  F rsq / rcp          measured native float32 for the input half (arith2.table_lookup)
  F rne16 / rne16w     float32 (a value / four bytes) -> binary16 RNE
Contraction (decided statically from the SASS, never from numbers): an F add node listed in `contract` as k evaluates
add(mul(x0, x0), mul(x1, x1)) as fma(xk, xk, mul(x(1-k), x(1-k))) -- one rounding of xk*xk + t, t rounded separately.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import numpy as np

import arith as AR
import arith2 as A2

LANE = np.arange(32)


class MapError(Exception):
    pass


def execute(wiring, arrays, input_bytes, weights, output_len, tables, contract=None, stats=None):
    nodes = [tuple(n) if n[0] != 'F' else (n[0], n[1], tuple(n[2])) for n in wiring['nodes']]
    contract = {int(k): int(v) for k, v in (contract or {}).items()}
    wl = arrays['wload_off']
    src = arrays['sread_src']
    go = arrays['gstore_off']
    NC, NY, NL = wl.shape[:3]
    if NL != 32:
        raise MapError('lane count %d != 32' % NL)
    inp = np.frombuffer(input_bytes, np.uint8) if isinstance(input_bytes, (bytes, bytearray)) else np.asarray(input_bytes, np.uint8)
    W = np.frombuffer(weights, np.uint8) if isinstance(weights, (bytes, bytearray)) else np.asarray(weights, np.uint8)
    memo = {}
    used_contract = set()
    counts = dict(contracted_adds=0, rsq_calls=0, rcp_calls=0)

    def value(i):
        v = memo.get(i)
        if v is not None:
            return v
        n = nodes[i]
        kind = n[0]
        if kind == 'W':
            off = wl[..., n[1]]
            if off.min() < 0 or off.max() + n[2] >= W.size:
                raise MapError('weight load outside the tensor')
            v = W[off + n[2]]
        elif kind == 'S':
            s = src[..., n[1], n[2]]
            if s.min() < -1:
                raise MapError('shared read byte without a visible write (code %d)' % int(s.min()))
            if s.max() >= inp.size:
                raise MapError('shared read byte outside the input')
            v = np.where(s >= 0, inp[np.maximum(s, 0)], 0).astype(np.uint8)
        elif kind in ('X', 'XH'):
            src4 = arrays['xsite_src'][..., n[1], :]
            w4 = arrays['xsite_wsrc'][..., n[1], :]
            if src4.min() < -3 or (src4 == -2).any() or src4.max() >= inp.size or w4.max() >= W.size:
                raise MapError('guarded global word without a valid source')
            b4 = np.where(src4 >= 0, inp[np.maximum(src4, 0)], np.where(src4 == -3, W[np.maximum(w4, 0)], 0)).astype(np.uint8)
            if kind == 'X':
                v = b4[..., n[2]]
            else:
                h = n[2]
                v = (b4[..., 2 * h].astype(np.uint16) | (b4[..., 2 * h + 1].astype(np.uint16) << 8)).view(np.float16)
        elif kind == 'E':
            v = AR.decode_e4m3_f16(value(n[1]))
        elif kind == 'K':
            v = np.full((NC, NY, NL), np.array([n[1]], np.uint16).view(np.float16)[0], np.float16)
        elif kind == 'Q':
            v = AR.satfinite_e4m3(value(n[1]))
        elif kind == 'HB':
            v = (value(n[1]).astype(np.uint16) | (value(n[2]).astype(np.uint16) << 8)).view(np.float16)
        elif kind == 'B':
            v = value(n[2])[..., LANE ^ n[1]]
        elif kind == 'SI':
            idx = arrays['shfl_idx'][..., n[1]]
            if idx.min() < 0 or idx.max() > 31:
                raise MapError('shfl.idx index outside 0..31')
            v = np.take_along_axis(value(n[2]), idx, axis=-1)
        elif kind == 'PS':
            p = arrays['psel'][..., n[1]]
            if not np.isin(p, (0, 1)).all():
                raise MapError('data selp predicate not recorded')
            v = np.where(p == 1, value(n[2]), value(n[3]))
        elif kind == 'MV':
            cell = 2 * LANE + n[1]
            s = 8 * (cell % 8) + cell // 8
            a, b = value(n[2]), value(n[3])
            v = np.where(s % 2 == 0, a[..., s // 2], b[..., s // 2]).astype(np.float16)
        elif kind == 'WI':
            v = A2.word_op(n[1], n[2], value(n[3]), value(n[4]), n[5])
        elif kind == 'F':
            op = n[1]
            if op == 'add' and i in contract:
                m = [nodes[a] for a in n[2]]
                if not all(x[0] == 'F' and x[1] == 'mul' and x[2][0] == x[2][1] for x in m):
                    raise MapError('contraction listed for an add that is not add(mul(x,x), mul(y,y))')
                k = contract[i]
                xf = m[k][2][0]
                v = AR.f16_fma(value(xf), value(xf), value(n[2][1 - k]))
                used_contract.add(i)
                counts['contracted_adds'] += 1
            else:
                args = [value(a) for a in n[2]]
                if op == 'min':
                    v = AR.f16_min(*args)
                elif op == 'max':
                    v = AR.f16_max(*args)
                elif op == 'abs':
                    v = AR.f16_abs(*args)
                elif op == 'mul':
                    v = AR.f16_mul(*args)
                elif op == 'fma':
                    v = AR.f16_fma(*args)
                elif op == 'add':
                    v = A2.f16_add(*args)
                elif op in ('rsqrt', 'rcp'):
                    key = 'rsq' if op == 'rsqrt' else 'rcp'
                    v = A2.table_lookup(args[0], tables[key], key)
                    counts[key + '_calls'] += 1
                elif op == 'rne16':
                    v = A2.rne16(args[0])
                elif op == 'rne16w':
                    v = A2.rne16(A2.bytes_to_f32(*args))
                else:
                    raise MapError('f16 op %s' % op)
        elif kind == 'M':
            raise MapError('QMMA output %d read before its QMMA ran' % n[1])
        else:
            raise MapError('atom kind %r' % (kind,))
        memo[i] = v
        return v

    M_atom = {}
    for i, n in enumerate(nodes):
        if n[0] == 'M':
            M_atom[(n[1], n[2])] = i
    for o, m in enumerate(wiring['mmas']):
        A = np.stack([value(a) for a in m['A']], -1)
        B = np.stack([value(b) for b in m['B']], -1)
        C = np.stack([value(c) for c in m['C']], -1)
        if A.dtype != np.uint8 or B.dtype != np.uint8 or C.dtype != np.float16:
            raise MapError('QMMA %d operand dtypes %s %s %s' % (o, A.dtype, B.dtype, C.dtype))
        d = AR.qmma_step(AR.a_matrix(A), AR.b_matrix(B), AR.lanes_to_c(C), stats=stats)
        dl = AR.d_to_lanes(d).astype(np.float16)
        for k in range(4):
            if (o, k) in M_atom:
                memo[M_atom[(o, k)]] = dl[..., k]
    out = np.zeros(output_len, np.uint8)
    cover = np.zeros(output_len, np.int32)
    for j, pc in enumerate(wiring['gstore_pcs']):
        atoms = wiring['gstore_atoms'][str(pc)]
        vals = np.stack([value(a) for a in atoms], -1)
        if vals.dtype != np.uint8:
            raise MapError('store of non-byte atoms')
        off = go[..., j]
        live = off >= 0
        for b in range(16):
            dst = off[live] + b
            if dst.size and dst.max() >= output_len:
                raise MapError('store outside the output')
            out[dst] = vals[..., b][live]
            np.add.at(cover, dst, 1)
    missing = sorted(set(contract) - used_contract)
    if missing:
        raise MapError('%d listed contraction node(s) never evaluated' % len(missing))
    if stats is not None:
        stats.update(counts)
    return out, cover
