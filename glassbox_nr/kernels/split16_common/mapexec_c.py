"""r05 numeric executor for class maps ('opus-split16-ptx-map-classes/1', ptx_interp_c.py).  No file I/O.

Arithmetic is exactly mapexec2's: every node kind calls the same arith / arith2 function with the same arguments (the code
below is mapexec2.execute with two additions, marked r05).  Regression gate: on single-class B26 maps converted to this
format, execute() returns the same bytes and coverage as mapexec2.execute (regression/t2c-b26-executor-regression.json).
r05 additions:
  1. INVALID MASKS.  A site that a thread does not have (xsite -7, shfl_idx -2, psel -2: absent for that thread's class) is
     evaluated as a harmless placeholder and marked invalid for that thread; invalidity propagates to every node computed
     from it.  Any other bad code still raises exactly as in mapexec2.  A QMMA operand that is invalid for ANY thread raises
     (MMA structure is uniform in a class map, so an MMA may never depend on a class-specific site).
  2. STORES PER CLASS.  For class k, every store pc of that class is evaluated from the class's own atoms and written only
     for threads with thread_class == k and a store offset >= 0; a value that is invalid for such a thread raises.  A store
     offset >= 0 for a pc outside the thread's class raises.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import numpy as np

import arith as AR
import arith2 as A2

LANE = np.arange(32)
ABSENT_X, ABSENT_SI, ABSENT_PS = -7, -2, -2


class MapError(Exception):
    pass


def execute(wiring, arrays, input_bytes, weights, output_len, tables, contract=None, stats=None):
    nodes = [tuple(n) if n[0] != 'F' else (n[0], n[1], tuple(n[2])) for n in wiring['nodes']]
    contract = {int(k): int(v) for k, v in (contract or {}).items()}
    wl = arrays['wload_off']
    src = arrays['sread_src']
    go = arrays['gstore_off']
    tc = arrays['thread_class']
    NC, NY, NL = wl.shape[:3]
    if NL != 32:
        raise MapError('lane count %d != 32' % NL)
    inp = np.frombuffer(input_bytes, np.uint8) if isinstance(input_bytes, (bytes, bytearray)) else np.asarray(input_bytes, np.uint8)
    W = np.frombuffer(weights, np.uint8) if isinstance(weights, (bytes, bytearray)) else np.asarray(weights, np.uint8)
    memo, inval = {}, {}
    used_contract = set()
    counts = dict(contracted_adds=0, rsq_calls=0, rcp_calls=0)
    NONE = np.zeros((NC, NY, NL), bool)

    def value(i):
        v = memo.get(i)
        if v is not None:
            return v
        n = nodes[i]
        kind = n[0]
        bad = NONE
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
            absent = (src4 == ABSENT_X)                                          # r05
            if (absent.any(-1) != absent.all(-1)).any():
                raise MapError('xsite partly absent')
            bad = absent.all(-1)
            s4 = np.where(absent, -1, src4)
            if s4.min() < -3 or (s4 == -2).any() or s4.max() >= inp.size or w4.max() >= W.size:
                raise MapError('guarded global word without a valid source')
            b4 = np.where(s4 >= 0, inp[np.maximum(s4, 0)], np.where(s4 == -3, W[np.maximum(w4, 0)], 0)).astype(np.uint8)
            if kind == 'X':
                v = b4[..., n[2]]
            else:
                h = n[2]
                v = (b4[..., 2 * h].astype(np.uint16) | (b4[..., 2 * h + 1].astype(np.uint16) << 8)).view(np.float16)
        elif kind == 'E':
            v = AR.decode_e4m3_f16(value(n[1])); bad = inval[n[1]]
        elif kind == 'K':
            v = np.full((NC, NY, NL), np.array([n[1]], np.uint16).view(np.float16)[0], np.float16)
        elif kind == 'Q':
            v = AR.satfinite_e4m3(value(n[1])); bad = inval[n[1]]
        elif kind == 'HB':
            v = (value(n[1]).astype(np.uint16) | (value(n[2]).astype(np.uint16) << 8)).view(np.float16); bad = inval[n[1]] | inval[n[2]]
        elif kind == 'B':
            v = value(n[2])[..., LANE ^ n[1]]; bad = inval[n[2]][..., LANE ^ n[1]]
        elif kind == 'SI':
            idx = arrays['shfl_idx'][..., n[1]]
            absent = idx == ABSENT_SI                                            # r05
            ix = np.where(absent, 0, idx)
            if ix.min() < 0 or ix.max() > 31:
                raise MapError('shfl.idx index outside 0..31')
            x = value(n[2])
            v = np.take_along_axis(x, ix, axis=-1)
            bad = absent | np.take_along_axis(inval[n[2]], ix, axis=-1)
        elif kind == 'PS':
            p = arrays['psel'][..., n[1]]
            absent = p == ABSENT_PS                                              # r05
            pp = np.where(absent, 0, p)
            if not np.isin(pp, (0, 1)).all():
                raise MapError('data selp predicate not recorded')
            v = np.where(pp == 1, value(n[2]), value(n[3]))
            bad = absent | np.where(pp == 1, inval[n[2]], inval[n[3]])
        elif kind == 'MV':
            cell = 2 * LANE + n[1]
            s = 8 * (cell % 8) + cell // 8
            a, b = value(n[2]), value(n[3])
            v = np.where(s % 2 == 0, a[..., s // 2], b[..., s // 2]).astype(np.float16)
            bad = np.where(s % 2 == 0, inval[n[2]][..., s // 2], inval[n[3]][..., s // 2])
        elif kind == 'WI':
            v = A2.word_op(n[1], n[2], value(n[3]), value(n[4]), n[5]); bad = inval[n[3]] | inval[n[4]]
        elif kind == 'F':
            op = n[1]
            if op == 'add' and i in contract:
                m = [nodes[a] for a in n[2]]
                if not all(x[0] == 'F' and x[1] == 'mul' and x[2][0] == x[2][1] for x in m):
                    raise MapError('contraction listed for an add that is not add(mul(x,x), mul(y,y))')
                k = contract[i]
                xf = m[k][2][0]
                v = AR.f16_fma(value(xf), value(xf), value(n[2][1 - k]))
                bad = inval[xf] | inval[n[2][1 - k]]
                used_contract.add(i)
                counts['contracted_adds'] += 1
            else:
                args = [value(a) for a in n[2]]
                bad = np.logical_or.reduce([inval[a] for a in n[2]]) if n[2] else NONE
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
                    # r05: a table lookup of a placeholder value outside the class could be out of domain; look up valid threads only
                    a0 = np.where(bad, np.float16(1.0), args[0]).astype(np.float16)
                    v = A2.table_lookup(a0, tables[key], key)
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
        inval[i] = bad
        return v

    M_atom = {}
    for i, n in enumerate(nodes):
        if n[0] == 'M':
            M_atom[(n[1], n[2])] = i
    for o, m in enumerate(wiring['mmas']):
        A = np.stack([value(a) for a in m['A']], -1)
        B = np.stack([value(b) for b in m['B']], -1)
        C = np.stack([value(c) for c in m['C']], -1)
        if any(inval[x].any() for x in list(m['A']) + list(m['B']) + list(m['C'])):         # r05
            raise MapError('QMMA %d operand depends on a class-absent site' % o)
        if A.dtype != np.uint8 or B.dtype != np.uint8 or C.dtype != np.float16:
            raise MapError('QMMA %d operand dtypes %s %s %s' % (o, A.dtype, B.dtype, C.dtype))
        d = AR.qmma_step(AR.a_matrix(A), AR.b_matrix(B), AR.lanes_to_c(C), stats=stats)
        dl = AR.d_to_lanes(d).astype(np.float16)
        for k in range(4):
            if (o, k) in M_atom:
                memo[M_atom[(o, k)]] = dl[..., k]
                inval[M_atom[(o, k)]] = NONE
    out = np.zeros(output_len, np.uint8)
    cover = np.zeros(output_len, np.int32)
    col = {pc: j for j, pc in enumerate(wiring['gstore_pcs'])}
    classes = wiring['classes']
    if sorted(np.unique(tc).tolist()) != sorted(c['cls'] for c in classes):
        raise MapError('thread classes of the arrays != wiring classes')
    for c in classes:                                                                   # r05
        mask = tc == c['cls']
        for pc in wiring['gstore_pcs']:
            if pc not in c['gstore_pcs'] and (go[..., col[pc]][mask] >= 0).any():
                raise MapError('store offset for pc %d outside class %d' % (pc, c['cls']))
        for pc in c['gstore_pcs']:
            atoms = c['gstore_atoms'][str(pc)]
            vals = np.stack([value(a) for a in atoms], -1)
            bad = np.logical_or.reduce([inval[a] for a in atoms])
            if vals.dtype != np.uint8:
                raise MapError('store of non-byte atoms')
            off = go[..., col[pc]]
            live = (off >= 0) & mask
            if bad[live].any():
                raise MapError('store pc %d of class %d depends on an absent site' % (pc, c['cls']))
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
