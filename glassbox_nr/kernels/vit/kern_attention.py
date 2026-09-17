"""attention_chained kernel.

Evaluator and helpers copied line by line from experiments/parity-astra-opus-b31-l59-20260913/revisions/r04-launch62/predict_l62.py;
compute() = that predictor's PHASE 'compute' section (from the map-check test to EV.run).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import numpy as np

NC, NL = 32, 32
RCP_FIRST = 0x0410
LEAF_REGION = {'qload': 'q', 'kread': 'k', 'vread': 'v'}


def f16(b):
    return np.ascontiguousarray(b).view('<f2')


def to_bytes16(f):
    return np.ascontiguousarray(np.asarray(f).astype('<f2')).view(np.uint8)


def uint_of(b, nbytes):
    b = np.ascontiguousarray(b)
    return b.view({2: '<u2', 4: '<u4'}[nbytes]).astype(np.int64)


def bytes_of(u, nbytes):
    return np.ascontiguousarray((u & ((1 << (8 * nbytes)) - 1)).astype({2: '<u2', 4: '<u4'}[nbytes])).view(np.uint8)


class Evaluator:
    """Every node value is a little-endian byte array [NC, NL, nbytes] computed as if every lane consumed it."""

    def __init__(self, LANE, CORE, nodes, npz, gsrc, data, rcp_f32, stats):
        self.LANE, self.CORE, self.nodes, self.npz, self.gsrc, self.data, self.rcp_f32, self.stats = LANE, CORE, nodes, npz, gsrc, data, rcp_f32, stats
        self.memo, self.mma_out = {}, {}

    def leaf(self, n):
        _, kind, ep, slot, word = n
        if kind == 'mma':
            return self.mma_out[(ep, slot)][..., 4 * word:4 * word + 4]
        name = ('%s|%d|%d' % (kind, ep, slot)) if kind == 'qload' else ('%s_src|%d|%d' % (kind, ep, slot))
        off = self.npz[name].astype(np.int64)[:, 0, :]
        if (off < -1).any():
            raise RuntimeError('MAP_OFFSET_MISSING:%s' % name)
        zero = off < 0
        if kind == 'qload' and zero.any():
            raise RuntimeError('QUERY_LOAD_MISSING:%s' % name)
        self.stats['shared_zero_lanes'] = self.stats.get('shared_zero_lanes', 0) + int(zero.sum())
        idx = np.where(zero, 0, off)[..., None] + 4 * word + np.arange(4)
        return np.where(zero[..., None], 0, self.data[LEAF_REGION[kind]][idx]).astype(np.uint8)

    def ev(self, i):
        hit = self.memo.get(i)
        if hit is not None:
            return hit
        n = self.nodes[i]
        k = n[0]
        L, CORE = self.LANE, self.CORE
        if k == 'leaf':
            r = self.leaf(n)
        elif k == 'c16':
            r = np.broadcast_to(np.frombuffer(np.float16(float(n[1])).tobytes(), np.uint8), (NC, NL, 2))
        elif k == 'ibits':
            r = np.broadcast_to(np.frombuffer(np.uint32(n[1]).astype('<u4').tobytes(), np.uint8), (NC, NL, 4))
        elif k == 'cat':
            r = np.concatenate([self.ev(x) for x in n[1:]], -1)
        elif k == 'part':
            r = self.ev(n[1])[..., 2 * n[2]:2 * n[2] + 2]
        elif k == 'fop':
            op = n[1]
            vals = [f16(self.ev(x)).astype(np.float64) for x in n[2:]]
            with np.errstate(over='ignore', invalid='ignore'):
                if op == 'fma':
                    res = (vals[0] * vals[1] + vals[2]).astype(np.float16)
                elif op == 'add':
                    res = (vals[0] + vals[1]).astype(np.float16)
                elif op == 'sub':
                    res = (vals[0] - vals[1]).astype(np.float16)
                elif op == 'mul':
                    res = (vals[0] * vals[1]).astype(np.float16)
                elif op == 'max':
                    res = np.maximum(vals[0].astype(np.float16), vals[1].astype(np.float16))
                elif op == 'min':
                    res = np.minimum(vals[0].astype(np.float16), vals[1].astype(np.float16))
                else:
                    raise RuntimeError('FOP_UNKNOWN:%s' % op)
            if not np.isfinite(res).all():
                raise L.OutOfDomain('non-finite binary16 result in %s' % op)
            self.stats['fop_' + op] = self.stats.get('fop_' + op, 0) + 1
            r = to_bytes16(res)
        elif k in ('ishl', 'iadd'):
            nbytes = n[1] // 8
            src = self.ev(n[2])
            if src.shape[-1] != nbytes:
                raise RuntimeError('INTEGER_WIDTH_MISMATCH:%s' % k)
            u = uint_of(src, nbytes)
            u = (u << n[3]) if k == 'ishl' else (u + n[3])
            r = bytes_of(u, nbytes).reshape(NC, NL, nbytes)
        elif k == 'ilow16':
            src = self.ev(n[1])
            r = np.ascontiguousarray(src[..., 0:2])
        elif k == 'prmt':
            ctl, a, b = n[1], self.ev(n[2]), self.ev(n[3])
            if a.shape[-1] != 4 or b.shape[-1] != 4:
                raise RuntimeError('PRMT_WIDTH')
            parts = []
            for j in range(4):
                sel = (ctl >> (4 * j)) & 15
                if sel >= 8:
                    raise RuntimeError('PRMT_SIGN_REPLICATION_NOT_MODELLED')
                parts.append((a if sel < 4 else b)[..., sel % 4:sel % 4 + 1])
            r = np.concatenate(parts, -1)
        elif k == 'gather':
            pc, idx = n[1], n[2]
            src = self.ev(self.gsrc['%d|%d' % (pc, idx)])
            r = np.broadcast_to(src[:, idx:idx + 1, :], (NC, NL, src.shape[-1]))
        elif k == 'tof32':
            h = f16(self.ev(n[1]))
            r = np.ascontiguousarray(h.astype('<f4')).view(np.uint8)
        elif k == 'rcp':
            x32 = np.ascontiguousarray(self.ev(n[1])).view('<f4')
            h = x32.astype(np.float16)
            if not np.array_equal(h.astype(np.float32), x32):
                raise RuntimeError('RCP_INPUT_NOT_A_HALF')
            bits = h.view(np.uint16).astype(np.int64)
            if (bits < RCP_FIRST).any() or (bits >= 0x7C00).any():
                raise L.OutOfDomain('RCP input outside the measured table domain [0x0410, 0x7c00)')
            self.stats['rcp_bits_min'] = min(self.stats.get('rcp_bits_min', 0xFFFF), int(bits.min()))
            self.stats['rcp_bits_max'] = max(self.stats.get('rcp_bits_max', 0), int(bits.max()))
            r = np.ascontiguousarray(self.rcp_f32[bits - RCP_FIRST]).view(np.uint8)
        elif k == 'fmulc':
            a32 = np.ascontiguousarray(self.ev(n[1])).view('<f4')
            c32 = np.frombuffer(np.uint32(n[2]).astype('<u4').tobytes(), '<f4')[0]
            with np.errstate(over='ignore', under='ignore'):
                p = (a32 * c32).astype(np.float32)
            if not np.isfinite(p).all() or ((p != 0) & (np.abs(p) < np.finfo(np.float32).tiny)).any():
                raise L.OutOfDomain('mul.ftz.f32 outside the normal float32 domain')
            r = np.ascontiguousarray(p.astype('<f4')).view(np.uint8)
        elif k == 'f16of':
            x32 = np.ascontiguousarray(self.ev(n[1])).view('<f4')
            with np.errstate(over='ignore'):
                h = x32.astype(np.float16)
            if not np.isfinite(h).all():
                raise L.OutOfDomain('non-finite float32 -> float16')
            r = to_bytes16(h)
        elif k == 'sat':
            h = f16(self.ev(n[1]))
            if not np.isfinite(h).all():
                raise L.OutOfDomain('non-finite value published')
            r = np.ascontiguousarray(CORE.quantise_codes(h.astype(np.float32)))
        else:
            raise RuntimeError('TEMPLATE_NODE_UNKNOWN:%s' % k)
        self.memo[i] = r
        return r

    def run(self, templates):
        L = self.LANE
        wiring = templates['0|0']['wiring']
        if any(templates['0|%d' % l]['wiring'] != wiring for l in range(NL)):
            raise RuntimeError('QMMA_WIRING_NOT_LANE_UNIFORM')
        for ep, slot, pc, A, B, C in wiring:
            a_l = np.concatenate([self.ev(x) for x in A], -1).reshape(NC, NL, 16)
            b_l = np.concatenate([self.ev(x) for x in B], -1).reshape(NC, NL, 8)
            c_l = np.concatenate([self.ev(x) for x in C], -1).reshape(NC, NL, 8)
            C16 = L.lanes_to_c(f16(c_l).reshape(NC, NL, 4))
            D = L.qmma_step(L.a_matrix(a_l), L.b_matrix(b_l), C16, stats=self.stats)
            self.mma_out[(ep, slot)] = to_bytes16(L.d_to_lanes(D)).reshape(NC, NL, 8)
        out, cover = np.zeros(65536, np.uint8), np.zeros(65536, np.int32)
        for lane in range(NL):
            for kind, ep, slot, pc, node in templates['0|%d' % lane]['stores']:
                if kind != 'ostore':
                    raise RuntimeError('STORE_KIND_UNKNOWN')
                val = self.ev(node)
                if val.shape != (NC, NL, 16):
                    raise RuntimeError('STORE_WIDTH')
                off = self.npz['ostore|%d|%d' % (ep, slot)].astype(np.int64)[:, 0, lane]
                if (off < 0).any():
                    raise RuntimeError('STORE_OFFSET_MISSING')
                idx = off[:, None] + np.arange(16)
                out[idx] = val[:, lane, :]
                np.add.at(cover, idx.ravel(), 1)
        if not (cover == 1).all():
            raise RuntimeError('OUTPUT_COVERAGE_NOT_EXACTLY_ONCE')
        return out


def compute(LANE, CORE, data, tm, fc, npz, rcp_table_npz, rcp_table_sha256, stats):
    for n in ('q', 'k', 'v'):
        if data[n].size != 65536:
            raise RuntimeError('INPUT_LENGTH:%s' % n)
    if tm.get('status') != 'PASS_EXECUTABLE_MAP' or fc.get('status') != 'PASS_FAMILIES_EQUAL':
        raise RuntimeError('MAP_OR_FAMILY_CHECK_INVALID')
    if rcp_table_sha256 != CORE._TABLE_FILES['rcp'][1]:
        raise RuntimeError('RCP_TABLE_HASH_NOT_THE_BACKEND_PIN')
    t = rcp_table_npz
    in_bits, f_bits, h_bits = (np.array(t[k]) for k in ('input_half_bits', 'native_float_bits', 'native_half_bits'))
    if not np.array_equal(in_bits.astype(np.int64), np.arange(RCP_FIRST, RCP_FIRST + in_bits.size)) or in_bits.size < 0x7C00 - RCP_FIRST:
        raise RuntimeError('RCP_TABLE_INDEX_LAYOUT')
    rcp_f32 = np.ascontiguousarray(f_bits.astype('<u4')).view('<f4')
    stats['rcp_table_float_to_half_rne_equals_measured_half'] = bool(np.array_equal(rcp_f32.astype(np.float16).view(np.uint16), h_bits.astype(np.uint16)))
    EV = Evaluator(LANE, CORE, tm['nodes'], npz, tm['gather_sources'], data, rcp_f32, stats)
    out = EV.run(tm['templates'])
    return {'output': out.tobytes()}
