"""qkv_chained kernel.

Evaluator copied line by line from experiments/parity-astra-opus-b31-l59-20260913/revisions/r03-launch61/predict_l61.py;
compute() = that predictor's PHASE 'compute' section (from the map-check test to the coverage check).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import numpy as np

NC, NL = 16, 32
BUFFER_BYTES, WEIGHT_BYTES = 393216, 3145856
RSQ_FIRST = 0x0410


def f16(b):
    return np.ascontiguousarray(b).view('<f2')


def to_bytes16(f):
    return np.ascontiguousarray(np.asarray(f).astype('<f2')).view(np.uint8)


class Evaluator:
    """Evaluate one (z, warp) class; every value is a little-endian byte array [NC, NL, nbytes]."""

    def __init__(self, LANE, CORE, nodes, npz, data, rsq_f32, stats):
        self.LANE, self.CORE, self.nodes, self.npz, self.data, self.rsq_f32, self.stats = LANE, CORE, nodes, npz, data, rsq_f32, stats

    def offsets(self, name, z, y):
        off = self.npz[name].astype(np.int64)[:, z, y, :]
        return off

    def leaf(self, n, z, y):
        _, kind, ep, slot, word = n
        if kind == 'mma':
            return self.mma_out[(ep, slot)][..., 4 * word:4 * word + 4]
        name = ('src|%d|%d' % (ep, slot)) if kind == 'sread' else ('%s|%d|%d' % (kind, ep, slot))
        off = self.offsets(name, z, y)
        if kind == 'g32':
            if (off < 0).any():
                raise RuntimeError('MAP_OFFSET_MISSING:%s' % name)
            return self.data['weights'][off[..., None] + np.arange(4)]
        if kind == 'sread':
            if (off == -9).any() or (off < -1).any():
                raise RuntimeError('MAP_OFFSET_MISSING:%s' % name)
            zero = off < 0
            self.stats['sread_zero_lanes'] = self.stats.get('sread_zero_lanes', 0) + int(zero.sum())
            idx = np.where(zero, 0, off)[..., None] + 4 * word + np.arange(4)
            return np.where(zero[..., None], 0, self.data['input'][idx]).astype(np.uint8)
        if (off < 0).any():
            raise RuntimeError('MAP_OFFSET_MISSING:%s' % name)
        idx = off[..., None] + 4 * word + np.arange(4)
        if kind == 'gload':
            return self.data['weights'][idx]
        if kind == 'bufload':
            if not self.data['written'][idx].all():
                raise RuntimeError('BUFFER_READ_BEFORE_WRITE')
            return self.data['buffer'][idx]
        raise RuntimeError('LEAF_KIND_UNKNOWN:%s' % kind)

    def ev(self, i, z, y):
        hit = self.memo.get(i)
        if hit is not None:
            return hit
        n = self.nodes[i]
        k = n[0]
        L, CORE = self.LANE, self.CORE
        if k == 'leaf':
            r = self.leaf(n, z, y)
        elif k == 'c16':
            r = np.broadcast_to(np.frombuffer(np.float16(float(n[1])).tobytes(), np.uint8), (NC, NL, 2))
        elif k == 'cat':
            r = np.concatenate([self.ev(x, z, y) for x in n[1:]], -1)
        elif k == 'part':
            r = self.ev(n[1], z, y)[..., 2 * n[2]:2 * n[2] + 2]
        elif k == 'fop':
            op, a, b = n[1], n[2], n[3]
            na, nb = self.nodes[a], self.nodes[b]
            with np.errstate(over='ignore', invalid='ignore'):
                if op == 'add' and na[0] == 'fop' and na[1] == 'mul' and nb[0] == 'fop' and nb[1] == 'mul':
                    # H_FMA_OPERAND0: HMUL2 t = y*y (rounded), HFMA2 s = x*x + t (one rounding)
                    x0, x1 = f16(self.ev(na[2], z, y)).astype(np.float64), f16(self.ev(na[3], z, y)).astype(np.float64)
                    t = f16(self.ev(b, z, y)).astype(np.float64)
                    res = (x0 * x1 + t).astype(np.float16)
                    self.stats['fma_sites_evaluated'] = self.stats.get('fma_sites_evaluated', 0) + 1
                else:
                    fa, fb = f16(self.ev(a, z, y)).astype(np.float64), f16(self.ev(b, z, y)).astype(np.float64)
                    res = {'add': lambda: (fa + fb).astype(np.float16), 'mul': lambda: (fa * fb).astype(np.float16),
                           'max': lambda: np.maximum(fa.astype(np.float16), fb.astype(np.float16))}[op]()
            if not np.isfinite(res).all():
                raise L.OutOfDomain('non-finite binary16 result in %s' % op)
            if op == 'max':
                self.stats['max_input_min'] = min(self.stats.get('max_input_min', 1e9), float(f16(self.ev(a, z, y)).astype(np.float64).min()))
            r = to_bytes16(res)
        elif k == 'tof32':
            h = f16(self.ev(n[1], z, y))
            r = np.ascontiguousarray(h.astype('<f4')).view(np.uint8)
        elif k == 'rsq':
            x32 = np.ascontiguousarray(self.ev(n[1], z, y)).view('<f4')
            h = x32.astype(np.float16)
            if not np.array_equal(h.astype(np.float32), x32):
                raise RuntimeError('RSQ_INPUT_NOT_A_HALF')
            bits = h.view(np.uint16).astype(np.int64)
            if (bits < RSQ_FIRST).any() or (bits >= 0x7C00).any():
                raise L.OutOfDomain('RSQ input outside the measured table domain [0x0410, 0x7c00)')
            self.stats['rsq_bits_min'] = min(self.stats.get('rsq_bits_min', 0xFFFF), int(bits.min()))
            self.stats['rsq_bits_max'] = max(self.stats.get('rsq_bits_max', 0), int(bits.max()))
            r = np.ascontiguousarray(self.rsq_f32[bits - RSQ_FIRST]).view(np.uint8)
        elif k == 'f16of':
            x32 = np.ascontiguousarray(self.ev(n[1], z, y)).view('<f4')
            with np.errstate(over='ignore'):
                h = x32.astype(np.float16)
            if not np.isfinite(h).all():
                raise L.OutOfDomain('non-finite float32 -> float16')
            r = to_bytes16(h)
        elif k == 'bfly':
            src = self.ev(n[2], z, y)
            r = src[:, np.arange(NL) ^ n[1], :]
        elif k == 'movm':
            src = f16(self.ev(n[1], z, y)).reshape(NC, NL * 2)
            cell = np.arange(64)
            r = to_bytes16(src[:, 8 * (cell % 8) + cell // 8]).reshape(NC, NL, 2 * 2)
        elif k == 'sat':
            h = f16(self.ev(n[1], z, y))
            if not np.isfinite(h).all():
                raise L.OutOfDomain('non-finite value published')
            r = np.ascontiguousarray(CORE.quantise_codes(h.astype(np.float32)))
        else:
            raise RuntimeError('TEMPLATE_NODE_UNKNOWN:%s' % k)
        self.memo[i] = r
        return r

    def run_class(self, z, y, tmpl):
        L = self.LANE
        self.memo, self.mma_out = {}, {}
        NB = NC
        for ep, slot, pc, A, B, C in tmpl['wiring']:
            a_l = np.concatenate([self.ev(x, z, y) for x in A], -1).reshape(NB, NL, 16)
            b_l = np.concatenate([self.ev(x, z, y) for x in B], -1).reshape(NB, NL, 8)
            c_l = np.concatenate([self.ev(x, z, y) for x in C], -1).reshape(NB, NL, 8)
            C16 = L.lanes_to_c(f16(c_l).reshape(NB, NL, 4))
            D = L.qmma_step(L.a_matrix(a_l), L.b_matrix(b_l), C16, stats=self.stats)
            self.mma_out[(ep, slot)] = to_bytes16(L.d_to_lanes(D)).reshape(NB, NL, 8)
            # C of a later MMA may be this D: drop memoised values that depend on nothing newer (mma leaves are keyed per event)
        stores = []
        for kind, ep, slot, pc, node in tmpl['stores']:
            stores.append((kind, ep, slot, np.ascontiguousarray(self.ev(node, z, y))))
        return stores


def compute(LANE, CORE, inp, W, tm, sc, npz, rsq_table_npz, rsq_table_sha256, stats):
    if (tm.get('status') != 'PASS_EXECUTABLE_MAP' or sc.get('status') != 'PASS_H_FMA_OPERAND0' or inp.size != 65536 or W.size != WEIGHT_BYTES):
        raise RuntimeError('MAP_CHECK_OR_INPUT_INVALID')
    rsq_hash = CORE._TABLE_FILES['rsq'][1]
    if rsq_table_sha256 != rsq_hash:
        raise RuntimeError('RSQ_TABLE_HASH_NOT_THE_BACKEND_PIN')
    t = rsq_table_npz
    in_bits, f_bits, h_bits = (np.array(t[k]) for k in ('input_half_bits', 'native_float_bits', 'native_half_bits'))
    if not np.array_equal(in_bits.astype(np.int64), np.arange(RSQ_FIRST, RSQ_FIRST + in_bits.size)) or in_bits.size < 0x7C00 - RSQ_FIRST:
        raise RuntimeError('RSQ_TABLE_INDEX_LAYOUT')
    rsq_f32 = np.ascontiguousarray(f_bits.astype('<u4')).view('<f4')
    stats['rsq_table_float_to_half_rne_equals_measured_half'] = bool(np.array_equal(rsq_f32.astype(np.float16).view(np.uint16), h_bits.astype(np.uint16)))
    nodes = tm['nodes']
    buf, written = np.zeros(BUFFER_BYTES, np.uint8), np.zeros(BUFFER_BYTES, bool)
    outs = {n: np.zeros(65536, np.uint8) for n in ('q', 'k', 'v')}
    cover = {n: np.zeros(65536, np.int32) for n in ('q', 'k', 'v')}
    EV = Evaluator(LANE, CORE, nodes, npz, dict(input=inp, weights=W, buffer=buf, written=written), rsq_f32, stats)
    for z in (0, 1):                                                        # H_STAGE_ORDER: every z0 before any z1
        for y in (0, 1):
            t = tm['templates']['%d|%d' % (z, y)]
            before = stats.get('fma_sites_evaluated', 0)
            for kind, ep, slot, val in EV.run_class(z, y, t):
                off = npz['%s|%d|%d' % (kind, ep, slot)].astype(np.int64)[:, z, y, :]
                if (off < 0).any() or val.shape != (NC, NL, 16):
                    raise RuntimeError('STORE_OFFSET_OR_WIDTH')
                idx = off[..., None] + np.arange(16)
                if kind == 'bufstore':
                    if z != 0 or written[idx].any():
                        raise RuntimeError('BUFFER_STORE_UNEXPECTED')
                    buf[idx] = val; written[idx] = True
                elif kind in ('qstore', 'kstore', 'vstore'):
                    if z != 1:
                        raise RuntimeError('OUTPUT_STORE_UNEXPECTED')
                    outs[kind[0]][idx] = val
                    np.add.at(cover[kind[0]], idx.ravel(), 1)
                else:
                    raise RuntimeError('STORE_KIND_UNKNOWN')
            sites = stats.get('fma_sites_evaluated', 0) - before
            if sites != (32 if z == 1 else 0):
                raise RuntimeError('FMA_SITE_COUNT:%d|%d:%d' % (z, y, sites))
    if not written.all() or not all((cover[n] == 1).all() for n in cover):
        raise RuntimeError('COVERAGE_NOT_EXACTLY_ONCE')
    return {'q': outs['q'].tobytes(), 'k': outs['k'].tobytes(), 'v': outs['v'].tobytes(), 'buffer': buf.tobytes()}
