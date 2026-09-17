"""projection kernels (chained / wait).

Evaluator copied line by line from experiments/parity-astra-opus-b31-l59-20260913/revisions/r05-launch63/predict_l63.py;
compute() = that predictor's PHASE 'compute' section (from the map-check test to the coverage check).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import numpy as np

NC, NL = 8, 32
BUFFER_BYTES = 131072


def f16(b):
    return np.ascontiguousarray(b).view('<f2')


def to_bytes16(f):
    return np.ascontiguousarray(np.asarray(f).astype('<f2')).view(np.uint8)


class Evaluator:
    """Evaluate one (z, warp) class; every value is a little-endian byte array [NC, NL, nbytes]."""

    def __init__(self, LANE, CORE, nodes, npz, data, stats):
        self.LANE, self.CORE, self.nodes, self.npz, self.data, self.stats = LANE, CORE, nodes, npz, data, stats

    def leaf(self, n, z, y):
        _, kind, ep, slot, word = n
        if kind == 'mma':
            return self.mma_out[(ep, slot)][..., 4 * word:4 * word + 4]
        name = ('src|%d|%d' % (ep, slot)) if kind == 'sread' else ('%s|%d|%d' % (kind, ep, slot))
        off = self.npz[name].astype(np.int64)[:, z, y, :]
        if kind == 'sread':
            if (off < -1).any():
                raise RuntimeError('MAP_OFFSET_MISSING:%s' % name)
            zero = off < 0
            self.stats['sread_zero_lanes'] = self.stats.get('sread_zero_lanes', 0) + int(zero.sum())
            idx = np.where(zero, 0, off)[..., None] + 4 * word + np.arange(4)
            return np.where(zero[..., None], 0, self.data['input'][idx]).astype(np.uint8)
        if (off < 0).any():
            raise RuntimeError('MAP_OFFSET_MISSING:%s' % name)
        if kind == 'g32':
            return self.data['weights'][off[..., None] + np.arange(4)]
        idx = off[..., None] + 4 * word + np.arange(4)
        if kind == 'gload':
            return self.data['weights'][idx]
        if kind == 'cg':
            return self.data['residual'][idx]
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
        elif k == 'ibits':
            r = np.broadcast_to(np.frombuffer(np.uint32(n[1]).astype('<u4').tobytes(), np.uint8), (NC, NL, 4))
        elif k == 'cat':
            r = np.concatenate([self.ev(x, z, y) for x in n[1:]], -1)
        elif k == 'part':
            r = self.ev(n[1], z, y)[..., 2 * n[2]:2 * n[2] + 2]
        elif k == 'dq':
            codes = self.ev(n[1], z, y)
            if ((codes & 0x7F) == 0x7F).any():
                raise L.OutOfDomain('E4M3 NaN code in a decoded operand')
            self.stats['dq'] = self.stats.get('dq', 0) + int(codes.size)
            r = to_bytes16(CORE.decode_codes(codes).astype(np.float64))
        elif k == 'fop':
            a = f16(self.ev(n[2], z, y)).astype(np.float64)
            b = f16(self.ev(n[3], z, y)).astype(np.float64)
            with np.errstate(over='ignore'):
                res = (a * b if n[1] == 'mul' else a + b).astype(np.float16)
            if n[1] not in ('mul', 'add'):
                raise RuntimeError('FOP_UNKNOWN:%s' % n[1])
            if not np.isfinite(res).all():
                raise L.OutOfDomain('binary16 overflow in %s' % n[1])
            self.stats['fop_' + n[1]] = self.stats.get('fop_' + n[1], 0) + 1
            r = to_bytes16(res)
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
        for ep, slot, pc, A, B, C in tmpl['wiring']:
            a_l = np.concatenate([self.ev(x, z, y) for x in A], -1).reshape(NC, NL, 16)
            b_l = np.concatenate([self.ev(x, z, y) for x in B], -1).reshape(NC, NL, 8)
            c_l = np.concatenate([self.ev(x, z, y) for x in C], -1).reshape(NC, NL, 8)
            C16 = L.lanes_to_c(f16(c_l).reshape(NC, NL, 4))
            D = L.qmma_step(L.a_matrix(a_l), L.b_matrix(b_l), C16, stats=self.stats)
            self.mma_out[(ep, slot)] = to_bytes16(L.d_to_lanes(D)).reshape(NC, NL, 8)
        return [(kind, ep, slot, np.ascontiguousarray(self.ev(node, z, y))) for kind, ep, slot, pc, node in tmpl['stores']]


def compute(LANE, CORE, inp, res, W, tm, fc, npz, stats):
    if tm.get('status') != 'PASS_EXECUTABLE_MAP' or fc.get('status') != 'PASS_FAMILIES_EQUAL' or inp.size != 65536 or res.size != 65536 or W.size != 1050624:
        raise RuntimeError('MAP_CHECK_OR_INPUT_INVALID')
    buf, written = np.zeros(BUFFER_BYTES, np.uint8), np.zeros(BUFFER_BYTES, bool)
    out, cover = np.zeros(65536, np.uint8), np.zeros(65536, np.int32)
    red_count = np.zeros(BUFFER_BYTES, np.int32)
    EV = Evaluator(LANE, CORE, tm['nodes'], npz, dict(input=inp, residual=res, weights=W, buffer=buf, written=written), stats)
    for z in range(4):                                                      # H_STAGE_ORDER
        for y in (0, 1):
            for kind, ep, slot, val in EV.run_class(z, y, tm['templates']['%d|%d' % (z, y)]):
                off = npz['%s|%d|%d' % (kind, ep, slot)].astype(np.int64)[:, z, y, :]
                if (off < 0).any() or val.shape != (NC, NL, 16):
                    raise RuntimeError('STORE_OFFSET_OR_WIDTH')
                idx = off[..., None] + np.arange(16)
                if kind == 'bufstore':
                    if z != 0 or written[idx].any():
                        raise RuntimeError('BUFFER_STORE_UNEXPECTED')
                    buf[idx] = val; written[idx] = True
                elif kind == 'red':
                    if z not in (1, 2) or not written[idx].all():
                        raise RuntimeError('REDG_ON_UNWRITTEN_BUFFER')
                    cur = np.ascontiguousarray(buf[idx]).view('<f2').astype(np.float64)
                    add = val.view('<f2').astype(np.float64)
                    with np.errstate(over='ignore'):
                        r = (cur + add).astype(np.float16)
                    if not np.isfinite(r).all():
                        raise LANE.OutOfDomain('binary16 overflow in REDG add')
                    buf[idx] = np.ascontiguousarray(r.astype('<f2')).view(np.uint8)
                    np.add.at(red_count, idx.ravel(), 1)
                elif kind == 'outstore':
                    if z != 3:
                        raise RuntimeError('OUTPUT_STORE_UNEXPECTED')
                    out[idx] = val
                    np.add.at(cover, idx.ravel(), 1)
                else:
                    raise RuntimeError('STORE_KIND_UNKNOWN')
    if not (cover == 1).all() or not written.all() or not (red_count == 2).all():
        raise RuntimeError('COVERAGE_NOT_EXACT')
    return {'output': out.tobytes(), 'buffer': buf.tobytes()}
