"""ffn_contract_chained kernel.

Evaluator copied line by line from experiments/parity-astra-opus-b31-l59-20260913/revisions/r02-launch60/predict_l60.py;
compute() = that predictor's PHASE 'compute' section (from the MAP_OR_INPUT_INVALID check to the coverage check).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import numpy as np

NC, NY, NL = 8, 2, 32


class Evaluator:
    """Evaluate interpreter templates for one stage; values are little-endian byte arrays [NC, NY, NL, nbytes]."""

    def __init__(self, LANE, CORE, npz, buffers, stats):
        self.LANE, self.CORE, self.npz, self.buffers, self.stats = LANE, CORE, npz, buffers, stats
        self.cache = {}

    def offsets(self, kind, ep, slot, z):
        name = ('src|%d|%d' % (ep, slot)) if kind == 'sread' else ('%s|%d|%d' % (kind, ep, slot))
        if name not in self.cache:
            self.cache[name] = self.npz[name].astype(np.int64)
        off = self.cache[name][:, z, :NY, :]
        if (off < 0).any():
            raise RuntimeError('MAP_OFFSET_MISSING:%s' % name)
        return off

    def fetch(self, kind, ep, slot, z, width):
        src = self.buffers['input' if kind == 'sread' else {'gload': 'weights', 'g32': 'weights', 'cg': 'residual', 'bufload': 'buffer'}[kind]]
        idx = self.offsets(kind, ep, slot, z)[..., None] + np.arange(width)
        if kind == 'bufload' and not self.buffers['written'][idx].all():
            raise RuntimeError('BUFFER_READ_BEFORE_WRITE')
        return src[idx]

    def f16(self, b):
        return np.ascontiguousarray(b).view('<f2')

    def to_bytes(self, f):
        return np.ascontiguousarray(f.astype('<f2')).view(np.uint8)

    def ev(self, node, z, mma_out):
        k = node[0]
        if k in ('sread', 'gload', 'cg', 'bufload'):
            return self.fetch(k, node[1], node[2], z, 16)[..., 4 * node[3]:4 * node[3] + 4]
        if k == 'g32':
            return self.fetch(k, node[1], node[2], z, 4)
        if k == 'mma':
            return mma_out[(node[1], node[2])][..., 4 * node[3]:4 * node[3] + 4]
        if k == 'cat':
            return np.concatenate([self.ev(x, z, mma_out) for x in node[1:]], -1)
        if k == 'c16':
            b = np.frombuffer(np.float16(float(node[1])).tobytes(), np.uint8)
            return np.broadcast_to(b, (NC, NY, NL, 2))
        if k == 'part':
            x = self.ev(node[1], z, mma_out)
            return x[..., 2 * node[2]:2 * node[2] + 2]
        if k == 'dq':
            codes = self.ev(node[1], z, mma_out)
            if ((codes & 0x7F) == 0x7F).any():
                raise self.LANE.OutOfDomain('E4M3 NaN code in a decoded operand')
            self.stats['dq'] = self.stats.get('dq', 0) + int(codes.size)
            return self.to_bytes(self.CORE.decode_codes(codes).astype(np.float64))
        if k == 'sat':
            f = self.f16(self.ev(node[1], z, mma_out))
            if not np.isfinite(f).all():
                raise self.LANE.OutOfDomain('non-finite value published')
            return np.ascontiguousarray(self.CORE.quantise_codes(f.astype(np.float32)))
        if k == 'fop':
            a = self.f16(self.ev(node[2], z, mma_out)).astype(np.float64)
            b = self.f16(self.ev(node[3], z, mma_out)).astype(np.float64)
            with np.errstate(over='ignore'):
                r = (a * b if node[1] == 'mul' else a + b).astype(np.float16)
            if not np.isfinite(r).all():
                raise self.LANE.OutOfDomain('binary16 overflow in %s' % node[1])
            self.stats[node[1]] = self.stats.get(node[1], 0) + int(r.size)
            return self.to_bytes(r)
        raise RuntimeError('TEMPLATE_NODE_UNKNOWN:%s' % k)

    def stage(self, z, wiring):
        L = self.LANE
        mma_out = {}
        NB = NC * NY
        for ep, slot, pc, A, B, C in wiring:
            a_l = np.concatenate([self.ev(x, z, mma_out) for x in A], -1).reshape(NB, NL, 16)
            b_l = np.concatenate([self.ev(x, z, mma_out) for x in B], -1).reshape(NB, NL, 8)
            c_l = np.concatenate([self.ev(x, z, mma_out) for x in C], -1).reshape(NB, NL, 8)
            C16 = L.lanes_to_c(self.f16(c_l).reshape(NB, NL, 4))
            D = L.qmma_step(L.a_matrix(a_l), L.b_matrix(b_l), C16, stats=self.stats)
            mma_out[(ep, slot)] = self.to_bytes(L.d_to_lanes(D)).reshape(NC, NY, NL, 8)
        return mma_out


def compute(LANE, CORE, inp, res, W, tmpl, npz, stats):
    if tmpl.get('status') != 'PASS_EXECUTABLE_MAP' or inp.size != 262144 or res.size != 65536 or W.size != 4196352:
        raise RuntimeError('MAP_OR_INPUT_INVALID')
    buf, written = np.zeros(131072, np.uint8), np.zeros(131072, bool)
    out, cover = np.zeros(65536, np.uint8), np.zeros(65536, np.int32)
    EV = Evaluator(LANE, CORE, npz, dict(input=inp, residual=res, weights=W, buffer=buf, written=written), stats)

    def store_key(kind, slot):
        ks = [k for k in npz if k.startswith(kind + '|') and k.endswith('|%d' % slot)]
        if len(ks) != 1:
            raise RuntimeError('STORE_KEY_AMBIGUOUS:%s|%d' % (kind, slot))
        return ks[0]
    for z in range(4):
        t = tmpl['templates']['%d|live' % z]
        mma_out = EV.stage(z, t['wiring'])
        for kind, slot, pc, node in t['stores']:
            off = npz[store_key(kind, slot)].astype(np.int64)[:, z, :NY, :]
            if (off < 0).any():
                raise RuntimeError('STORE_OFFSET_MISSING')
            idx = off[..., None] + np.arange(16)
            val = np.ascontiguousarray(EV.ev(node, z, mma_out))
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
            elif kind == 'outstore':
                if z != 3:
                    raise RuntimeError('OUTPUT_STORE_UNEXPECTED')
                out[idx] = val
                np.add.at(cover, idx.ravel(), 1)
            else:
                raise RuntimeError('STORE_KIND_UNKNOWN')
    if not (cover == 1).all():
        raise RuntimeError('OUTPUT_COVERAGE_NOT_EXACTLY_ONCE')
    # the buffer is returned for the write-set record only; the B31 contract comparison region is 'output'
    return {'output': out.tobytes(), 'buffer_not_compared': buf.tobytes()}
