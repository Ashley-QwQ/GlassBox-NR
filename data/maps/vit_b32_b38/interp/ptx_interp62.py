# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

"""launch62 PTX address/provenance interpreter (derived from ../r03-launch61/ptx_interp61.py, which is unchanged).

Executes every thread of gridDim 32x1x1 (ctaid.x) x blockDim 32x4x1 (lane, warp) = 4,096 threads of
cc_vit_1d_attention_chained_fp8.  Integer/address instructions exactly; data symbolically (hash-consed DAG, no numbers).
Added relative to launch61: key and value shared storages, 8/16-bit integer instructions and casts, exact SYMBOLIC integer
nodes on half bit patterns (shl/add on b32 and b16, low-16 cast), fma/min/max/sub f16x2, rcp.approx.ftz.f32,
cvt.rn.f32.u32, mul.ftz.f32 by a constant, prmt.b32, and shfl.sync.idx.b32 with a {reg|pred} destination.
Joint hypotheses (as before): H_TID_X_ZERO, H_ELECT_ANY, H_BARRIER_VIS, H_BRACE_LE; new:
  H_SHFL_LANE   shfl.sync.idx.b32 d|p, s, idx, 31, -1: d(lane) = s(lane idx & 31) at that statement, same warp thread group
                (PTX documentation reading used by the B39 round; not measured on sm_120).  The DAG records a gather node
                (pc, idx); the source node of every lane at that pc is tabulated so the predictor can resolve it.
Templates are classed per (warp, lane) because lane-dependent predicates select different wiring per lane.
Output: maps/l62-address-map.npz (per-event offsets), maps/l62-templates-and-checks.json.
"""
import sys
sys.dont_write_bytecode = True
import collections, hashlib, json, re, struct, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_A = sys.argv


def _arg(k, d=None):
    if k in _A:
        return _A[_A.index(k) + 1]
    if d is None:
        raise SystemExit('missing argument %s' % k)
    return d
PTX = Path(_arg('--ptx'))
SASS = Path(_arg('--sass'))
S0 = Path(_arg('--s0'))
OUT_NPZ = Path(_arg('--out-npz'))
OUT_JSON = Path(_arg('--out-json'))
FUNC = _arg('--func', 'cc_vit_1d_attention_chained_fp8')
KEY_BASE, VAL_BASE, BARRIER_BASE = 0x10000000, 0x18000000, 0x20000000
LOOP_HEAD = _arg('--loop-head')
M16, M32, M64 = 0xFFFF, 0xFFFFFFFF, (1 << 64) - 1
SPECIALS = ('%ctaid.x', '%ctaid.y', '%ctaid.z', '%tid.x', '%tid.y', '%tid.z', '%ntid.x', '%ntid.y', '%ntid.z', '%laneid')
MMA = 'mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16'
RELEASE_DONE = 0x7FFFFFF0
LEAF_KINDS = ('qload', 'kread', 'vread')
STORE_KINDS = ('ostore',)
TYPE_BITS = {'u8': 8, 's8': 8, 'u16': 16, 's16': 16, 'u32': 32, 's32': 32, 'u64': 64, 's64': 64}


class Unsupported(Exception):
    pass


def split_top(s):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in '{[':
            depth += 1
        elif ch in '}]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(cur)); cur = []
        else:
            cur.append(ch)
    if ''.join(cur).strip():
        out.append(''.join(cur))
    return [x.strip() for x in out]


def parse_operand(tok):
    tok = tok.strip()
    if tok.startswith('{'):
        return ('grp', tuple(parse_operand(t) for t in split_top(tok[1:-1])))
    if tok.startswith('['):
        inner = tok[1:-1].strip()
        m = re.fullmatch(FUNC + r'_param_0(?:\+(\d+))?', inner)
        if m:
            return ('pmem', int(m.group(1) or 0))
        m = re.fullmatch(r'(%\w+)\+(\d+)', inner)
        if m:
            return ('memoff', parse_operand(m.group(1)), int(m.group(2)))
        return ('mem', parse_operand(inner))
    if tok in SPECIALS:
        return ('spc', tok)
    m = re.fullmatch(r'(%r\d+)\|(%p\d+)', tok)
    if m:
        return ('pairdst', m.group(1), m.group(2))
    if tok.startswith('_|'):
        return ('reg', tok[2:])
    if re.fullmatch(r'-?\d+', tok):
        return ('int', int(tok))
    m = re.fullmatch(r'0x([0-9a-fA-F]+)U?', tok)
    if m:
        return ('int', int(m.group(1), 16))
    if tok.startswith('_ZZ') and tok.endswith('E18shared_key_storage'):
        return ('int', KEY_BASE)
    if tok.startswith('_ZZ') and tok.endswith('E20shared_value_storage'):
        return ('int', VAL_BASE)
    if tok.startswith('_ZZ') and tok.endswith('E30key_value_copy_barrier_storage'):
        return ('int', BARRIER_BASE)
    if re.fullmatch(r'%(r|rd|rs|p)\d+', tok) or re.fullmatch(r'[A-Za-z_]\w*', tok):
        return ('reg', tok)
    raise Unsupported('operand %r' % tok)


def parse(text):
    b0 = text.index('{', text.index('.maxnreg')) + 1
    src = text[b0:text.rindex('}')]
    stmts, labels, buf, depth = [], {}, [], 0
    for ch in src:
        if ch == '{':
            if depth == 0 and not ''.join(buf).strip():
                continue
            depth += 1; buf.append(ch); continue
        if ch == '}':
            if depth == 0:
                if ''.join(buf).strip():
                    raise Unsupported('block close inside statement')
                continue
            depth -= 1; buf.append(ch); continue
        if ch == '\n':
            t = ''.join(buf).strip()
            if depth == 0 and re.fullmatch(r'\$L__\w+:', t):
                labels[t[:-1]] = len(stmts); buf = []; continue
            buf.append(' '); continue
        if ch == ';' and depth == 0:
            t = ' '.join(''.join(buf).split()); buf = []
            if t and not t.startswith('.'):
                stmts.append(t)
            continue
        buf.append(ch)
    prog = []
    for t in stmts:
        pred = None
        m = re.match(r'@(%p\d+)\s+(.*)$', t)
        if m:
            pred, t = m.group(1), m.group(2)
        parts = t.split(' ', 1)
        mn = parts[0]
        ops = [] if len(parts) == 1 else split_top(parts[1])
        if mn in ('bra', 'bra.uni'):
            prog.append((mn, pred, ops[0].strip(), t)); continue
        prog.append((mn, pred, tuple(parse_operand(o) for o in ops), t))
    return prog, labels


def width(name):
    if name.startswith('%rd'):
        return 64
    if name.startswith('%rs'):
        return 16
    if name.startswith('%r'):
        return 32
    return None


def signed(x, bits):
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


def s32(x):
    return signed(x, 32)


def f32_bits_to_f16(bits):
    v32 = np.frombuffer(struct.pack('<I', bits & M32), '<f4')[0]
    h = np.float16(v32)
    return float(h), bool(float(h) == float(v32))


class Thread:
    def __init__(self, prog, labels, params, regions, c, y, lane, marks, tid_x=0):
        self.prog, self.labels, self.params, self.regions, self.marks = prog, labels, params, regions, marks
        self.spc = {'%ctaid.x': c, '%ctaid.y': 0, '%ctaid.z': 0, '%tid.x': tid_x, '%tid.y': y, '%tid.z': 0,
                    '%ntid.x': 1, '%ntid.y': 4, '%ntid.z': 1, '%laneid': lane}
        self.R, self.events, self.epoch, self.evid = {}, [], -1, 0
        self.back_taken = collections.Counter()
        self.const_f16 = []
        self.shfl_src = {}

    def val(self, op):
        k = op[0]
        if k == 'reg':
            try:
                return self.R[op[1]]
            except KeyError:
                raise Unsupported('read of undefined register %s' % op[1])
        if k == 'int':
            return op[1]
        if k == 'spc':
            return self.spc[op[1]]
        if k == 'grp':
            return ('cat', tuple(self.val(o) for o in op[1]))
        raise Unsupported('value of %r' % (op,))

    def ival(self, op):
        v = self.val(op)
        if not isinstance(v, int):
            raise Unsupported('integer expected, got %r' % (v,))
        return v

    def put(self, op, v):
        w = width(op[1])
        if w is not None and isinstance(v, int):
            v &= (1 << w) - 1
        self.R[op[1]] = v

    def addr(self, op):
        if op[0] == 'mem':
            return self.ival(op[1])
        if op[0] == 'memoff':
            return self.ival(op[1]) + op[2]
        raise Unsupported('memory operand %r' % (op,))

    def region(self, a, n):
        for name, (base, size) in self.regions.items():
            if base <= a and a + n <= base + size:
                return name, a - base
        raise Unsupported('address %x+%d outside every declared buffer' % (a, n))

    def shared(self, a, n):
        for name, base in (('key', KEY_BASE), ('val', VAL_BASE)):
            if base <= a and a + n <= base + 4096:
                return name, a - base
        raise Unsupported('shared address %x+%d outside the key/value storages' % (a, n))

    def emit(self, kind, pc, *payload):
        self.evid += 1
        self.events.append((kind, self.epoch, pc, self.evid) + payload)
        return self.evid

    def run(self, max_steps=400000):
        prog, labels, marks = self.prog, self.labels, self.marks
        pc, steps, n = 0, 0, len(prog)
        while pc < n:
            steps += 1
            if steps > max_steps:
                raise Unsupported('step limit')
            if pc in marks:
                self.epoch += 1
            mn, pred, ops, text = prog[pc]
            here = pc
            pc += 1
            if pred is not None and not self.R[pred]:
                continue
            if mn in ('bra', 'bra.uni'):
                tgt = labels[ops]
                if tgt <= here:
                    self.back_taken[ops] += 1
                pc = tgt
                continue
            if mn.startswith('mov.') and mn != MMA:
                if ops[0][0] == 'grp':
                    v = self.val(ops[1])
                    if isinstance(v, tuple) and v[0] == 'cat' and len(v[1]) == len(ops[0][1]):
                        for d, x in zip(ops[0][1], v[1]):
                            self.R[d[1]] = x
                    elif isinstance(v, int):
                        raise Unsupported('integer split into a register group')
                    else:
                        for i, d in enumerate(ops[0][1]):
                            self.R[d[1]] = ('part', v, i)
                else:
                    self.put(ops[0], self.val(ops[1]))
            elif mn in ('add.s32', 'add.s64', 'add.s16'):
                a, b = self.val(ops[1]), self.val(ops[2])
                if isinstance(a, int) and isinstance(b, int):
                    self.put(ops[0], a + b)
                elif isinstance(b, int):
                    self.R[ops[0][1]] = ('iadd', 32 if mn == 'add.s32' else 16, a, b & M32)
                else:
                    raise Unsupported('symbolic integer add form %s' % text)
            elif mn in ('sub.s32', 'sub.s16'):
                self.put(ops[0], self.ival(ops[1]) - self.ival(ops[2]))
            elif mn in ('mul.lo.s32', 'mul.lo.s16'):
                self.put(ops[0], self.ival(ops[1]) * self.ival(ops[2]))
            elif mn == 'mad.lo.s32':
                self.put(ops[0], self.ival(ops[1]) * self.ival(ops[2]) + self.ival(ops[3]))
            elif mn == 'mul.wide.s32':
                self.put(ops[0], s32(self.ival(ops[1])) * s32(self.ival(ops[2])))
            elif mn == 'mul.wide.s16':
                self.put(ops[0], signed(self.ival(ops[1]), 16) * signed(self.ival(ops[2]), 16))
            elif mn == 'mul.wide.u32':
                self.put(ops[0], (self.ival(ops[1]) & M32) * (self.ival(ops[2]) & M32))
            elif mn in ('shl.b32', 'shl.b64', 'shl.b16'):
                a, nsh = self.val(ops[1]), self.ival(ops[2])
                if isinstance(a, int):
                    self.put(ops[0], a << nsh)
                elif mn in ('shl.b32', 'shl.b16'):
                    self.R[ops[0][1]] = ('ishl', 32 if mn == 'shl.b32' else 16, a, nsh)
                else:
                    raise Unsupported('symbolic 64-bit shift')
            elif mn == 'shr.s32':
                self.put(ops[0], s32(self.ival(ops[1])) >> self.ival(ops[2]))
            elif mn == 'shr.s16':
                self.put(ops[0], signed(self.ival(ops[1]), 16) >> self.ival(ops[2]))
            elif mn == 'shr.u32':
                self.put(ops[0], (self.ival(ops[1]) & M32) >> self.ival(ops[2]))
            elif mn == 'shr.u16':
                self.put(ops[0], (self.ival(ops[1]) & M16) >> self.ival(ops[2]))
            elif mn in ('and.b32', 'and.b16'):
                self.put(ops[0], self.ival(ops[1]) & self.ival(ops[2]))
            elif mn == 'or.b32':
                self.put(ops[0], self.ival(ops[1]) | self.ival(ops[2]))
            elif mn == 'xor.b32':
                self.put(ops[0], self.ival(ops[1]) ^ self.ival(ops[2]))
            elif mn in ('or.pred', 'not.pred'):
                self.R[ops[0][1]] = bool(self.R[ops[1][1]] or self.R[ops[2][1]]) if mn == 'or.pred' else (not self.R[ops[1][1]])
            elif mn in ('max.s32', 'min.s32'):
                a, b = s32(self.ival(ops[1])), s32(self.ival(ops[2]))
                self.put(ops[0], max(a, b) if mn == 'max.s32' else min(a, b))
            elif mn == 'div.s32':
                a, b = s32(self.ival(ops[1])), s32(self.ival(ops[2]))
                q = abs(a) // abs(b)
                self.put(ops[0], q if (a < 0) == (b < 0) else -q)
            elif mn == 'div.u32':
                self.put(ops[0], (self.ival(ops[1]) & M32) // (self.ival(ops[2]) & M32))
            elif mn.startswith('cvta.'):
                self.put(ops[0], self.ival(ops[1]))
            elif mn == 'cvt.rn.f32.u32':
                self.put(ops[0], struct.unpack('<I', struct.pack('<f', float(self.ival(ops[1]) & M32)))[0])
            elif re.fullmatch(r'cvt\.([us](?:8|16|32|64))\.([us](?:8|16|32|64))', mn):
                dt, st_ = mn.split('.')[1:]
                v = self.val(ops[1])
                if not isinstance(v, int):
                    if mn == 'cvt.u16.u32':
                        self.R[ops[0][1]] = ('ilow16', v)
                        continue
                    raise Unsupported('symbolic cast %s' % mn)
                v &= (1 << TYPE_BITS[st_]) - 1
                if st_[0] == 's':
                    v = signed(v, TYPE_BITS[st_])
                v &= (1 << TYPE_BITS[dt]) - 1
                if dt[0] == 's':
                    v = signed(v, TYPE_BITS[dt])
                self.put(ops[0], v)
            elif mn.startswith('setp.'):
                _, cc, ty = mn.split('.')
                a, b = self.ival(ops[1]), self.ival(ops[2])
                if ty == 's32':
                    a, b = s32(a), s32(b)
                elif ty in ('u32', 'b32'):
                    a, b = a & M32, b & M32
                elif ty == 'b16':
                    a, b = a & M16, b & M16
                elif ty == 'b64':
                    a, b = a & M64, b & M64
                else:
                    raise Unsupported(mn)
                if ty.startswith('b') and cc not in ('eq', 'ne'):
                    raise Unsupported(mn)
                self.R[ops[0][1]] = {'eq': a == b, 'ne': a != b, 'ge': a >= b, 'gt': a > b, 'lt': a < b, 'le': a <= b}[cc]
            elif mn in ('selp.b32', 'selp.b64', 'selp.b16'):
                self.put(ops[0], self.val(ops[1]) if self.R[ops[3][1]] else self.val(ops[2]))
            elif mn == 'ld.param.b64':
                self.put(ops[0], struct.unpack_from('<Q', self.params, ops[1][1])[0])
            elif mn == 'ld.param.v2.b32':
                for i, r in enumerate(ops[0][1]):
                    self.put(r, struct.unpack_from('<I', self.params, ops[1][1] + 4 * i)[0])
            elif mn == 'ld.weak.global.ca.v4.u32':
                reg, off = self.region(self.addr(ops[1]), 16)
                if reg != 'q':
                    raise Unsupported('weak load from %s' % reg)
                e = self.emit('qload', here, off, 16)
                for i, r in enumerate(ops[0][1]):
                    self.R[r[1]] = ('ev', e, i)
            elif mn == 'ld.shared::cta.v4.u32':
                store, off = self.shared(self.addr(ops[1]), 16)
                e = self.emit({'key': 'kread', 'val': 'vread'}[store], here, off, 16)
                for i, r in enumerate(ops[0][1]):
                    self.R[r[1]] = ('ev', e, i)
            elif mn == 'st.shared::cta.v4.u32':
                store, off = self.shared(self.addr(ops[0]), 16)
                self.emit('sstore', here, store, off, 16, fold_zero(self.val(ops[1])))
            elif mn == MMA:
                d, a, b, cc = ops
                e = self.emit('mma', here, tuple(self.val(o) for o in a[1]), tuple(self.val(o) for o in b[1]),
                              tuple(self.val(o) for o in cc[1]))
                for i, r in enumerate(d[1]):
                    self.R[r[1]] = ('ev', e, i)
            elif mn in ('mul.f16x2', 'add.f16x2', 'max.f16x2', 'min.f16x2', 'sub.f16x2'):
                self.R[ops[0][1]] = ('fop', mn.split('.')[0], (self.val(ops[1]), self.val(ops[2])))
            elif mn == 'fma.rn.f16x2':
                self.R[ops[0][1]] = ('fop', 'fma', (self.val(ops[1]), self.val(ops[2]), self.val(ops[3])))
            elif mn == 'cvt.rn.f16.f32':
                v = self.val(ops[1])
                if isinstance(v, int):
                    h, exact = f32_bits_to_f16(v)
                    self.const_f16.append((here, v, h, exact))
                    self.R[ops[0][1]] = ('c16', h)
                else:
                    self.R[ops[0][1]] = ('f16of', v)
            elif mn == 'cvt.f32.f16':
                v = self.val(ops[1])
                if isinstance(v, int):
                    raise Unsupported('integer half-to-float of a constant')
                self.R[ops[0][1]] = ('tof32', v)
            elif mn == 'rcp.approx.ftz.f32':
                self.R[ops[0][1]] = ('rcp', self.val(ops[1]))
            elif mn == 'mul.ftz.f32':
                a, b = self.val(ops[1]), self.val(ops[2])
                if isinstance(a, int) or not isinstance(b, int):
                    raise Unsupported('mul.ftz.f32 form %s' % text)
                self.R[ops[0][1]] = ('fmulc', a, b & M32)
            elif mn == 'cvt.rn.satfinite.e4m3x2.f16x2':
                self.R[ops[0][1]] = ('sat', self.val(ops[1]))
            elif mn == 'prmt.b32':
                ctl = self.ival(ops[3])
                self.R[ops[0][1]] = ('prmt', ctl, self.val(ops[1]), self.val(ops[2]))
            elif mn == 'shfl.sync.idx.b32':
                dst = ops[0]
                if dst[0] != 'pairdst' or self.ival(ops[3]) != 31 or s32(self.ival(ops[4])) != -1:
                    raise Unsupported('shfl form %r' % (text,))
                idx = self.ival(ops[2]) & 31
                self.shfl_src[here] = self.val(ops[1])
                self.R[dst[1]] = ('gather', here, idx)
                self.R[dst[2]] = True
            elif mn == 'st.global.L1::no_allocate.b128':
                reg, off = self.region(self.addr(ops[0]), 16)
                if reg != 'output':
                    raise Unsupported('b128 store into %s' % reg)
                self.emit('ostore', here, off, 16, self.val(ops[1]))
            elif mn == 'st.global.b32':
                reg, off = self.region(self.addr(ops[0]), 4)
                self.emit('zstore', here, reg, off, self.ival(ops[1]))
            elif mn == 'cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes':
                store, dst = self.shared(self.ival(ops[0][1]), self.ival(ops[2]))
                reg, off = self.region(self.ival(ops[1][1]), self.ival(ops[2]))
                if (store, reg) not in (('key', 'k'), ('val', 'v')):
                    raise Unsupported('copy %s <- %s' % (store, reg))
                self.emit('copy', here, store, dst, off, self.ival(ops[2]), self.ival(ops[3][1]) - BARRIER_BASE)
            elif mn == 'elect.sync':
                self.R[ops[0][1]] = True
            elif mn == 'mbarrier.try_wait.shared::cta.b64':
                self.emit('wait', here, self.addr(ops[1]) - BARRIER_BASE)
                self.R[ops[0][1]] = True
            elif mn == 'mbarrier.arrive.shared::cta.b64':
                self.put(ops[0], 0)
            elif mn in ('mbarrier.init.shared.b64', 'mbarrier.expect_tx.relaxed.cta.shared::cta.b64', 'bar.sync'):
                pass
            elif mn == 'ld.relaxed.gpu.global.L1::no_allocate.s32':
                reg, off = self.region(self.addr(ops[1]), 4)
                self.emit('relread', here, reg, off)
                self.put(ops[0], RELEASE_DONE)
            elif mn == 'nanosleep.u32':
                raise Unsupported('nanosleep executed: a release wait was not satisfied under the completed-release policy')
            elif mn == 'st.release.gpu.global.L1::no_allocate.s32':
                reg, off = self.region(self.addr(ops[0]), 4)
                self.emit('release', here, reg, off, s32(self.ival(ops[1])))
            elif mn == 'ret':
                return
            else:
                raise Unsupported('mnemonic %s' % mn)
        raise Unsupported('fell off the end')


def fold_zero(sym):
    if sym[0] == 'cat':
        return b''.join(fold_zero(x) for x in sym[1])
    if sym[0] == 'sat' and sym[1][0] == 'cat' and all(h == ('c16', 0.0) and str(h[1]) == '0.0' for h in sym[1][1]):
        return b'\x00' * len(sym[1][1])
    raise Unsupported('non-zero shared store %r' % (sym,))


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Interner:
    def __init__(self):
        self.ids, self.nodes = {}, []

    def get(self, node):
        i = self.ids.get(node)
        if i is None:
            i = len(self.nodes)
            self.ids[node] = i
            self.nodes.append(node)
        return i


def main():
    t0 = time.time()
    s0 = json.loads(S0.read_text(encoding='utf-8'))
    if fsha(PTX) != s0['static']['ptx_sha256'] or s0['status'] != 'PASS_IDENTITY':
        raise SystemExit('PTX or S0 identity mismatch')
    params = bytes.fromhex(s0['params']['c1-C-img-full']['hex'])
    grid, block = s0['launch_rows']['c1-C-img-full']['grid'], s0['launch_rows']['c1-C-img-full']['block_dim']
    if grid != [32, 1, 1] or block != [32, 4, 1]:
        raise SystemExit('unexpected geometry')
    w = struct.unpack('<8Q', params)
    regions = dict(q=(w[0], 65536), k=(w[1], 65536), v=(w[2], 65536), output=(w[3], 65536), release62=(w[5], 64), release63=(w[6], 128))
    prog, labels = parse(PTX.read_text(encoding='utf-8'))
    head = labels[LOOP_HEAD]
    backs = [i for i, p in enumerate(prog) if p[0] == 'bra' and p[2] == LOOP_HEAD and labels[LOOP_HEAD] <= i]
    if len(backs) != 1:
        raise SystemExit('loop back-edge not unique')
    marks = {head, backs[0] + 1}
    NC, NY, NL = grid[0], block[1], block[0]
    OFF = {}
    fails = collections.Counter()
    tmpl = {}
    gsrc = {}
    copies = collections.defaultdict(collections.Counter)
    sstores = collections.defaultdict(collections.Counter)
    waits_ref = {}
    releases, relreads, zstores = collections.Counter(), collections.Counter(), collections.Counter()
    back_taken, ev_counts, consts = collections.Counter(), collections.Counter(), collections.Counter()
    NI = Interner()
    ibits_uses = [0]

    def canon_thread(th):
        slot, per = {}, collections.defaultdict(int)
        for ev in th.events:
            kind, ep, pc, eid = ev[:4]
            if kind in LEAF_KINDS + ('mma',) + STORE_KINDS:
                slot[eid] = (kind, ep, per[(kind, ep)])
                per[(kind, ep)] += 1
        memo = {}

        def canon(sym):
            key = id(sym)
            hit = memo.get(key)
            if hit is not None and hit[0] is sym:
                return hit[1]
            if isinstance(sym, int):
                # an integer register used as data (e.g. mov.b32 %r1441, 0 seeding an f16 accumulator): its 32 bits ARE the data
                ibits_uses[0] += 1
                return NI.get(('ibits', sym & M32))
            k = sym[0]
            if k == 'ev':
                node = ('leaf',) + slot[sym[1]] + (sym[2],)
            elif k == 'c16':
                node = ('c16', repr(float(sym[1])))
            elif k in ('sat', 'tof32', 'rcp', 'f16of', 'ilow16'):
                node = (k, canon(sym[1]))
            elif k in ('ishl', 'iadd'):
                node = (k, sym[1], canon(sym[2]), sym[3])
            elif k == 'fmulc':
                node = ('fmulc', canon(sym[1]), sym[2])
            elif k == 'prmt':
                node = ('prmt', sym[1], canon(sym[2]), canon(sym[3]))
            elif k == 'gather':
                node = ('gather', sym[1], sym[2])
            elif k == 'part':
                node = ('part', canon(sym[1]), sym[2])
            elif k == 'cat':
                node = ('cat',) + tuple(canon(x) for x in sym[1])
            elif k == 'fop':
                node = ('fop', sym[1]) + tuple(canon(x) for x in sym[2])
            else:
                raise Unsupported('symbol %r' % (k,))
            i = NI.get(node)
            memo[key] = (sym, i)
            return i
        wiring, stores = [], []
        for ev in th.events:
            kind, ep, pc, eid = ev[:4]
            if kind == 'mma':
                wiring.append((ep, slot[eid][2], pc, tuple(canon(x) for x in ev[4]), tuple(canon(x) for x in ev[5]), tuple(canon(x) for x in ev[6])))
            elif kind in STORE_KINDS:
                stores.append((kind, ep, slot[eid][2], pc, canon(ev[6])))
        shfl = {pc: canon(v) for pc, v in th.shfl_src.items()}
        return wiring, stores, shfl

    for c in range(NC):
        for y in range(NY):
            for lane in range(NL):
                th = Thread(prog, labels, params, regions, c, y, lane, marks)
                th.run()
                back_taken.update(th.back_taken)
                for pc, v, h, exact in th.const_f16:
                    consts[(pc, str(v), h, exact)] += 1
                per = collections.defaultdict(int)
                waits = []
                for ev in th.events:
                    kind, ep, pc, eid = ev[:4]
                    ev_counts[kind] += 1
                    if kind in LEAF_KINDS + STORE_KINDS:
                        key = (kind, ep, per[(kind, ep)])
                        per[(kind, ep)] += 1
                        arr = OFF.setdefault(key, np.full((NC, NY, NL), -1, np.int64))
                        arr[c, y, lane] = ev[4]
                    elif kind == 'copy':
                        copies[c][(ep, pc) + ev[4:]] += 1
                    elif kind == 'sstore':
                        sstores[c][(ep, pc) + ev[4:]] += 1
                    elif kind == 'wait':
                        waits.append((ep, pc, ev[4]))
                    elif kind == 'release':
                        releases[(y, ev[4], ev[5] - 4 * c, ev[6])] += 1
                    elif kind == 'relread':
                        relreads[(y, ep, ev[4], ev[5] - 4 * (c >> 1))] += 1
                    elif kind == 'zstore':
                        zstores[(y, ev[4])] += 1
                wr = waits_ref.setdefault(y, waits)
                if wr != waits:
                    fails['wait_sequence_differs_within_warp'] += 1
                wiring, stores, shfl = canon_thread(th)
                cls = (y, lane)
                for pc, nid in shfl.items():
                    prev = gsrc.setdefault((y, pc, lane), nid)
                    if prev != nid:
                        fails['gather_source_differs_across_ctaid_x_%d' % y] += 1
                if cls not in tmpl:
                    tmpl[cls] = dict(wiring=wiring, stores=stores)
                else:
                    if tmpl[cls]['wiring'] != wiring:
                        fails['wiring_differs_across_ctaid_x_warp%d' % y] += 1
                    if tmpl[cls]['stores'] != stores:
                        fails['stores_differ_across_ctaid_x_warp%d' % y] += 1
        print('ctaid.x', c, 'nodes', len(NI.nodes), 'seconds', round(time.time() - t0, 1), flush=True)

    # the QMMA wiring of the output warp must be lane-uniform (one batched evaluation per MMA)
    w0 = tmpl[(0, 0)]['wiring']
    lane_uniform_wiring = all(tmpl[(0, l)]['wiring'] == w0 for l in range(NL))
    gather_pcs = sorted({pc for (y, pc, l) in gsrc if y == 0})
    gather_complete = all((0, pc, l) in gsrc for pc in gather_pcs for l in range(NL))

    def data(th):
        return [ev[:3] + tuple(x for x in ev[4:] if isinstance(x, (int, bytes, str))) for ev in th.events if ev[0] not in ('relread', 'release')]
    pa = Thread(prog, labels, params, regions, 3, 0, 9, marks); pa.run()
    pb = Thread(prog, labels, params, regions, 3, 0, 9, marks, tid_x=5); pb.run()
    tidx_independent = data(pa) == data(pb)

    # shared replay per CTA and per storage; positions (epoch, pc); visibility H_BARRIER_VIS
    replay = collections.Counter()
    SRC = {}
    th = Thread(prog, labels, params, regions, 0, 0, 0, marks); th.run()
    per, read_pc = collections.defaultdict(int), {}
    for ev in th.events:
        if ev[0] in ('kread', 'vread'):
            read_pc[(ev[0], ev[1], per[(ev[0], ev[1])])] = ev[2]; per[(ev[0], ev[1])] += 1
    waits = waits_ref[0]
    for kind, store in (('kread', 'key'), ('vread', 'val')):
        keys = sorted(k for k in OFF if k[0] == kind)
        for key in keys:
            _, ep, j = key
            rpos = (ep, read_pc[key])
            arr = np.full((NC, NY, NL), -9, np.int64)
            for c in range(NC):
                cps = [(k, n) for k, n in sorted(copies[c].items()) if k[2] == store]
                zs = [(k, n) for k, n in sorted(sstores[c].items()) if k[2] == store]
                for y in range(NY):
                    base = OFF[key][c, y]
                    if np.any(base != base[0] + 16 * np.arange(NL)):
                        replay['read_not_lane_contiguous'] += 1
                        continue
                    idx = base[0] + np.arange(16 * NL)
                    src = np.full(idx.shape, -2, np.int64)
                    writes = [((we, wp), dst, ln, None) for (we, wp, st_, dst, ln, data_), n in zs if (we, wp) < rpos]
                    for (we, wp, st_, dst, s_, ln, bar), n in cps:
                        if (we, wp) >= rpos:
                            continue
                        if any((we, wp) < (a, b) < rpos and br == bar for a, b, br in waits):
                            writes.append(((we, wp), dst, ln, s_))
                        elif np.any((idx >= dst) & (idx < dst + ln)):
                            replay['race_pending_copy_overlaps_read'] += 1
                    writes.sort(key=lambda q: q[0])
                    pos = np.full(idx.shape, -10 ** 9, np.int64)
                    for p, dst, ln, s_ in writes:
                        m = (idx >= dst) & (idx < dst + ln)
                        if not m.any():
                            continue
                        val = np.full(idx.shape, -1, np.int64) if s_ is None else s_ + (idx - dst)
                        key_p = p[0] * 100000 + p[1]
                        if np.any(m & (pos == key_p) & (src != val)):
                            replay['conflicting_writes'] += 1
                        src = np.where(m, val, src)
                        pos = np.where(m, key_p, pos)
                    if np.any(src == -2):
                        replay['uninitialised_read_bytes'] += int((src == -2).sum())
                    lanes = src.reshape(NL, 16)
                    zero = (lanes == -1).all(1)
                    datal = (lanes >= 0).all(1) & np.all(np.diff(lanes, axis=1) == 1, axis=1)
                    if np.any(~zero & ~datal):
                        replay['lane_mixes_zero_and_data'] += 1
                    arr[c, y] = np.where(zero, -1, lanes[:, 0])
            SRC[(kind, ep, j)] = arr

    # closed forms written from the PTX reading before this interpreter was first run (see REPORT section 3)
    cc, yy, ll = np.meshgrid(np.arange(NC), np.arange(NY), np.arange(NL), indexing='ij')
    closed = collections.OrderedDict()

    def cmp(name, keys, arrays, formula):
        good = total = 0
        for key in keys:
            got, exp = arrays[key], formula(key)
            good += int((got == exp).sum()); total += int(got.size)
        closed[name] = dict(equal=good, elements=total, keys=len(keys))
    qk = sorted(k for k in OFF if k[0] == 'qload')
    cmp('qload', qk, OFF, lambda k: np.where(yy == 0, 16384 * k[2] + 512 * cc + 16 * ll, -1))
    kk = sorted(k for k in SRC if k[0] == 'kread')
    cmp('kread_src', kk, SRC, lambda k: np.where(np.full(yy.shape, k[1] == 0), 16384 * k[2] + 512 * cc + 16 * ll, -1))
    vk = sorted(k for k in SRC if k[0] == 'vread')
    cmp('vread_src', vk, SRC, lambda k: np.where(np.full(yy.shape, k[1] == 0), 32768 * (k[2] >> 1) + 1024 * cc + 512 * (k[2] & 1) + 16 * ll, -1))
    ok_ = sorted(k for k in OFF if k[0] == 'ostore')
    cmp('ostore', ok_, OFF, lambda k: np.where(yy == 0, 16384 * k[2] + 512 * cc + 16 * ll, -1))

    def cover(keys, arrays, size):
        cnt = np.zeros(size, np.int32)
        for k in keys:
            a = arrays[k][:, 0, :]
            a = a[a >= 0]
            for b in range(16):
                np.add.at(cnt, a + b, 1)
        return cnt
    cov = dict(output_exactly_once=[int((cover(ok_, OFF, 65536) == 1).sum()), 65536],
               q_read_exactly_once_by_warp0=[int((cover(qk, OFF, 65536) == 1).sum()), 65536],
               k_read_exactly_once_by_warp0=[int((cover(kk, SRC, 65536) == 1).sum()), 65536],
               v_read_exactly_once_by_warp0=[int((cover(vk, SRC, 65536) == 1).sum()), 65536])

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    npz = {'%s|%d|%d' % k: v.astype(np.int32) for k, v in OFF.items()}
    npz.update({'%s_src|%d|%d' % k: v.astype(np.int32) for k, v in SRC.items()})
    np.savez_compressed(OUT_NPZ, **npz)

    sass = SASS.read_text(encoding='utf-8')
    sass_lits = sorted({float(x) for x in re.findall(r'(?:HFMA2|HMNMX2|HMUL2|HADD2)[^;]*?,\s*(-?\d+\.\d+(?:e[-+]\d+)?)', sass)})
    const_rows = [dict(pc=k[0], operand=k[1], f16=k[2], exact=k[3], count=n) for k, n in sorted(consts.items(), key=lambda q: (q[0][0], q[0][1]))]
    ptx_nonzero = sorted({r['f16'] for r in const_rows if r['f16'] != 0.0})
    # SASS prints HFMA2 as "HFMA2 Rd, Ra, Rb.H0_H0, lit, lit": the multiplier comes from a constant register (not listed as a
    # literal) and the addend is a literal.  So a PTX constant may be absent from the literal set only if it is exclusively an fma
    # multiplier AND every SASS HFMA2 has a register multiplier AND its addend literal is the PTX fma addend.
    fma_nodes = [n for n in NI.nodes if n[0] == 'fop' and n[1] == 'fma']
    def c16_of(i):
        n = NI.nodes[i]
        if n[0] == 'c16':
            return float(n[1])
        if n[0] == 'cat' and len({NI.nodes[x] for x in n[1:]}) == 1 and NI.nodes[n[1]][0] == 'c16':   # {low, low} broadcast
            return float(NI.nodes[n[1]][1])
        return None
    fma_mult = sorted({c16_of(n[3]) for n in fma_nodes if c16_of(n[3]) is not None})
    fma_add = sorted({c16_of(n[4]) for n in fma_nodes if c16_of(n[4]) is not None})
    # first operand may be RZ.H0_H0 (the tail fma(0, mult, add) of the padding correction)
    hfma = re.findall(r'HFMA2 R\d+, (?:R\d+|RZ)(?:\.reuse)?(?:\.H0_H0)?, (\S+?), (-?\d+\.\d+(?:e[-+]\d+)?), (-?\d+\.\d+(?:e[-+]\d+)?)\s', sass)
    hfma_register_mult = bool(hfma) and all(re.fullmatch(r'R\d+(?:\.reuse)?\.H0_H0', m) for m, _, _ in hfma)
    hfma_add_lits = sorted({float(x) for _, x, y2 in hfma if x == y2})
    missing = sorted(set(ptx_nonzero) - set(sass_lits))
    other_uses = {r['f16'] for r in const_rows} - set(fma_mult)
    const_ok = (all(v in fma_mult and v not in other_uses for v in missing) and hfma_register_mult and hfma_add_lits == fma_add
                and len(hfma) == sass.count('HFMA2 '))
    const_detail = dict(missing_from_literals=missing, fma_multipliers=fma_mult, fma_addends=fma_add, sass_hfma2_lines=len(hfma),
                        sass_hfma2_register_multiplier=hfma_register_mult, sass_hfma2_addend_literals=hfma_add_lits)

    def jl(x):
        return [jl(i) for i in x] if isinstance(x, (list, tuple)) else x
    mma_per_epoch = dict(collections.Counter(wr[0] for wr in w0))
    closed_ok = all(v['equal'] == v['elements'] and v['keys'] > 0 for v in closed.values())
    cov_ok = all(v[0] == v[1] for v in cov.values())
    ok = (sum(fails.values()) == 0 and sum(replay.values()) == 0 and tidx_independent and closed_ok and cov_ok and const_ok
          # epochs = r1388 = ((r3-1)>>6)+1 with r3 = 8*8 = 64 -> 1 (attempt 2 wrongly expected 2; the interpreter was right)
          and lane_uniform_wiring and gather_complete and sum(zstores.values()) == 0 and mma_per_epoch == {0: 64})
    rep = dict(format='opus-b31-l62-address-map/1', valid=ok, match=None, status='PASS_EXECUTABLE_MAP' if ok else 'MAP_CHECK_FAILED',
               ptx_sha256=fsha(PTX), sass_sha256=fsha(SASS), geometry=dict(grid=grid, block_dim=block, threads=NC * NY * NL),
               regions={k: [hex(b), n] for k, (b, n) in regions.items()}, interpreter_failures=dict(fails), shared_replay=dict(replay),
               tid_x_independence_probe=tidx_independent, backward_branches_taken=dict(back_taken), event_counts=dict(ev_counts),
               epoch_marks=sorted(marks), mma_per_epoch_warp0=mma_per_epoch, lane_uniform_qmma_wiring=lane_uniform_wiring,
               gather_pcs=gather_pcs, gather_sources_complete=gather_complete, zero_stores_executed=sum(zstores.values()),
               integer_bit_constants_used_as_data=dict(uses=ibits_uses[0], values=sorted({n[1] for n in NI.nodes if n[0] == 'ibits'})),
               closed_form_cross_check=closed, coverage=cov,
               constants=dict(rows=const_rows, ptx_nonzero_f16=ptx_nonzero, sass_float_literals=sass_lits, consistent_with_sass=const_ok, detail=const_detail),
               releases=[dict(y=k[0], region=k[1], offset_minus_4c=k[2], value=k[3], count=n) for k, n in sorted(releases.items())],
               release_reads=[dict(y=k[0], epoch=k[1], region=k[2], offset_minus_4half_c=k[3], count=n) for k, n in sorted(relreads.items())],
               nodes=[jl(n) for n in NI.nodes],
               templates={'0|%d' % l: dict(wiring=jl(tmpl[(0, l)]['wiring']), stores=jl(tmpl[(0, l)]['stores'])) for l in range(NL)},
               gather_sources={'%d|%d' % (pc, l): gsrc[(0, pc, l)] for pc in gather_pcs for l in range(NL)},
               map_npz=str(OUT_NPZ), map_npz_sha256=fsha(OUT_NPZ),
               hypotheses=['H_TID_X_ZERO', 'H_ELECT_ANY', 'H_BARRIER_VIS', 'H_BRACE_LE', 'H_SHFL_LANE'],
               what_a_false_pass_would_look_like={
                   'integer_semantics_wrong_but_consistent': 'caught where addresses change: independent closed forms over every event',
                   'shfl_index_semantics_wrong': 'NOT caught here: gather indices are exact integers, but d(lane) = s(idx) is a PTX reading',
                   'brace_order_reversed': 'NOT caught (consistent relabelling); the paired-LEA integer trick would change value though',
                   'ptx_arithmetic_differs_from_executed_sass': 'NOT caught here; see sass_family_check_l62.py'},
               seconds=round(time.time() - t0, 1))
    OUT_JSON.write_text(json.dumps(rep) + '\n', encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ('status', 'interpreter_failures', 'shared_replay', 'tid_x_independence_probe', 'backward_branches_taken',
                                          'mma_per_epoch_warp0', 'lane_uniform_qmma_wiring', 'gather_sources_complete', 'zero_stores_executed',
                                          'closed_form_cross_check', 'coverage', 'releases', 'seconds')}, indent=1))
    print(json.dumps(dict(ptx_nonzero_f16=ptx_nonzero, sass_literals=sass_lits, const_ok=const_ok, const_detail=const_detail, nodes=len(NI.nodes),
                          gather_pcs=len(gather_pcs), ibits=rep['integer_bit_constants_used_as_data']), indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
