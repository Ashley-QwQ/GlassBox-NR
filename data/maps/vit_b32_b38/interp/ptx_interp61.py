# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

"""launch61 PTX address/provenance interpreter (derived from ../r02-launch60/ptx_interp60.py, which is unchanged).

Executes every thread of gridDim 16x1x2 (ctaid.x, ctaid.z) x blockDim 32x4x1 (lane, warp) = 4,096 threads of
cc_vit_1d_qkv_chained_fp8.  Integer/address instructions exactly; data symbolically (hash-consed DAG, no numbers).
Added relative to launch60: xor.b32, setp.*.b64, setp.lt.u32, selp.b64, WARP_SZ, 0f<hex> float literals, max.f16x2,
cvt.f32.f16, rsqrt.approx.ftz.f32, sqrt.approx.ftz.f32 (constant operand only), cvt.rn.f16.f32 of a symbolic float,
shfl.sync.bfly.b32 (lane gather), movmatrix.sync.trans.aligned.m8n8.b16 (lane-matrix move), fence.release.gpu.
Joint hypotheses (as launch60): H_TID_X_ZERO, H_ELECT_ANY, H_BARRIER_VIS, H_BRACE_LE, H_STAGE_ORDER (z1 of a ctaid.x runs
after z0 wrote release61[ctaid.x]); new:
  H_WARP_SZ_32   WARP_SZ = 32 lanes (the SASS SHFL.BFLY clamp field is 0x1f).
The data DAG records WHICH values are combined; lane-gather semantics (H_SHFL_BFLY_XOR, H_MOVM_TRANSPOSE) and the SASS
contraction (H_FMA_OPERAND0) are applied by the predictor, not here.
Output: maps/l61-address-map.npz (per-event offsets), maps/l61-templates-and-checks.json.
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
FUNC = _arg('--func', 'cc_vit_1d_qkv_chained_fp8')
SHARED_BASE, BARRIER_BASE = 0x10000000, 0x20000000
LOOP_HEAD = _arg('--loop-head')
M16, M32, M64 = 0xFFFF, 0xFFFFFFFF, (1 << 64) - 1
SPECIALS = ('%ctaid.x', '%ctaid.y', '%ctaid.z', '%tid.x', '%tid.y', '%tid.z', '%ntid.x', '%ntid.y', '%ntid.z', '%laneid')
MMA = 'mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16'
RELEASE_DONE = 0x7FFFFFF0
WARP_SZ = 32
BUFFER_BYTES, WEIGHT_BYTES = 393216, 3145856
LEAF_KINDS = ('sread', 'gload', 'g32', 'bufload')
STORE_KINDS = ('bufstore', 'qstore', 'kstore', 'vstore')


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
    if tok == 'WARP_SZ':
        return ('int', WARP_SZ)
    if tok.startswith('_|'):
        return ('reg', tok[2:])
    if re.fullmatch(r'-?\d+', tok):
        return ('int', int(tok))
    m = re.fullmatch(r'0f([0-9a-fA-F]{8})', tok)
    if m:
        return ('int', int(m.group(1), 16))
    if tok.startswith('_ZZ') and tok.endswith('E20shared_input_storage'):
        return ('int', SHARED_BASE)
    if tok.startswith('_ZZ') and tok.endswith('E27shared_copy_barrier_storage'):
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


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def s64(x):
    x &= M64
    return x - (1 << 64) if x >> 63 else x


def f32_bits_to_f16(bits):
    """cvt.rn.f16.f32 of an integer f32 bit pattern: IEEE round-to-nearest-even (numpy float32 -> float16)."""
    v32 = np.frombuffer(struct.pack('<I', bits & M32), '<f4')[0]
    h = np.float16(v32)
    return float(h), bool(float(h) == float(v32))


class Thread:
    def __init__(self, prog, labels, params, regions, c, z, y, lane, marks, tid_x=0):
        self.prog, self.labels, self.params, self.regions, self.marks = prog, labels, params, regions, marks
        self.spc = {'%ctaid.x': c, '%ctaid.y': 0, '%ctaid.z': z, '%tid.x': tid_x, '%tid.y': y, '%tid.z': 0,
                    '%ntid.x': 1, '%ntid.y': 4, '%ntid.z': 1, '%laneid': lane}
        self.R, self.events, self.epoch, self.evid = {}, [], -1, 0
        self.back_taken = collections.Counter()
        self.const_f16 = []

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

    def emit(self, kind, pc, *payload):
        self.evid += 1
        self.events.append((kind, self.epoch, pc, self.evid) + payload)
        return self.evid

    def run(self, max_steps=600000):
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
            elif mn in ('add.s32', 'add.s64'):
                self.put(ops[0], self.ival(ops[1]) + self.ival(ops[2]))
            elif mn in ('sub.s32', 'sub.s16'):
                self.put(ops[0], self.ival(ops[1]) - self.ival(ops[2]))
            elif mn in ('mul.lo.s32', 'mul.lo.s16'):
                self.put(ops[0], self.ival(ops[1]) * self.ival(ops[2]))
            elif mn == 'mad.lo.s32':
                self.put(ops[0], self.ival(ops[1]) * self.ival(ops[2]) + self.ival(ops[3]))
            elif mn == 'mul.wide.s32':
                self.put(ops[0], s32(self.ival(ops[1])) * s32(self.ival(ops[2])))
            elif mn == 'mul.wide.u32':
                self.put(ops[0], (self.ival(ops[1]) & M32) * (self.ival(ops[2]) & M32))
            elif mn in ('shl.b32', 'shl.b64'):
                self.put(ops[0], self.ival(ops[1]) << self.ival(ops[2]))
            elif mn == 'shr.s32':
                self.put(ops[0], s32(self.ival(ops[1])) >> self.ival(ops[2]))
            elif mn == 'shr.u32':
                self.put(ops[0], (self.ival(ops[1]) & M32) >> self.ival(ops[2]))
            elif mn in ('and.b32', 'and.b16'):
                self.put(ops[0], self.ival(ops[1]) & self.ival(ops[2]))
            elif mn == 'or.b32':
                self.put(ops[0], self.ival(ops[1]) | self.ival(ops[2]))
            elif mn == 'xor.b32':
                self.put(ops[0], self.ival(ops[1]) ^ self.ival(ops[2]))
            elif mn == 'not.b32':
                self.put(ops[0], ~self.ival(ops[1]))
            elif mn == 'or.pred':
                self.R[ops[0][1]] = bool(self.R[ops[1][1]] or self.R[ops[2][1]])
            elif mn == 'div.s32':
                a, b = s32(self.ival(ops[1])), s32(self.ival(ops[2]))
                q = abs(a) // abs(b)
                self.put(ops[0], q if (a < 0) == (b < 0) else -q)
            elif mn.startswith('cvta.'):
                self.put(ops[0], self.ival(ops[1]))
            elif mn.startswith('setp.'):
                _, cc, ty = mn.split('.')
                a, b = self.ival(ops[1]), self.ival(ops[2])
                if ty == 's32':
                    a, b = s32(a), s32(b)
                elif ty in ('u32', 'b32'):
                    a, b = a & M32, b & M32
                elif ty == 'b64':
                    a, b = a & M64, b & M64
                else:
                    raise Unsupported(mn)
                if ty.startswith('b') and cc not in ('eq', 'ne'):
                    raise Unsupported(mn)
                self.R[ops[0][1]] = {'eq': a == b, 'ne': a != b, 'ge': a >= b, 'gt': a > b, 'lt': a < b}[cc]
            elif mn in ('selp.b32', 'selp.b64'):
                self.put(ops[0], self.val(ops[1]) if self.R[ops[3][1]] else self.val(ops[2]))
            elif mn == 'ld.param.b64':
                self.put(ops[0], struct.unpack_from('<Q', self.params, ops[1][1])[0])
            elif mn == 'ld.param.v2.b32':
                for i, r in enumerate(ops[0][1]):
                    self.put(r, struct.unpack_from('<I', self.params, ops[1][1] + 4 * i)[0])
            elif mn == 'ld.weak.global.ca.v4.u32':
                reg, off = self.region(self.addr(ops[1]), 16)
                kind = {'weights': 'gload', 'buffer': 'bufload'}.get(reg)
                if kind is None:
                    raise Unsupported('weak load from %s' % reg)
                e = self.emit(kind, here, off, 16)
                for i, r in enumerate(ops[0][1]):
                    self.R[r[1]] = ('ev', e, i)
            elif mn == 'ld.global.b32':
                reg, off = self.region(self.addr(ops[1]), 4)
                if reg != 'weights':
                    raise Unsupported('b32 load outside weights')
                e = self.emit('g32', here, off, 4)
                self.R[ops[0][1]] = ('ev', e, 0)
            elif mn == 'ld.shared::cta.v4.u32':
                e = self.emit('sread', here, self.addr(ops[1]) - SHARED_BASE, 16)
                for i, r in enumerate(ops[0][1]):
                    self.R[r[1]] = ('ev', e, i)
            elif mn == 'st.shared::cta.v4.u32':
                self.emit('sstore', here, self.addr(ops[0]) - SHARED_BASE, 16, fold_zero(self.val(ops[1])))
            elif mn == MMA:
                d, a, b, cc = ops
                e = self.emit('mma', here, tuple(self.val(o) for o in a[1]), tuple(self.val(o) for o in b[1]),
                              tuple(self.val(o) for o in cc[1]))
                for i, r in enumerate(d[1]):
                    self.R[r[1]] = ('ev', e, i)
            elif mn in ('mul.f16x2', 'add.f16x2', 'max.f16x2'):
                self.R[ops[0][1]] = ('fop', mn.split('.')[0], (self.val(ops[1]), self.val(ops[2])))
            elif mn == 'cvt.rn.f16.f32':
                v = self.val(ops[1])
                if isinstance(v, int):
                    h, exact = f32_bits_to_f16(v)
                    self.const_f16.append((here, v, h, exact))
                    self.R[ops[0][1]] = ('c16', h)
                elif isinstance(v, tuple) and v[0] == 'csqrt':
                    x32 = np.frombuffer(struct.pack('<I', v[1]), '<f4')[0]
                    s = np.sqrt(np.float32(x32))
                    h = float(np.float16(s))
                    self.const_f16.append((here, ('sqrt', v[1]), h, False))
                    self.R[ops[0][1]] = ('c16', h)
                else:
                    self.R[ops[0][1]] = ('f16of', v)
            elif mn == 'cvt.f32.f16':
                self.R[ops[0][1]] = ('tof32', self.val(ops[1]))
            elif mn == 'rsqrt.approx.ftz.f32':
                self.R[ops[0][1]] = ('rsq', self.val(ops[1]))
            elif mn == 'sqrt.approx.ftz.f32':
                v = self.val(ops[1])
                if not isinstance(v, int):
                    raise Unsupported('sqrt of a non-constant')
                self.R[ops[0][1]] = ('csqrt', v & M32)
            elif mn == 'cvt.rn.satfinite.e4m3x2.f16x2':
                self.R[ops[0][1]] = ('sat', self.val(ops[1]))
            elif mn == 'shfl.sync.bfly.b32':
                dist, ctl, mask = self.ival(ops[2]), self.ival(ops[3]), self.ival(ops[4])
                if ctl != 31 or s32(mask) != -1 or dist not in (1, 2):
                    raise Unsupported('shfl form %r' % (text,))
                self.R[ops[0][1]] = ('bfly', dist, self.val(ops[1]))
            elif mn == 'movmatrix.sync.trans.aligned.m8n8.b16':
                self.R[ops[0][1]] = ('movm', self.val(ops[1]))
            elif mn == 'st.global.L1::no_allocate.b128':
                reg, off = self.region(self.addr(ops[0]), 16)
                kind = {'buffer': 'bufstore', 'q': 'qstore', 'k': 'kstore', 'v': 'vstore'}.get(reg)
                if kind is None:
                    raise Unsupported('b128 store into %s' % reg)
                self.emit(kind, here, off, 16, self.val(ops[1]))
            elif mn == 'cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes':
                reg, off = self.region(self.ival(ops[1][1]), self.ival(ops[2]))
                if reg != 'input':
                    raise Unsupported('copy source outside launch61 input')
                self.emit('copy', here, self.ival(ops[0][1]) - SHARED_BASE, off, self.ival(ops[2]), self.ival(ops[3][1]) - BARRIER_BASE)
            elif mn == 'elect.sync':
                self.R[ops[0][1]] = True
            elif mn == 'mbarrier.try_wait.shared::cta.b64':
                self.emit('wait', here, self.addr(ops[1]) - BARRIER_BASE)
                self.R[ops[0][1]] = True
            elif mn == 'mbarrier.arrive.shared::cta.b64':
                self.put(ops[0], 0)
            elif mn in ('mbarrier.init.shared.b64', 'mbarrier.expect_tx.relaxed.cta.shared::cta.b64', 'bar.sync', 'fence.release.gpu'):
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
    """Global hash-consing of canonical nodes; identical structures across threads get identical ids."""

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
    if grid != [16, 1, 2] or block != [32, 4, 1]:
        raise SystemExit('unexpected geometry')
    w = struct.unpack('<10Q', params)
    if s0['weights']['bytes'] != WEIGHT_BYTES:
        raise SystemExit('unexpected weight tensor size')
    regions = dict(input=(w[0], 65536), q=(w[1], 65536), k=(w[2], 65536), v=(w[3], 65536), weights=(w[4], WEIGHT_BYTES),
                   release61=(w[5], 64), buffer=(w[6], BUFFER_BYTES), release60=(w[7], 32), release62=(w[8], 64))
    prog, labels = parse(PTX.read_text(encoding='utf-8'))
    head = labels[LOOP_HEAD]
    backs = [i for i, p in enumerate(prog) if p[0] == 'bra' and p[2] == LOOP_HEAD and labels[LOOP_HEAD] <= i]
    if len(backs) != 1:
        raise SystemExit('loop back-edge not unique')
    marks = {head, backs[0] + 1}
    NC, NZ, NY, NL = grid[0], grid[2], block[1], block[0]
    OFF = {}
    fails = collections.Counter()
    templates = {}
    copies = collections.defaultdict(collections.Counter)
    sstores = collections.defaultdict(collections.Counter)
    waits_by_z = {}
    releases, relreads = collections.Counter(), collections.Counter()
    back_taken, ev_counts = collections.Counter(), collections.Counter()
    consts = collections.Counter()
    NI = Interner()

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
                raise Unsupported('integer inside data expression')
            k = sym[0]
            if k == 'ev':
                node = ('leaf',) + slot[sym[1]] + (sym[2],)
            elif k == 'c16':
                node = ('c16', repr(float(sym[1])))
            elif k in ('sat', 'tof32', 'rsq', 'f16of', 'movm'):
                node = (k, canon(sym[1]))
            elif k == 'bfly':
                node = ('bfly', sym[1], canon(sym[2]))
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
        return wiring, stores

    for c in range(NC):
        for z in range(NZ):
            for y in range(NY):
                for lane in range(NL):
                    th = Thread(prog, labels, params, regions, c, z, y, lane, marks)
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
                            arr = OFF.setdefault(key, np.full((NC, NZ, NY, NL), -1, np.int64))
                            arr[c, z, y, lane] = ev[4]
                        elif kind == 'copy':
                            copies[(c, z)][(ep, pc) + ev[4:]] += 1
                        elif kind == 'sstore':
                            sstores[(c, z)][(ep, pc) + ev[4:]] += 1
                        elif kind == 'wait':
                            waits.append((ep, pc, ev[4]))
                        elif kind == 'release':
                            releases[(z, y, ev[4], ev[5] - 4 * c, ev[6])] += 1
                        elif kind == 'relread':
                            relreads[(z, ep, ev[4], ev[5] - (4 * c if ev[4] == 'release61' else 0))] += 1
                    wz = waits_by_z.setdefault(z, waits)
                    if wz != waits:
                        fails['wait_sequence_differs_within_stage'] += 1
                    wiring, stores = canon_thread(th)
                    cls = (z, str(y) if y < 2 else 'dead')
                    if cls not in templates:
                        templates[cls] = dict(wiring=wiring, stores=stores, first=(c, lane))
                    else:
                        if templates[cls]['wiring'] != wiring:
                            fails['wiring_differs_%s' % (cls,)] += 1
                        if templates[cls]['stores'] != stores:
                            fails['store_templates_differ_%s' % (cls,)] += 1
        print('ctaid.x', c, 'nodes', len(NI.nodes), 'seconds', round(time.time() - t0, 1), flush=True)

    # tid.x independence
    def data(th):
        return [ev[:3] + tuple(x for x in ev[4:] if isinstance(x, (int, bytes))) for ev in th.events if ev[0] not in ('relread', 'release')]
    pa = Thread(prog, labels, params, regions, 3, 1, 1, 9, marks); pa.run()
    pb = Thread(prog, labels, params, regions, 3, 1, 1, 9, marks, tid_x=5); pb.run()
    tidx_independent = data(pa) == data(pb)

    # shared replay per CTA (ctaid.x, ctaid.z); program positions (epoch, pc); visibility H_BARRIER_VIS
    replay = collections.Counter()
    read_keys = sorted(k for k in OFF if k[0] == 'sread')
    read_pc = {}
    th = Thread(prog, labels, params, regions, 0, 0, 0, 0, marks); th.run()
    per = collections.defaultdict(int)
    for ev in th.events:
        if ev[0] == 'sread':
            read_pc[(ev[1], per[ev[1]])] = ev[2]; per[ev[1]] += 1
    SRC = {key: np.full((NC, NZ, NY, NL), -9, np.int64) for key in read_keys}
    for c in range(NC):
        for z in range(NZ):
            cps, zs, waits = sorted(copies[(c, z)].items()), sorted(sstores[(c, z)].items()), waits_by_z[z]
            for key in read_keys:
                _, ep, j = key
                rpos = (ep, read_pc[(ep, j)])
                for y in range(NY):
                    base = OFF[key][c, z, y]
                    if np.any(base != base[0] + 16 * np.arange(NL)):
                        replay['read_not_lane_contiguous'] += 1
                        continue
                    idx = base[0] + np.arange(16 * NL)
                    src = np.full(idx.shape, -2, np.int64)
                    writes = [((we, wp), dst, ln, None) for (we, wp, dst, ln, data_), n in zs if (we, wp) < rpos]
                    for (we, wp, dst, s_, ln, bar), n in cps:
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
                    SRC[key][c, z, y] = np.where(zero, -1, lanes[:, 0])

    # closed forms written from the PTX reading before this interpreter was first run (see REPORT section 3)
    cc, zz, yy, ll = np.meshgrid(np.arange(NC), np.arange(NZ), np.arange(NY), np.arange(NL), indexing='ij')
    pp = yy % 2
    live = yy < 2
    closed = collections.OrderedDict()

    def cmp(name, kind, formula, mask=None):
        good = total = 0
        keys = [k for k in (read_keys if kind == 'sread' else OFF) if k[0] == kind]
        for key in keys:
            got = SRC[key] if kind == 'sread' else OFF[key]
            exp = formula(key)
            m = np.ones(got.shape, bool) if mask is None else mask(key)
            good += int(((got == exp) | ~m).sum()); total += int(got.size)
        closed[name] = dict(equal=good, elements=total, keys=len(keys))

    cmp('sread_src', 'sread', lambda k: np.where(live, 16384 * k[2] + 8192 * zz + 512 * k[1] + 16 * ll, -1))
    cmp('gload', 'gload', lambda k: 128 + 512 * (192 * (k[1] + 1) + 3072 * zz + 12 * cc + 6 * pp + k[2]) + 16 * ll)
    cmp('g32_scale_z1', 'g32', lambda k: np.where(zz == 1, 4 * (2 * cc + pp), -1))
    buf = lambda k: 131072 * (k[2] // 8) + 32768 * ((k[2] % 8) // 2) + 512 * ((k[2] % 8) % 2) + 1024 * pp + 2048 * cc + 16 * ll  # noqa: E731
    cmp('bufstore_z0', 'bufstore', lambda k: np.where((zz == 0) & live, buf(k), -1))
    cmp('bufload_z1', 'bufload', lambda k: np.where((zz == 1) & live, buf(k), -1))
    cmp('qstore_z1', 'qstore', lambda k: np.where((zz == 1) & live, 16384 * k[2] + 1024 * cc + 512 * pp + 16 * ll, -1))
    cmp('kstore_z1', 'kstore', lambda k: np.where((zz == 1) & live, 16384 * k[2] + 1024 * cc + 512 * pp + 16 * ll, -1))
    cmp('vstore_z1', 'vstore', lambda k: np.where((zz == 1) & live, 32768 * (k[2] // 2) + 512 * (k[2] % 2) + 1024 * pp + 2048 * cc + 16 * ll, -1))

    def cover(kind, size, zsel, width_=16):
        cnt = np.zeros(size, np.int32)
        keys = read_keys if kind == 'sread' else [k for k in OFF if k[0] == kind]
        for k in keys:
            a = (SRC[k] if kind == 'sread' else OFF[k])[:, zsel, :2]
            a = a[a >= 0]
            for b in range(width_):
                np.add.at(cnt, a + b, 1)
        return cnt
    cov = {}
    for kind in ('qstore', 'kstore', 'vstore'):
        cnt = cover(kind, 65536, [1])
        cov['%s_exactly_once' % kind] = [int((cnt == 1).sum()), 65536]
    cov['bufstore_z0_exactly_once'] = [int((cover('bufstore', BUFFER_BYTES, [0]) == 1).sum()), BUFFER_BYTES]
    cov['bufload_z1_exactly_once'] = [int((cover('bufload', BUFFER_BYTES, [1]) == 1).sum()), BUFFER_BYTES]
    wc = cover('gload', WEIGHT_BYTES, [0, 1])
    cov['weights_matrix_read_exactly_once_by_live_warps'] = [int((wc[128:] == 1).sum()), WEIGHT_BYTES - 128]
    cov['weights_header_read_by_gload'] = int((wc[:128] != 0).sum())
    sc = cover('g32', 128, [1], 4)
    cov['scale_header_bytes_read_counts_live_z1'] = sorted(set(sc.tolist()))
    ic = cover('sread', 65536, [0, 1])
    cov['input_bytes_read_counts_live'] = sorted(set(ic.tolist()))
    cov['buffer_read_after_write_same_cta'] = bool(all(
        np.array_equal(OFF[('bufstore', 15, j)][:, 0, :2], OFF[('bufload', 15, j)][:, 1, :2]) for j in range(24)))

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    npz = {'%s|%d|%d' % k: v.astype(np.int32) for k, v in OFF.items()}
    npz.update({'src|%d|%d' % k[1:]: v.astype(np.int32) for k, v in SRC.items()})
    np.savez_compressed(OUT_NPZ, **npz)

    # SASS literals that the compiler folded (cross-check of the PTX constant path)
    sass = SASS.read_text(encoding='utf-8')
    sass_floor = sorted(set(re.findall(r'HMNMX2 R\d+, R\d+, ([0-9.e+-]+), ([0-9.e+-]+), !PT', sass)))
    sass_sqrt = sorted(set(re.findall(r'HMUL2 R\d+, R\d+, ([0-9.e+-]+), ([0-9.e+-]+) ', sass)))
    const_rows = [dict(pc=k[0], operand=k[1], f16=k[2], exact=k[3], count=n) for k, n in sorted(consts.items(), key=lambda q: str(q[0]))]
    floor_vals = sorted({r['f16'] for r in const_rows if r['operand'] == str(948045311)})
    sqrt_vals = sorted({r['f16'] for r in const_rows if r['operand'].startswith('(')})
    const_ok = (floor_vals == [float(sass_floor[0][0])] and len(sass_floor) == 1 and sass_floor[0][0] == sass_floor[0][1]
                and sqrt_vals == [float(sass_sqrt[0][0])] and len(sass_sqrt) == 1 and sass_sqrt[0][0] == sass_sqrt[0][1])
    s32v = float(np.sqrt(np.float32(32.0)))
    ulp = 2.0 ** (int(np.floor(np.log2(s32v))) - 10)
    sqrt_margin = abs((s32v / ulp) - (np.floor(s32v / ulp) + 0.5))

    def jl(x):
        return [jl(i) for i in x] if isinstance(x, (list, tuple)) else x
    mma_per_epoch = {'%d|%s' % k: dict(collections.Counter(wr[0] for wr in v['wiring'])) for k, v in templates.items() if k[1] != 'dead'}
    closed_ok = all(v['equal'] == v['elements'] and v['keys'] > 0 for v in closed.values())
    cov_ok = (all(v[0] == v[1] for k, v in cov.items() if isinstance(v, list) and len(v) == 2 and k.endswith('exactly_once') or k.endswith('live_warps'))
              and cov['buffer_read_after_write_same_cta'])
    ok = (sum(fails.values()) == 0 and sum(replay.values()) == 0 and tidx_independent and closed_ok and cov_ok and const_ok
          and all(len(v) == 16 and all(n == 48 for n in v.values()) for v in mma_per_epoch.values()))
    rep = dict(format='opus-b31-l61-address-map/1', valid=ok, match=None, status='PASS_EXECUTABLE_MAP' if ok else 'MAP_CHECK_FAILED',
               ptx_sha256=fsha(PTX), sass_sha256=fsha(SASS), geometry=dict(grid=grid, block_dim=block, threads=NC * NZ * NY * NL),
               regions={k: [hex(b), n] for k, (b, n) in regions.items()}, interpreter_failures=dict(fails), shared_replay=dict(replay),
               tid_x_independence_probe=tidx_independent, backward_branches_taken=dict(back_taken), event_counts=dict(ev_counts),
               epoch_marks=sorted(marks), mma_per_epoch_live=mma_per_epoch, closed_form_cross_check=closed, coverage=cov,
               constants=dict(rows=const_rows, sass_floor_literals=sass_floor, sass_sqrt_literals=sass_sqrt, consistent_with_sass=const_ok,
                              sqrt32_f32=s32v, sqrt32_distance_to_f16_rounding_boundary_in_f16_ulps=float(sqrt_margin)),
               releases=[dict(z=k[0], y=k[1], region=k[2], offset_minus_4c=k[3], value=k[4], count=n) for k, n in sorted(releases.items())],
               release_reads=[dict(z=k[0], epoch=k[1], region=k[2], offset=k[3], count=n) for k, n in sorted(relreads.items())],
               copies_cta00=[list(k) + [n] for k, n in sorted(copies[(0, 0)].items())], waits_by_z={str(z): v for z, v in waits_by_z.items()},
               nodes=[jl(n) for n in NI.nodes],
               templates={'%d|%s' % k: dict(wiring=jl(v['wiring']), stores=jl(v['stores'])) for k, v in templates.items()},
               map_npz=str(OUT_NPZ), map_npz_sha256=fsha(OUT_NPZ),
               hypotheses=['H_TID_X_ZERO', 'H_ELECT_ANY', 'H_BARRIER_VIS', 'H_BRACE_LE', 'H_STAGE_ORDER', 'H_WARP_SZ_32'],
               what_a_false_pass_would_look_like={
                   'integer_semantics_wrong_but_consistent': 'caught where addresses change: independent closed forms over every event',
                   'stage_order_wrong': 'NOT caught here: the release protocol is read from the PTX, not measured',
                   'brace_order_reversed': 'NOT caught (consistent relabelling)',
                   'ptx_arithmetic_differs_from_executed_sass': 'NOT caught here (the DAG is PTX); see sass_numeric_check_l61.py'},
               seconds=round(time.time() - t0, 1))
    OUT_JSON.write_text(json.dumps(rep) + '\n', encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ('status', 'interpreter_failures', 'shared_replay', 'tid_x_independence_probe', 'backward_branches_taken',
                                          'mma_per_epoch_live', 'closed_form_cross_check', 'coverage', 'releases', 'seconds')}, indent=1))
    print(json.dumps(dict(constants={k: v for k, v in rep['constants'].items() if k != 'rows'}, const_rows=const_rows[:12], nodes=len(NI.nodes)), indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
