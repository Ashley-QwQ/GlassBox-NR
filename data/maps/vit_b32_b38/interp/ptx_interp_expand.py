# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

"""S1/S2 launch59 PTX address interpreter and executable map builder.

Bounded concrete interpreter for the ONE function cc_vit_1d_ffn_expand_publish_fp8 (static/launch59.ptx.txt, copied
by s0_identity.py with its sha256). Every thread of the measured geometry is executed: gridDim 32x1x1 (tsv `launch`
row) x blockDim (32,4,1) (tsv `geometry` row) = 32 CTAs x 4 warps (tid.y) x 32 lanes. Integer/address instructions are
executed exactly; data instructions (loads, MMA, half ops, F2FP) are executed SYMBOLICALLY, so the output of this stage
is provenance (which byte feeds which operand slot), never a numeric value. No target, oracle, checkpoint or old
prediction is opened; the only inputs are the PTX text, the launch59 params, and the tsv geometry recorded by S0.

Declared joint hypotheses (evidence tiers in RESULT_ZH.md):
  H_TID_X_ZERO      tid.x = tid.z = ctaid.z = 0 for every lane; blockDim.x=32 is carried by %laneid (ARCHITECTURE 5.2:
                    TID.Y = warp).  Checked here: data events do not depend on tid.x (a probe thread with tid.x=5).
  H_ELECT_ANY       elect.sync picks at least one thread; interpreter policy ALL executes every candidate copy and
                    requires all candidates to be identical, so the result does not depend on which thread wins.
  H_BARRIER_VIS     an async shared copy becomes visible to a reading thread only after that thread has completed a
                    mbarrier.try_wait on the copy's barrier word positioned after the launch; synchronous st.shared
                    is visible to every later program position.  Program positions of different warps are compared
                    as (loop epoch, statement index) (warps meet at WARPSYNC for every QMMA).  A copy launched
                    before a read but not yet awaited that overlaps the read is a RACE (hard error), not ignored.
  H_BRACE_LE        in a PTX brace group {x0,x1,..} x0 occupies the lowest bytes (v4.u32 load/store, b128 store,
                    b32 = {b16 lo, b16 hi}, f16x2 half 0 = low 16 bits).
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
S0 = Path(_arg('--s0'))
OUT_NPZ = Path(_arg('--out-npz'))
OUT_JSON = Path(_arg('--out-json'))

SHARED_BASE = 0x10000000
BARRIER_BASE = 0x20000000
SHARED_BYTES, BARRIER_BYTES = 24576, 24
LOOP_HEAD = _arg('--loop-head')
FUNC = _arg('--func', 'cc_vit_1d_ffn_expand_publish_fp8')
RELEASE_DONE = 0x7FFFFFF0
M8, M16, M32, M64 = 0xFF, 0xFFFF, 0xFFFFFFFF, (1 << 64) - 1
SPECIALS = ('%ctaid.x', '%ctaid.y', '%ctaid.z', '%tid.x', '%tid.y', '%tid.z', '%ntid.x', '%ntid.y', '%ntid.z', '%laneid')
MMA_MNEMONIC = 'mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16'


class Unsupported(Exception):
    pass


# ------------------------------------------------------------------------------------------------------------ parsing
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
        m = re.fullmatch(re.escape(FUNC) + r'_param_0(?:\+(\d+))?', inner)
        if m:
            return ('pmem', int(m.group(1) or 0))
        return ('mem', parse_operand(inner))
    if tok in SPECIALS:
        return ('spc', tok)
    if tok.startswith('_|'):
        return ('reg', tok[2:])
    if re.fullmatch(r'-?\d+', tok):
        return ('int', int(tok))
    if tok.startswith('_ZZ') and tok.endswith('E20shared_input_storage'):
        return ('int', SHARED_BASE)
    if tok.startswith('_ZZ') and tok.endswith('E27shared_copy_barrier_storage'):
        return ('int', BARRIER_BASE)
    if re.fullmatch(r'%(r|rd|rs|p)\d+', tok) or re.fullmatch(r'[A-Za-z_]\w*', tok):
        return ('reg', tok)
    raise Unsupported('operand %r' % tok)


def parse(text):
    start = text.index('.maxnreg')
    b0 = text.index('{', start) + 1
    b1 = text.rindex('}')
    src = text[b0:b1]
    stmts, labels, buf, depth = [], {}, [], 0
    for ch in src:
        if ch == '{':
            if depth == 0 and not ''.join(buf).strip():
                continue
            depth += 1; buf.append(ch); continue
        if ch == '}':
            if depth == 0:
                if ''.join(buf).strip():
                    raise Unsupported('block close inside statement: %r' % ''.join(buf))
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
    if ''.join(buf).strip():
        raise Unsupported('trailing text %r' % ''.join(buf))
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


# ----------------------------------------------------------------------------------------------------------- helpers
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


def s16(x):
    x &= M16
    return x - (1 << 16) if x >> 15 else x


def f32_bits_to_f16(bits):
    v = struct.unpack('<f', struct.pack('<I', bits & M32))[0]
    if float(np.float16(v)) != v:
        raise Unsupported('f32 constant %r not exact in binary16' % v)
    return v


# ------------------------------------------------------------------------------------------------------ interpreter
class Thread:
    def __init__(self, prog, labels, params, c, y, lane, tid_x=0):
        self.prog, self.labels, self.params = prog, labels, params
        self.spc = {'%ctaid.x': c, '%ctaid.y': 0, '%ctaid.z': 0, '%tid.x': tid_x, '%tid.y': y, '%tid.z': 0,
                    '%ntid.x': 1, '%ntid.y': 4, '%ntid.z': 1, '%laneid': lane}
        self.R = {}
        self.events = []
        self.epoch = -1
        self.back_taken = collections.Counter()
        self.evid = 0

    def val(self, op):
        k = op[0]
        if k == 'reg':
            return self.R[op[1]]
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
            raise Unsupported('integer expected, got symbol %r' % (v,))
        return v

    def put(self, op, v):
        name = op[1]
        w = width(name)
        if w is not None and isinstance(v, int):
            v &= (1 << w) - 1
        self.R[name] = v

    def addr(self, op):
        if op[0] != 'mem':
            raise Unsupported('memory operand expected: %r' % (op,))
        return self.ival(op[1])

    def emit(self, kind, pc, *payload):
        self.evid += 1
        self.events.append((kind, self.epoch, pc, self.evid) + payload)
        return self.evid

    def run(self, max_steps=200000):
        prog, labels = self.prog, self.labels
        head = labels[LOOP_HEAD]
        pc, steps = 0, 0
        n = len(prog)
        while pc < n:
            steps += 1
            if steps > max_steps:
                raise Unsupported('step limit')
            if pc == head:                      # counts fall-through entry AND back edges
                self.epoch += 1
            mn, pred, ops, text = prog[pc]
            here = pc
            pc += 1
            if pred is not None and not self.R[pred]:
                continue
            if mn == 'bra' or mn == 'bra.uni':
                tgt = labels[ops]
                if tgt <= here:
                    self.back_taken[ops] += 1
                pc = tgt
                continue
            if mn.startswith('mov.'):
                if ops[1][0] == 'spc':
                    self.put(ops[0], self.spc[ops[1][1]])
                else:
                    self.put(ops[0], self.val(ops[1]))
            elif mn == 'add.s32' or mn == 'add.s64':
                self.put(ops[0], self.ival(ops[1]) + self.ival(ops[2]))
            elif mn == 'sub.s32' or mn == 'sub.s16':
                self.put(ops[0], self.ival(ops[1]) - self.ival(ops[2]))
            elif mn == 'mul.lo.s32' or mn == 'mul.lo.s16':
                self.put(ops[0], self.ival(ops[1]) * self.ival(ops[2]))
            elif mn == 'mad.lo.s32':
                self.put(ops[0], self.ival(ops[1]) * self.ival(ops[2]) + self.ival(ops[3]))
            elif mn == 'mul.wide.s32':
                self.put(ops[0], s32(self.ival(ops[1])) * s32(self.ival(ops[2])))
            elif mn == 'mul.wide.u32':
                self.put(ops[0], (self.ival(ops[1]) & M32) * (self.ival(ops[2]) & M32))
            elif mn == 'mul.wide.u16':
                self.put(ops[0], (self.ival(ops[1]) & M16) * (self.ival(ops[2]) & M16))
            elif mn == 'shl.b32':
                self.put(ops[0], self.ival(ops[1]) << self.ival(ops[2]))
            elif mn == 'shr.s32':
                self.put(ops[0], s32(self.ival(ops[1])) >> self.ival(ops[2]))
            elif mn == 'shr.u32':
                self.put(ops[0], (self.ival(ops[1]) & M32) >> self.ival(ops[2]))
            elif mn == 'shr.u16':
                self.put(ops[0], (self.ival(ops[1]) & M16) >> self.ival(ops[2]))
            elif mn in ('and.b32', 'and.b16'):
                self.put(ops[0], self.ival(ops[1]) & self.ival(ops[2]))
            elif mn == 'or.b32':
                self.put(ops[0], self.ival(ops[1]) | self.ival(ops[2]))
            elif mn == 'div.s32':
                a, b = s32(self.ival(ops[1])), s32(self.ival(ops[2]))
                if b == 0:
                    raise Unsupported('division by zero')
                q = abs(a) // abs(b)
                self.put(ops[0], q if (a < 0) == (b < 0) else -q)
            elif mn.startswith('cvt.u') or mn.startswith('cvta.'):
                self.put(ops[0], self.ival(ops[1]))
            elif mn.startswith('setp.'):
                _, cc, ty = mn.split('.')
                a, b = self.ival(ops[1]), self.ival(ops[2])
                if ty == 's32':
                    a, b = s32(a), s32(b)
                elif ty == 's16':
                    a, b = s16(a), s16(b)
                elif ty == 'u32':
                    a, b = a & M32, b & M32
                elif ty == 'b16':
                    a, b = a & M16, b & M16
                elif ty == 'b32':
                    if cc not in ('eq', 'ne'):
                        raise Unsupported(mn)
                    a, b = a & M32, b & M32
                else:
                    raise Unsupported(mn)
                self.R[ops[0][1]] = {'eq': a == b, 'ne': a != b, 'ge': a >= b, 'gt': a > b, 'lt': a < b}[cc]
            elif mn in ('selp.b32', 'selp.b16'):
                self.put(ops[0], self.val(ops[1]) if self.R[ops[3][1]] else self.val(ops[2]))
            elif mn == 'ld.param.b64':
                off = ops[1][1]
                self.put(ops[0], struct.unpack_from('<Q', self.params, off)[0])
            elif mn == 'ld.param.v2.b32':
                off = ops[1][1]
                for i, r in enumerate(ops[0][1]):
                    self.put(r, struct.unpack_from('<I', self.params, off + 4 * i)[0])
            elif mn == 'ld.weak.global.ca.v4.u32' or mn == 'ld.shared::cta.v4.u32':
                kind = 'gload' if mn.startswith('ld.weak') else 'sread'
                e = self.emit(kind, here, self.addr(ops[1]), 16)
                for i, r in enumerate(ops[0][1]):
                    self.R[r[1]] = ('ev', e, i)          # H_BRACE_LE: brace position i = bytes 4i..4i+3
            elif mn == 'st.shared::cta.v4.u32':
                self.emit('sstore', here, self.addr(ops[0]), 16, fold_bytes(self.val(ops[1])))
            elif mn == MMA_MNEMONIC:
                d, a, b, cc = ops
                e = self.emit('mma', here, tuple(self.val(o) for o in a[1]), tuple(self.val(o) for o in b[1]),
                              tuple(self.val(o) for o in cc[1]))
                for i, r in enumerate(d[1]):
                    self.R[r[1]] = ('ev', e, i)
            elif mn in ('min.f16x2', 'max.f16x2', 'mul.f16x2', 'fma.rn.f16x2', 'abs.f16x2'):
                self.R[ops[0][1]] = ('fop', mn.split('.')[0], tuple(self.val(o) for o in ops[1:]))
            elif mn == 'cvt.rn.f16.f32':
                self.R[ops[0][1]] = ('c16', f32_bits_to_f16(self.ival(ops[1])))
            elif mn == 'cvt.rn.satfinite.e4m3x2.f16x2':
                self.R[ops[0][1]] = ('sat', self.val(ops[1]))
            elif mn == 'st.global.L1::no_allocate.b128':
                self.emit('gstore', here, self.addr(ops[0]), 16, self.val(ops[1]))
            elif mn == 'cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes':
                self.emit('copy', here, self.ival(ops[0][1]), self.ival(ops[1][1]), self.ival(ops[2]), self.ival(ops[3][1]))
            elif mn == 'elect.sync':
                self.R[ops[0][1]] = True                  # H_ELECT_ANY, policy ALL
            elif mn == 'mbarrier.try_wait.shared::cta.b64':
                self.emit('wait', here, self.addr(ops[1]))
                self.R[ops[0][1]] = True
            elif mn == 'mbarrier.arrive.shared::cta.b64':
                self.put(ops[0], 0)
                self.emit('arrive', here, self.addr(ops[1]))
            elif mn in ('mbarrier.init.shared.b64', 'mbarrier.expect_tx.relaxed.cta.shared::cta.b64'):
                self.emit(mn.split('.')[1], here, self.addr(ops[0]))
            elif mn == 'bar.sync':
                self.emit('bar_sync', here)
            elif mn == 'st.release.gpu.global.L1::no_allocate.s32':
                self.emit('release', here, self.addr(ops[0]), self.ival(ops[1]))
            elif mn == 'or.pred':
                self.R[ops[0][1]] = bool(self.R[ops[1][1]] or self.R[ops[2][1]])
            elif mn == 'min.s32':
                self.put(ops[0], min(s32(self.ival(ops[1])), s32(self.ival(ops[2]))))
            elif mn == 'ld.relaxed.gpu.global.L1::no_allocate.s32':
                self.emit('relread', here, self.addr(ops[1]))
                self.put(ops[0], RELEASE_DONE)
            elif mn == 'nanosleep.u32':
                raise Unsupported('nanosleep executed: a release poll was not satisfied under the completed-release policy')
            elif mn == 'ret':
                return
            else:
                raise Unsupported('mnemonic %s' % mn)
        raise Unsupported('fell off the end')


def fold_bytes(sym):
    """Constant-fold a zero-store value: only +0.0 through satfinite E4M3 is supported (0x00)."""
    if sym[0] == 'cat':
        return b''.join(fold_bytes(x) for x in sym[1])
    if sym[0] == 'sat':
        inner = sym[1]
        if inner[0] != 'cat' or len(inner[1]) != 2:
            raise Unsupported('sat operand %r' % (inner,))
        out = b''
        for h in inner[1]:
            if h != ('c16', 0.0) or str(h[1]) != '0.0':
                raise Unsupported('non +0 constant through satfinite: %r' % (h,))
            out += b'\x00'
        return out
    raise Unsupported('cannot fold %r' % (sym,))


# ----------------------------------------------------------------------------------------------- gate/publish decode
def const_pair(sym):
    if sym[0] == 'cat' and len(sym[1]) == 2 and sym[1][0] == sym[1][1] and sym[1][0][0] == 'c16':
        return sym[1][0][1]
    return None


def decode_gate(f):
    """f = mul(X, fma(max(min(X,4),-4), fma(-0.055908203125, abs(Y), 0.447265625), 0.89453125)) -> X (an 'ev')."""
    try:
        assert f[0] == 'fop' and f[1] == 'mul'
        x, t = f[2]
        assert x[0] == 'ev'
        assert t[0] == 'fop' and t[1] == 'fma'
        y, l1, k2 = t[2]
        assert const_pair(k2) == 0.89453125
        assert y[0] == 'fop' and y[1] == 'max' and const_pair(y[2][1]) == -4.0
        mn = y[2][0]
        assert mn[0] == 'fop' and mn[1] == 'min' and mn[2][0] == x and const_pair(mn[2][1]) == 4.0
        assert l1[0] == 'fop' and l1[1] == 'fma'
        k0, ab, k1 = l1[2]
        assert const_pair(k0) == -0.055908203125 and const_pair(k1) == 0.447265625
        assert ab[0] == 'fop' and ab[1] == 'abs' and ab[2][0] == y
    except (AssertionError, IndexError, TypeError, ValueError):
        return None
    return x


def decode_store(v):
    """b128 store value -> 16 byte provenances (mma_evid, D reg index, half) under H_BRACE_LE."""
    if v[0] != 'cat' or len(v[1]) != 4:
        raise Unsupported('store value shape')
    out = []
    for w in v[1]:
        if w[0] != 'cat' or len(w[1]) != 2:
            raise Unsupported('b32 store word shape')
        for rs in w[1]:
            if rs[0] != 'sat':
                raise Unsupported('store byte pair not from satfinite')
            x = decode_gate(rs[1])
            if x is None:
                raise Unsupported('store half not the gate template')
            out.append((x[1], x[2], 0))
            out.append((x[1], x[2], 1))
    return out


# ------------------------------------------------------------------------------------------------------- map builder
def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    t0 = time.time()
    s0 = json.loads(S0.read_text(encoding='utf-8'))
    if sha256(PTX) != s0['static']['ptx_sha256']:
        raise SystemExit('PTX hash differs from S0 record')
    params = bytes.fromhex(s0['params']['c1-C-img-full']['hex'])
    # single params payload used for published map
    grid = s0['launch_rows']['c1-C-img-full']['grid']
    block = s0['launch_rows']['c1-C-img-full']['block_dim']
    if grid != [32, 1, 1] or block != [32, 4, 1]:
        raise SystemExit('unexpected geometry %r %r' % (grid, block))
    rd1, rd2, rd3, rd4 = (struct.unpack_from('<Q', params, o)[0] for o in (0, 16, 24, 56))
    rpoll = struct.unpack_from('<Q', params, 48)[0]
    relreads = collections.Counter()
    limit = int(sys.argv[sys.argv.index('--ctas') + 1]) if '--ctas' in sys.argv else grid[0]
    prog, labels = parse(PTX.read_text(encoding='utf-8'))
    mn_census = collections.Counter(p[0] for p in prog)
    NC, NY, NL, NE = limit, block[1], block[0], 16

    sread_addr = np.full((NC, NY, NE, 8, NL), -1, np.int64)
    gload_addr = np.full((NC, NY, NE, 8, NL), -1, np.int64)      # event epoch -1..14 -> index 0..15
    store_addr = np.full((NC, NY, 8, NL), -1, np.int64)
    fails = collections.Counter()
    ref_wiring, ref_store, ref_waits = None, None, None
    copies = collections.defaultdict(collections.Counter)         # c -> (epoch, pc, dst, src, len, bar) -> count
    sstores = collections.defaultdict(collections.Counter)
    back_taken, releases, ev_counts = collections.Counter(), [], collections.Counter()
    threads = 0

    def abstract(th):
        """Per-thread events -> thread-independent wiring (event ids replaced by (kind, epoch, slot, index))."""
        slot = {}
        per = collections.defaultdict(int)
        for ev in th.events:
            kind, ep, pc, eid = ev[:4]
            if kind in ('sread', 'gload', 'mma', 'gstore'):
                key = (kind, ep)
                slot[eid] = (kind, ep, per[key], pc)
                per[key] += 1

        def ref(sym):
            if sym[0] == 'ev':
                k, ep, sl, _ = slot[sym[1]]
                return (k, ep, sl, sym[2])
            if sym[0] == 'cat' and len(sym[1]) == 2 and all(x == ('c16', 0.0) for x in sym[1]):
                return ('zero_f16x2',)
            raise Unsupported('operand symbol %r' % (sym,))
        wiring, stores = [], []
        for ev in th.events:
            kind, ep, pc, eid = ev[:4]
            if kind == 'mma':
                wiring.append((ep, slot[eid][2], pc, tuple(ref(s) for s in ev[4]), tuple(ref(s) for s in ev[5]),
                               tuple(ref(s) for s in ev[6])))
            elif kind == 'gstore':
                prov = decode_store(ev[6])
                stores.append((slot[eid][2], pc, tuple((slot[m][1], slot[m][2], r, h) for m, r, h in prov)))
        return wiring, stores

    for c in range(NC):
        for y in range(NY):
            for lane in range(NL):
                th = Thread(prog, labels, params, c, y, lane)
                th.run()
                threads += 1
                back_taken.update(th.back_taken)
                per = collections.defaultdict(int)
                waits = []
                for ev in th.events:
                    kind, ep, pc, eid = ev[:4]
                    ev_counts[kind] += 1
                    if kind == 'sread':
                        j = per[('sread', ep)]; per[('sread', ep)] += 1
                        if READ_PC.setdefault(j, pc) != pc:
                            fails['read_slot_pc_not_static'] += 1
                        sread_addr[c, y, ep, j, lane] = ev[4] - SHARED_BASE
                    elif kind == 'gload':
                        q = per[('gload', ep)]; per[('gload', ep)] += 1
                        gload_addr[c, y, ep + 1, q, lane] = ev[4] - rd3
                    elif kind == 'gstore':
                        s = per[('gstore', ep)]; per[('gstore', ep)] += 1
                        if ep != 15:
                            fails['gstore_not_after_last_epoch'] += 1
                        store_addr[c, y, s, lane] = ev[4] - rd2
                    elif kind == 'copy':
                        copies[c][(ep, pc, ev[4] - SHARED_BASE, ev[5] - rd1, ev[6], ev[7] - BARRIER_BASE)] += 1
                    elif kind == 'sstore':
                        sstores[c][(ep, pc, ev[4] - SHARED_BASE, ev[5], ev[6])] += 1
                    elif kind == 'wait':
                        waits.append((ep, pc, ev[4] - BARRIER_BASE))
                    elif kind == 'release':
                        releases.append(dict(c=c, y=y, lane=lane, offset_from_rd4=ev[4] - rd4, value=ev[5]))
                    elif kind == 'relread':
                        relreads[(ep, ev[4] - rpoll)] += 1
                w, st = abstract(th)
                if ref_wiring is None:
                    ref_wiring, ref_store, ref_waits = w, st, waits
                    ref_store_y = {y: st}
                else:
                    if w != ref_wiring:
                        fails['mma_wiring_differs_from_thread0'] += 1
                    if waits != ref_waits:
                        fails['wait_sequence_differs_from_thread0'] += 1
                    if st:
                        if y not in ref_store_y:
                            ref_store_y[y] = st
                        elif [(a, b) for a, _, b in st] != [(a, b) for a, _, b in ref_store_y[y]]:
                            fails['store_provenance_differs'] += 1
        print('cta', c, 'threads', threads, 'seconds', round(time.time() - t0, 1), flush=True)

    # tid.x independence probe (H_TID_X_ZERO): data events must be identical except init/release bookkeeping
    def data_events(th):
        keep = ('sread', 'gload', 'mma', 'gstore', 'copy', 'sstore', 'wait')
        return [ev[:3] + tuple(x for x in ev[4:] if not isinstance(x, tuple)) for ev in th.events if ev[0] in keep]
    pa = Thread(prog, labels, params, 0, 1, 7, tid_x=0); pa.run()
    pb = Thread(prog, labels, params, 0, 1, 7, tid_x=5); pb.run()
    tidx_independent = data_events(pa) == data_events(pb)

    # ---------------------------------------------------------------- shared replay (H_BARRIER_VIS)
    sread_src = np.full((NC, NY, NE, 8, NL), -9, np.int64)   # input byte offset of byte 0; -1 zero store
    replay = collections.Counter()
    for c in range(NC):
        cps = sorted(copies[c].items())
        zs = sorted(sstores[c].items())
        # identical-candidate requirement: one (dst,len) at one position may not carry two different sources
        seen = {}
        for (ep, pc, dst, src, ln, bar), n in cps:
            k = (ep, pc, dst, ln)
            if k in seen and seen[k] != (src, bar):
                replay['copy_candidates_disagree'] += 1
            seen[k] = (src, bar)
        for y in range(NY):
            for ep in range(NE):
                for j in range(8):
                    base = sread_addr[c, y, ep, j]
                    if np.any(base != base[0] + 16 * np.arange(NL)):
                        replay['read_not_lane_contiguous'] += 1
                        continue
                    idx = base[0] + np.arange(16 * NL)
                    rpos = (ep, ref_pc_sread(ref_wiring, ep, j))
                    src = np.full(idx.shape, -2, np.int64)
                    pos = np.full(idx.shape, -1, np.int64)
                    writes = []
                    for (wep, wpc, dst, ln, data), n in zs:
                        if (wep, wpc) < rpos:
                            writes.append(((wep, wpc), dst, ln, None, data))
                    for (wep, wpc, dst, s_, ln, bar), n in cps:
                        if (wep, wpc) >= rpos:
                            continue
                        vis = any((wep, wpc) < (a, b) < rpos and br == bar for a, b, br in ref_waits)
                        if vis:
                            writes.append(((wep, wpc), dst, ln, s_, None))
                        else:
                            m = (idx >= dst) & (idx < dst + ln)
                            if m.any():
                                replay['race_pending_copy_overlaps_read_bytes'] += int(m.sum())
                    writes.sort(key=lambda w: w[0])
                    for order, (p, dst, ln, s_, data) in enumerate(writes):
                        m = (idx >= dst) & (idx < dst + ln)
                        if not m.any():
                            continue
                        key = p[0] * 100000 + p[1]
                        same = m & (pos == key)
                        if s_ is None:
                            if any(b != 0 for b in data):
                                replay['nonzero_constant_store'] += 1
                            val = np.full(idx.shape, -1, np.int64)
                        else:
                            val = s_ + (idx - dst)
                        if np.any(same & (src != val)):
                            replay['conflicting_writes_same_position'] += int((same & (src != val)).sum())
                        src = np.where(m, val, src)
                        pos = np.where(m, key, pos)
                    if np.any(src == -2):
                        replay['uninitialised_read_bytes'] += int((src == -2).sum())
                    lanes = src.reshape(NL, 16)
                    zero = (lanes == -1).all(1)
                    data = (lanes >= 0).all(1) & np.all(np.diff(lanes, axis=1) == 1, axis=1)
                    if np.any(~zero & ~data):
                        replay['lane_mixes_zero_and_data_or_noncontiguous'] += int((~zero & ~data).sum())
                    sread_src[c, y, ep, j] = np.where(zero, -1, lanes[:, 0])
                    if np.any((lanes[data] < 0) | (lanes[data] + 0 >= 65536)):
                        replay['source_outside_input'] += 1

    # ---------------------------------------------------------------- closed-form cross-checks (hand derivation)
    cy_, ce, cj, cl = np.meshgrid(np.arange(NY), np.arange(NE), np.arange(8), np.arange(NL), indexing='ij')
    exp_src = np.where(cy_ < 2, 16384 * (cj // 2) + 1024 * ce + 512 * (cj % 2) + 16 * cl, -1)
    cc_, cy2, cE, cq, cl2 = np.meshgrid(np.arange(NC), np.arange(NY), np.arange(NE), np.arange(8), np.arange(NL), indexing='ij')
    qoff = np.array([0, 512, 1024, 1536, 131072, 131584, 132096, 132608])
    exp_w = 262144 * cE + 2048 * (cy2 % 2) + 4096 * cc_ + qoff[cq] + 16 * cl2
    live = np.arange(NY) < 2
    so = store_addr[:, :2]
    closed = dict(
        a_source_equals_closed_form=int((sread_src == exp_src[None]).sum()), a_elements=int(sread_src.size),
        weight_offset_equals_closed_form=int((gload_addr == exp_w).sum()), weight_elements=int(gload_addr.size),
        dead_warps_store_nothing=bool((store_addr[:, 2:] == -1).all()),
    )
    # coverage
    cover = np.zeros(262144, np.int32)
    if (so >= 0).all():
        for s_ in range(8):
            for b in range(16):
                np.add.at(cover, (so[:, :, s_, :] + b).ravel(), 1)
    wcover = np.zeros(4194320, np.int32)
    ga = gload_addr[:, :2]
    if (ga >= 0).all() and (ga + 16 <= 4194320).all():
        for b in range(16):
            np.add.at(wcover, (ga + b).ravel(), 1)
    icover = np.zeros(65536, np.int32)
    sa = sread_src[:, :2]
    if (sa >= 0).all():
        for b in range(16):
            np.add.at(icover, (sa + b).ravel(), 1)
    coverage = dict(output_bytes=262144, output_written_exactly_once=int((cover == 1).sum()),
                    output_written_more_than_once=int((cover > 1).sum()), output_unwritten=int((cover == 0).sum()),
                    weight_bytes_read_by_live_warps_exactly_once=int((wcover[:4194304] == 1).sum()),
                    weight_bytes_0_4194304=4194304, weight_bytes_beyond_4194304_read=int(wcover[4194304:].sum()),
                    input_bytes_read_per_live_cta_warp_counts=sorted(set(icover.tolist())),
                    ctas=NC)

    # wiring facts
    mma_eps = collections.Counter(w[0] for w in ref_wiring)
    chains = build_chains(ref_wiring)
    store_table = [dict(slot=s_, pc=pc, bytes=[dict(mma_epoch=a, mma_slot=b, d_reg=r, half=h) for a, b, r, h in prov])
                   for s_, pc, prov in ref_store_y.get(0, [])]
    store_same_y1 = [(a, b) for a, _, b in ref_store_y.get(0, [])] == [(a, b) for a, _, b in ref_store_y.get(1, [])]

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, sread_src=sread_src.astype(np.int32), sread_shared_offset=sread_addr.astype(np.int32),
                        gload_weight_offset=gload_addr.astype(np.int64), store_output_offset=store_addr.astype(np.int64))
    errors = sum(fails.values()) + sum(v for k, v in replay.items())
    ok = (errors == 0 and tidx_independent and closed['a_source_equals_closed_form'] == closed['a_elements']
          and closed['weight_offset_equals_closed_form'] == closed['weight_elements'] and closed['dead_warps_store_nothing']
          and coverage['output_written_exactly_once'] == 262144 and coverage['weight_bytes_read_by_live_warps_exactly_once'] == 4194304
          and coverage['weight_bytes_beyond_4194304_read'] == 0 and len(chains['chains']) == 32
          and all(v == 64 for v in mma_eps.values()) and len(mma_eps) == 16
          and all(len(ch['steps']) == 32 for ch in chains['chains']) and chains['problems'] == [] and store_same_y1
          and len(store_table) == 8 and NC == 32)
    rep = dict(
        format='opus-b31-l59-address-map/1', valid=ok, match=None,
        status='PASS_EXECUTABLE_MAP' if ok else 'MAP_CHECK_FAILED',
        stage='S1/S2 PTX interpreter (symbolic data, exact addresses); no numeric value, no target read',
        ptx_sha256=sha256(PTX), params_hex=params.hex(), geometry=dict(grid=grid, block_dim=block, threads=threads),
        param_pointers=dict(rd1_input=hex(rd1), rd2_output=hex(rd2), rd3_weights=hex(rd3), rd4_release=hex(rd4)),
        mnemonic_census=dict(mn_census), event_counts=dict(ev_counts), backward_branches_taken=dict(back_taken),
        loop_head=LOOP_HEAD, mma_per_epoch=dict(mma_eps), releases=releases[:4], release_count=len(releases),
        release_reads=[dict(epoch=k[0], offset_from_word6=k[1], count=n) for k, n in sorted(relreads.items())], word6=hex(rpoll),
        tid_x_independence_probe=tidx_independent,
        interpreter_failures=dict(fails), shared_replay=dict(replay),
        closed_form_cross_check=closed, coverage=coverage,
        copies_per_cta0=[dict(epoch=k[0], pc=k[1], shared_dst=k[2], input_src=k[3], bytes=k[4], barrier=k[5], candidates=n)
                         for k, n in sorted(copies[0].items())],
        waits_thread0=[dict(epoch=a, pc=b, barrier=c) for a, b, c in ref_waits],
        chains=chains, store_table=store_table, store_table_same_for_warp1=store_same_y1,
        map_npz=str(OUT_NPZ), map_npz_sha256=sha256(OUT_NPZ),
        hypotheses=['H_TID_X_ZERO', 'H_ELECT_ANY', 'H_BARRIER_VIS', 'H_BRACE_LE'],
        what_a_false_pass_would_look_like={
            'interpreter_mis-implements_an_integer_op': 'caught only where it changes addresses: closed-form equality over all '
                                                        '491,520 A and weight elements, exactly-once output/weight coverage',
            'hand_formula_copied_from_interpreter': 'no: closed forms are written independently in this file and compared',
            'barrier_visibility_wrong': 'NOT caught if hardware exposes a half-finished copy; the PTX order makes every read '
                                        'follow a wait on its slot barrier, which is what H_BARRIER_VIS encodes',
            'brace_order_reversed': 'NOT caught here (a consistent relabelling keeps coverage); only a numeric DIFF would show it',
            'empty_set_pass': 'element counts are reported next to every equality',
        },
        seconds=round(time.time() - t0, 1))
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=str) + '\n', encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ('status', 'geometry', 'interpreter_failures', 'shared_replay', 'closed_form_cross_check',
                                          'coverage', 'tid_x_independence_probe', 'backward_branches_taken', 'seconds')}, indent=1))
    return 0 if ok else 1


def ref_pc_sread(wiring, ep, j):
    """program position of the j-th shared read in epoch ep, taken from the A operands of the reference wiring."""
    return READ_PC[j]


READ_PC = {}


def build_chains(wiring):
    by_ref = {(ep, sl): (pc, a, b, cc) for ep, sl, pc, a, b, cc in wiring}
    problems = []
    final = [(ep, sl) for ep, sl, *_ in wiring if ep == 15]
    consumed = set()
    for ep, sl, pc, a, b, cc in wiring:
        for x in cc:
            if x[0] == 'mma':
                consumed.add((x[1], x[2]))
    heads = [k for k in final if k not in consumed]
    chains = []
    for head in heads:
        steps, cur = [], head
        while True:
            pc, a, b, cc = by_ref[cur]
            if not (len(a) == 4 and all(x[0] == 'sread' and x[1] == cur[0] and x[3] == i for i, x in enumerate(a))
                    and len({x[2] for x in a}) == 1):
                problems.append(('A_not_one_same_epoch_read_in_brace_order', cur))
            if len(b) != 2:
                problems.append(('B_width', cur))
            if not all(x[0] == 'gload' and x[1] == cur[0] - 1 for x in b) or len({x[2] for x in b}) != 1:
                problems.append(('B_not_one_previous_load', cur))
            steps.append(dict(epoch=cur[0], mma_slot=cur[1], pc=pc, a_read_slot=a[0][2], b_load_slot=b[0][2],
                              b_reg_index=[x[3] for x in b]))
            if all(x == ('zero_f16x2',) for x in cc):
                break
            if not all(x[0] == 'mma' for x in cc) or len({(x[1], x[2]) for x in cc}) != 1 or [x[3] for x in cc] != [0, 1]:
                problems.append(('C_not_one_previous_D_pair_in_order', cur)); break
            cur = (cc[0][1], cc[0][2])
        steps.reverse()
        chains.append(dict(final=head, steps=steps))
    return dict(chains=chains, problems=problems, heads=len(heads), mma_total=len(wiring))


if __name__ == '__main__':
    raise SystemExit(main())
