"""D19 predictions.  gates | B6 --sample S | B7 --sample S | chain --sample S | diag --sample S --which DIAG1|DIAG2
NO ORACLE IS OPENED.  2 CPU threads.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, hashlib, json, platform, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
D19 = HERE.parent
ROOT = D19.parents[1]
E = ROOT / 'experiments'
D13 = E / 'parity-astra-d13-b234-sm120-cpu-chain-20260912'
D17 = E / 'parity-astra-d17-b5-sm120-cpu-chain-20260912'
SAMPLES = ('c1-C-img-full',)
OPENED = {}
_hashing = [False]


def read_capable(mode):
    """True if an open with this mode/flags can READ.  Text/binary modes: 'r' or '+'.
    Integer os.open flags: access mode O_RDONLY or O_RDWR."""
    if isinstance(mode, int):
        acc = mode & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR)
        return acc in (os.O_RDONLY, os.O_RDWR)
    m = str(mode)
    return ('r' in m) or ('+' in m)


def _audit(ev, args):
    if ev != 'open' or _hashing[0] or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        p = str(Path(os.fsdecode(args[0])).resolve())
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else None
        mode = args[1] if args[1] is not None else flags
        OPENED.setdefault(p, []).append(mode if isinstance(mode, int) else str(mode))
    except Exception:
        pass


sys.addaudithook(_audit)
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _no_gpu(*a, **k):
    raise RuntimeError('GPU forbidden')


torch.cuda.init = _no_gpu
torch.cuda._lazy_init = _no_gpu
sys.path.insert(0, str(HERE))
import d19_backend as BW  # noqa: E402
import d17_backend as B5  # noqa: E402
import d13_backend as DB  # noqa: E402
import d13_predict as P13  # noqa: E402
torch.set_num_threads(2)    # d13_predict sets 4 at import
S = BW.S
SAFETENSORS = ROOT / 'weights/dlssnr-weights-logical.safetensors'
FREEZE = D19 / 'dependency-freeze.json'
DENY_DIR = ('\\oracle\\', '/oracle/')
DENY_SUFFIX = ('-Y.TIN', 'full.TIN', 'pooled.raw', '-Y.tin64')   # '.tin64' alone also matched this round's inputs (attempt 1)
NATIVE_X_FORBIDDEN_IN_CHAIN = ('B5-X.inpview64', 'B6-X.tin64', 'B7-X.tin64')


def sha_file(p):
    _hashing[0] = True
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    finally:
        _hashing[0] = False


def shab(b):
    return hashlib.sha256(b).hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace('\\', '/')


def dep_files():
    fs = [HERE / n for n in ('d19_backend.py', 'd19_predict.py', 'd19_sm75_on_C.py')]
    fs += [D19 / 'chain-and-hypotheses.json', D19 / 'static-chains/static-chains.json',
           D17 / 'scripts/d17_backend.py', D17 / 'dependency-freeze.json',
           ROOT / 'mlxdlss/sm120_b1.py', ROOT / 'mlxdlss/model.py', ROOT / 'mlxdlss/__init__.py',
           ROOT / 'mlxdlss/resources/rsq-domain.npz', ROOT / 'mlxdlss/resources/rcp-domain.npz', SAFETENSORS,
           E / 'parity-b5-branched-ffn/TIN64-physical-to-logical.npy', D13 / 'scripts/d13_backend.py', D13 / 'scripts/d13_predict.py']
    fs += [Path(p) for p in P13.PACKAGES.values()]
    for s in SAMPLES:
        fs += [D19 / 'interfaces' / s / 'interface.json', D19 / 'interfaces' / s / 'inputs/B6-X.tin64',
               D19 / 'interfaces' / s / 'inputs/B7-X.tin64', D17 / 'interfaces' / s / 'interface.json',
               D17 / 'interfaces' / s / 'inputs/B5-X.inpview64', D17 / 'predictions' / s / 'chain/manifest.json',
               D13 / 'interfaces' / s / 'inputs/B1-X.inpview']
    return fs


def verify_freeze():
    fz = json.loads(FREEZE.read_text())
    drift = {k: v for k, v in fz['files'].items() if sha_file(ROOT / k) != v}
    if drift:
        raise SystemExit('INVALID: freeze drift %s' % drift)
    return sha_file(FREEZE)


def denied_reads():
    out = []
    for p, modes in OPENED.items():
        if any(read_capable(m if not (isinstance(m, str) and m.lstrip('-').isdigit()) else int(m)) for m in modes):
            if any(d in p for d in DENY_DIR) or p.endswith(DENY_SUFFIX):
                out.append(p)
    return sorted(out)


def finish(outdir, manifest):
    denied = denied_reads()
    manifest.update(opened_files_count=len(OPENED), denied_opens=denied, invalid=bool(denied), finished_utc=utc())
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    (outdir / 'opened.json').write_text(json.dumps({k: [str(x) for x in v] for k, v in sorted(OPENED.items())}, indent=0))
    if denied:
        raise SystemExit('INVALID: oracle-like reads %s' % denied)


def logical(block):
    from safetensors.numpy import load_file
    return {k: v for k, v in load_file(str(SAFETENSORS)).items() if k.startswith('block%d.' % block)}


def chain_b1_b5(up, tables, raw1):
    wts = {k: P13.logical_weights(up, k)[0] for k in (1, 2, 3, 4)}
    _, y1 = S.SM120B1Reference(wts[1], tables, weights_id='normal').run_codes(S.inpview_decode_codes(raw1, 160, 160))
    files = {'B1-Y.TIN': S.tin_encode_codes(y1)}
    carried = S.tin_decode_codes(files['B1-Y.TIN'], 160, 160)
    for k in (2, 3, 4):
        r = DB.SingleWindowSm120(k, wts[k], tables).run_codes(carried)
        if k == 4:
            files['B4-full.TIN'] = S.tin_encode_codes(r['Y'])
            files['B4-pooled.raw'] = DB.pooled_encode(r['pooled'])
        else:
            files['B%d-Y.TIN' % k] = S.tin_encode_codes(r['Y'])
            carried = S.tin_decode_codes(files['B%d-Y.TIN' % k], 160, 160)
    x5 = B5.inpview64_decode(files['B4-pooled.raw'])
    r5 = B5.B5Sm120(logical(5), tables).run_codes(x5)
    files['B5-Y.tin64'] = B5.tin64_encode(r5['Y'])
    return files


def stage_gates():
    out = D19 / 'gates'
    out.mkdir(exist_ok=False)
    started = utc()
    FREEZE.write_text(json.dumps(dict(created_utc=started, note='freeze file not listed in itself', python=sys.version,
                                      platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                                      files={rel(p): sha_file(p) for p in dep_files()}), indent=1))
    gates = {}
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    # G0 adapter equivalence at the B5 configuration
    g0 = {}
    ad5 = BW.BranchedWindow2hSm120(5, logical(5), tables)
    ref5 = B5.B5Sm120(logical(5), tables)
    for s in SAMPLES:
        x = B5.inpview64_decode((D17 / 'interfaces' / s / 'inputs/B5-X.inpview64').read_bytes())
        a, b = ad5.run_codes(x), ref5.run_codes(x)
        g0[s] = dict(Y=bool(np.array_equal(a['Y'], b['Y'])), QZ=bool(np.array_equal(a['QZ'], b['QZ'])),
                     AV=bool(np.array_equal(a['AV'], b['AV'])), canvas=a['canvas_hw'], Y_sha256=shab(B5.tin64_encode(a['Y'])))
    gates['G0_adapter_equals_D17_at_B5'] = dict(detail=g0, passed=all(v['Y'] and v['QZ'] and v['AV'] for v in g0.values()))
    # G1 B1->B5 reproduces D17's frozen chain prediction hashes
    g1 = {}
    for s in SAMPLES:
        want = {k: v['sha256'] for k, v in json.loads((D17 / 'predictions' / s / 'chain/manifest.json').read_text())['predictions'].items()}
        got = {k: shab(v) for k, v in chain_b1_b5(up, tables, (D13 / 'interfaces' / s / 'inputs/B1-X.inpview').read_bytes()).items()}
        g1[s] = dict(equal=got == want, n=len(got))
    gates['G1_d17_chain_reproduced'] = dict(detail=g1, passed=all(v['equal'] and v['n'] == 6 for v in g1.values()))
    # G2 weights / geometry / codecs
    g2 = {}
    for b in (6, 7):
        wl = logical(b)
        g2['block%d_shapes' % b] = all(tuple(wl['block%d.layer0.%s' % (b, n)].shape) == sh for n, sh in B5.SHAPES.items())
        g2['block%d_differs_from_block5' % b] = not np.array_equal(wl['block%d.layer0.qkv_weight' % b], logical(5)['block5.layer0.qkv_weight'])
    for s in SAMPLES:
        iface = json.loads((D19 / 'interfaces' / s / 'interface.json').read_text())
        for b in (6, 7):
            gx, gy = BW.canvas_grid(tuple(iface['blocks'][str(b)]['origin_xy']))
            g2['%s_B%d_grid' % (s, b)] = [gx, gy, 1] == iface['blocks'][str(b)]['grid_xyz'] and tuple(iface['blocks'][str(b)]['origin_xy']) == BW.BLOCKS[b]
            raw = (D19 / 'interfaces' / s / 'inputs' / ('B%d-X.tin64' % b)).read_bytes()
            g2['%s_B%d_X_tin64_roundtrip' % (s, b)] = B5.tin64_encode(B5.tin64_decode(raw)) == raw
    g2['transposed_B7_origin_would_mispredict_grid'] = list(BW.canvas_grid((0, -4))) + [1] != [11, 10, 1]
    gates['G2_weights_geometry_codecs'] = dict(detail=g2, passed=all(g2.values()))
    # G3 audit classifier self-test
    cases = {'rb': True, 'r': True, 'r+b': True, 'rb+': True, 'wb': False, 'ab': False, 'xb': False, 'w+b': True,
             'O_RDONLY': True, 'O_RDWR': True, 'O_WRONLY': False, 'O_WRONLY|O_CREAT': False, 'O_RDONLY|O_BINARY': True}
    flag = dict(O_RDONLY=os.O_RDONLY, O_RDWR=os.O_RDWR, O_WRONLY=os.O_WRONLY, O_CREAT=os.O_CREAT, O_BINARY=getattr(os, 'O_BINARY', 0))

    def ev(k):
        if k.startswith('O_'):
            v = 0
            for part in k.split('|'):
                v |= flag[part]
            return read_capable(v)
        return read_capable(k)
    g3 = {k: ev(k) == want for k, want in cases.items()}
    gates['G3_audit_classifier'] = dict(detail=g3, passed=all(g3.values()))
    # G4 D17 freeze still valid for inherited files
    fz17 = json.loads((D17 / 'dependency-freeze.json').read_text())
    inh = ['experiments/parity-astra-d17-b5-sm120-cpu-chain-20260912/scripts/d17_backend.py', 'mlxdlss/sm120_b1.py', 'mlxdlss/model.py']
    gates['G4_inherited_hashes_equal_D17_freeze'] = dict(detail={k: sha_file(ROOT / k) == fz17['files'][k] for k in inh},
                                                        passed=all(sha_file(ROOT / k) == fz17['files'][k] for k in inh))
    passed = all(g['passed'] for g in gates.values())
    finish(out, dict(stage='gates', started_utc=started, freeze_sha256=sha_file(FREEZE), gates=gates, all_gates_passed=passed))
    print(json.dumps({k: (v['passed'], v['detail']) for k, v in gates.items()}, indent=1, default=str))
    if not passed:
        raise SystemExit('GATES FAILED')


def stage_block(b, sample, pad_mode='canvas_before_ffn', layout='tin64', subdir=None, control=True):
    fz = verify_freeze()
    gm = json.loads((D19 / 'gates/manifest.json').read_text())
    if not gm['all_gates_passed'] or gm['invalid']:
        raise SystemExit('gates not passed')
    subdir = subdir or ('B%d' % b)
    out = D19 / 'predictions' / sample / subdir
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    iface = json.loads((D19 / 'interfaces' / sample / 'interface.json').read_text())
    be = BW.BranchedWindow2hSm120(b, logical(b), tables, pad_mode=pad_mode)
    bi = iface['blocks'][str(b)]
    be.validate_invocation(block_index=b, head_count=2, window_size=8, origin_xy=bi['origin_xy'],
                           grid_xy=bi['grid_xyz'][:2], shape=(bi['extent_wh'][1], bi['extent_wh'][0], 64))
    raw = (D19 / 'interfaces' / sample / 'inputs' / ('B%d-X.tin64' % b)).read_bytes()
    x = B5.tin64_decode(raw) if layout == 'tin64' else B5.inpview64_decode(raw)
    t = time.perf_counter()
    r = be.run_codes(x)
    secs = time.perf_counter() - t
    yb = B5.tin64_encode(r['Y'])
    name = 'B%d-Y.tin64' % b
    (out / name).write_bytes(yb)
    finish(out, dict(stage='B%d' % b, sample=sample, subdir=subdir, pad_mode=pad_mode, input_layout=layout,
                     teacher_forced_input='native B%d X (%s)' % (b, layout), started_utc=started, predictions_written_utc=utc(),
                     freeze_sha256=fz, input_sha256=shab(raw), backend=be.describe(), seconds=secs,
                     predictions={name: dict(sha256=shab(yb), bytes=len(yb))}, prediction_equals_input=yb == raw,
                     stats=dict(Y=P13.code_stats(r['Y']), Y_half=P13.half_stats(r['Y_half']), QZ=P13.code_stats(r['QZ'])),
                     canvas_hw=r['canvas_hw'], n_windows=r['n_windows'], native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='B%d' % b, sample=sample, subdir=subdir, sha=shab(yb), seconds=secs, canvas=r['canvas_hw']), indent=1))
    if control:
        cp = subprocess.run([sys.executable, '-B', str(HERE / 'd19_sm75_on_C.py'), sample, str(b)], capture_output=True, text=True)
        (out / 'sm75-control.log').write_text(cp.stdout + cp.stderr)
        print('sm75 control exit', cp.returncode)


def stage_chain(sample):
    fz = verify_freeze()
    for b in (6, 7):
        c = D19 / 'comparisons' / sample / ('B%d.json' % b)
        if not c.exists() or json.loads(c.read_text())['verdict'] != 'MATCH':
            raise SystemExit('chain refused: %s not MATCH' % c)
    out = D19 / 'predictions' / sample / 'chain'
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    raw1 = (D13 / 'interfaces' / sample / 'inputs/B1-X.inpview').read_bytes()
    t = time.perf_counter()
    files = chain_b1_b5(up, tables, raw1)
    carried = files['B5-Y.tin64']
    log = [dict(step='B1..B5', from_='native B1 X only', B5_Y_sha256=shab(carried))]
    for b in (6, 7):
        x = B5.tin64_decode(carried)                     # CPU output bytes of the previous block
        r = BW.BranchedWindow2hSm120(b, logical(b), tables).run_codes(x)
        carried = B5.tin64_encode(r['Y'])
        files['B%d-Y.tin64' % b] = carried
        log.append(dict(step='B%d' % b, input_sha256=shab(B5.tin64_encode(x)), output_sha256=shab(carried)))
    secs = time.perf_counter() - t
    for n, b_ in files.items():
        (out / n).write_bytes(b_)
    leaked = [p for p in OPENED if p.endswith(NATIVE_X_FORBIDDEN_IN_CHAIN)]
    if leaked:
        raise SystemExit('INVALID: chain opened native intermediate X %s' % leaked)
    (out / 'chain-log.json').write_text(json.dumps(log, indent=1))
    finish(out, dict(stage='chain', sample=sample, only_native_input='D13 interfaces/%s/inputs/B1-X.inpview' % sample,
                     started_utc=started, predictions_written_utc=utc(), freeze_sha256=fz, input_sha256=shab(raw1), seconds=secs,
                     predictions={n: dict(sha256=shab(v), bytes=len(v)) for n, v in files.items()},
                     native_intermediate_X_opened=False, native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='chain', sample=sample, seconds=secs, predictions={n: shab(v) for n, v in files.items()}), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gates', 'B6', 'B7', 'chain', 'diag'])
    ap.add_argument('--sample', choices=SAMPLES)
    ap.add_argument('--which', choices=['DIAG1', 'DIAG2'])
    ap.add_argument('--block', type=int, default=6)
    a = ap.parse_args()
    if a.stage == 'gates':
        stage_gates()
    elif a.stage in ('B6', 'B7'):
        stage_block(int(a.stage[1]), a.sample)
    elif a.stage == 'diag':
        if a.which == 'DIAG1':
            stage_block(a.block, a.sample, layout='inpview64', subdir='B%d-DIAG1-inpview64' % a.block, control=False)
        else:
            stage_block(a.block, a.sample, pad_mode='pad_after_ffn', subdir='B%d-DIAG2-pad-after-ffn' % a.block, control=False)
    else:
        stage_chain(a.sample)


if __name__ == '__main__':
    main()
