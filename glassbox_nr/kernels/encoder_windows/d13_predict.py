"""D13 predictions. Stages: gates | B2 | B3 | B4 | chain.  NO ORACLE IS OPENED HERE.

gates  dependency freeze (no self-hash of the freeze file) + G1-G4
B2/B3/B4  verify freeze unchanged, teacher-forced single-block prediction from
       the native X of the same capture, plus the sm_75 path on the same X as a
       discriminating control. One stage per process, output dir exclusive.
chain  only after comparisons/B2,B3,B4 are MATCH (or --diagnostic): CPU B1 from
       native B1 X, then B2,B3,B4 fed only by CPU outputs.

Every open() is recorded by an audit hook; any open under an oracle directory or
of a native-output file name makes the stage INVALID and exit non-zero.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, hashlib, importlib.util, json, platform, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
D13 = HERE.parent
ROOT = D13.parents[1]
E = ROOT / 'experiments'

OPENED = {}
DENY_PARTS = ('oracle',)
DENY_NAMES = ('-Y.TIN', 'full.TIN', 'pooled.raw')
_hashing = [False]


def _audit(ev, args):
    if ev != 'open' or _hashing[0] or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        OPENED.setdefault(str(Path(os.fsdecode(args[0])).resolve()), str(args[1]))
    except Exception:
        pass


sys.addaudithook(_audit)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _no_gpu(*a, **k):
    raise RuntimeError('GPU forbidden in D13 CPU prediction')


torch.cuda.init = _no_gpu
torch.cuda._lazy_init = _no_gpu
torch.set_num_threads(4)

sys.path.insert(0, str(HERE))
import d13_backend as DB  # noqa: E402
S = DB.S

SAMPLE = 'c1-C-img-full'
IFACE = D13 / 'interfaces' / SAMPLE
PACKAGES = {
    1: E / 'parity-astra-d03-b1-sm120-boundary-20260912/weights/normal.weights',
    2: E / 'parity-b2-texture-align/weights/B2.weights',
    3: E / 'parity-b3-close/weights/B3.weights',
    4: E / 'parity-b4-pre-color/weights/B4.weights',
}
SAFETENSORS = ROOT / 'weights/dlssnr-weights-logical.safetensors'
G1_INPUT = E / 'parity-astra-d01-b1-sm120-20260912/sm120-input/B1-X.inpview'
G1_EXPECT = E / 'parity-astra-d11-b1-sm120-core-integration-20260912/revisions/r01/predictions/D07-normal.TIN'
UNPACKER = ROOT / 'research-upstream/python/mlxdlss/tools/unpack_dlssnr_weights.py'
FREEZE = D13 / 'dependency-freeze.json'


def sha_file(p):
    _hashing[0] = True
    try:
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()
    finally:
        _hashing[0] = False


def shab(b):
    return hashlib.sha256(b).hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_unpacker():
    spec = importlib.util.spec_from_file_location('d13_unpack', UNPACKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def logical_weights(up, k):
    raw = Path(PACKAGES[k]).read_bytes()
    return up.unpack_known_tensor('block%d.layer0.layer' % k, np.frombuffer(raw, np.uint8)), shab(raw)


def dep_files():
    files = [HERE / 'd13_backend.py', HERE / 'd13_predict.py', D13 / 'chain-and-hypotheses.json',
             D13 / 'static-chains/static-chains.json', IFACE / 'interface.json',
             ROOT / 'mlxdlss/sm120_b1.py', ROOT / 'mlxdlss/model.py', ROOT / 'mlxdlss/__init__.py',
             ROOT / 'mlxdlss/resources/rsq-domain.npz', ROOT / 'mlxdlss/resources/rcp-domain.npz',
             UNPACKER, SAFETENSORS, G1_INPUT, G1_EXPECT]
    files += [Path(p) for p in PACKAGES.values()]
    files += sorted((IFACE / 'inputs').iterdir())
    return files


def sm75_modules():
    """The frozen sm_75 B2/B3/B4 closures, used ONLY as a discriminating control."""
    for name in ('parity-attention-temporal-literal', 'parity-b3-close', 'parity-b2-texture-align',
                 'parity-b4-pre-color'):
        sys.path.insert(0, str(E / name))
    from b2_candidate import b2
    from b3_candidate import b3
    from b4_candidate import b4
    return b2, b3, b4


def code_stats(codes):
    u = np.asarray(codes, np.uint8)
    return dict(size=int(u.size), nan_code=int(np.count_nonzero((u & 0x7F) == 0x7F)),
                pos_zero=int(np.count_nonzero(u == 0)), neg_zero=int(np.count_nonzero(u == 0x80)),
                sat_448=int(np.count_nonzero((u & 0x7F) == 0x7E)))


def half_stats(h):
    a = np.asarray(h, np.float16)
    f = a.astype(np.float32)
    return dict(size=int(a.size), nan=int(np.isnan(f).sum()), inf=int(np.isinf(f).sum()),
                neg_zero=int(np.count_nonzero(a.view(np.uint16) == 0x8000)),
                over_448=int(np.count_nonzero(np.abs(f) > 448)))


def verify_freeze():
    fz = json.loads(FREEZE.read_text())
    drift = {k: (v, sha_file(ROOT / k)) for k, v in fz['files'].items() if sha_file(ROOT / k) != v}
    if drift:
        raise SystemExit('INVALID: dependency drift since freeze: %s' % drift)
    return sha_file(FREEZE)


def finish(outdir, manifest):
    # only READ opens count: this process legitimately WRITES files named *-Y.TIN / full.TIN / pooled.raw
    reads = [p for p, mode in OPENED.items() if not any(c in mode for c in 'wax+')]
    denied = sorted(p for p in reads if any(('\\%s\\' % d) in p or ('/%s/' % d) in p for d in DENY_PARTS)
                    or p.endswith(DENY_NAMES))
    manifest['opened_files_count'] = len(OPENED)
    manifest['denied_opens'] = denied
    manifest['invalid'] = bool(denied)
    manifest['finished_utc'] = utc()
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    (outdir / 'opened.json').write_text(json.dumps(sorted(OPENED), indent=0))
    if denied:
        raise SystemExit('INVALID: oracle-like paths opened: %s' % denied)


def stage_gates():
    outdir = D13 / 'gates'
    outdir.mkdir(exist_ok=False)
    started = utc()
    files = dep_files()
    freeze = dict(created_utc=started, note='sha256 of every dependency; the freeze file itself is not listed',
                  python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                  files={str(p.resolve().relative_to(ROOT)).replace('\\', '/'): sha_file(p) for p in files})
    FREEZE.write_text(json.dumps(freeze, indent=1))
    up = load_unpacker()
    tables = S.Sm120Tables.load()
    from safetensors.numpy import load_file
    logical = load_file(str(SAFETENSORS))
    gates = {}
    wts = {}
    # G3 weights
    g3 = {}
    for k in (1, 2, 3, 4):
        w, pkg_sha = logical_weights(up, k)
        wts[k] = w
        pref = 'block%d.layer0.' % k
        want = {n: v for n, v in logical.items() if n.startswith(pref)}
        eq = set(want) == set(w) and all(np.array_equal(w[n], want[n]) for n in want)
        g3['B%d' % k] = dict(package_sha256=pkg_sha, tensors=sorted(want), equal_to_logical_safetensors=bool(eq))
    gates['G3_weights'] = dict(detail=g3, passed=all(v['equal_to_logical_safetensors'] for v in g3.values()))
    # G1 core regression
    core = S.SM120B1Reference(wts[1], tables, weights_id='normal')
    g1_raw = G1_INPUT.read_bytes()
    t = time.perf_counter()
    g1_out = core.run_inpview(g1_raw)
    gates['G1_core_regression'] = dict(input_sha256=shab(g1_raw), output_sha256=shab(g1_out),
                                       expected_sha256=sha_file(G1_EXPECT), elements=len(g1_out),
                                       seconds=time.perf_counter() - t,
                                       passed=g1_out == G1_EXPECT.read_bytes())
    # G2 parameterisation: adapter block 1 == core, on D01 X and on the C B1 X
    adapter1 = DB.SingleWindowSm120(1, wts[1], tables)
    g2 = {}
    for name, raw in (('D01_X', g1_raw), ('C_B1_X', (IFACE / 'inputs/B1-X.inpview').read_bytes())):
        codes = S.inpview_decode_codes(raw, 160, 160)
        av_c, y_c = core.run_codes(codes)
        t = time.perf_counter()
        r = adapter1.run_codes(codes)
        g2[name] = dict(core_Y_sha256=shab(S.tin_encode_codes(y_c)), adapter_Y_sha256=shab(S.tin_encode_codes(r['Y'])),
                        Y_equal=bool(np.array_equal(r['Y'], y_c)), AV_equal=bool(np.array_equal(r['AV'], av_c)),
                        elements=int(y_c.size), adapter_seconds=time.perf_counter() - t)
    gates['G2_parameterisation'] = dict(detail=g2, passed=all(v['Y_equal'] and v['AV_equal'] for v in g2.values()))
    # G4 codecs
    sys.path.insert(0, str(E / 'parity-b1-isolate'))
    from interface import tin_decode as old_tin_decode
    g4 = {}
    for name in ('B2-X.TIN', 'B3-X.TIN', 'B4-X.TIN'):
        raw = (IFACE / 'inputs' / name).read_bytes()
        c = S.tin_decode_codes(raw, 160, 160)
        g4[name] = dict(roundtrip=S.tin_encode_codes(c) == raw,
                        equals_sm75_interface_decode=bool(np.array_equal(np.asarray(old_tin_decode(raw, 160, 160), np.uint8), c)))
    raw1 = (IFACE / 'inputs/B1-X.inpview').read_bytes()
    g4['B1-X.inpview'] = dict(roundtrip=S.inpview_encode_codes(S.inpview_decode_codes(raw1, 160, 160)) == raw1)
    gates['G4_codecs'] = dict(detail=g4, passed=all(all(v.values()) for v in g4.values()))
    passed = all(g['passed'] for g in gates.values())
    manifest = dict(stage='gates', started_utc=started, freeze_sha256=sha_file(FREEZE), gates=gates,
                    all_gates_passed=passed)
    finish(outdir, manifest)
    print(json.dumps(manifest, indent=1, default=str))
    if not passed:
        raise SystemExit('GATES FAILED')


def stage_block(k):
    freeze_sha = verify_freeze()
    gm = json.loads((D13 / 'gates/manifest.json').read_text())
    if not gm['all_gates_passed'] or gm['invalid']:
        raise SystemExit('gates not passed')
    outdir = D13 / 'predictions' / ('single-B%d' % k)
    outdir.mkdir(parents=True, exist_ok=False)
    started = utc()
    up = load_unpacker()
    tables = S.Sm120Tables.load()
    w, pkg_sha = logical_weights(up, k)
    backend = DB.SingleWindowSm120(k, w, tables)
    backend.validate_invocation(block_index=k, head_count=1, window_size=8,
                                origin_xy=DB.BLOCKS[k]['origin_xy'], shape=(160, 160, 32))
    iface = json.loads((IFACE / 'interface.json').read_text())
    got_origin = tuple(iface['blocks'][str(k)]['origin_xy'])
    if got_origin != DB.BLOCKS[k]['origin_xy']:
        raise SystemExit('INVALID: capture origin %s != candidate %s' % (got_origin, DB.BLOCKS[k]['origin_xy']))
    xname = 'B%d-X.TIN' % k
    raw = (IFACE / 'inputs' / xname).read_bytes()
    codes = S.tin_decode_codes(raw, 160, 160)
    t = time.perf_counter()
    r = backend.run_codes(codes)
    secs = time.perf_counter() - t
    files = {}
    if k == 4:
        files['B4-full.TIN'] = S.tin_encode_codes(r['Y'])
        files['B4-pooled.raw'] = DB.pooled_encode(r['pooled'])
    else:
        files['B%d-Y.TIN' % k] = S.tin_encode_codes(r['Y'])
    for n, b in files.items():
        (outdir / n).write_bytes(b)
    written = utc()
    # sm_75 discriminating control on the same X (after the candidate is written)
    b2, b3, b4 = sm75_modules()
    t = time.perf_counter()
    if k == 2:
        y75, _ = b2(codes)
        ctrl = {'B2-Y.TIN': S.tin_encode_codes(np.asarray(y75, np.uint8))}
    elif k == 3:
        y75, _ = b3(codes)
        ctrl = {'B3-Y.TIN': S.tin_encode_codes(np.asarray(y75, np.uint8))}
    else:
        f75, p75, _ = b4(codes)
        ctrl = {'B4-full.TIN': S.tin_encode_codes(np.asarray(f75, np.uint8)),
                'B4-pooled.raw': DB.pooled_encode(np.asarray(p75, np.uint8))}
    ctrl_dir = outdir / 'sm75-control'
    ctrl_dir.mkdir()
    for n, b in ctrl.items():
        (ctrl_dir / n).write_bytes(b)
    manifest = dict(stage='B%d' % k, started_utc=started, predictions_written_utc=written,
                    freeze_sha256=freeze_sha, sample=SAMPLE, teacher_forced_input=xname,
                    input_sha256=shab(raw), weights_package_sha256=pkg_sha, backend=backend.describe(),
                    info=r['info'], seconds=secs, sm75_control_seconds=time.perf_counter() - t,
                    predictions={n: dict(sha256=shab(b), bytes=len(b)) for n, b in files.items()},
                    sm75_control={n: dict(sha256=shab(b), bytes=len(b)) for n, b in ctrl.items()},
                    prediction_equals_input={n: b == raw for n, b in files.items()},
                    stats=dict(Y_codes=code_stats(r['Y']), Y_half=half_stats(r['Y_half']), AV_codes=code_stats(r['AV']),
                               **({'pooled_codes': code_stats(r['pooled']), 'pool_half': half_stats(r['pool_half'])}
                                  if k == 4 else {})),
                    native_intermediate_used=False, gpu_used=False)
    finish(outdir, manifest)
    print(json.dumps({k2: manifest[k2] for k2 in ('stage', 'seconds', 'predictions', 'sm75_control', 'invalid')}, indent=1))


def stage_chain(diagnostic):
    freeze_sha = verify_freeze()
    for k in (2, 3, 4):
        p = D13 / 'comparisons' / ('B%d.json' % k)
        v = json.loads(p.read_text())['verdict'] if p.exists() else None
        if v != 'MATCH' and not diagnostic:
            raise SystemExit('chain refused: comparisons/B%d verdict %s' % (k, v))
    outdir = D13 / 'predictions' / ('chain' if not diagnostic else 'chain-diagnostic')
    outdir.mkdir(parents=True, exist_ok=False)
    started = utc()
    up = load_unpacker()
    tables = S.Sm120Tables.load()
    wts = {k: logical_weights(up, k) for k in (1, 2, 3, 4)}
    core = S.SM120B1Reference(wts[1][0], tables, weights_id='normal')
    raw1 = (IFACE / 'inputs/B1-X.inpview').read_bytes()
    t = time.perf_counter()
    _, y1 = core.run_codes(S.inpview_decode_codes(raw1, 160, 160))
    files = {'B1-Y.TIN': S.tin_encode_codes(y1)}
    carried = S.tin_decode_codes(files['B1-Y.TIN'], 160, 160)
    assert np.array_equal(carried, y1)
    for k in (2, 3, 4):
        r = DB.SingleWindowSm120(k, wts[k][0], tables).run_codes(carried)
        if k == 4:
            files['B4-full.TIN'] = S.tin_encode_codes(r['Y'])
            files['B4-pooled.raw'] = DB.pooled_encode(r['pooled'])
        else:
            files['B%d-Y.TIN' % k] = S.tin_encode_codes(r['Y'])
            carried = S.tin_decode_codes(files['B%d-Y.TIN' % k], 160, 160)
    secs = time.perf_counter() - t
    for n, b in files.items():
        (outdir / n).write_bytes(b)
    manifest = dict(stage='chain', diagnostic=diagnostic, started_utc=started, predictions_written_utc=utc(),
                    freeze_sha256=freeze_sha, sample=SAMPLE, only_native_input='B1-X.inpview',
                    input_sha256=shab(raw1), weights_package_sha256={k: v[1] for k, v in wts.items()},
                    seconds=secs, predictions={n: dict(sha256=shab(b), bytes=len(b)) for n, b in files.items()},
                    native_intermediate_used=False, gpu_used=False)
    finish(outdir, manifest)
    print(json.dumps({k2: manifest[k2] for k2 in ('stage', 'seconds', 'predictions', 'invalid')}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gates', 'B2', 'B3', 'B4', 'chain'])
    ap.add_argument('--diagnostic', action='store_true')
    a = ap.parse_args()
    if a.stage == 'gates':
        stage_gates()
    elif a.stage == 'chain':
        stage_chain(a.diagnostic)
    else:
        stage_block(int(a.stage[1]))


if __name__ == '__main__':
    main()
