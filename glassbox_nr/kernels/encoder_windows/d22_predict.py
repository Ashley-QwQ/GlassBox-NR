"""D22 predictions.  gates | B9 --sample S | chain --sample S | diag --sample S --which DIAG1|DIAG2
NO ORACLE IS OPENED.  2 CPU threads.  Helpers reused read-only from D21 (which reuses D19/D17/D13).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, hashlib, json, platform, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
D22 = HERE.parent
ROOT = D22.parents[1]
E = ROOT / 'experiments'
D13 = E / 'parity-astra-d13-b234-sm120-cpu-chain-20260912'
D17 = E / 'parity-astra-d17-b5-sm120-cpu-chain-20260912'
D19 = E / 'parity-astra-d19-b67-sm120-cpu-chain-20260912'
D21 = E / 'parity-astra-d21-b8-downsample-cpu-20260912'
SAMPLES = ('c1-C-img-full',)
sys.path.insert(0, str(HERE))
import d21_predict as P21  # noqa: E402
import d22_backend as B9  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
torch.set_num_threads(2)
P19, B8, S, BW, B5, P13 = P21.P19, P21.B8, B9.S, P21.BW, P21.B5, P21.P13
OPENED = {}
_hashing = [False]


def _audit(ev, args):
    if ev != 'open' or _hashing[0] or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        p = str(Path(os.fsdecode(args[0])).resolve())
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else None
        mode = args[1] if args[1] is not None else flags
        OPENED.setdefault(p, []).append(mode)
    except Exception:
        pass


sys.addaudithook(_audit)
FREEZE = D22 / 'dependency-freeze.json'
DENY_DIR = ('\\oracle\\', '/oracle/')
DENY_SUFFIX = ('-Y.TIN', 'full.TIN', 'pooled.raw', '-Y.tin64', 'B8-full.bin', 'B8-pooled.bin', 'B9-Y.bin', 'Y.native-physical')
NATIVE_X_FORBIDDEN_IN_CHAIN = ('B5-X.inpview64', 'B6-X.tin64', 'B7-X.tin64', 'B8-X.tin64', 'B9-X.bin')


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
    fs = [HERE / n for n in ('d22_backend.py', 'd22_predict.py')]
    fs += [D22 / 'chain-and-hypotheses.json', D22 / 'static-chains/static-chains.json', D22 / 'controls/sm75-b9-control/result.json',
           D21 / 'scripts/d21_backend.py', D21 / 'scripts/d21_predict.py', D21 / 'dependency-freeze.json',
           D19 / 'scripts/d19_backend.py', D19 / 'scripts/d19_predict.py', D17 / 'scripts/d17_backend.py',
           D13 / 'scripts/d13_backend.py', D13 / 'scripts/d13_predict.py',
           ROOT / 'mlxdlss/sm120_b1.py', ROOT / 'mlxdlss/model.py', ROOT / 'mlxdlss/__init__.py',
           ROOT / 'mlxdlss/resources/rsq-domain.npz', ROOT / 'mlxdlss/resources/rcp-domain.npz', P19.SAFETENSORS,
           E / 'parity-b5-branched-ffn/TIN64-physical-to-logical.npy', E / 'parity-b9-h4-ffn/TIN128-physical-to-logical.npy']
    fs += [Path(p) for p in P13.PACKAGES.values()]
    for s in SAMPLES:
        fs += [D22 / 'interfaces' / s / 'interface.json', D22 / 'interfaces' / s / 'inputs/B9-X.bin',
               D21 / 'predictions' / s / 'chain/manifest.json', D13 / 'interfaces' / s / 'inputs/B1-X.inpview',
               D17 / 'interfaces' / s / 'inputs/B5-X.inpview64', D19 / 'interfaces' / s / 'inputs/B6-X.tin64']
    return fs


def verify_freeze():
    fz = json.loads(FREEZE.read_text())
    drift = {k: v for k, v in fz['files'].items() if sha_file(ROOT / k) != v}
    if drift:
        raise SystemExit('INVALID: freeze drift %s' % drift)
    return sha_file(FREEZE)


def denied_reads():
    return sorted(p for p, modes in OPENED.items()
                  if any(P19.read_capable(m) for m in modes) and (any(d in p for d in DENY_DIR) or p.endswith(DENY_SUFFIX)))


def finish(outdir, manifest):
    denied = denied_reads()
    manifest.update(opened_files_count=len(OPENED), denied_opens=denied, invalid=bool(denied), finished_utc=utc())
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    (outdir / 'opened.json').write_text(json.dumps({k: [str(x) for x in v] for k, v in sorted(OPENED.items())}, indent=0))
    if denied:
        raise SystemExit('INVALID: oracle-like reads %s' % denied)


def chain_b1_b8(up, tables, raw1):
    files = P19.chain_b1_b5(up, tables, raw1)
    carried = files['B5-Y.tin64']
    log = [dict(step='B1..B5', from_='native B1 X only', B5_Y_sha256=shab(carried))]
    for b in (6, 7):
        x = B5.tin64_decode(carried)
        r = BW.BranchedWindow2hSm120(b, P19.logical(b), tables).run_codes(x)
        carried = B5.tin64_encode(r['Y'])
        files['B%d-Y.tin64' % b] = carried
        log.append(dict(step='B%d' % b, input_sha256=shab(B5.tin64_encode(x)), output_sha256=shab(carried)))
    r8 = B8.B8Sm120(P19.logical(8), tables).run_codes(B5.tin64_decode(carried))
    files.update(P21.b8_files(r8))
    log.append(dict(step='B8', input_sha256=shab(carried), full_sha256=shab(files['B8-full.tin64']),
                    pooled_sha256=shab(files['B8-pooled.inpview128'])))
    return files, r8, log


def b9_backend(tables, tail_seed='published_qz'):
    return B9.BranchedWindowSm120(9, 4, 40, (0, 0), P19.logical(9), tables, tail_seed=tail_seed)


def stage_gates():
    out = D22 / 'gates'
    out.mkdir(exist_ok=False)
    started = utc()
    FREEZE.write_text(json.dumps(dict(created_utc=started, note='freeze file not listed in itself', python=sys.version,
                                      platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                                      files={rel(p): sha_file(p) for p in dep_files()}), indent=1))
    gates = {}
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    # G0 compatibility of the generic class at already-verified 2-head configurations (non-empty: 2 samples x 2 configs)
    g0 = {}
    for s in SAMPLES:
        x5 = B5.inpview64_decode((D17 / 'interfaces' / s / 'inputs/B5-X.inpview64').read_bytes())
        ref5 = B5.B5Sm120(P19.logical(5), tables).run_codes(x5)
        got5 = B9.BranchedWindowSm120(5, 2, 80, (0, 0), P19.logical(5), tables).run_codes(x5)
        x6 = B5.tin64_decode((D19 / 'interfaces' / s / 'inputs/B6-X.tin64').read_bytes())
        ref6 = BW.BranchedWindow2hSm120(6, P19.logical(6), tables).run_codes(x6)
        got6 = B9.BranchedWindowSm120(6, 2, 80, (-4, -4), P19.logical(6), tables).run_codes(x6)
        g0[s] = dict(B5_Y=bool(np.array_equal(ref5['Y'], got5['Y'])), B5_QZ=bool(np.array_equal(ref5['QZ'], got5['QZ'])),
                     B5_AV=bool(np.array_equal(ref5['AV'], got5['AV'])), B6_Y=bool(np.array_equal(ref6['Y'], got6['Y'])),
                     B6_Y_half_bits=bool(np.array_equal(ref6['Y_half'].view(np.uint16), got6['Y_half'].view(np.uint16))))
    gates['G0_generic_class_equals_D17_D19_at_2h'] = dict(detail=g0, n_cases=2 * len(SAMPLES), passed=all(all(v.values()) for v in g0.values()))
    # G1 D21 chain reproduced (10 exits x 2 samples)
    g1 = {}
    for s in SAMPLES:
        want = {k: v['sha256'] for k, v in json.loads((D21 / 'predictions' / s / 'chain/manifest.json').read_text())['predictions'].items()}
        files, _, _ = chain_b1_b8(up, tables, (D13 / 'interfaces' / s / 'inputs/B1-X.inpview').read_bytes())
        got = {k: shab(v) for k, v in files.items()}
        g1[s] = dict(equal=got == want, n=len(got))
    gates['G1_d21_chain_reproduced'] = dict(detail=g1, passed=all(v['equal'] and v['n'] == 10 for v in g1.values()))
    # G2 weights, layouts, geometry
    g2 = {}
    wl = P19.logical(9)
    g2['block9_shapes'] = all(tuple(wl['block9.layer0.%s' % n].shape) == sh for n, sh in B9.shapes_for(4).items())
    g2['block9_uses_logical_bias'] = not B9.M.uses_fragment_swizzle(9, 4)
    g2['tin128_map_is_permutation'] = bool(np.array_equal(np.sort(B9.TIN128_MAP), np.arange(204800)))
    g2['tin128_map_not_identity'] = bool(not np.array_equal(B9.TIN128_MAP, np.arange(204800)))
    sys.path.insert(0, str(E / 'parity-window-origin-20260911'))
    import wo_spec
    g2['tin128_recorded_map_equals_closed_form'] = bool(np.array_equal(B9.TIN128_MAP, wo_spec.tin_map(40, 40, 128)))
    rng = np.random.default_rng(22)
    probe = rng.integers(0, 256, (40, 40, 128), dtype=np.uint8)
    g2['tin128_roundtrip'] = bool(np.array_equal(B9.tin128_decode(B9.tin128_encode(probe)), probe))
    for s in SAMPLES:
        i = json.loads((D22 / 'interfaces' / s / 'interface.json').read_text())['b9']
        g2['%s_geometry' % s] = (tuple(i['origin_xy']) == (0, 0) and i['grid_xyz'] == [5, 5, 1] and i['block_dim'] == [32, 4, 1]
                                 and i['derived_bytes']['X'] == 204800 and i['derived_bytes']['Y'] == 204800)
        raw = (D22 / 'interfaces' / s / 'inputs/B9-X.bin').read_bytes()
        a = B9.inpview128_decode(raw)
        g2['%s_X_inpview128_roundtrip' % s] = B9.inpview128_encode(a) == raw
        g2['%s_X_layouts_distinguishable' % s] = bool(np.count_nonzero(a != B9.tin128_decode(raw)) > 0)
        g2['%s_X_no_nan_codes' % s] = bool(not np.any((a & 0x7F) == 0x7F))
    gates['G2_weights_layouts_geometry'] = dict(detail=g2, passed=all(g2.values()))
    ctrl = json.loads((D22 / 'controls/sm75-b9-control/result.json').read_text())
    gates['G3_sm75_positive_control'] = dict(elements=ctrl['elements'], mismatch=ctrl['physical_mismatch_vs_native'],
                                             passed=ctrl['matches_recorded_cpu_y'] and ctrl['physical_mismatch_vs_native'] == 0 and ctrl['elements'] == 204800)
    g4 = {k: P19.read_capable(v) == want for k, v, want in (('rb', 'rb', True), ('rb+', 'rb+', True), ('wb', 'wb', False),
                                                            ('O_RDONLY', os.O_RDONLY, True), ('O_RDWR', os.O_RDWR, True), ('O_WRONLY', os.O_WRONLY, False))}
    gates['G4_audit_classifier'] = dict(detail=g4, passed=all(g4.values()))
    passed = all(g['passed'] for g in gates.values())
    finish(out, dict(stage='gates', started_utc=started, freeze_sha256=sha_file(FREEZE), gates=gates, all_gates_passed=passed))
    print(json.dumps({k: (v['passed'], v.get('detail')) for k, v in gates.items()}, indent=1, default=str))
    if not passed:
        raise SystemExit('GATES FAILED')


def stage_b9(sample, layout='inpview128', tail_seed='published_qz', subdir='B9'):
    fz = verify_freeze()
    gm = json.loads((D22 / 'gates/manifest.json').read_text())
    if not gm['all_gates_passed'] or gm['invalid']:
        raise SystemExit('gates not passed')
    out = D22 / 'predictions' / sample / subdir
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    i = json.loads((D22 / 'interfaces' / sample / 'interface.json').read_text())['b9']
    be = b9_backend(tables, tail_seed)
    be.validate_invocation(block_index=9, head_count=4, origin_xy=i['origin_xy'], grid_xy=i['grid_xyz'][:2], shape=(40, 40, 128))
    raw = (D22 / 'interfaces' / sample / 'inputs/B9-X.bin').read_bytes()
    x = B9.inpview128_decode(raw) if layout == 'inpview128' else B9.tin128_decode(raw)
    t = time.perf_counter()
    r = be.run_codes(x)
    secs = time.perf_counter() - t
    yb = B9.tin128_encode(r['Y'])
    (out / 'B9-Y.tin128').write_bytes(yb)
    finish(out, dict(stage='B9', sample=sample, subdir=subdir, input_layout=layout, tail_seed=tail_seed,
                     teacher_forced_input='native B9 X (%s)' % layout, started_utc=started, predictions_written_utc=utc(),
                     freeze_sha256=fz, input_sha256=shab(raw), backend=be.describe(), seconds=secs,
                     predictions={'B9-Y.tin128': dict(sha256=shab(yb), bytes=len(yb))}, prediction_equals_input=yb == raw,
                     stats=dict(Y=P13.code_stats(r['Y']), Y_half=P13.half_stats(r['Y_half']), QZ=P13.code_stats(r['QZ'])),
                     canvas_hw=r['canvas_hw'], n_windows=r['n_windows'], native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='B9', sample=sample, subdir=subdir, sha=shab(yb), seconds=secs, windows=r['n_windows']), indent=1))


def stage_chain(sample):
    fz = verify_freeze()
    c = D22 / 'comparisons' / sample / 'B9.json'
    if not c.exists() or json.loads(c.read_text())['verdict'] != 'MATCH':
        raise SystemExit('chain refused: %s not MATCH' % c)
    out = D22 / 'predictions' / sample / 'chain'
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    raw1 = (D13 / 'interfaces' / sample / 'inputs/B1-X.inpview').read_bytes()
    t = time.perf_counter()
    files, r8, log = chain_b1_b8(up, tables, raw1)
    x9 = B9.inpview128_decode(files['B8-pooled.inpview128'])     # CPU B8 pooled bytes are the B9 input buffer
    assert np.array_equal(x9, r8['pooled'])
    r9 = b9_backend(tables).run_codes(x9)
    files['B9-Y.tin128'] = B9.tin128_encode(r9['Y'])
    log.append(dict(step='B9', input_sha256=shab(files['B8-pooled.inpview128']), output_sha256=shab(files['B9-Y.tin128'])))
    secs = time.perf_counter() - t
    for n, b in files.items():
        (out / n).write_bytes(b)
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
    ap.add_argument('stage', choices=['gates', 'B9', 'chain', 'diag'])
    ap.add_argument('--sample', choices=SAMPLES)
    ap.add_argument('--which', choices=['DIAG1', 'DIAG2'])
    a = ap.parse_args()
    if a.stage == 'gates':
        stage_gates()
    elif a.stage == 'B9':
        stage_b9(a.sample)
    elif a.stage == 'diag':
        if a.which == 'DIAG1':
            stage_b9(a.sample, layout='tin128', subdir='B9-DIAG1-input-tin128')
        else:
            stage_b9(a.sample, tail_seed='unquantised_z', subdir='B9-DIAG2-tail-seed-unquantised-z')
    else:
        stage_chain(a.sample)


if __name__ == '__main__':
    main()
