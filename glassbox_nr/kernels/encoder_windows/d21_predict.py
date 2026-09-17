"""D21 predictions.  gates | B8 --sample S | chain --sample S | diag --sample S --which DIAG1|DIAG2
NO ORACLE IS OPENED.  2 CPU threads.  Helpers reused read-only from D19's d19_predict.
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
D21 = HERE.parent
ROOT = D21.parents[1]
E = ROOT / 'experiments'
D13 = E / 'parity-astra-d13-b234-sm120-cpu-chain-20260912'
D17 = E / 'parity-astra-d17-b5-sm120-cpu-chain-20260912'
D19 = E / 'parity-astra-d19-b67-sm120-cpu-chain-20260912'
SAMPLES = ('c1-C-img-full',)
sys.path.insert(0, str(HERE))
import d19_predict as P19  # noqa: E402  (installs its own audit hook; we keep ours separately)
import d21_backend as B8  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
torch.set_num_threads(2)
S, BW, B5, DB, P13 = B8.S, P19.BW, P19.B5, P19.DB, P19.P13
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
FREEZE = D21 / 'dependency-freeze.json'
DENY_DIR = ('\\oracle\\', '/oracle/')
DENY_SUFFIX = ('-Y.TIN', 'full.TIN', 'pooled.raw', '-Y.tin64', 'B8-full.bin', 'B8-pooled.bin')
NATIVE_X_FORBIDDEN_IN_CHAIN = ('B5-X.inpview64', 'B6-X.tin64', 'B7-X.tin64', 'B8-X.tin64')


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
    fs = [HERE / n for n in ('d21_backend.py', 'd21_predict.py')]
    fs += [D21 / 'chain-and-hypotheses.json', D21 / 'static-chains/static-chains.json',
           D19 / 'scripts/d19_backend.py', D19 / 'scripts/d19_predict.py', D19 / 'dependency-freeze.json',
           D17 / 'scripts/d17_backend.py', D13 / 'scripts/d13_backend.py', D13 / 'scripts/d13_predict.py',
           ROOT / 'mlxdlss/sm120_b1.py', ROOT / 'mlxdlss/model.py', ROOT / 'mlxdlss/__init__.py',
           ROOT / 'mlxdlss/resources/rsq-domain.npz', ROOT / 'mlxdlss/resources/rcp-domain.npz',
           P19.SAFETENSORS, E / 'parity-b5-branched-ffn/TIN64-physical-to-logical.npy']
    fs += [Path(p) for p in P13.PACKAGES.values()]
    for s in SAMPLES:
        fs += [D21 / 'interfaces' / s / 'interface.json', D21 / 'interfaces' / s / 'inputs/B8-X.tin64',
               D19 / 'predictions' / s / 'chain/manifest.json', D13 / 'interfaces' / s / 'inputs/B1-X.inpview',
               D13 / 'interfaces' / s / 'inputs/B4-X.TIN', D13 / 'interfaces' / s / 'interface.json',
               D19 / 'interfaces' / s / 'inputs/B7-X.tin64']
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


def stage_gates():
    out = D21 / 'gates'
    out.mkdir(exist_ok=False)
    started = utc()
    FREEZE.write_text(json.dumps(dict(created_utc=started, note='freeze file not listed in itself', python=sys.version,
                                      platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                                      files={rel(p): sha_file(p) for p in dep_files()}), indent=1))
    gates = {}
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    # G0 subclass mechanism: an instance built like B8Sm120 but at block 7 config equals D19's class on B7 X
    g0 = {}
    for s in SAMPLES:
        x = B5.tin64_decode((D19 / 'interfaces' / s / 'inputs/B7-X.tin64').read_bytes())
        ref = BW.BranchedWindow2hSm120(7, P19.logical(7), tables).run_codes(x)
        sub = B8.B8Sm120.__new__(B8.B8Sm120)
        BW.BranchedWindow2hSm120.__init__(sub, 7, P19.logical(7), tables)
        got = BW.BranchedWindow2hSm120.run_codes(sub, x)
        g0[s] = dict(Y=bool(np.array_equal(ref['Y'], got['Y'])), Y_half_bits=bool(np.array_equal(ref['Y_half'].view(np.uint16), got['Y_half'].view(np.uint16))))
    gates['G0_subclass_body_equals_D19_at_B7'] = dict(detail=g0, passed=all(all(v.values()) for v in g0.values()))
    # G1 D19 chain B1..B7 reproduced
    g1 = {}
    for s in SAMPLES:
        want = {k: v['sha256'] for k, v in json.loads((D19 / 'predictions' / s / 'chain/manifest.json').read_text())['predictions'].items()}
        files = P19.chain_b1_b5(up, tables, (D13 / 'interfaces' / s / 'inputs/B1-X.inpview').read_bytes())
        carried = files['B5-Y.tin64']
        for b in (6, 7):
            r = BW.BranchedWindow2hSm120(b, P19.logical(b), tables).run_codes(B5.tin64_decode(carried))
            carried = B5.tin64_encode(r['Y'])
            files['B%d-Y.tin64' % b] = carried
        got = {k: shab(v) for k, v in files.items()}
        g1[s] = dict(equal=got == want, n=len(got))
    gates['G1_d19_chain_reproduced'] = dict(detail=g1, passed=all(v['equal'] and v['n'] == 8 for v in g1.values()))
    # G2 generic pool tail at D13's B4 config reproduces D13 B4 pooled (known answer on sm120)
    g2 = {}
    w4 = P13.logical_weights(up, 4)[0]
    for s in SAMPLES:
        x4 = S.tin_decode_codes((D13 / 'interfaces' / s / 'inputs/B4-X.TIN').read_bytes(), 160, 160)
        ref = DB.SingleWindowSm120(4, w4, tables).run_codes(x4)
        _, pooled = B8.pool_tail(ref['Y_half'], w4['block4.layer0.weight0'], [(0, 32)])
        d13rec = json.loads((D13 / 'interfaces' / s / 'interface.json').read_text())['sha256']['oracle/B4-pooled.raw']
        g2[s] = dict(pooled_equals_D13_backend=bool(np.array_equal(pooled, ref['pooled'])),
                     encoded_sha256_equals_D13_recorded_native=shab(DB.pooled_encode(pooled)) == d13rec)
    gates['G2_pool_tail_known_answer_B4'] = dict(detail=g2, passed=all(all(v.values()) for v in g2.values()))
    # G3 geometry, layouts, weights
    g3 = {}
    wl = P19.logical(8)
    g3['block8_weight0_shape'] = tuple(wl['block8.layer0.weight0'].shape) == (64, 128)
    g3['block8_body_shapes'] = all(tuple(wl['block8.layer0.%s' % n].shape) == sh for n, sh in B5.SHAPES.items())
    for s in SAMPLES:
        i = json.loads((D21 / 'interfaces' / s / 'interface.json').read_text())['b8']
        gx, gy = BW.canvas_grid(tuple(i['origin_xy']))
        g3['%s_grid' % s] = [gx, gy, 1] == i['grid_xyz'] and tuple(i['origin_xy']) == (0, -4)
        g3['%s_bytes' % s] = i['derived_bytes'] == dict(X=409600, full=409600, pooled=204800, rule=i['derived_bytes']['rule'])
        raw = (D21 / 'interfaces' / s / 'inputs/B8-X.tin64').read_bytes()
        g3['%s_X_tin64_roundtrip' % s] = B5.tin64_encode(B5.tin64_decode(raw)) == raw
    rng = np.random.default_rng(21)
    probe = rng.integers(0, 256, (40, 40, 128), dtype=np.uint8)
    g3['inpview128_roundtrip'] = bool(np.array_equal(B8.inpview128_decode(B8.inpview128_encode(probe)), probe))
    g3['inpview128_not_identity'] = B8.inpview128_encode(probe) != probe.tobytes()
    p64 = rng.integers(0, 256, (80, 80, 64), dtype=np.uint8)
    # same channel-order formula family: D13 pooled encode at 64 ch == D17 INPVIEW64 encode, and the 128-ch
    # order restricted to its first 64 entries equals the 64-ch order
    ch64 = np.arange(64)
    order64 = (ch64 // 16) * 16 + ((ch64 % 16) // 4) * 2 + ch64 % 2 + ((ch64 % 4) // 2) * 8
    g3['same_formula_as_d13_pooled_at_64ch'] = (DB.pooled_encode(p64) == B5.inpview64_encode(p64)) and \
        bool(np.array_equal(B8._ORDER128[:64], order64))
    # the two pool groupings must differ on ONE shared input, otherwise DIAG1 would be vacuous
    probe_h = (rng.random((8, 8, 4)) * 1000 + 0.37).astype(np.float16)
    g3['row_vs_column_pairs_distinguishable'] = not np.array_equal(
        B8.pool_half(probe_h, 'row_pairs').view(np.uint16), B8.pool_half(probe_h, 'column_pairs').view(np.uint16))
    gates['G3_geometry_layout_weights'] = dict(detail=g3, passed=all(g3.values()))
    g4 = {k: P19.read_capable(v) == want for k, v, want in
          (('rb', 'rb', True), ('rb+', 'rb+', True), ('wb', 'wb', False), ('O_RDONLY', os.O_RDONLY, True),
           ('O_RDWR', os.O_RDWR, True), ('O_WRONLY', os.O_WRONLY, False))}
    gates['G4_audit_classifier'] = dict(detail=g4, passed=all(g4.values()))
    passed = all(g['passed'] for g in gates.values())
    finish(out, dict(stage='gates', started_utc=started, freeze_sha256=sha_file(FREEZE), gates=gates, all_gates_passed=passed))
    print(json.dumps({k: (v['passed'], v['detail']) for k, v in gates.items()}, indent=1, default=str))
    if not passed:
        raise SystemExit('GATES FAILED')


def b8_files(r):
    return {'B8-full.tin64': B5.tin64_encode(r['full']), 'B8-pooled.inpview128': B8.inpview128_encode(r['pooled'])}


def stage_b8(sample, grouping='row_pairs', source='unpublished', subdir='B8'):
    fz = verify_freeze()
    gm = json.loads((D21 / 'gates/manifest.json').read_text())
    if not gm['all_gates_passed'] or gm['invalid']:
        raise SystemExit('gates not passed')
    out = D21 / 'predictions' / sample / subdir
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    i = json.loads((D21 / 'interfaces' / sample / 'interface.json').read_text())['b8']
    be = B8.B8Sm120(P19.logical(8), tables, grouping=grouping, source=source)
    be.validate_invocation(block_index=8, head_count=2, window_size=8, origin_xy=i['origin_xy'], grid_xy=i['grid_xyz'][:2],
                           shape=(i['extent_wh'][1], i['extent_wh'][0], 64))
    raw = (D21 / 'interfaces' / sample / 'inputs/B8-X.tin64').read_bytes()
    t = time.perf_counter()
    r = be.run_codes(B5.tin64_decode(raw))
    secs = time.perf_counter() - t
    files = b8_files(r)
    for n, b in files.items():
        (out / n).write_bytes(b)
    finish(out, dict(stage='B8', sample=sample, subdir=subdir, grouping=grouping, pool_source=source,
                     teacher_forced_input='native B8 X (TIN64)', started_utc=started, predictions_written_utc=utc(),
                     freeze_sha256=fz, input_sha256=shab(raw), backend=be.describe(), seconds=secs,
                     predictions={n: dict(sha256=shab(b), bytes=len(b)) for n, b in files.items()},
                     stats=dict(full=P13.code_stats(r['full']), pooled=P13.code_stats(r['pooled']), pool_half=P13.half_stats(r['pool_half'])),
                     canvas_hw=r['canvas_hw'], native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='B8', sample=sample, subdir=subdir, seconds=secs, canvas=r['canvas_hw'],
                          predictions={n: shab(b) for n, b in files.items()}), indent=1))


def stage_chain(sample):
    fz = verify_freeze()
    c = D21 / 'comparisons' / sample / 'B8.json'
    if not c.exists() or json.loads(c.read_text())['verdict'] != 'MATCH':
        raise SystemExit('chain refused: %s not MATCH' % c)
    out = D21 / 'predictions' / sample / 'chain'
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    raw1 = (D13 / 'interfaces' / sample / 'inputs/B1-X.inpview').read_bytes()
    t = time.perf_counter()
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
    files.update(b8_files(r8))
    log.append(dict(step='B8', input_sha256=shab(carried), full_sha256=shab(files['B8-full.tin64']),
                    pooled_sha256=shab(files['B8-pooled.inpview128'])))
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
    ap.add_argument('stage', choices=['gates', 'B8', 'chain', 'diag'])
    ap.add_argument('--sample', choices=SAMPLES)
    ap.add_argument('--which', choices=['DIAG1', 'DIAG2'])
    a = ap.parse_args()
    if a.stage == 'gates':
        stage_gates()
    elif a.stage == 'B8':
        stage_b8(a.sample)
    elif a.stage == 'diag':
        if a.which == 'DIAG1':
            stage_b8(a.sample, grouping='column_pairs', subdir='B8-DIAG1-column-pairs')
        else:
            stage_b8(a.sample, source='published', subdir='B8-DIAG2-pool-published-y')
    else:
        stage_chain(a.sample)


if __name__ == '__main__':
    main()
