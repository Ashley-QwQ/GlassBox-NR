"""D25 predictions.  gates | B14 --sample S | chain --sample S | diag --sample S --which DIAG1|DIAG2
NO ORACLE IS OPENED.  2 CPU threads.  Helpers reused read-only from D24 (-> D22/D21/D19/D17/D13).
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
D25 = HERE.parent
ROOT = D25.parents[1]
E = ROOT / 'experiments'
D13 = E / 'parity-astra-d13-b234-sm120-cpu-chain-20260912'
D21 = E / 'parity-astra-d21-b8-downsample-cpu-20260912'
D22 = E / 'parity-astra-d22-b9-4h-sm120-cpu-chain-20260912'
D24 = E / 'parity-astra-d24-b10-b13-chained-cpu-20260912'
SAMPLES = ('c1-C-img-full',)
sys.path.insert(0, str(HERE))
import d24_predict as P24  # noqa: E402
import d25_backend as B14  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
torch.set_num_threads(2)
P22, P19, P13, S, B9 = P24.P22, P24.P19, P24.P13, P24.S, P24.B9
P21 = P22.P21
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
FREEZE = D25 / 'dependency-freeze.json'
DENY_DIR = ('\\oracle\\', '/oracle/')
DENY_SUFFIX = ('-Y.TIN', 'full.TIN', 'pooled.raw', '-Y.tin64', '-full.bin', '-pooled.bin', '-Y.bin', 'Y.native-physical')
NATIVE_X_FORBIDDEN_IN_CHAIN = P24.NATIVE_X_FORBIDDEN_IN_CHAIN + ('B14-X.bin',)


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
    fs = [HERE / 'd25_backend.py', HERE / 'd25_predict.py', D25 / 'chain-and-hypotheses.json', D25 / 'static-chains/static-chains.json',
          D24 / 'scripts/d24_predict.py', D24 / 'dependency-freeze.json',
          D22 / 'scripts/d22_backend.py', D22 / 'scripts/d22_predict.py', D21 / 'scripts/d21_backend.py', D21 / 'scripts/d21_predict.py',
          E / 'parity-astra-d19-b67-sm120-cpu-chain-20260912/scripts/d19_backend.py', E / 'parity-astra-d19-b67-sm120-cpu-chain-20260912/scripts/d19_predict.py',
          E / 'parity-astra-d17-b5-sm120-cpu-chain-20260912/scripts/d17_backend.py', D13 / 'scripts/d13_backend.py', D13 / 'scripts/d13_predict.py',
          ROOT / 'mlxdlss/sm120_b1.py', ROOT / 'mlxdlss/model.py', ROOT / 'mlxdlss/__init__.py',
          ROOT / 'mlxdlss/resources/rsq-domain.npz', ROOT / 'mlxdlss/resources/rcp-domain.npz', P19.SAFETENSORS,
          E / 'parity-b5-branched-ffn/TIN64-physical-to-logical.npy', E / 'parity-b9-h4-ffn/TIN128-physical-to-logical.npy']
    fs += [Path(p) for p in P13.PACKAGES.values()]
    for s in SAMPLES:
        fs += [D25 / 'interfaces' / s / 'interface.json', D25 / 'interfaces' / s / 'inputs/B14-X.bin',
               D24 / 'predictions' / s / 'chain/manifest.json', D24 / 'predictions' / s / 'B13/manifest.json', D24 / 'interfaces' / s / 'inputs/B13-X.bin',
               D21 / 'predictions' / s / 'B8/manifest.json', E / 'parity-astra-d21-b8-downsample-cpu-20260912/interfaces' / s / 'inputs/B8-X.tin64',
               D13 / 'interfaces' / s / 'interface.json', D13 / 'interfaces' / s / 'inputs/B4-X.TIN', D13 / 'interfaces' / s / 'inputs/B1-X.inpview']
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


def b14_files(r):
    return {'B14-full.tin128': B9.tin128_encode(r['full']), 'B14-pooled.inpview256': B14.inpview_encode(r['pooled'])}


def chain_b1_b13(up, tables, raw1):
    files, log = P24.chain_b1_b9(up, tables, raw1)
    carried = files['B9-Y.tin128']
    for b in P24.BLOCKS:
        r = P24.backend(b, tables).run_codes(B9.tin128_decode(carried))
        nxt = B9.tin128_encode(r['Y'])
        files['B%d-Y.tin128' % b] = nxt
        log.append(dict(step='B%d' % b, input_sha256=shab(carried), output_sha256=shab(nxt)))
        carried = nxt
    return files, log


def stage_gates():
    out = D25 / 'gates'
    out.mkdir(exist_ok=False)
    started = utc()
    FREEZE.write_text(json.dumps(dict(created_utc=started, note='freeze file not listed in itself', python=sys.version,
                                      platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                                      files={rel(p): sha_file(p) for p in dep_files()}), indent=1))
    gates = {}
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    # G0 body reuse: D24 path at B13 reproduces D24 frozen B13 prediction (non-empty, 2 samples)
    g0 = {}
    for s in SAMPLES:
        want = json.loads((D24 / 'predictions' / s / 'B13/manifest.json').read_text())['predictions']['B13-Y.tin128']['sha256']
        r = P24.backend(13, tables).run_codes(B9.tin128_decode((D24 / 'interfaces' / s / 'inputs/B13-X.bin').read_bytes()))
        g0[s] = dict(B13_equals_D24_frozen=shab(B9.tin128_encode(r['Y'])) == want)
    gates['G0_body_reuse_B13'] = dict(detail=g0, n_cases=len(SAMPLES), passed=all(all(v.values()) for v in g0.values()))
    # G1 D24 chain reproduced (15 exits x 2 samples)
    g1 = {}
    for s in SAMPLES:
        want = {k: v['sha256'] for k, v in json.loads((D24 / 'predictions' / s / 'chain/manifest.json').read_text())['predictions'].items()}
        files, _ = chain_b1_b13(up, tables, (D13 / 'interfaces' / s / 'inputs/B1-X.inpview').read_bytes())
        got = {k: shab(v) for k, v in files.items()}
        g1[s] = dict(equal=got == want, n=len(got))
    gates['G1_d24_chain_reproduced'] = dict(detail=g1, passed=all(v['equal'] and v['n'] == 15 for v in g1.values()))
    # G2 generic pool tail + INPVIEW codec against known answers: B8 (D21 frozen, 128ch) and B4 (D13 recorded native, 64ch)
    g2 = {}
    w4 = P13.logical_weights(up, 4)[0]
    for s in SAMPLES:
        want8 = json.loads((D21 / 'predictions' / s / 'B8/manifest.json').read_text())['predictions']['B8-pooled.inpview128']['sha256']
        r8 = P21.B8.B8Sm120(P19.logical(8), tables).run_codes(P24.B5.tin64_decode((D21 / 'interfaces' / s / 'inputs/B8-X.tin64').read_bytes()))
        _, p8 = B14.pool_tail(r8['Y_half'], P19.logical(8)['block8.layer0.weight0'])
        g2['%s_B8_generic_tail_equals_D21_frozen' % s] = shab(B14.inpview_encode(p8)) == want8
        x4 = S.tin_decode_codes((D13 / 'interfaces' / s / 'inputs/B4-X.TIN').read_bytes(), 160, 160)
        r4 = P24.P22.P21.P19.DB.SingleWindowSm120(4, w4, tables).run_codes(x4)
        _, p4 = B14.pool_tail(r4['Y_half'], w4['block4.layer0.weight0'])
        rec4 = json.loads((D13 / 'interfaces' / s / 'interface.json').read_text())['sha256']['oracle/B4-pooled.raw']
        g2['%s_B4_generic_tail_equals_D13_native' % s] = shab(B14.inpview_encode(p4)) == rec4
    rng = np.random.default_rng(25)
    pr128 = rng.integers(0, 256, (40, 40, 128), dtype=np.uint8)
    pr64 = rng.integers(0, 256, (80, 80, 64), dtype=np.uint8)
    pr256 = rng.integers(0, 256, (20, 20, 256), dtype=np.uint8)
    g2['codec_128_equals_d21'] = B14.inpview_encode(pr128) == P21.B8.inpview128_encode(pr128)
    g2['codec_64_equals_d13_pooled'] = B14.inpview_encode(pr64) == P19.DB.pooled_encode(pr64)
    g2['codec_256_roundtrip'] = bool(np.array_equal(B14.inpview_decode(B14.inpview_encode(pr256), 20, 20, 256), pr256))
    g2['codec_256_not_identity'] = B14.inpview_encode(pr256) != pr256.tobytes()
    g2['codec_256_order_prefix_equals_128'] = bool(np.array_equal(B14.inpview_order(256)[:128], B14.inpview_order(128)))
    probe_h = (rng.random((8, 8, 4)) * 1000 + 0.37).astype(np.float16)
    g2['pool_groupings_distinguishable'] = not np.array_equal(P21.B8.pool_half(probe_h, 'row_pairs').view(np.uint16), P21.B8.pool_half(probe_h, 'column_pairs').view(np.uint16))
    gates['G2_pool_tail_codec_known_answers'] = dict(detail=g2, n_known_answer_cases=2 * len(SAMPLES), passed=all(g2.values()))
    # G3 weights and geometry
    g3 = {}
    wl = P19.logical(14)
    g3['block14_body_shapes'] = all(tuple(wl['block14.layer0.%s' % n].shape) == sh for n, sh in B9.shapes_for(4).items())
    g3['block14_weight0_128x256'] = tuple(wl['block14.layer0.weight0'].shape) == (128, 256)
    g3['block14_logical_bias'] = not B9.M.uses_fragment_swizzle(14, 4)
    tensor_sha = shab(b''.join(np.ascontiguousarray(wl['block14.layer0.%s' % n]).tobytes() for n in sorted(list(B9.shapes_for(4)) + ['weight0'])))
    for s in SAMPLES:
        i = json.loads((D25 / 'interfaces' / s / 'interface.json').read_text())['b14']
        g3['%s_geometry' % s] = (tuple(i['origin_xy']) == (-4, -4) and i['grid_xyz'] == [6, 6, 1] and i['canvas_hw'] == [48, 48]
                                 and i['derived_bytes']['X'] == 204800 and i['derived_bytes']['full'] == 204800 and i['derived_bytes']['pooled'] == 102400
                                 and i['pooled_extent_wh'] == [20, 20])
        raw = (D25 / 'interfaces' / s / 'inputs/B14-X.bin').read_bytes()
        t = B9.tin128_decode(raw)
        g3['%s_X_tin128_roundtrip' % s] = B9.tin128_encode(t) == raw
        g3['%s_X_no_nan' % s] = bool(not np.any((t & 0x7F) == 0x7F))
    gates['G3_weights_geometry'] = dict(detail=g3, block14_logical_tensor_sha256=tensor_sha, passed=all(g3.values()))
    g4 = {k: P19.read_capable(v) == want for k, v, want in (('rb', 'rb', True), ('rb+', 'rb+', True), ('wb', 'wb', False),
                                                            ('O_RDONLY', os.O_RDONLY, True), ('O_RDWR', os.O_RDWR, True), ('O_WRONLY', os.O_WRONLY, False))}
    gates['G4_audit_classifier'] = dict(detail=g4, passed=all(g4.values()))
    passed = all(g['passed'] for g in gates.values())
    finish(out, dict(stage='gates', started_utc=started, freeze_sha256=sha_file(FREEZE), gates=gates, all_gates_passed=passed))
    print(json.dumps({k: (v['passed'], v.get('detail')) for k, v in gates.items()}, indent=1, default=str))
    if not passed:
        raise SystemExit('GATES FAILED')


def stage_b14(sample, grouping='row_pairs', source='unpublished', subdir='B14'):
    fz = verify_freeze()
    gm = json.loads((D25 / 'gates/manifest.json').read_text())
    if not gm['all_gates_passed'] or gm['invalid']:
        raise SystemExit('gates not passed')
    out = D25 / 'predictions' / sample / subdir
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    i = json.loads((D25 / 'interfaces' / sample / 'interface.json').read_text())['b14']
    be = B14.B14Sm120(P19.logical(14), tables, grouping=grouping, source=source)
    be.validate_invocation(block_index=14, head_count=4, origin_xy=i['origin_xy'], grid_xy=i['grid_xyz'][:2], shape=(40, 40, 128))
    raw = (D25 / 'interfaces' / sample / 'inputs/B14-X.bin').read_bytes()
    t = time.perf_counter()
    r = be.run_codes(B9.tin128_decode(raw))
    secs = time.perf_counter() - t
    files = b14_files(r)
    for n, b in files.items():
        (out / n).write_bytes(b)
    finish(out, dict(stage='B14', sample=sample, subdir=subdir, grouping=grouping, pool_source=source,
                     teacher_forced_input='native B14 X (TIN128)', started_utc=started, predictions_written_utc=utc(),
                     freeze_sha256=fz, input_sha256=shab(raw), backend=be.describe(), seconds=secs,
                     predictions={n: dict(sha256=shab(b), bytes=len(b)) for n, b in files.items()},
                     stats=dict(full=P13.code_stats(r['full']), pooled=P13.code_stats(r['pooled']), pool_half=P13.half_stats(r['pool_half'])),
                     canvas_hw=r['canvas_hw'], native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='B14', sample=sample, subdir=subdir, seconds=secs, predictions={n: shab(b) for n, b in files.items()}), indent=1))


def stage_chain(sample):
    fz = verify_freeze()
    c = D25 / 'comparisons' / sample / 'B14.json'
    if not c.exists() or json.loads(c.read_text())['verdict'] != 'MATCH':
        raise SystemExit('chain refused: %s not MATCH' % c)
    out = D25 / 'predictions' / sample / 'chain'
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    raw1 = (D13 / 'interfaces' / sample / 'inputs/B1-X.inpview').read_bytes()
    t = time.perf_counter()
    files, log = chain_b1_b13(up, tables, raw1)
    carried = files['B13-Y.tin128']
    r = B14.B14Sm120(P19.logical(14), tables).run_codes(B9.tin128_decode(carried))
    files.update(b14_files(r))
    log.append(dict(step='B14', input_sha256=shab(carried), full_sha256=shab(files['B14-full.tin128']), pooled_sha256=shab(files['B14-pooled.inpview256'])))
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
    print(json.dumps(dict(stage='chain', sample=sample, seconds=secs, n=len(files)), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gates', 'B14', 'chain', 'diag'])
    ap.add_argument('--sample', choices=SAMPLES)
    ap.add_argument('--which', choices=['DIAG1', 'DIAG2'])
    a = ap.parse_args()
    if a.stage == 'gates':
        stage_gates()
    elif a.stage == 'B14':
        stage_b14(a.sample)
    elif a.stage == 'diag':
        if a.which == 'DIAG1':
            stage_b14(a.sample, grouping='column_pairs', subdir='B14-DIAG1-column-pairs')
        else:
            stage_b14(a.sample, source='published', subdir='B14-DIAG2-pool-published-y')
    else:
        stage_chain(a.sample)


if __name__ == '__main__':
    main()
