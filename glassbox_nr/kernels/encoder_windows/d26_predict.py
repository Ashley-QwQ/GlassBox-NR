"""D26 predictions.  gates | B15 --sample S | chain --sample S | diag --sample S --which DIAG1|DIAG2
NO ORACLE IS OPENED.  2 CPU threads.  Helpers reused read-only from D25 (-> D24/D22/D21/D19/D17/D13).
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
D26 = HERE.parent
ROOT = D26.parents[1]
E = ROOT / 'experiments'
D13 = E / 'parity-astra-d13-b234-sm120-cpu-chain-20260912'
D17 = E / 'parity-astra-d17-b5-sm120-cpu-chain-20260912'
D19 = E / 'parity-astra-d19-b67-sm120-cpu-chain-20260912'
D21 = E / 'parity-astra-d21-b8-downsample-cpu-20260912'
D22 = E / 'parity-astra-d22-b9-4h-sm120-cpu-chain-20260912'
D24 = E / 'parity-astra-d24-b10-b13-chained-cpu-20260912'
D25 = E / 'parity-astra-d25-b14-downsample-cpu-20260912'
WO = E / 'parity-window-origin-20260911'
SAMPLES = ('c1-C-img-full',)
sys.path.insert(0, str(HERE))
import d25_predict as P25  # noqa: E402
import d26_backend as N  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
torch.set_num_threads(2)
P24, S, B9, B14 = P25.P24, P25.S, P25.B9, P25.B14
P22, P19, P13, B5 = P24.P22, P24.P19, P24.P13, P24.B5
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
FREEZE = D26 / 'dependency-freeze.json'
DENY_DIR = ('\\oracle\\', '/oracle/')
DENY_SUFFIX = ('-Y.TIN', 'full.TIN', 'pooled.raw', '-Y.tin64', '-full.bin', '-pooled.bin', '-Y.bin', 'Y.native-physical')
NATIVE_X_FORBIDDEN_IN_CHAIN = P25.NATIVE_X_FORBIDDEN_IN_CHAIN + ('B15-X.bin',)


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
    fs = [HERE / 'd26_backend.py', HERE / 'd26_predict.py', D26 / 'chain-and-hypotheses.json', D26 / 'static-chains/static-chains.json',
          D25 / 'scripts/d25_predict.py', D25 / 'scripts/d25_backend.py', D25 / 'dependency-freeze.json',
          D24 / 'scripts/d24_predict.py', D22 / 'scripts/d22_backend.py', D22 / 'scripts/d22_predict.py',
          D21 / 'scripts/d21_backend.py', D21 / 'scripts/d21_predict.py', D19 / 'scripts/d19_backend.py', D19 / 'scripts/d19_predict.py',
          D17 / 'scripts/d17_backend.py', D13 / 'scripts/d13_backend.py', D13 / 'scripts/d13_predict.py', WO / 'wo_spec.py',
          ROOT / 'mlxdlss/sm120_b1.py', ROOT / 'mlxdlss/model.py', ROOT / 'mlxdlss/__init__.py',
          ROOT / 'mlxdlss/resources/rsq-domain.npz', ROOT / 'mlxdlss/resources/rcp-domain.npz', P19.SAFETENSORS,
          E / 'parity-b5-branched-ffn/TIN64-physical-to-logical.npy', E / 'parity-b9-h4-ffn/TIN128-physical-to-logical.npy']
    fs += [Path(p) for p in P13.PACKAGES.values()]
    for s in SAMPLES:
        fs += [D26 / 'interfaces' / s / 'interface.json', D26 / 'interfaces' / s / 'inputs/B15-X.bin',
               D25 / 'predictions' / s / 'chain/manifest.json', D25 / 'predictions' / s / 'B14/manifest.json',
               D22 / 'predictions' / s / 'B9/manifest.json', D22 / 'interfaces' / s / 'inputs/B9-X.bin',
               D17 / 'interfaces' / s / 'inputs/B5-X.inpview64', D13 / 'interfaces' / s / 'inputs/B1-X.inpview']
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


def b15_backend(tables, tail_seed='published_qz'):
    return N.BranchedWindowSm120N(15, 8, 20, (0, 0), P19.logical(15), tables, tail_seed=tail_seed)


def canvas_coverage(origin_xy, side):
    ox, oy = origin_xy
    gx, gy = N.BW.canvas_grid(origin_xy, side)
    H, W = gy * 8, gx * 8
    canvas = np.full((H, W), -1, np.int64)
    canvas[-oy:-oy + side, -ox:-ox + side] = np.arange(side * side).reshape(side, side)
    vals = canvas[canvas >= 0]
    return dict(canvas_hw=[H, W], windows=gx * gy, each_valid_pixel_once=bool(np.array_equal(np.sort(vals), np.arange(side * side))),
                pad_tokens=int((canvas < 0).sum()), pad_at_end_only=bool((canvas[:side, :side] >= 0).all()))


def stage_gates():
    out = D26 / 'gates'
    out.mkdir(exist_ok=False)
    started = utc()
    FREEZE.write_text(json.dumps(dict(created_utc=started, note='freeze file not listed in itself', python=sys.version,
                                      platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                                      files={rel(p): sha_file(p) for p in dep_files()}), indent=1))
    gates = {}
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    # G0 copied class reproduces verified 2-head B5 (D17 live backend) and 4-head B9 (D22 frozen + D22 class) on both samples
    g0 = {}
    for s in SAMPLES:
        x5 = B5.inpview64_decode((D17 / 'interfaces' / s / 'inputs/B5-X.inpview64').read_bytes())
        ref5 = B5.B5Sm120(P19.logical(5), tables).run_codes(x5)
        got5 = N.BranchedWindowSm120N(5, 2, 80, (0, 0), P19.logical(5), tables).run_codes(x5)
        x9 = B9.inpview128_decode((D22 / 'interfaces' / s / 'inputs/B9-X.bin').read_bytes())
        want9 = json.loads((D22 / 'predictions' / s / 'B9/manifest.json').read_text())['predictions']['B9-Y.tin128']['sha256']
        ref9 = P22.b9_backend(tables).run_codes(x9)
        got9 = N.BranchedWindowSm120N(9, 4, 40, (0, 0), P19.logical(9), tables).run_codes(x9)
        g0[s] = dict(B5_Y=bool(np.array_equal(ref5['Y'], got5['Y'])), B5_QZ=bool(np.array_equal(ref5['QZ'], got5['QZ'])),
                     B5_AV=bool(np.array_equal(ref5['AV'], got5['AV'])),
                     B9_Y_equals_D22_frozen=shab(B9.tin128_encode(got9['Y'])) == want9,
                     B9_Y_half_bits_equal_D22_class=bool(np.array_equal(ref9['Y_half'].view(np.uint16), got9['Y_half'].view(np.uint16))),
                     B9_QZ_AV_equal_D22_class=bool(np.array_equal(ref9['QZ'], got9['QZ']) and np.array_equal(ref9['AV'], got9['AV'])))
    refused = {}
    for hs in ((8, 40), (4, 20), (16, 10), (1, 160)):
        try:
            N.BranchedWindowSm120N(15, hs[0], hs[1], (0, 0), P19.logical(15), tables)
            refused['%dh_side%d' % hs] = False
        except S.UnsupportedByBackend:
            refused['%dh_side%d' % hs] = True
    g0['unsupported_domain_refused'] = refused
    gates['G0_copied_class_regression_2h_4h'] = dict(detail=g0, n_cases=2 * len(SAMPLES),
                                                    passed=all(all(v.values()) for v in g0.values()))
    # G1 D25 17-exit chain reproduced (both samples)
    g1 = {}
    for s in SAMPLES:
        want = {k: v['sha256'] for k, v in json.loads((D25 / 'predictions' / s / 'chain/manifest.json').read_text())['predictions'].items()}
        files, _ = P25.chain_b1_b13(up, tables, (D13 / 'interfaces' / s / 'inputs/B1-X.inpview').read_bytes())
        r14 = B14.B14Sm120(P19.logical(14), tables).run_codes(B9.tin128_decode(files['B13-Y.tin128']))
        files.update(P25.b14_files(r14))
        got = {k: shab(v) for k, v in files.items()}
        g1[s] = dict(equal=got == want, n=len(got))
    gates['G1_d25_chain_reproduced'] = dict(detail=g1, passed=all(v['equal'] and v['n'] == 17 for v in g1.values()))
    # G2 weights, geometry, input
    g2 = {}
    wl = P19.logical(15)
    g2['block15_shapes_for_8'] = all(tuple(wl['block15.layer0.%s' % n].shape) == sh for n, sh in N.shapes_for(8).items())
    g2['block15_no_extra_tensors'] = sorted(k.split('.', 2)[2] for k in wl) == sorted(N.shapes_for(8))
    g2['block15_logical_bias'] = not N.M.uses_fragment_swizzle(15, 8)
    g2['block15_attn_scale_finite_positive'] = bool(np.all(np.isfinite(wl['block15.layer0.attn_scale'])) and np.all(wl['block15.layer0.attn_scale'] > 0))
    tensor_sha = shab(b''.join(np.ascontiguousarray(wl['block15.layer0.%s' % n]).tobytes() for n in sorted(N.shapes_for(8))))
    cov = canvas_coverage((0, 0), 20)
    g2['canvas_each_valid_pixel_once'] = cov['each_valid_pixel_once'] and cov['pad_at_end_only']
    g2['canvas_pad_tokens_176'] = cov['pad_tokens'] == 176
    x_layout = {}
    for s in SAMPLES:
        i = json.loads((D26 / 'interfaces' / s / 'interface.json').read_text())
        b = i['b15']
        g2['%s_geometry' % s] = (tuple(b['origin_xy']) == (0, 0) and b['grid_xyz'] == [3, 3, 1] and b['canvas_hw'] == cov['canvas_hw']
                                 and b['block_dim'] == [32, 8, 1] and b['extent_wh'] == [20, 20]
                                 and b['derived_bytes']['X'] == 102400 and b['derived_bytes']['Y'] == 102400 and b['module'].endswith('1010'))
        raw = (D26 / 'interfaces' / s / 'inputs/B15-X.bin').read_bytes()
        g2['%s_X_sha_equals_record' % s] = shab(raw) == i['sha256']['inputs/B15-X.bin']
        d25p = json.loads((D25 / 'predictions' / s / 'B14/manifest.json').read_text())['predictions']['B14-pooled.inpview256']['sha256']
        g2['%s_X_equals_D25_frozen_B14_pooled_prediction' % s] = shab(raw) == d25p
        a = N.inpview256_decode(raw)
        t = N.tin256_decode(raw)
        g2['%s_X_inpview256_roundtrip' % s] = N.inpview256_encode(a) == raw
        g2['%s_X_no_nan' % s] = bool(not np.any((a & 0x7F) == 0x7F))
        x_layout[s] = dict(inpview256_vs_tin256_logical_differences=int(np.count_nonzero(a != t)))
        g2['%s_X_layouts_distinguishable' % s] = x_layout[s]['inpview256_vs_tin256_logical_differences'] > 0
    gates['G2_weights_geometry_input'] = dict(detail=g2, block15_logical_tensor_sha256=tensor_sha, canvas=cov, x_layout=x_layout, passed=all(g2.values()))
    # G3 layout codecs
    g3 = {}
    m = N.tin256_map()
    g3['tin256_is_permutation'] = bool(np.array_equal(np.sort(m), np.arange(102400)))
    g3['tin256_not_identity'] = bool(not np.array_equal(m, np.arange(102400)))
    g3['closed_form_equals_TIN128_npy'] = bool(np.array_equal(N.wo_spec.tin_map(40, 40, 128), B9.TIN128_MAP))
    g3['closed_form_equals_TIN64_npy'] = bool(np.array_equal(N.wo_spec.tin_map(80, 80, 64), np.load(E / 'parity-b5-branched-ffn/TIN64-physical-to-logical.npy')))
    rng = np.random.default_rng(26)
    pr = rng.integers(0, 256, (20, 20, 256), dtype=np.uint8)
    g3['tin256_roundtrip'] = bool(np.array_equal(N.tin256_decode(N.tin256_encode(pr)), pr))
    g3['inpview256_roundtrip'] = bool(np.array_equal(N.inpview256_decode(N.inpview256_encode(pr)), pr))
    g3['inpview256_not_identity'] = N.inpview256_encode(pr) != pr.tobytes()
    g3['inpview256_ne_tin256'] = N.inpview256_encode(pr) != N.tin256_encode(pr)
    gates['G3_layout_codecs'] = dict(detail=g3, passed=all(g3.values()))
    # G4 static structure consistent with frozen candidate
    st = json.loads((D26 / 'static-chains/static-chains.json').read_text())
    sb = st['kernels']['B15']
    g5 = dict(calibration_passed=st['calibration_gate_B9_vs_D25']['passed'],
              loop_iterations_4_8_8_8=[l.get('decoded_iterations') for l in sb['arithmetic_loops']] == [4, 8, 8, 8],
              expansion_chains_8x256_rz_16=sb['chain_signature'].get('8|256|RZ') == 16,
              no_wait_loops=len(sb['wait_sync_loops']) == 0, header_once=sb['header_occurrences'] == 1,
              candidate_k256_steps_8=len(N.splits32(256)) == 8, candidate_branch_steps_4=len(B5.K128) == 4,
              output_projection_carried_16_equals_loop2=sb['arithmetic_loops'][1]['mma'] == 16 and sb['arithmetic_loops'][1]['loop_carried_c'] == 16,
              tail_loop_carried_16=sb['arithmetic_loops'][3]['mma'] == 16 and sb['arithmetic_loops'][3]['loop_carried_c'] == 16)
    gates['G4_static_consistency'] = dict(detail=g5, passed=all(g5.values()))
    g6 = {k: P19.read_capable(v) == want for k, v, want in (('rb', 'rb', True), ('rb+', 'rb+', True), ('wb', 'wb', False),
                                                            ('O_RDONLY', os.O_RDONLY, True), ('O_RDWR', os.O_RDWR, True), ('O_WRONLY', os.O_WRONLY, False))}
    gates['G5_audit_classifier'] = dict(detail=g6, passed=all(g6.values()))
    passed = all(g['passed'] for g in gates.values())
    finish(out, dict(stage='gates', started_utc=started, freeze_sha256=sha_file(FREEZE), gates=gates, all_gates_passed=passed))
    print(json.dumps({k: (v['passed'], v.get('detail')) for k, v in gates.items()}, indent=1, default=str))
    if not passed:
        raise SystemExit('GATES FAILED')


def stage_b15(sample, layout='inpview256', tail_seed='published_qz', subdir='B15'):
    fz = verify_freeze()
    gm = json.loads((D26 / 'gates/manifest.json').read_text())
    if not gm['all_gates_passed'] or gm['invalid']:
        raise SystemExit('gates not passed')
    out = D26 / 'predictions' / sample / subdir
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    i = json.loads((D26 / 'interfaces' / sample / 'interface.json').read_text())['b15']
    be = b15_backend(tables, tail_seed)
    be.validate_invocation(block_index=15, head_count=i['head_count'], origin_xy=i['origin_xy'], grid_xy=i['grid_xyz'][:2], shape=(20, 20, 256))
    raw = (D26 / 'interfaces' / sample / 'inputs/B15-X.bin').read_bytes()
    x = N.inpview256_decode(raw) if layout == 'inpview256' else N.tin256_decode(raw)
    t = time.perf_counter()
    r = be.run_codes(x)
    secs = time.perf_counter() - t
    yb = N.tin256_encode(r['Y'])
    (out / 'B15-Y.tin256').write_bytes(yb)
    finish(out, dict(stage='B15', sample=sample, subdir=subdir, input_layout=layout, tail_seed=tail_seed,
                     teacher_forced_input='native B15 X (%s)' % layout, started_utc=started, predictions_written_utc=utc(),
                     freeze_sha256=fz, input_sha256=shab(raw), backend=be.describe(), seconds=secs,
                     predictions={'B15-Y.tin256': dict(sha256=shab(yb), bytes=len(yb))}, prediction_equals_input=yb == raw,
                     stats=dict(Y=P13.code_stats(r['Y']), Y_half=P13.half_stats(r['Y_half']), QZ=P13.code_stats(r['QZ'])),
                     canvas_hw=r['canvas_hw'], n_windows=r['n_windows'], native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='B15', sample=sample, subdir=subdir, sha=shab(yb), seconds=secs, canvas=r['canvas_hw'], windows=r['n_windows']), indent=1))


def stage_chain(sample):
    fz = verify_freeze()
    c = D26 / 'comparisons' / sample / 'B15.json'
    if not c.exists() or json.loads(c.read_text())['verdict'] != 'MATCH':
        raise SystemExit('chain refused: %s not MATCH' % c)
    out = D26 / 'predictions' / sample / 'chain'
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    raw1 = (D13 / 'interfaces' / sample / 'inputs/B1-X.inpview').read_bytes()
    t = time.perf_counter()
    files, log = P25.chain_b1_b13(up, tables, raw1)
    carried = files['B13-Y.tin128']
    r14 = B14.B14Sm120(P19.logical(14), tables).run_codes(B9.tin128_decode(carried))
    files.update(P25.b14_files(r14))
    log.append(dict(step='B14', input_sha256=shab(carried), full_sha256=shab(files['B14-full.tin128']), pooled_sha256=shab(files['B14-pooled.inpview256'])))
    x15 = N.inpview256_decode(files['B14-pooled.inpview256'])     # CPU B14 pooled bytes are the B15 input buffer
    assert np.array_equal(x15, r14['pooled'])
    r15 = b15_backend(tables).run_codes(x15)
    files['B15-Y.tin256'] = N.tin256_encode(r15['Y'])
    log.append(dict(step='B15', input_sha256=shab(files['B14-pooled.inpview256']), output_sha256=shab(files['B15-Y.tin256']), canvas_hw=r15['canvas_hw']))
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
    ap.add_argument('stage', choices=['gates', 'B15', 'chain', 'diag'])
    ap.add_argument('--sample', choices=SAMPLES)
    ap.add_argument('--which', choices=['DIAG1', 'DIAG2'])
    a = ap.parse_args()
    if a.stage == 'gates':
        stage_gates()
    elif a.stage == 'B15':
        stage_b15(a.sample)
    elif a.stage == 'diag':
        if a.which == 'DIAG1':
            stage_b15(a.sample, layout='tin256', subdir='B15-DIAG1-input-tin256')
        else:
            stage_b15(a.sample, tail_seed='unquantised_z', subdir='B15-DIAG2-tail-seed-unquantised-z')
    else:
        stage_chain(a.sample)


if __name__ == '__main__':
    main()
