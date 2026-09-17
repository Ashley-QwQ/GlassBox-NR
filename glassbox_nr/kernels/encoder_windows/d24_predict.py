"""D24 predictions.  gates | B10|B11|B12|B13 --sample S | chain --sample S | diag --sample S --block B --which DIAG1|DIAG2
NO ORACLE IS OPENED.  2 CPU threads.  Candidate = D22 BranchedWindowSm120 (unchanged) at heads 4, side 40,
per-block origin; chained X decoded as TIN128 (hypothesis). Helpers reused read-only from D22.
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
D24 = HERE.parent
ROOT = D24.parents[1]
E = ROOT / 'experiments'
D13 = E / 'parity-astra-d13-b234-sm120-cpu-chain-20260912'
D19 = E / 'parity-astra-d19-b67-sm120-cpu-chain-20260912'
D21 = E / 'parity-astra-d21-b8-downsample-cpu-20260912'
D22 = E / 'parity-astra-d22-b9-4h-sm120-cpu-chain-20260912'
SAMPLES = ('c1-C-img-full',)
BLOCKS = (10, 11, 12, 13)
ORIGINS = {10: (-4, -4), 11: (-4, 0), 12: (0, -4), 13: (0, 0)}
sys.path.insert(0, str(HERE))
import d22_predict as P22  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
torch.set_num_threads(2)
B9, P19, P13, BW, B5, S = P22.B9, P22.P19, P22.P13, P22.BW, P22.B5, P22.S
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
FREEZE = D24 / 'dependency-freeze.json'
DENY_DIR = ('\\oracle\\', '/oracle/')
DENY_SUFFIX = ('-Y.TIN', 'full.TIN', 'pooled.raw', '-Y.tin64', '-full.bin', '-pooled.bin', '-Y.bin', 'Y.native-physical')
NATIVE_X_FORBIDDEN_IN_CHAIN = ('B5-X.inpview64', 'B6-X.tin64', 'B7-X.tin64', 'B8-X.tin64', 'B9-X.bin',
                               'B10-X.bin', 'B11-X.bin', 'B12-X.bin', 'B13-X.bin')


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
    fs = [HERE / 'd24_predict.py', D24 / 'chain-and-hypotheses.json', D24 / 'static-chains/static-chains.json',
          D24 / 'controls/sm75-b10-b13-control/result.json',
          D22 / 'scripts/d22_backend.py', D22 / 'scripts/d22_predict.py', D22 / 'dependency-freeze.json',
          D21 / 'scripts/d21_backend.py', D21 / 'scripts/d21_predict.py', D19 / 'scripts/d19_backend.py', D19 / 'scripts/d19_predict.py',
          E / 'parity-astra-d17-b5-sm120-cpu-chain-20260912/scripts/d17_backend.py', D13 / 'scripts/d13_backend.py', D13 / 'scripts/d13_predict.py',
          ROOT / 'mlxdlss/sm120_b1.py', ROOT / 'mlxdlss/model.py', ROOT / 'mlxdlss/__init__.py',
          ROOT / 'mlxdlss/resources/rsq-domain.npz', ROOT / 'mlxdlss/resources/rcp-domain.npz', P19.SAFETENSORS,
          E / 'parity-b5-branched-ffn/TIN64-physical-to-logical.npy', E / 'parity-b9-h4-ffn/TIN128-physical-to-logical.npy']
    fs += [Path(p) for p in P13.PACKAGES.values()]
    for s in SAMPLES:
        fs += [D24 / 'interfaces' / s / 'interface.json', D22 / 'predictions' / s / 'chain/manifest.json',
               D22 / 'predictions' / s / 'B9/manifest.json', D22 / 'interfaces' / s / 'inputs/B9-X.bin',
               D19 / 'interfaces' / s / 'inputs/B6-X.tin64', D13 / 'interfaces' / s / 'inputs/B1-X.inpview']
        fs += [D24 / 'interfaces' / s / 'inputs' / ('B%d-X.bin' % b) for b in BLOCKS]
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


def backend(b, tables, tail_seed='published_qz'):
    return B9.BranchedWindowSm120(b, 4, 40, ORIGINS[b], P19.logical(b), tables, tail_seed=tail_seed)


def chain_b1_b9(up, tables, raw1):
    files, r8, log = P22.chain_b1_b8(up, tables, raw1)
    x9 = B9.inpview128_decode(files['B8-pooled.inpview128'])
    assert np.array_equal(x9, r8['pooled'])
    r9 = P22.b9_backend(tables).run_codes(x9)
    files['B9-Y.tin128'] = B9.tin128_encode(r9['Y'])
    log.append(dict(step='B9', input_sha256=shab(files['B8-pooled.inpview128']), output_sha256=shab(files['B9-Y.tin128'])))
    return files, log


def canvas_coverage(origin_xy):
    """Place a 40x40 index grid on the canvas; every valid pixel must appear exactly once, the rest is pad."""
    ox, oy = origin_xy
    gx, gy = BW.canvas_grid(origin_xy, 40)
    H, W = gy * 8, gx * 8
    canvas = np.full((H, W), -1, np.int64)
    canvas[-oy:-oy + 40, -ox:-ox + 40] = np.arange(1600).reshape(40, 40)
    vals = canvas[canvas >= 0]
    return dict(canvas_hw=[H, W], windows=gx * gy, each_valid_pixel_once=bool(np.array_equal(np.sort(vals), np.arange(1600))),
                pad_tokens=int((canvas < 0).sum()), pad_bytes=int((canvas < 0).sum()) * 128)


def stage_gates():
    out = D24 / 'gates'
    out.mkdir(exist_ok=False)
    started = utc()
    FREEZE.write_text(json.dumps(dict(created_utc=started, note='freeze file not listed in itself', python=sys.version,
                                      platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                                      files={rel(p): sha_file(p) for p in dep_files()}), indent=1))
    gates = {}
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    # G0 candidate reproduces D22 at the verified B9 configuration, and D19 at the 2-head chained B6 configuration
    g0 = {}
    for s in SAMPLES:
        want9 = json.loads((D22 / 'predictions' / s / 'B9/manifest.json').read_text())['predictions']['B9-Y.tin128']['sha256']
        r9 = P22.b9_backend(tables).run_codes(B9.inpview128_decode((D22 / 'interfaces' / s / 'inputs/B9-X.bin').read_bytes()))
        x6 = B5.tin64_decode((D19 / 'interfaces' / s / 'inputs/B6-X.tin64').read_bytes())
        ref6 = BW.BranchedWindow2hSm120(6, P19.logical(6), tables).run_codes(x6)
        got6 = B9.BranchedWindowSm120(6, 2, 80, (-4, -4), P19.logical(6), tables).run_codes(x6)
        g0[s] = dict(B9_equals_D22_frozen=shab(B9.tin128_encode(r9['Y'])) == want9, B6_2h_chained_equals_D19=bool(np.array_equal(ref6['Y'], got6['Y'])))
    gates['G0_candidate_regression'] = dict(detail=g0, n_cases=2 * len(SAMPLES), passed=all(all(v.values()) for v in g0.values()))
    # G1 D22 chain reproduced (11 exits x 2 samples)
    g1 = {}
    for s in SAMPLES:
        want = {k: v['sha256'] for k, v in json.loads((D22 / 'predictions' / s / 'chain/manifest.json').read_text())['predictions'].items()}
        files, _ = chain_b1_b9(up, tables, (D13 / 'interfaces' / s / 'inputs/B1-X.inpview').read_bytes())
        got = {k: shab(v) for k, v in files.items()}
        g1[s] = dict(equal=got == want, n=len(got))
    gates['G1_d22_chain_reproduced'] = dict(detail=g1, passed=all(v['equal'] and v['n'] == 11 for v in g1.values()))
    # G2 weights / canvas / layouts / geometry
    g2 = {}
    tensor_hashes = {}
    for b in BLOCKS:
        wl = P19.logical(b)
        g2['block%d_shapes' % b] = all(tuple(wl['block%d.layer0.%s' % (b, n)].shape) == sh for n, sh in B9.shapes_for(4).items())
        g2['block%d_logical_bias' % b] = not B9.M.uses_fragment_swizzle(b, 4)
        tensor_hashes[b] = shab(b''.join(np.ascontiguousarray(wl['block%d.layer0.%s' % (b, n)]).tobytes() for n in sorted(B9.shapes_for(4))))
        cov = canvas_coverage(ORIGINS[b])
        g2['block%d_canvas_each_valid_pixel_once' % b] = cov['each_valid_pixel_once']
        for s in SAMPLES:
            i = json.loads((D24 / 'interfaces' / s / 'interface.json').read_text())['blocks'][str(b)]
            g2['%s_B%d_origin_grid_canvas' % (s, b)] = (tuple(i['origin_xy']) == ORIGINS[b] and i['canvas_hw'] == cov['canvas_hw']
                                                          and [BW.canvas_grid(ORIGINS[b], 40)[0], BW.canvas_grid(ORIGINS[b], 40)[1], 1] == i['grid_xyz']
                                                          and i['bytes'] == 204800)
            raw = (D24 / 'interfaces' / s / 'inputs' / ('B%d-X.bin' % b)).read_bytes()
            t = B9.tin128_decode(raw)
            g2['%s_B%d_X_tin128_roundtrip' % (s, b)] = B9.tin128_encode(t) == raw
            g2['%s_B%d_X_tin_vs_inpview_distinguishable' % (s, b)] = bool(np.count_nonzero(t != B9.inpview128_decode(raw)) > 0)
            g2['%s_B%d_X_no_nan_codes' % (s, b)] = bool(not np.any((t & 0x7F) == 0x7F))
        g2['block%d_pad_bytes_recorded' % b] = cov['pad_bytes']
    g2['block_tensor_contents_all_distinct'] = len(set(tensor_hashes.values())) == 4
    pad = {b: g2.pop('block%d_pad_bytes_recorded' % b) for b in BLOCKS}
    g2['pad_bytes_match_sm75_recorded_4h'] = pad == {10: 90112, 11: 40960, 12: 40960, 13: 0}
    gates['G2_weights_canvas_layouts'] = dict(detail=g2, block_tensor_sha256=tensor_hashes, pad_bytes=pad, passed=all(g2.values()))
    ctrl = json.loads((D24 / 'controls/sm75-b10-b13-control/result.json').read_text())
    gates['G3_sm75_positive_control'] = dict(n_blocks=len(ctrl['results']), passed=ctrl['all_passed'] and len(ctrl['results']) == 4)
    g4 = {k: P19.read_capable(v) == want for k, v, want in (('rb', 'rb', True), ('rb+', 'rb+', True), ('wb', 'wb', False),
                                                            ('O_RDONLY', os.O_RDONLY, True), ('O_RDWR', os.O_RDWR, True), ('O_WRONLY', os.O_WRONLY, False))}
    gates['G4_audit_classifier'] = dict(detail=g4, passed=all(g4.values()))
    passed = all(g['passed'] for g in gates.values())
    finish(out, dict(stage='gates', started_utc=started, freeze_sha256=sha_file(FREEZE), gates=gates, all_gates_passed=passed))
    print(json.dumps({k: (v['passed'], v.get('detail')) for k, v in gates.items()}, indent=1, default=str))
    if not passed:
        raise SystemExit('GATES FAILED')


def stage_block(b, sample, layout='tin128', tail_seed='published_qz', subdir=None):
    fz = verify_freeze()
    gm = json.loads((D24 / 'gates/manifest.json').read_text())
    if not gm['all_gates_passed'] or gm['invalid']:
        raise SystemExit('gates not passed')
    subdir = subdir or ('B%d' % b)
    out = D24 / 'predictions' / sample / subdir
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    i = json.loads((D24 / 'interfaces' / sample / 'interface.json').read_text())['blocks'][str(b)]
    be = backend(b, tables, tail_seed)
    be.validate_invocation(block_index=b, head_count=4, origin_xy=i['origin_xy'], grid_xy=i['grid_xyz'][:2], shape=(40, 40, 128))
    raw = (D24 / 'interfaces' / sample / 'inputs' / ('B%d-X.bin' % b)).read_bytes()
    x = B9.tin128_decode(raw) if layout == 'tin128' else B9.inpview128_decode(raw)
    t = time.perf_counter()
    r = be.run_codes(x)
    secs = time.perf_counter() - t
    yb = B9.tin128_encode(r['Y'])
    name = 'B%d-Y.tin128' % b
    (out / name).write_bytes(yb)
    finish(out, dict(stage='B%d' % b, sample=sample, subdir=subdir, input_layout=layout, tail_seed=tail_seed,
                     teacher_forced_input='native B%d X (%s)' % (b, layout), started_utc=started, predictions_written_utc=utc(),
                     freeze_sha256=fz, input_sha256=shab(raw), backend=be.describe(), seconds=secs,
                     predictions={name: dict(sha256=shab(yb), bytes=len(yb))}, prediction_equals_input=yb == raw,
                     stats=dict(Y=P13.code_stats(r['Y']), Y_half=P13.half_stats(r['Y_half'])),
                     canvas_hw=r['canvas_hw'], n_windows=r['n_windows'], native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='B%d' % b, sample=sample, subdir=subdir, sha=shab(yb), seconds=secs, canvas=r['canvas_hw'], windows=r['n_windows']), indent=1))


def stage_chain(sample):
    fz = verify_freeze()
    for b in BLOCKS:
        c = D24 / 'comparisons' / sample / ('B%d.json' % b)
        if not c.exists() or json.loads(c.read_text())['verdict'] != 'MATCH':
            raise SystemExit('chain refused: %s not MATCH' % c)
    out = D24 / 'predictions' / sample / 'chain'
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    up = P13.load_unpacker()
    raw1 = (D13 / 'interfaces' / sample / 'inputs/B1-X.inpview').read_bytes()
    t = time.perf_counter()
    files, log = chain_b1_b9(up, tables, raw1)
    carried = files['B9-Y.tin128']
    for b in BLOCKS:
        x = B9.tin128_decode(carried)
        r = backend(b, tables).run_codes(x)
        nxt = B9.tin128_encode(r['Y'])
        files['B%d-Y.tin128' % b] = nxt
        log.append(dict(step='B%d' % b, input_sha256=shab(carried), output_sha256=shab(nxt), origin_xy=ORIGINS[b], canvas_hw=r['canvas_hw']))
        carried = nxt
    secs = time.perf_counter() - t
    for n, bb in files.items():
        (out / n).write_bytes(bb)
    leaked = [p for p in OPENED if p.endswith(NATIVE_X_FORBIDDEN_IN_CHAIN)]
    if leaked:
        raise SystemExit('INVALID: chain opened native intermediate X %s' % leaked)
    (out / 'chain-log.json').write_text(json.dumps(log, indent=1))
    finish(out, dict(stage='chain', sample=sample, only_native_input='D13 interfaces/%s/inputs/B1-X.inpview' % sample,
                     started_utc=started, predictions_written_utc=utc(), freeze_sha256=fz, input_sha256=shab(raw1), seconds=secs,
                     predictions={n: dict(sha256=shab(v), bytes=len(v)) for n, v in files.items()},
                     native_intermediate_X_opened=False, native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='chain', sample=sample, seconds=secs, n=len(files), predictions={n: shab(v) for n, v in files.items()}), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gates', 'B10', 'B11', 'B12', 'B13', 'chain', 'diag'])
    ap.add_argument('--sample', choices=SAMPLES)
    ap.add_argument('--block', type=int, choices=BLOCKS)
    ap.add_argument('--which', choices=['DIAG1', 'DIAG2'])
    a = ap.parse_args()
    if a.stage == 'gates':
        stage_gates()
    elif a.stage == 'chain':
        stage_chain(a.sample)
    elif a.stage == 'diag':
        if a.which == 'DIAG1':
            stage_block(a.block, a.sample, layout='inpview128', subdir='B%d-DIAG1-input-inpview128' % a.block)
        else:
            stage_block(a.block, a.sample, tail_seed='unquantised_z', subdir='B%d-DIAG2-tail-seed-unquantised-z' % a.block)
    else:
        stage_block(int(a.stage[1:]), a.sample)


if __name__ == '__main__':
    main()
