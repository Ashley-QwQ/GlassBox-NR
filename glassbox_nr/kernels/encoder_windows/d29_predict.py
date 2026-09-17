"""D29 CPU prediction process.

Commands: single --sample S --block B16..B21, chain --sample S, uploaded.
The process is deliberately unable to open native answers, rental captures,
historical predictions, or another intermediate block input.
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
ROUND = HERE.parent
ROOT = ROUND.parents[1]
E = ROOT / 'experiments'
D13 = E / 'parity-astra-d13-b234-sm120-cpu-chain-20260912'
D19 = E / 'parity-astra-d19-b67-sm120-cpu-chain-20260912'
D20 = E / 'parity-astra-d20-b0-reset-cpu-20260912'
DIMG = E / 'parity-astra-image-to-b15-20260913'
D21 = E / 'parity-astra-d21-b8-downsample-cpu-20260912'
D22 = E / 'parity-astra-d22-b9-4h-sm120-cpu-chain-20260912'
D25 = E / 'parity-astra-d25-b14-downsample-cpu-20260912'
D29 = ROUND
SAMPLES = ('c1-C-img-full',)
sys.path.insert(0, str(HERE))
import d25_predict as P25  # noqa: E402
import d29_backend as N  # noqa: E402
import d20_backend as B0  # noqa: E402
import d21_backend as B8  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
torch.set_num_threads(2)
P24, S, B9, B14 = P25.P24, P25.S, P25.B9, P25.B14
P22, P19, P13, B5 = P24.P22, P24.P19, P24.P13, P24.B5

OPENED = {}
HASHING = [False]
ALLOWED_READS = set()

def _audit(ev, args):
    if ev != 'open' or HASHING[0] or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        p = str(Path(os.fsdecode(args[0])).resolve()).replace('\\', '/').lower()
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else None
        mode = args[1] if len(args) > 1 and args[1] is not None else flags
        OPENED.setdefault(p, []).append(mode)
        writing = (isinstance(mode, str) and ('r' not in mode and '+' not in mode)) or (isinstance(mode, int) and (mode & os.O_ACCMODE) == os.O_WRONLY)
        if writing:
            return
        protected_suffix = p.endswith(('b1-x.inpview', 'b2-x.tin', 'b3-x.tin', 'b4-x.tin', 'b5-x.tin64',
                                 'b6-x.tin64', 'b7-x.tin64', 'b8-x.tin64', 'b9-x.bin', 'b10-x.bin',
                                 'b11-x.bin', 'b12-x.bin', 'b13-x.bin', 'b14-x.bin', 'b15-x.bin',
                                 'b16-x.tin256', 'b17-x.tin256', 'b18-x.tin256', 'b19-x.tin256',
                                 'b20-x.tin256', 'b21-x.tin256'))
        forbidden = ('/oracle/' in p or '/rental-transfer/' in p or '/predictions/' in p or
                     (protected_suffix and p not in ALLOWED_READS))
        if forbidden:
            raise RuntimeError('Forbidden prediction-time read: ' + p)
    except RuntimeError:
        raise
    except Exception:
        pass

sys.addaudithook(_audit)

def sha_file(p):
    HASHING[0] = True
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    finally:
        HASHING[0] = False

def shab(b):
    return hashlib.sha256(b).hexdigest()

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

FREEZE = D29 / 'dependency-freeze.json'

def verify_freeze():
    fz = json.loads(FREEZE.read_text(encoding='utf-8'))
    drift = {k: v for k, v in fz['files'].items() if sha_file(ROOT / k) != v}
    if drift:
        raise SystemExit('INVALID: freeze drift %s' % drift)
    return sha_file(FREEZE)

def finish(out, manifest):
    manifest.update(opened_files_count=len(OPENED), opened_files=sorted(OPENED),
                    denied_opens=[], invalid=False, finished_utc=utc())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str), encoding='utf-8')
    (out / 'opened.json').write_text(json.dumps({k: [str(x) for x in v] for k, v in sorted(OPENED.items())}, indent=1), encoding='utf-8')

def logical(block):
    return P19.logical(block)

def backend(block, origin, tables):
    return N.BranchedWindowSm120N(block, 8, 20, tuple(origin), logical(block), tables)

def block_info(sample, block):
    return json.loads((D29 / 'interfaces' / sample / 'interface.json').read_text(encoding='utf-8'))['blocks'][block]

def stage_single(sample, block):
    if sample not in SAMPLES or block not in ('B16', 'B17', 'B18', 'B19', 'B20', 'B21'):
        raise SystemExit('invalid sample/block')
    freeze = verify_freeze()
    out = D29 / 'predictions' / sample / block
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    info = block_info(sample, block)
    inp = D29 / 'interfaces' / sample / info['input_file']
    ALLOWED_READS.add(str(inp.resolve()).replace('\\', '/').lower())
    raw = inp.read_bytes()
    tables = S.Sm120Tables.load()
    be = backend(info['block'], info['origin_xy'], tables)
    be.validate_invocation(block_index=info['block'], head_count=8, origin_xy=info['origin_xy'], grid_xy=info['grid_xyz'][:2], shape=(20, 20, 256))
    t = time.perf_counter()
    result = be.run_codes(N.tin256_decode(raw))
    seconds = time.perf_counter() - t
    pred = N.tin256_encode(result['Y'])
    name = '%s-Y.tin256' % block
    (out / name).write_bytes(pred)
    finish(out, dict(stage='single', sample=sample, block=block, started_utc=started, predictions_written_utc=utc(),
        freeze_sha256=freeze, input_file=str(inp.relative_to(ROOT)).replace('\\', '/'), input_sha256=shab(raw),
        backend=be.describe(), seconds=seconds, predictions={name: dict(sha256=shab(pred), bytes=len(pred))},
        native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='single', sample=sample, block=block, sha256=shab(pred), seconds=seconds), indent=1))

def run_b16_b21(carried, sample, tables, files, log):
    for n in range(16, 22):
        b = 'B%d' % n
        info = block_info(sample, b)
        raw_in = carried
        result = backend(n, info['origin_xy'], tables).run_codes(N.tin256_decode(carried))
        carried = N.tin256_encode(result['Y'])
        name = '%s-Y.tin256' % b
        files[name] = carried
        log.append(dict(step=b, input_sha256=shab(raw_in), output_sha256=shab(carried), origin_xy=info['origin_xy'], canvas_hw=result['canvas_hw']))
    return carried

def stage_chain(sample):
    if sample not in SAMPLES:
        raise SystemExit('invalid sample')
    freeze = verify_freeze()
    out = D29 / 'predictions' / sample / 'chain'
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    start_input = D13 / 'interfaces' / sample / 'inputs/B1-X.inpview'
    ALLOWED_READS.add(str(start_input.resolve()).replace('\\', '/').lower())
    raw1 = start_input.read_bytes()
    files, log = P25.chain_b1_b13(P13.load_unpacker(), tables, raw1)
    carried = files['B13-Y.tin128']
    r14 = B14.B14Sm120(logical(14), tables).run_codes(B9.tin128_decode(carried))
    files.update(P25.b14_files(r14))
    log.append(dict(step='B14', input_sha256=shab(carried), full_sha256=shab(files['B14-full.tin128']), pooled_sha256=shab(files['B14-pooled.inpview256'])))
    r15 = backend(15, (0, 0), tables).run_codes(N.inpview256_decode(files['B14-pooled.inpview256']))
    carried = N.tin256_encode(r15['Y'])
    files['B15-Y.tin256'] = carried
    log.append(dict(step='B15', input_sha256=shab(files['B14-pooled.inpview256']), output_sha256=shab(carried), canvas_hw=r15['canvas_hw']))
    run_b16_b21(carried, sample, tables, files, log)
    required = 24
    if len(files) != required:
        raise SystemExit('INVALID: chain output count %d != %d' % (len(files), required))
    for name, data in files.items():
        (out / name).write_bytes(data)
    (out / 'chain-log.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
    finish(out, dict(stage='chain', sample=sample, only_native_input=str((D13 / 'interfaces' / sample / 'inputs/B1-X.inpview').relative_to(ROOT)).replace('\\', '/'),
        started_utc=started, predictions_written_utc=utc(), freeze_sha256=freeze, input_sha256=shab(raw1),
        seconds=None, predictions={n: dict(sha256=shab(v), bytes=len(v)) for n, v in files.items()},
        native_intermediate_X_opened=False, native_intermediate_used=False, gpu_used=False, n_predictions=len(files)))
    print(json.dumps(dict(stage='chain', sample=sample, n_predictions=len(files)), indent=1))

def stage_uploaded():
    freeze = verify_freeze()
    out = D29 / 'predictions' / 'c1-C-img-full' / 'uploaded'
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    tables = S.Sm120Tables.load()
    up = DIMG / 'inputs'
    # This is the controlled D20 uploaded-colour entry, not a saved feature input.
    from safetensors.numpy import load_file
    all_weights = load_file(str(ROOT / 'weights/dlssnr-weights-logical.safetensors'))
    colour = up / 'colour.rgba16'; params = up / 'pre.params'
    ALLOWED_READS.update(str(p.resolve()).replace('\\', '/').lower() for p in (colour, params))
    b0 = B0.B0ResetCandidate({k: v for k, v in all_weights.items() if k.startswith('block0.layer0.')}, tables).run(
        colour.read_bytes(), params.read_bytes())
    files = {'b0_full.tin': b0['full_tin'], 'b0_pooled.inpview': b0['pooled_inpview']}
    files.update(P19.chain_b1_b5(P13.load_unpacker(), tables, b0['pooled_inpview']))
    carried = files['B5-Y.tin64']
    for block in (6, 7):
        r = P19.BW.BranchedWindow2hSm120(block, logical(block), tables).run_codes(P19.B5.tin64_decode(carried))
        carried = P19.B5.tin64_encode(r['Y']); files['B%d-Y.tin64' % block] = carried
    r8 = B8.B8Sm120(logical(8), tables).run_codes(P19.B5.tin64_decode(carried))
    files['B8-full.tin64'] = P19.B5.tin64_encode(r8['full']); files['B8-pooled.inpview128'] = B8.inpview128_encode(r8['pooled'])
    r9 = B9.BranchedWindowSm120(9, 4, 40, (0, 0), logical(9), tables).run_codes(r8['pooled'])
    carried = B9.tin128_encode(r9['Y']); files['B9-Y.tin128'] = carried
    for bi, origin in ((10, (-4, -4)), (11, (-4, 0)), (12, (0, -4)), (13, (0, 0))):
        rb = B9.BranchedWindowSm120(bi, 4, 40, origin, logical(bi), tables).run_codes(B9.tin128_decode(carried))
        carried = B9.tin128_encode(rb['Y']); files['B%d-Y.tin128' % bi] = carried
    r14 = B14.B14Sm120(logical(14), tables).run_codes(B9.tin128_decode(carried))
    files['B14-full.tin128'] = B9.tin128_encode(r14['full']); files['B14-pooled.inpview256'] = B14.inpview_encode(r14['pooled'])
    r15 = backend(15, (0, 0), tables).run_codes(N.inpview256_decode(files['B14-pooled.inpview256']))
    carried = N.tin256_encode(r15['Y']); files['B15-Y.tin256'] = carried
    log = [dict(step='B15', input_sha256=shab(files['B14-pooled.inpview256']), output_sha256=shab(carried), canvas_hw=r15['canvas_hw'])]
    run_b16_b21(carried, 'c1-C-img-full', tables, files, log)
    if len(files) != 26:
        raise SystemExit('INVALID: uploaded output count %d != 26' % len(files))
    for name, data in files.items(): (out / name).write_bytes(data)
    (out / 'chain-log.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
    finish(out, dict(stage='uploaded', sample='c1-C-img-full', scope='uploaded colour -> CPU B0..B21, reset counter0, 320x240',
        started_utc=started, predictions_written_utc=utc(), freeze_sha256=freeze,
        input_files=[str((up / x).relative_to(ROOT)).replace('\\', '/') for x in ('colour.rgba16', 'pre.params')],
        predictions={n: dict(sha256=shab(v), bytes=len(v)) for n, v in files.items()}, n_predictions=len(files),
        native_intermediate_used=False, gpu_used=False))
    print(json.dumps(dict(stage='uploaded', n_predictions=len(files)), indent=1))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='stage', required=True)
    s = sub.add_parser('single'); s.add_argument('--sample', required=True); s.add_argument('--block', required=True)
    c = sub.add_parser('chain'); c.add_argument('--sample', required=True)
    sub.add_parser('uploaded')
    a = ap.parse_args()
    if a.stage == 'single': stage_single(a.sample, a.block)
    elif a.stage == 'chain': stage_chain(a.sample)
    else: stage_uploaded()

if __name__ == '__main__':
    main()
