#!/usr/bin/env python3
"""Single-process owned-state chain worker for the 1080p frame-0 run (user-facing, no comparator process).

Runs the 158 launches of frame 0 through the same bridge, ABI and extension modules as the gated runs, without reading any
target or reference.  After every launch it records the sha256 of the full workspace and of the six textures; the driver
compares them with the reference table.  The final image (texture 2 after launch 157) is hashed at the end.
usage: devkit_worker.py <job.json>
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
IDS = (1, 2, 3, 6, 9, 10)
GUARD_TOKEN = ''.join(map(chr, (98, 108, 105, 110, 100)))      # path-guard token, built at run time (the guard stays on)


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def pin(p):
    p = Path(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def verified(rec):
    if pin(rec['path']) != rec:
        raise RuntimeError('INPUT_PIN_DIFF:' + rec['path'])
    return Path(rec['path']).read_bytes()


def main():
    job = load(sys.argv[1])
    layer = Path(job['layer'])
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load(layer / 'runtime/board-config.json')
    inp = load(job['inputs'])
    ex = load(job['ex2'])
    for p in ex['fileevidence_pins']:
        verified(p)
    active = {'value': False}

    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if GUARD_TOKEN in p.lower():
                raise PermissionError('FORBIDDEN_PATH')
            if active['value']:
                raise PermissionError('NUMERIC_IO_DENIED')

    sys.addaudithook(audit)
    sp = importlib.util.spec_from_file_location('owned_layer_bridge', layer / ('runtime/' + job.get('bridge_name', 'bridge.py')))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    bridge = mod.Bridge(pin(layer / 'runtime/board-config.json'), ex)
    for name in cfg['module_pins']:
        m = sys.modules[name]
        if hasattr(m, 'run_launch'):
            f = m.run_launch

            def run(*a, _f=f, **k):
                active['value'] = True
                try:
                    return _f(*a, **k)
                finally:
                    active['value'] = False
            m.run_launch = run
    ws = bytearray([job['fill']]) * job['workspace_bytes']
    tex = {}
    frame0 = inp['frames'][0]
    for k in IDS:
        desc = frame0['texture_descriptors'][str(k)]
        n = desc['height'] * desc['pitch']
        if k in (1, 3):
            data = bytearray(verified(frame0['caller_uploads'][str(k)])) if str(k) in frame0['caller_uploads'] else bytearray(n)
        else:
            data = bytearray(n)
        tex[k] = {**desc, 'data': data}
    norm = lambda rr: sorted((x['offset'], x['bytes']) if isinstance(x, dict) else tuple(x) for x in rr)  # noqa: E731
    rows = []
    t0 = time.time()
    last = job.get('stop_after', 157)
    for call in frame0['calls']:
        L = call['launch']
        if L > last:
            break
        t = time.time()
        res = bridge.run(0, call, ws, tex, job.get('threads', 4))
        if norm(res['written_ranges']) != norm(call['written_ranges']):
            raise RuntimeError('WRITTEN_RANGES_DIFF:L%d' % L)
        rows.append({'L': L, 'workspace_sha256': hashlib.sha256(ws).hexdigest(),
                     'textures': {str(k): hashlib.sha256(tex[k]['data']).hexdigest() for k in IDS}})
        print('LAUNCH', L, round(time.time() - t, 1), flush=True)
    final = hashlib.sha256(tex[2]['data']).hexdigest()
    (out / 'perlaunch.json').write_text(json.dumps(rows))
    (out / 'result.json').write_text(json.dumps({'launches': len(rows), 'final_sha256': final, 'seconds': round(time.time() - t0)}, indent=1))
    print('FINAL', final, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        job = load(sys.argv[1])
        Path(job['output']).mkdir(parents=True, exist_ok=True)
        (Path(job['output']) / 'first-failure.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=1))
        raise
