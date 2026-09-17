"""Launch100 (B39) CPU predictor (PREDICTION stage).

Reads ONLY: its frozen inputs (main.bin, skip.bin), the block39 weight tensor bytes, the frozen executable map, the reused
arithmetic modules (L59 l59_lane.py, mlxdlss.sm120_b1) and the Python runtime.  Never opens a target, checkpoint,
native capture, identity record, comparison, D32/D34 directory or earlier prediction: an audit hook installed before
any other import raises on such a path and the run exits 2 without sealing.

    python -B predict_b39.py --sample-id c1-C-img-full --freeze freeze/<name>.json --out-dir predictions/<run>/<sample>

Exit: 0 prediction sealed; 2 INVALID (freeze/identity/audit failure, nothing sealed); 3 OUT_OF_DOMAIN (named refusal).

Candidate (frozen before any B39 numeric run; every step from the executable map, no free parameter):
  chain  D = 8 x QMMA.16832 K=32 (l59_lane slot layout, W27_trunc, C=D layout), seed +0, A/B from the map
  reduce red[h] = add16(add16(D_z0, D_z1), D_z2)                         (store, reduce-add, reduce-add in z order)
  merge  out[b] = sat_e4m3( fma16( e4m3(skip[s]), sin16(w), add16(red[r], D_z3[src lane]) ) )      (revision r1; r2 was
         sat_e4m3(add16(add16(red, D_z3), mul16(e4m3(skip), sin16))) and is kept as a DIFF record)
  counters c[0x3a0][cta] = 3 (last release, z order), c[0x3a8][cta] = 0
add16/fma16 = float64 then binary16 RNE (G1-calibrated HADD2 / HFMA2 register forms); sat_e4m3 = sm120_b1.quantise_codes
(self-check only).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
AUDIT = []
PHASE = ['startup']
FORBIDDEN_MARKERS = ('\\targets\\', 'checkpoint', '\\native\\', '\\identity\\', 'comparison', '\\acceptance', 'negative',
                     'parity-astra-d32', 'parity-astra-d34', 'rental-transfer', 'oracle', '.zip', 'invalid-runs')
ALLOWED_WRITE_DIR = [None]
FORBIDDEN_ATTEMPTS = []
_PROJECT_LOW = os.path.normcase(os.path.abspath(str(PROJECT))).lower()
_VENV_LOW = os.path.normcase(os.path.abspath(str(PROJECT / '.venv'))).lower()
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def _norm(p):
    try:
        return os.path.normcase(os.path.abspath(os.path.normpath(os.fsdecode(p)))).lower()
    except Exception:
        return repr(p).lower()


def _inside(p, root):
    return p == root or p.startswith(root + os.sep)


def _is_write(mode, flags):
    m = mode if isinstance(mode, str) else ''
    f = flags if isinstance(flags, int) else 0
    return any(c in m for c in 'wax+') or bool(f & _WRITE_FLAGS)


def _hook(event, args):
    if event == 'open':
        path, mode, flags = args[0], args[1], args[2]
        if isinstance(path, int):
            return
        p = _norm(path)
        w = _is_write(mode, flags)
        AUDIT.append((p, str(mode), int(flags) if isinstance(flags, int) else -1, PHASE[0], w))
        out_dir = ALLOWED_WRITE_DIR[0]
        if out_dir is not None and _inside(p, out_dir):
            return
        if w:
            FORBIDDEN_ATTEMPTS.append('write:' + p)
            raise PermissionError('predictor may not write outside its output directory: %s' % p)
        inside = _inside(p, _PROJECT_LOW) and not _inside(p, _VENV_LOW)
        if inside and any(m in p[len(_PROJECT_LOW):] for m in FORBIDDEN_MARKERS):
            FORBIDDEN_ATTEMPTS.append(p)
            raise PermissionError('predictor may not open %s' % p)
    elif event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.spawn', 'os.startfile', 'os.posix_spawn'):
        FORBIDDEN_ATTEMPTS.append(event)
        raise PermissionError('predictor may not start processes')


sys.addaudithook(_hook)

import argparse, hashlib, json, uuid  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

SAMPLES = ('c1-C-img-full',)
FREEZE_FORMAT = 'opus-b39-pre-run-freeze/1'
L59_LANE = 'experiments/parity-astra-opus-b31-l59-20260913/l59_lane.py'
NEED_ROUND = ['predict_b39.py', 'maps/b39-map.npz', 'maps/b39-map.json', 'weights/block39.layer0.layer.bin']
NEED_PROJECT = [L59_LANE, 'mlxdlss/__init__.py', 'mlxdlss/sm120_b1.py', 'mlxdlss/model.py']
OUTLETS = ('out', 'red', 'ctr', 'ctr2')


class OutOfDomain(Exception):
    pass


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def unique(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError('duplicate JSON key %r' % k)
        d[k] = v
    return d


def strict_json(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)
    if not isinstance(obj, dict):
        raise RuntimeError('JSON_ROOT_NOT_OBJECT:%s' % path)
    return obj


def classify(path, frozen, out_dir):
    if path in frozen:
        return 'frozen:' + frozen[path]
    if out_dir and _inside(path, out_dir):
        return 'own_output'
    if (_inside(path, _PROJECT_LOW) and not _inside(path, _VENV_LOW) and '\\__pycache__\\' in path and path.endswith('.pyc')
            and not os.path.exists(path)):
        return 'bytecode_probe_absent'
    for pref, name in ((_norm(sys.base_prefix), 'python_base'), (_norm(sys.prefix), 'python_venv'),
                       (_norm(os.environ.get('SystemRoot', r'C:\Windows')), 'system')):
        if _inside(path, pref):
            return name
    return 'other'


def add16(a, b):
    with np.errstate(over='ignore', invalid='ignore'):
        r = (np.asarray(a, np.float16).astype(np.float64) + np.asarray(b, np.float16).astype(np.float64)).astype(np.float16)
    if not np.isfinite(r).all():
        raise OutOfDomain('non-finite add16 result')
    return r


def mul16(a, b):
    with np.errstate(over='ignore', invalid='ignore'):
        r = (np.asarray(a, np.float16).astype(np.float64) * np.asarray(b, np.float16).astype(np.float64)).astype(np.float16)
    if not np.isfinite(r).all():
        raise OutOfDomain('non-finite mul16 result')
    return r


def fma16(a, b, c):
    with np.errstate(over='ignore', invalid='ignore'):
        r = (np.asarray(a, np.float16).astype(np.float64) * np.asarray(b, np.float16).astype(np.float64)
             + np.asarray(c, np.float16).astype(np.float64)).astype(np.float16)
    if not np.isfinite(r).all():
        raise OutOfDomain('non-finite fma16 result')
    return r


def compute(LANE, CORE, main_b, skip_b, W, mp, mj, stats):
    A_src, B_off = mp['A_src'].astype(np.int64), mp['B_off'].astype(np.int64)
    OUTMAP, RED_OFF, REDMAP = mp['OUTMAP'].astype(np.int64), mp['RED_OFF'].astype(np.int64), mp['REDMAP'].astype(np.int64)
    if (A_src < 0).any() or (B_off < 0).any() or (OUTMAP[:, 0] < 0).any():
        raise RuntimeError('MAP_HAS_UNRESOLVED_OR_ZERO_OPERANDS')
    lead = A_src.shape[:5]
    NS = A_src.shape[5]
    C = np.zeros(lead + (16, 8), np.float16)
    for s in range(NS):
        a_l = main_b[A_src[..., s, :][..., None] + np.arange(16)]
        b_l = W[B_off[..., s, :][..., None] + np.arange(8)]
        C = LANE.qmma_step(LANE.a_matrix(a_l), LANE.b_matrix(b_l), C, stats=stats)
    D = LANE.d_to_lanes(C).astype(np.float16).reshape((-1,) + lead[4:] + (32, 4))      # [thread, chain, lane, slot]
    stats['threads'] = int(D.shape[0])
    v = [D[REDMAP[:, z, 0], REDMAP[:, z, 1], REDMAP[:, z, 2], REDMAP[:, z, 3]] for z in range(3)]
    red = add16(add16(v[0], v[1]), v[2])
    red_lo = int(RED_OFF.min())
    red_bytes = np.zeros(int(RED_OFF.max()) + 2 - red_lo, np.uint8)
    rb = red.astype('<f2').view(np.uint8).reshape(-1, 2)
    red_bytes[RED_OFF - red_lo] = rb[:, 0]
    red_bytes[RED_OFF - red_lo + 1] = rb[:, 1]
    red_cover = np.zeros(red_bytes.size, np.int32)
    np.add.at(red_cover, RED_OFF - red_lo, 1)
    np.add.at(red_cover, RED_OFF - red_lo + 1, 1)
    if not (red_cover == 1).all():
        raise RuntimeError('RED_COVERAGE_NOT_EXACTLY_ONCE')
    ridx = np.searchsorted(RED_OFF, OUTMAP[:, 0])
    if not np.array_equal(RED_OFF[ridx], OUTMAP[:, 0]):
        raise RuntimeError('OUT_RED_OFFSET_NOT_IN_RED_SET')
    d3 = D[OUTMAP[:, 1], OUTMAP[:, 2], OUTMAP[:, 3], OUTMAP[:, 4]]
    if (OUTMAP[:, 5] < 0).any():
        raise RuntimeError('CONST_SKIP_NOT_MODELLED_IN_THIS_CANDIDATE')
    skip_dec = CORE.decode_codes(skip_b[OUTMAP[:, 5]]).astype(np.float16)
    if not np.array_equal(skip_dec.astype(np.float64), CORE.decode_codes(skip_b[OUTMAP[:, 5]]).astype(np.float64)):
        raise OutOfDomain('E4M3 decode not exact in binary16')
    sin = np.stack([W[OUTMAP[:, 6]], W[OUTMAP[:, 6] + 1]], 1).copy().view('<f2').reshape(-1)
    # revision r1 (R1_HFMA2_FUSED_MERGE): the executed SASS fuses the PTX mul.f16x2 + add.f16x2 pair into HFMA2
    # (pc 0x3400 `HFMA2 R96, R96, R21, R99` = e4m3(skip) * sin + gather, one binary16 rounding; 128 HFMA2 / 0 HMUL2 / 32 HADD2).
    # r2 used add16(g, mul16(skip, sin)) and gave DIFF 232/73728 on out only (kept in predictions/r2).
    merged = fma16(skip_dec, sin, add16(red[ridx], d3))
    out = CORE.quantise_codes(merged.astype(np.float32))
    if out.size != 73728:
        raise RuntimeError('OUT_SIZE')
    cw = mj['coverage']['counter_writes']
    ctr_vals = {k: sorted(v) for k, v in cw.items()}
    ctr = np.zeros(max(int(k.split('+')[1]) for k in cw if k.startswith('ctr+')) + 4, np.uint8)
    ctr2 = np.zeros(max(int(k.split('+')[1]) for k in cw if k.startswith('ctr2+')) + 4, np.uint8)
    for k, vals in ctr_vals.items():
        name, off = k.split('+')
        off = int(off)
        if name == 'ctr':
            if vals != [0, 1, 2, 3]:
                raise RuntimeError('COUNTER_SEQUENCE_UNEXPECTED')
            ctr[off:off + 4] = np.frombuffer(np.int32(3).tobytes(), np.uint8)
        else:
            if vals != [0]:
                raise RuntimeError('COUNTER2_UNEXPECTED')
    stats['merge_abs_max'] = float(np.abs(merged.astype(np.float64)).max())
    stats['red_abs_max'] = float(np.abs(red.astype(np.float64)).max())
    stats['saturated_bytes'] = int((np.abs(merged.astype(np.float64)) > 448).sum())
    return dict(out=out.astype(np.uint8).tobytes(), red=red_bytes.tobytes(), ctr=ctr.tobytes(), ctr2=ctr2.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-id', required=True)
    ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--test-open', action='append', default=[],
                    help='negative test only: MODE::PATH attempted after the freeze check; MODE in r, rb, r+, w, a, '
                         'osrdonly, osrdwr, oswronly, oscreat')
    a = ap.parse_args()
    run_id = str(uuid.uuid4())
    t0 = datetime.now(timezone.utc)
    out_dir = a.out_dir.resolve()
    ALLOWED_WRITE_DIR[0] = _norm(out_dir)
    failures, status, code = [], 'PREDICTION_SEALED', 0
    frozen_lookup, stats, entries = {}, {}, {}
    need = []
    try:
        PHASE[0] = 'freeze_check'
        if a.sample_id not in SAMPLES:
            raise RuntimeError('UNKNOWN_SAMPLE_ID')
        if out_dir.exists() and any(out_dir.iterdir()):
            raise RuntimeError('OUTPUT_DIRECTORY_NOT_EMPTY')
        stale = [str(c) for d in (PROJECT / 'mlxdlss', HERE, (PROJECT / L59_LANE).parent) for c in d.glob('__pycache__')]
        if stale:
            raise RuntimeError('PYCACHE_PRESENT_BESIDE_FROZEN_SOURCES:' + ';'.join(stale))
        freeze = strict_json(a.freeze)
        if freeze.get('format') != FREEZE_FORMAT or not isinstance(freeze.get('files'), list):
            raise RuntimeError('FREEZE_SCHEMA_INVALID')
        receipt_path = a.freeze.with_name(a.freeze.stem + '.receipt.json')
        frec = strict_json(receipt_path)
        if frec.get('freeze_sha256') != fsha(a.freeze):
            raise RuntimeError('FREEZE_RECEIPT_MISMATCH')
        for e in freeze['files']:
            if not isinstance(e, dict) or set(e) != {'base', 'path', 'bytes', 'sha256'} or e['base'] not in ('round', 'project') \
                    or not isinstance(e['path'], str) or type(e['bytes']) is not int or not isinstance(e['sha256'], str):
                raise RuntimeError('FREEZE_ENTRY_SCHEMA')
            if e['path'] in entries:
                raise RuntimeError('FREEZE_DUPLICATE_ENTRY')
            p = ((HERE if e['base'] == 'round' else PROJECT) / e['path']).resolve()
            entries[e['path']] = (p, e)
        need = NEED_ROUND + NEED_PROJECT + ['inputs/%s/main.bin' % a.sample_id, 'inputs/%s/skip.bin' % a.sample_id]
        for rel in need:
            if rel not in entries:
                raise RuntimeError('FREEZE_MISSING_REQUIRED:' + rel)
            p, e = entries[rel]
            if not p.is_file() or p.stat().st_size != e['bytes'] or fsha(p) != e['sha256']:
                raise RuntimeError('FREEZE_CONTENT_MISMATCH:' + rel)
        if fsha(Path(__file__)) != entries['predict_b39.py'][1]['sha256']:
            raise RuntimeError('RUNNING_PREDICTOR_NOT_THE_FROZEN_ONE')
        frozen_lookup = {_norm(p): rel for rel, (p, _) in entries.items()}
        frozen_lookup[_norm(a.freeze)] = 'FREEZE_MANIFEST'
        frozen_lookup[_norm(receipt_path)] = 'FREEZE_RECEIPT'
        for spec in a.test_open:
            mode, _, tpath = spec.partition('::')
            osflags = dict(osrdonly=os.O_RDONLY, osrdwr=os.O_RDWR, oswronly=os.O_WRONLY, oscreat=os.O_WRONLY | os.O_CREAT)
            if mode in osflags:
                fd = os.open(tpath, osflags[mode] | getattr(os, 'O_BINARY', 0))
                os.close(fd)
            elif mode in ('r', 'rb', 'r+', 'w', 'a'):
                open(tpath, mode + ('b' if 'b' not in mode else '')).close()
            else:
                raise RuntimeError('TEST_OPEN_MODE_UNKNOWN:%s' % mode)

        PHASE[0] = 'import'
        sys.path.insert(0, str(PROJECT))
        sys.path.insert(0, str((PROJECT / L59_LANE).parent))
        import l59_lane as LANE  # noqa: E402
        from mlxdlss import sm120_b1 as CORE  # noqa: E402

        PHASE[0] = 'compute'
        main_b = np.frombuffer(entries['inputs/%s/main.bin' % a.sample_id][0].read_bytes(), np.uint8)
        skip_b = np.frombuffer(entries['inputs/%s/skip.bin' % a.sample_id][0].read_bytes(), np.uint8)
        W = np.frombuffer(entries['weights/block39.layer0.layer.bin'][0].read_bytes(), np.uint8)
        mj = strict_json(entries['maps/b39-map.json'][0])
        if mj.get('status') != 'PASS_EXECUTABLE_MAP' or len(main_b) != 65536 or len(skip_b) != 73728 or len(W) != 525312:
            raise RuntimeError('MAP_OR_INPUT_INVALID')
        with np.load(entries['maps/b39-map.npz'][0]) as m:
            mp = {k: m[k] for k in ('A_src', 'B_off', 'OUTMAP', 'RED_OFF', 'REDMAP')}
        try:
            outs = compute(LANE, CORE, main_b, skip_b, W, mp, mj, stats)
        except LANE.OutOfDomain as exc:
            raise OutOfDomain(str(exc))
        loaded = sorted({_norm(mod.__file__) for mod in list(sys.modules.values())
                         if isinstance(getattr(mod, '__file__', None), str) and os.path.isabs(mod.__file__)
                         and os.path.isfile(mod.__file__) and _inside(_norm(mod.__file__), _PROJECT_LOW)})
        unfrozen = [p for p in loaded if p not in frozen_lookup and not _inside(p, _VENV_LOW)]
        if unfrozen:
            raise RuntimeError('UNFROZEN_PROJECT_MODULE_LOADED:' + ';'.join(unfrozen))
        PHASE[0] = 'finalize'
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in OUTLETS:
            (out_dir / ('%s.bin.partial' % k)).write_bytes(outs[k])
    except OutOfDomain as exc:
        status, code = 'OUT_OF_DOMAIN', 3
        failures.append(str(exc))
    except PermissionError as exc:
        status, code = 'INVALID', 2
        failures.append('FORBIDDEN_ACCESS_BLOCKED:%s' % exc)
    except Exception as exc:  # noqa: BLE001
        status, code = 'INVALID', 2
        failures.append('%s:%s' % (type(exc).__name__, exc))
    t1 = datetime.now(timezone.utc)
    PHASE[0] = 'finalize'
    events = {}
    for row in list(AUDIT):
        events[row] = events.get(row, 0) + 1
    ev_rows = [dict(path=p, mode=mode, flags=flags, phase=ph, write=w, count=n, cls=classify(p, frozen_lookup, ALLOWED_WRITE_DIR[0]))
               for (p, mode, flags, ph, w), n in sorted(events.items())]
    others = [r for r in ev_rows if r['cls'] == 'other']
    if others and code == 0:
        status, code = 'INVALID', 2
        failures.append('UNCLASSIFIED_OPEN:%d' % len(others))
    if FORBIDDEN_ATTEMPTS and code == 0:
        status, code = 'INVALID', 2
        failures.append('FORBIDDEN_ATTEMPTS')
    audit = dict(format='opus-b39-predict-audit/1', run_id=run_id, sample_id=a.sample_id, exit_status=status, exit_code=code,
                 started_utc=t0.isoformat(), ended_utc=t1.isoformat(), pid=os.getpid(), hook='sys.addaudithook before numpy/torch import',
                 forbidden_markers=list(FORBIDDEN_MARKERS), forbidden_attempts=list(FORBIDDEN_ATTEMPTS),
                 open_events=ev_rows, other_opens=others, failures=failures,
                 blind_spots=['opens by native extension loaders that bypass the Python open audit event are not recorded',
                              'memory-mapped reads after open are not recorded separately'])
    audit_dir = out_dir if code == 0 else out_dir / ('failed-' + run_id[:8])
    audit_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for k in OUTLETS:
        partial = out_dir / ('%s.bin.partial' % k)
        if code != 0 and partial.exists():
            os.replace(partial, audit_dir / ('unsealed-%s.bin' % k))
            moved.append(k)
    audit['unsealed_outputs_moved_to_failed_dir'] = moved
    audit_path = audit_dir / 'predict-audit.json'
    audit_path.write_text(json.dumps(audit, indent=1) + '\n', encoding='utf-8')
    if code == 0:
        preds = {}
        for k in OUTLETS:
            pred = out_dir / ('%s.bin' % k)
            if pred.exists():
                raise SystemExit('refusing to overwrite an existing prediction %s' % pred)
            os.replace(out_dir / ('%s.bin.partial' % k), pred)
            preds[k] = dict(file=pred.name, sha256=fsha(pred), bytes=pred.stat().st_size)
        rec = dict(format='opus-b39-prediction/1', valid=True, match=None, status=status, run_id=run_id, sample_id=a.sample_id,
                   function='cc_dec_input_upsample_1024_512_tilesync_fp8', launch=100, created_utc=t1.isoformat(), native_target_read=False,
                   freeze=str(a.freeze.resolve()), freeze_sha256=fsha(a.freeze),
                   inputs={rel: dict(path=str(entries[rel][0]), sha256=entries[rel][1]['sha256']) for rel in need},
                   outlets=preds, audit=str(audit_path), audit_sha256=fsha(audit_path),
                   domain_stats={k: v for k, v in stats.items() if k != 'log2_addend_hist'}, log2_addend_hist=stats.get('log2_addend_hist'),
                   environment=dict(python=sys.version, numpy=np.__version__, torch=sys.modules['torch'].__version__),
                   what_a_false_pass_would_look_like={
                       'predictor_read_target': 'blocked by the audit hook and recorded; the acceptance driver re-checks the audit',
                       'model_tuned_on_target': 'not possible inside this process; across revisions only freeze timestamps show it'})
        (out_dir / 'prediction.json').write_text(json.dumps(rec, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=status, exit=code, run_id=run_id, failures=failures,
                          domain_stats={k: v for k, v in stats.items() if k != 'log2_addend_hist'}, audit=str(audit_path)), indent=1))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
