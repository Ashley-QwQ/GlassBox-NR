"""Which per-launch map may be used: a record counts only if it is VERIFIED, never because a file exists.

A record (maps/L<nn>/maps.json or maps/L<nn>/maps-aNN.json) is verified when:
  status == PASS_FAMILY_EQUAL and files is non-empty;
  every file in `files` exists and its sha256 equals file_sha256 (a half-written npz/json fails here);
  the npz loads and every key reads; the templates/wiring json parses;
  interp_sha256 equals the current interpreter copy and shim_sha256 equals the current S0 shim.
select(L) returns the newest verified record (by attempt order), or None with the reasons for every rejected record.
No I/O besides reading; importable by derive_maps.py, drive.py, write_registration.py, verify_maps.py.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import hashlib, json
from pathlib import Path

import numpy as np

ROUND = Path(__file__).resolve().parents[3] / "data" / "maps" / "vit_b32_b38"  # see MODIFICATIONS note: the maps/ and interp/ data tree this points at has not been migrated yet
INTERP = {'cc_vit_1d_ffn_expand_chained_fp8': 'ptx_interp_expand.py', 'cc_vit_1d_ffn_contract_chained_fp8': 'ptx_interp60.py',
          'cc_vit_1d_qkv_chained_fp8': 'ptx_interp61.py', 'cc_vit_1d_attention_chained_fp8': 'ptx_interp62.py',
          'cc_vit_1d_projection_chained_fp8': 'ptx_interp63.py', 'cc_vit_1d_projection_wait_fp8': 'ptx_interp63.py'}


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def records(L):
    d = ROUND / 'maps' / ('L%02d' % L)
    out = []
    if (d / 'maps.json').exists():
        out.append(d / 'maps.json')
    out += sorted(d.glob('maps-a[0-9][0-9].json'))
    return out


def verify_record(p):
    reasons = []
    try:
        doc = json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001
        return False, None, ['RECORD_UNREADABLE:%s' % exc]
    if doc.get('status') != 'PASS_FAMILY_EQUAL':
        reasons.append('STATUS:%s' % doc.get('status'))
    files, shas = doc.get('files') or {}, doc.get('file_sha256') or {}
    if not files:
        reasons.append('NO_FILES')
    for role, rel in files.items():
        f = ROUND / rel
        if not f.is_file():
            reasons.append('MISSING:%s' % rel)
            continue
        if fsha(f) != shas.get(role):
            reasons.append('SHA_MISMATCH:%s' % rel)
            continue
        try:
            if rel.endswith('.npz'):
                with np.load(f) as m:
                    if not m.files:
                        reasons.append('NPZ_EMPTY:%s' % rel)
                    for k in m.files:
                        m[k]
            else:
                json.loads(f.read_text(encoding='utf-8'))
        except Exception as exc:  # noqa: BLE001
            reasons.append('UNREADABLE:%s:%s' % (rel, type(exc).__name__))
    interp = INTERP.get(doc.get('function'))
    if interp is None or doc.get('interp_sha256') != fsha(ROUND / 'interp' / interp):
        reasons.append('INTERP_SHA_NOT_CURRENT')
    shim = ROUND / 'interp' / 's0' / ('s0-launch%02d.json' % doc.get('launch', -1))
    if not shim.is_file() or doc.get('shim_sha256') != fsha(shim):
        reasons.append('SHIM_SHA_NOT_CURRENT')
    return not reasons, doc, reasons


def select(L):
    verified, rejected = [], []
    for p in records(L):
        ok, doc, reasons = verify_record(p)
        (verified if ok else rejected).append((p, doc, reasons))
    if verified:
        p, doc, _ = verified[-1]
        return dict(record=p.relative_to(ROUND).as_posix(), record_sha256=fsha(p), files=doc['files'], doc=doc,
                    rejected=[(q.relative_to(ROUND).as_posix(), r) for q, _, r in rejected])
    return None, [(q.relative_to(ROUND).as_posix(), r) for q, _, r in rejected]


def next_attempt(L):
    d = ROUND / 'maps' / ('L%02d' % L)
    used = {p.name for p in d.glob('a[0-9][0-9]') if p.is_dir()} | {p.stem.split('-')[-1] for p in d.glob('maps-a[0-9][0-9].json')}
    n = 1
    while 'a%02d' % n in used:
        n += 1
    return 'a%02d' % n
