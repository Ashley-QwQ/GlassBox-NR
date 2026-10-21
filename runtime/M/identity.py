"""Pinned, path-explicit loading and per-launch actual route receipts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import derived_pins

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
ROUND = REPO / 'experiments/parity-cpp-opt-merge-r01-20261002-codex'

def safe(path):
    p = Path(path)
    if any(''.join(map(chr,(98,108,105,110,100,))) in part.lower() for part in p.parts):
        raise RuntimeError('FORBIDDEN_PATH')
    return p

def load(path):
    return json.loads(safe(path).read_text(encoding='utf-8-sig'))

def pin(path):
    p = safe(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}

def verify(item):
    if not derived_pins.same(pin(item['path']), item):
        raise RuntimeError('SEALED_IDENTITY_DRIFT:' + str(item['path']))
    return item

def module(item, name):
    verify(item)
    existing = sys.modules.get(name)
    if existing is not None:
        if not derived_pins.same(pin(existing.__file__), item):
            raise RuntimeError('LOADED_MODULE_IDENTITY_DIFF:' + name)
        return existing
    spec = importlib.util.spec_from_file_location(name, safe(item['path']))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if not derived_pins.same(pin(mod.__file__), item):
        raise RuntimeError('LOADED_MODULE_IDENTITY_DIFF:' + name)
    return mod

def check_actual(row, mod, loaders, assets, auxiliary=(), actual_abi=None):
    if actual_abi is None:
        raise RuntimeError('ACTUAL_ABI_IDENTITY_REQUIRED')
    actual = {'launch': row['launch'], 'module': pin(mod.__file__),
              'loaders': [pin(m.__file__) for m in loaders],
              'auxiliary_modules': [pin(m.__file__) for m in auxiliary],
              'ABI': actual_abi, 'panel_mode': 'decoded', 'assets': assets}
    expected = {k: row[k] for k in actual}
    if not derived_pins.same_deep(actual, expected):
        raise RuntimeError('ACTUAL_ROUTE_IDENTITY_DIFF:L' + str(row['launch']))
    return actual

def write(path, obj):
    p = safe(path).resolve()
    if not (p.is_relative_to(ROOT) or p.is_relative_to(ROUND)):
        raise RuntimeError('WRITE_OUTSIDE_MERGE_ROUND')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
