"""Derived-identity map for the M2 release set.

Some released files are not byte-identical to the sealed originals the configuration pins refer to:
  * stripped_json   data files from which derived analysis fields were removed (the runtime reads only what is kept)
  * patched_source  loader modules whose forbidden-path literal / pin check was changed
  * rebuilt_module  compiled modules rebuilt from the changed sources
DERIVED_MAP.json lists, for each such file, the original and the derived identity (sha256, bytes).  A pin check that
expects an original identity accepts the file only if it is the original or the registered derivative of exactly that
original.  The map file is itself pinned by DERIVED_MAP_SHA256 (set when the release is sealed).  If no digest is set
the layer is empty and every pin check is strict.
"""
import hashlib
import json
import os
from pathlib import Path

DERIVED_MAP_SHA256 = '20c54612b575c85b88cf0ff18e02e98880b54da600c7f0b7eb51adc4491ba168'          # filled in at seal time; empty means "no map is expected" (strict mode)
_STATE = {}


def _load():
    if 'by_orig' in _STATE:
        return _STATE
    by_orig, by_derived = {}, {}
    path = Path(os.environ.get('DLSSNR_DERIVED_MAP') or Path(__file__).with_name('DERIVED_MAP.json'))
    if DERIVED_MAP_SHA256:
        if not path.exists():
            raise RuntimeError('DERIVED_MAP_MISSING')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != DERIVED_MAP_SHA256:
            raise RuntimeError('DERIVED_MAP_SHA_MISMATCH')
        for e in json.loads(raw.decode('utf-8'))['entries']:
            if e['original_sha256'] in by_orig:
                raise RuntimeError('DERIVED_MAP_DUPLICATE_ORIGINAL')
            by_orig[e['original_sha256']] = e
            by_derived[e['derived_sha256']] = e
    _STATE['by_orig'], _STATE['by_derived'] = by_orig, by_derived
    return _STATE


def accepts_sha(expected_sha, actual_sha):
    """True if the file with actual_sha may stand in for the original expected_sha."""
    if actual_sha == expected_sha:
        return True
    e = _load()['by_orig'].get(expected_sha)
    return e is not None and e['derived_sha256'] == actual_sha


def accepts(expected_sha, expected_bytes, actual_sha, actual_bytes):
    if actual_sha == expected_sha:
        return actual_bytes == expected_bytes
    e = _load()['by_orig'].get(expected_sha)
    return (e is not None and e['derived_sha256'] == actual_sha and e['derived_bytes'] == actual_bytes
            and e['original_bytes'] == expected_bytes)


def same(actual, expected):
    """Identity dicts {path, bytes, sha256}: equal, or `actual` is the registered derivative of exactly `expected`."""
    if actual == expected:
        return True
    if actual.get('path') != expected.get('path'):
        return False
    return accepts(expected.get('sha256'), expected.get('bytes'), actual.get('sha256'), actual.get('bytes'))


def same_deep(actual, expected):
    """Parallel walk of two route records; identity dicts are compared with same(), everything else for equality."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if 'sha256' in expected and 'bytes' in expected and 'path' in expected:
            return same(actual, expected)
        return actual.keys() == expected.keys() and all(same_deep(actual[k], expected[k]) for k in expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(same_deep(a, b) for a, b in zip(actual, expected))
    return actual == expected


_load()   # eager: the map file is read at import time, before any asset audit is active
