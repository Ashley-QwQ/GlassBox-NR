#!/usr/bin/env python3
"""Self-test of the receipt comparison in run_1080_f0.py (no payload, weights or model needed).

  python devkit/test_run_1080_f0.py

Builds receipts from data/reference_f0.json and checks the exit code compare() returns for complete, incomplete,
duplicated, reordered and altered receipts. Exit code 0 = every case behaves as expected.
"""
import contextlib
import io
import json
import os
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
from run_1080_f0 import compare  # noqa: E402

ref = json.load(open(os.path.join(KIT, 'data', 'reference_f0.json')))
rows = [{'L': r['L'], 'workspace_sha256': r['workspace_sha256'], 'textures': dict(r['textures'])} for r in ref['launches']]
final = ref['final_sha256']


def receipt(got, final_sha=final):
    return got, {'launches': len(got), 'final_sha256': final_sha}


altered = [dict(r) for r in rows]
altered[57] = dict(altered[57], workspace_sha256='0' * 64)
CASES = [   # name, rows, result.json, stop_after, expected exit code
    ('complete 158', *receipt(rows), 157, 0),
    ('empty', *receipt([]), 157, 4),
    ('missing L2', *receipt(rows[:2] + rows[3:]), 157, 4),
    ('L0 repeated 158 times', *receipt([rows[0]] * 158), 157, 4),
    ('reversed', *receipt(rows[::-1]), 157, 4),
    ('extra row after L157', *receipt(rows + [rows[-1]]), 157, 4),
    ('result.json count differs', rows, {'launches': 157, 'final_sha256': final}, 157, 4),
    ('segment 0..9 complete', *receipt(rows[:10]), 9, 0),
    ('segment 0..9 missing L9', *receipt(rows[:9]), 9, 4),
    ('L57 workspace hash altered', *receipt(altered), 157, 2),
    ('final image differs', *receipt(rows, '0' * 64), 157, 3),
]
failed = 0
for name, got, result, stop_after, want in CASES:
    with contextlib.redirect_stdout(io.StringIO()) as text:
        code = compare(got, result, ref, stop_after)
    ok = code == want
    failed += not ok
    print('%-4s %-30s exit %d (expected %d)  %s' % ('ok' if ok else 'FAIL', name, code, want, text.getvalue().strip().splitlines()[0]))
print('ALL %d CASES PASS' % len(CASES) if not failed else '%d CASES FAIL' % failed)
sys.exit(1 if failed else 0)
