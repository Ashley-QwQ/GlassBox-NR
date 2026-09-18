"""Check a run's outputs against the published reference hash list.

    python -B tools/verify_reference_hashes.py runs/<run-tag>
    python -B tools/verify_reference_hashes.py runs/<run-tag>/<sample>/run_manifest.json ...

Reads reference/reference_hashes.json (the sha256 of the reference runtime's
output bytes at every compared exit -- hashes only, no reference data) and
one or more run_manifest.json files written by tools/run_pipeline.py, and
compares every exit's byte count and sha256.

Prints one line per mismatching or missing exit, then "N/154 MATCH". Exits
non-zero unless every one of the 154 listed exits is present in the given
manifests and matches. A run that covers only one sample therefore reports
77/154 and fails; pass both samples' manifests (or the run-tag directory
that contains both) for a full check.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import argparse
import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PKG_ROOT / "reference" / "reference_hashes.json"


def collect_manifests(paths):
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(sorted(p.glob("*/run_manifest.json")))
            if (p / "run_manifest.json").exists():
                found.append(p / "run_manifest.json")
        else:
            found.append(p)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run-tag directory, sample directory, or run_manifest.json path(s)")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    expected = {(e["sample"], e["node"], e["output"]): e for e in ref["exits"]}
    total = len(expected)

    manifests = collect_manifests(args.runs)
    if not manifests:
        print("no run_manifest.json found under the given path(s)")
        return 2

    got = {}
    for mf in manifests:
        m = json.loads(mf.read_text(encoding="utf-8"))
        sample = m["sample"]
        if m.get("status") != "completed":
            print(f"note: {mf} has status {m.get('status')!r}")
        for node, rec in m["nodes"].items():
            for out_name, meta in rec["outputs"].items():
                key = (sample, node, out_name)
                if key in got and got[key] != (meta["bytes"], meta["sha256"]):
                    print(f"CONFLICT {sample} {node}/{out_name}: two manifests disagree")
                    return 2
                got[key] = (meta["bytes"], meta["sha256"])

    match = 0
    for key, e in expected.items():
        sample, node, out_name = key
        if key not in got:
            print(f"MISSING  {sample} {node}/{out_name}")
            continue
        n_bytes, sha = got[key]
        if n_bytes == e["bytes"] and sha == e["sha256"]:
            match += 1
        else:
            print(f"MISMATCH {sample} {node}/{out_name}: got {n_bytes} B {sha}, reference {e['bytes']} B {e['sha256']}")

    print(f"{match}/{total} MATCH")
    return 0 if match == total else 1


if __name__ == "__main__":
    sys.exit(main())
