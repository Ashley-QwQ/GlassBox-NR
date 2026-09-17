"""Verifies the GlassBox-NR code-commitment hash and (once published-files-map.json
is filled in) that every published file can be traced back to its committed
original.

Usage:
  python verify_commitment.py                 # just recompute the commitment hash
  python verify_commitment.py --check-files    # also verify published files against the map
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMMITTED_VALUE = "83d1a5c274652471487dcc770cfbcba1627e072452cb70551a0b3484c3502c5a"


def sha256_of_json_object(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_commitment_hash() -> bool:
    manifest = json.loads((HERE / "freeze-code-manifest.json").read_text(encoding="utf-8"))
    got = sha256_of_json_object(manifest)
    ok = got == COMMITTED_VALUE
    print(f"commitment hash: {'MATCH' if ok else 'MISMATCH'} (got {got}, want {COMMITTED_VALUE})")
    print(f"manifest entries: {manifest.get('total_files')}")
    return ok


def verify_published_files(package_root: Path) -> bool:
    map_path = HERE / "published-files-map.json"
    if not map_path.exists():
        print("published-files-map.json not found -- nothing to check")
        return True
    entries = json.loads(map_path.read_text(encoding="utf-8"))
    ok = True
    for entry in entries:
        published_path = package_root / entry["published_path"]
        if not published_path.exists():
            print(f"MISSING: {entry['published_path']}")
            ok = False
            continue
        got = sha256_of_file(published_path)
        want = entry["published_sha256"]
        if got != want:
            print(f"HASH MISMATCH: {entry['published_path']} (got {got}, want {want})")
            ok = False
    print(f"published files checked: {len(entries)}, all {'MATCH' if ok else 'have mismatches'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-files", action="store_true", help="also verify published files against published-files-map.json")
    parser.add_argument("--package-root", type=Path, default=HERE.parent, help="root of the published package (default: parent of commitment/)")
    args = parser.parse_args()

    ok = verify_commitment_hash()
    if args.check_files:
        ok = verify_published_files(args.package_root) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
