"""Extracts WEIGHTS_HT from a user-supplied nvngx_dlssnr.dll into this
package's local weights directory. Never run without the user's own,
lawfully obtained DLL; never downloads or bundles one.

Produces, under `glassbox_nr/../weights/` (see paths.py):
  - weights-packed.safetensors   (opaque packed format, 153 tensors)
  - dlssnr-weights-logical.safetensors  (decoded logical format, 649 tensors)
  - <tensor-name>.bin, one per packed tensor  (raw bytes of each packed
    tensor, individually -- several kernel modules read one of these by
    path rather than through the logical/packed safetensors API; see
    release/_private/weight_derived_files.json for the registry of which
    modules read which slice)

Nothing under weights/ is ever published; see .gitignore.

Usage: python tools/extract_weights.py path/to/your/nvngx_dlssnr.dll
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glassbox_nr.paths as paths  # noqa: E402

sys.path.insert(0, str(paths.PKG_ROOT / "third_party" / "mlxdlss" / "tools"))
# The extraction/unpacking tool itself is third-party (MLX-DLSS, Apache-2.0);
# see third_party/mlxdlss/LICENSE and MODIFICATIONS.md.
import extract_dlssnr_weights as extractor  # noqa: E402
import unpack_dlssnr_weights as unpacker  # noqa: E402

EXPECTED_WEIGHTS_HT_SHA256 = "836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dll", type=Path, help="path to your own nvngx_dlssnr.dll")
    args = ap.parse_args()

    paths.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = extractor.extract_dll(args.dll, paths.WEIGHTS_PACKED_FILE)
    if summary["resourceSHA256"] != EXPECTED_WEIGHTS_HT_SHA256:
        print(f"REFUSING: WEIGHTS_HT sha256 {summary['resourceSHA256']} does not match "
              f"the expected {EXPECTED_WEIGHTS_HT_SHA256}. This DLL's weights are not "
              f"the ones this package's node implementations were verified against.",
              file=sys.stderr)
        return 1
    print(f"WEIGHTS_HT sha256 OK: {summary['resourceSHA256']}")

    logical_summary = unpacker.convert(paths.WEIGHTS_PACKED_FILE, paths.WEIGHTS_LOGICAL_FILE)
    print(f"logical format: {logical_summary['decodedTensorCount']} tensors decoded")

    from safetensors.numpy import safe_open
    with safe_open(str(paths.WEIGHTS_PACKED_FILE), framework="numpy") as f:
        keys = sorted(f.keys())
        for k in keys:
            data = f.get_tensor(k).tobytes()
            (paths.WEIGHTS_DIR / f"{k}.bin").write_bytes(data)
    print(f"wrote {len(keys)} individual tensor .bin files under {paths.WEIGHTS_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
