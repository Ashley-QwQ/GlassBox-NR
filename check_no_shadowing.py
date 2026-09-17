"""Asserts that a set of modules, once imported, resolve to the expected
files inside this package -- i.e. nothing got shadowed by a same-named module
elsewhere on sys.path. Run with this package's root on sys.path.

Usage: extend EXPECTED as each migration group lands, then run:
  python check_no_shadowing.py
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import importlib
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent

# module import name -> expected file path, relative to PKG_ROOT
EXPECTED = {
    "third_party.mlxdlss": "third_party/mlxdlss/__init__.py",
    "third_party.mlxdlss.model": "third_party/mlxdlss/model.py",
    "glassbox_nr.kernels.core.sm120_b1": "glassbox_nr/kernels/core/sm120_b1.py",
}

# Group (b): the encoder/window backend family uses bare (non-package) sys.path
# imports, matching the original code's own convention -- these resolve by
# name only once glassbox_nr/kernels/encoder_windows/ itself is on sys.path.
ENCODER_WINDOWS_DIR = "glassbox_nr/kernels/encoder_windows"
EXPECTED_BARE = {name: f"{ENCODER_WINDOWS_DIR}/{name}.py" for name in [
    "d13_backend", "d13_predict", "d17_backend", "d19_backend", "d19_predict",
    "d20_backend", "d21_backend", "d21_predict", "d22_backend", "d22_predict",
    "d24_predict", "d25_backend", "d25_predict", "d26_backend", "d26_predict",
    "d29_backend", "d29_predict", "d31_backend", "d31_predict", "wo_spec",
]}


def main() -> int:
    sys.path.insert(0, str(PKG_ROOT))
    sys.path.insert(0, str(PKG_ROOT / ENCODER_WINDOWS_DIR))
    sys.path.insert(0, str(PKG_ROOT / "glassbox_nr/kernels/split16_common"))
    sys.path.insert(0, str(PKG_ROOT / "glassbox_nr/kernels/vit"))
    sys.path.insert(0, str(PKG_ROOT / "glassbox_nr/kernels/b39"))
    sys.path.insert(0, str(PKG_ROOT / "glassbox_nr/kernels/transitions"))
    sys.path.insert(0, str(PKG_ROOT / "glassbox_nr/kernels/luna"))
    sys.path.insert(0, str(PKG_ROOT / "glassbox_nr/kernels/b63_s157"))
    sys.path.insert(0, str(PKG_ROOT / "glassbox_nr/kernels/b66"))
    sys.path.insert(0, str(PKG_ROOT / "glassbox_nr/kernels/b6x"))
    sys.path.insert(0, str(PKG_ROOT / "tools"))
    ok = True
    for name, rel in {**EXPECTED, **EXPECTED_BARE, "arith": "glassbox_nr/kernels/split16_common/arith.py",
                      "arith2": "glassbox_nr/kernels/split16_common/arith2.py",
                      "mapexec_c": "glassbox_nr/kernels/split16_common/mapexec_c.py",
                      "kern_repack": "glassbox_nr/kernels/vit/kern_repack.py",
                      "kern_expand": "glassbox_nr/kernels/vit/kern_expand.py",
                      "kern_contract": "glassbox_nr/kernels/vit/kern_contract.py",
                      "kern_qkv": "glassbox_nr/kernels/vit/kern_qkv.py",
                      "kern_attention": "glassbox_nr/kernels/vit/kern_attention.py",
                      "kern_projection": "glassbox_nr/kernels/vit/kern_projection.py",
                      "l59_lane": "glassbox_nr/kernels/vit/l59_lane.py",
                      "map_record": "glassbox_nr/kernels/vit/map_record.py",
                      "predict_b39": "glassbox_nr/kernels/b39/predict_b39.py",
                      "transition_backend": "glassbox_nr/kernels/transitions/transition_backend.py",
                      "d36_backend": "glassbox_nr/kernels/luna/d36_backend.py",
                      "stage_backends": "glassbox_nr/kernels/b63_s157/stage_backends.py",
                      "b66_backend": "glassbox_nr/kernels/b66/b66_backend.py",
                      "b66_backend_h4": "glassbox_nr/kernels/b66/b66_backend_h4.py",
                      "b6x_backend": "glassbox_nr/kernels/b6x/b6x_backend.py",
                      "glassbox_nr.node_executor": "glassbox_nr/node_executor.py",
                      "glassbox_nr.paths": "glassbox_nr/paths.py",
                      "s0_grand": "glassbox_nr/s0_grand.py",
                      "glassbox_nr.kernels.core.rsq_domain_patch": "glassbox_nr/kernels/core/rsq_domain_patch.py",
                      "run_pipeline": "tools/run_pipeline.py",
                      "noise_reference": "glassbox_nr/kernels/encoder_windows/noise_reference.py",
                      "mufu_tables": "glassbox_nr/kernels/encoder_windows/mufu_tables.py"}.items():
        mod = importlib.import_module(name)
        got = Path(mod.__file__).resolve()
        want = (PKG_ROOT / rel).resolve()
        match = got == want
        ok = ok and match
        print(f"{name}: {'OK' if match else 'SHADOWED'} (got {got}, want {want})")
    print("ALL OK" if ok else "SHADOWING DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
