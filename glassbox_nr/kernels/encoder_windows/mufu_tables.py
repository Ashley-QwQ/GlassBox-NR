"""Consumer side of this round's measured MUFU tables: look up, or REFUSE.

A domain-restricted table is only safe if it throws outside its domain.  The
pattern is the project's existing one -- `parity-qk-score-resource/
norm_trace.py:98-101` raises `Unresolved` when the input leaves the
exhaustively measured range.

Each table stores its own input bit patterns next to its results, so the
lookup is an exact membership test, not an interval test.  There is no
interpolation and no nearest-neighbour fallback: an input that was not
measured produces an exception.

Argument conventions, which differ per variant and are the thing most likely
to be got wrong (this round got it wrong once -- see a2b-correction.json):

    LG2   index with the fp32 value fed to MUFU.LG2 directly.
    SQRT  index with the fp32 value fed to MUFU.SQRT directly.
    COS   index with the value BEFORE the FMUL.RZ by 1/(2*pi) -- i.e. the
    SIN   target kernel's 0x820/0x890 output.  The probe kernel performs that
          reduction itself with the same instruction and immediate as the
          target's 0x830/0x8c0, so the table covers the pair as one unit.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np

import glassbox_nr.paths as _paths  # d20_backend.py puts the package root on sys.path before loading this module

TABLES = _paths.TABLES_DIR

FILES = {"LG2": "lg2-boxmuller-domain.npz",
         "SQRT": "sqrt-boxmuller-domain.npz",
         "COS": "cos-boxmuller-preduction.npz",
         "SIN": "sin-boxmuller-preduction.npz"}

# COS/SIN take the pre-reduction argument; LG2/SQRT take theirs directly.
PRE_REDUCTION = {"COS", "SIN"}


class OutsideMeasuredDomain(Exception):
    """Raised instead of returning a value that was never measured."""


class Table:
    def __init__(self, variant):
        self.variant = variant
        self.path = TABLES / FILES[variant]
        z = np.load(self.path)
        self.inputs = z["input_float_bits"]
        self.results = z["native_float_bits"]
        self.meta = json.loads(bytes(z["meta"]).decode())
        order = np.argsort(self.inputs, kind="stable")
        self.keys = self.inputs[order]
        self.vals = self.results[order]
        assert self.keys.size == self.vals.size

    def __call__(self, x):
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        bits = x.view(np.uint32).reshape(-1)
        pos = np.searchsorted(self.keys, bits)
        pos_c = np.clip(pos, 0, self.keys.size - 1)
        miss = self.keys[pos_c] != bits
        if miss.any():
            i = int(np.argmax(miss))
            raise OutsideMeasuredDomain(
                "MUFU.%s: input 0x%08x (%r) was never measured; this table holds "
                "%d inputs from %s"
                % (self.variant, int(bits[i]), float(x.reshape(-1)[i]),
                   self.keys.size, self.path.name))
        return self.vals[pos_c].view(np.float32).reshape(x.shape)


_CACHE = {}


def mufu(variant, x):
    """The callable `noise_reference.gaussians` expects."""
    if variant not in _CACHE:
        _CACHE[variant] = Table(variant)
    return _CACHE[variant](x)


def selftest():
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": str(detail)})
        if not ok:
            raise AssertionError("%s: %s" % (name, detail))

    for v in FILES:
        t = Table(v)
        ck("%s table loads" % v, t.keys.size > 0, t.keys.size)
        # a value inside the table round-trips
        probe = t.keys[t.keys.size // 3].view(np.uint32)
        got = t(np.array([probe], dtype=np.uint32).view(np.float32))
        ck("%s lookup returns the stored word" % v,
           got.view(np.uint32)[0] == t.vals[t.keys.size // 3])
        # a value NOT in the table must raise, not interpolate
        bad = np.array([np.float32(1e30)], dtype=np.float32)
        try:
            t(bad)
            ck("%s refuses an unmeasured input" % v, False, "returned a value")
        except OutsideMeasuredDomain:
            ck("%s refuses an unmeasured input" % v, True)
        # a batch with one bad element must be refused wholesale
        good = np.array([probe], dtype=np.uint32).view(np.float32)
        try:
            t(np.concatenate([good, bad]))
            ck("%s refuses a mixed batch" % v, False, "accepted it")
        except OutsideMeasuredDomain:
            ck("%s refuses a mixed batch" % v, True)

    # the convention check that caught the a2 error: cos must track cos(x)
    t = Table("COS")
    x = t.keys[::4001].view(np.float32).astype(np.float64)
    y = t.vals[::4001].view(np.float32).astype(np.float64)
    d_rad = float(np.abs(y - np.cos(x)).max())
    d_rev = float(np.abs(y - np.cos(2 * np.pi * x)).max())
    ck("COS table is indexed by the PRE-reduction argument",
       d_rad < d_rev and d_rad < 1e-5, {"as_radians": d_rad, "as_revolutions": d_rev})

    return {"all_passed": all(c["passed"] for c in checks),
            "count": len(checks), "checks": checks}


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=1))
