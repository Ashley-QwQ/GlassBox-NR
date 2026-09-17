"""Per-instance B67 / B68 / B69 binding of the unchanged D13 single-head arithmetic (isolated; not registered anywhere).

The D13 class keeps its block table in a module-global dict (d13_backend.BLOCKS) and refuses unknown blocks.  The
r02 B67 adapter extended that dict in-process.  Here nothing global is touched: SingleWindowB6x subclasses
SingleWindowSm120, binds its spec to the instance, and inherits _body / run_codes / validate_invocation unchanged.
Only weight loading (a copy of D13 __init__ lines, no arithmetic) is re-expressed so that the spec comes from the
instance.  gate_equivalence() proves on D13's own B2 / B3 configurations that this constructor yields byte-identical
weights and bias layout to D13's constructor, and that d13_backend.BLOCKS is unchanged afterwards.

SPECS are frozen from the trusted anchors (params word4: low 32 = x, high 32 = y):
  B67 origin (-4,-4)  pad rows 4 / cols 4      (r02 audited two-sample hash reproduction is the compat gate)
  B68 origin (-4, 0)  pad cols 4               (same function as B67; same config shape as D13 B3)
  B69 origin ( 0,-4)  pad rows 4, no pool      (outview_wait function; publication map is a separate frozen file)
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
D13_SCRIPTS = ROOT / "glassbox_nr/kernels/encoder_windows"
if str(D13_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(D13_SCRIPTS))
import d13_backend as D13  # noqa: E402

S = D13.S
M = D13.M
Unsupported = S.UnsupportedByBackend

SPECS = {
    67: dict(origin_xy=(-4, -4), pad_rows=4, pad_cols=4, pooled=False),
    68: dict(origin_xy=(-4, 0), pad_rows=0, pad_cols=4, pooled=False),
    69: dict(origin_xy=(0, -4), pad_rows=4, pad_cols=0, pooled=False),
}


class SingleWindowB6x(D13.SingleWindowSm120):
    def __init__(self, block_index: int, spec: dict, weights, tables):  # noqa: D401 - no super().__init__ (global table)
        want = dict(origin_xy=tuple(spec["origin_xy"]), pad_rows=int(spec["pad_rows"]), pad_cols=int(spec["pad_cols"]),
                    pooled=bool(spec["pooled"]))
        for axis, pad in ((0, want["pad_cols"]), (1, want["pad_rows"])):
            if pad != (4 if want["origin_xy"][axis] == -4 else 0) or want["origin_xy"][axis] not in (0, -4):
                raise Unsupported("spec pad does not follow origin: %r" % (want,))
        if want["pooled"]:
            raise Unsupported("pooled tail is not part of B67-B69")
        self.block_index = block_index
        self.spec = want
        self.tables = tables
        prefix = "block%d.layer0." % block_index
        self._w = {}
        for name, shape in D13.SHAPES.items():
            key = prefix + name
            if key not in weights:
                raise Unsupported("missing logical weight %r" % key)
            arr = np.array(S._as_numpy(weights[key]), dtype=np.float64, copy=True)
            if tuple(arr.shape) != shape:
                raise Unsupported("%s has shape %s, expected %s" % (key, arr.shape, shape))
            arr.setflags(write=False)
            self._w[name] = arr
        bias = torch.from_numpy(np.ascontiguousarray(self._w["attn_bias"], dtype=np.float32))
        laid = M.recover_attention_bias_layout(bias).reshape(64, 64).numpy().astype(np.float16)
        laid.setflags(write=False)
        self._bias_qk = laid

    def describe(self):
        d = dict(backend="b6x SingleWindowB6x (per-instance spec over unchanged D13 SingleWindowSm120 arithmetic)",
                 block_index=self.block_index, spec=dict(self.spec), accumulator_model=self.accumulator_model,
                 core_module=S.__file__, model_module=M.__file__, d13_module=D13.__file__)
        return d


def load_candidate(block_index: int, weights, tables):
    if block_index not in SPECS:
        raise Unsupported("b6x covers B67/B68/B69 only, not %r" % (block_index,))
    spec = SPECS[block_index]
    cand = SingleWindowB6x(block_index, spec, weights, tables)
    cand.validate_invocation(block_index=block_index, head_count=1, window_size=8, origin_xy=spec["origin_xy"], shape=(160, 160, 32))
    return cand


def gate_equivalence(weights, tables):
    """Constructor equivalence on D13's own configurations; returns a dict of booleans (all must be True)."""
    snapshot = {k: dict(v) for k, v in D13.BLOCKS.items()}
    res = {}
    for b in (2, 3):
        ref = D13.SingleWindowSm120(b, weights, tables)
        mine = SingleWindowB6x(b, D13.BLOCKS[b], weights, tables)
        res["B%d_spec" % b] = dict(mine.spec) == dict(origin_xy=tuple(ref.spec["origin_xy"]), pad_rows=ref.spec["pad_rows"],
                                                   pad_cols=ref.spec["pad_cols"], pooled=ref.spec["pooled"])
        res["B%d_weights" % b] = sorted(ref._w) == sorted(mine._w) and all(
            ref._w[k].tobytes() == mine._w[k].tobytes() for k in ref._w)
        res["B%d_bias" % b] = ref._bias_qk.tobytes() == mine._bias_qk.tobytes()
    res["d13_blocks_unchanged"] = {k: dict(v) for k, v in D13.BLOCKS.items()} == snapshot and 67 not in D13.BLOCKS
    return res
