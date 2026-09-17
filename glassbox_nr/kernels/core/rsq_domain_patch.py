"""F31: mlxdlss/sm120_b1.py's _rsq() guard (line 376) is one code point too
narrow: `bits >= 0x7C00` excludes 0x7C00 (+inf) even though the frozen,
measured table mlxdlss/resources/rsq-domain.npz has 30,705 rows and its LAST
row is exactly bits=0x7C00 -> native_half_bits=0 (i.e. rsq(+inf) = +0,
independently confirmed by parity-mufu-probe-20260912's out-of-domain
sweep). The real hardware doesn't saturate on overflow either (ARCHITECTURE
line 492: sm_120 HMMA overflow gives +inf, not a saturated finite value), so
this round's B04 _norm_squared producing +inf for some windows on a real
photograph is itself correct behaviour -- the table already has the answer,
the guard just never let it through.

We do NOT edit sm120_b1.py itself: many already-audited rounds froze its
sha256, and editing it would invalidate all of their checks. Instead this
module is imported by grand_node_executor.py and replaces CORE._rsq with a
version whose only difference is the guard's upper bound: <= 0x7C00 instead
of < 0x7C00. Same table, same index formula, same everything else. Anything
above 0x7C00 (NaN payloads) is still rejected exactly as before.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import numpy as np


def apply(core_module):
    """Replace core_module._rsq in place. Called once by grand_node_executor.py."""
    orig_first_code = core_module._RSQ_FIRST_CODE

    def patched_rsq(norm: np.ndarray, tables) -> np.ndarray:
        bits = np.asarray(norm, dtype=np.float16).view(np.uint16)
        if np.any(bits < orig_first_code) or np.any(bits > 0x7C00):
            raise core_module.UnsupportedByBackend(
                "RSQ table domain is positive half [0x0410, 0x7c00] (F31: 0x7c00/+inf included, table has that row)"
            )
        return tables.rsq_half_bits[bits - orig_first_code].view(np.float16)

    core_module._rsq = patched_rsq
