"""ffn_expand kernels (publish / chained).

compute() body copied line by line from experiments/parity-astra-opus-b31-l59-20260913/predict_l59.py (main, PHASE 'compute'
section from `inp = ...` to the coverage check); inputs, weights and the map are parameters.  No file I/O here.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import json

import numpy as np


def compute(LANE, inp, W, m, wiring, stats):
    src = m['sread_src'].astype(np.int64)
    wo = m['gload_weight_offset'].astype(np.int64)
    so = m['store_output_offset'].astype(np.int64)
    if wiring.get('status') != 'PASS_EXECUTABLE_MAP' or len(inp) != 65536 or len(W) < 4194304:
        raise RuntimeError('MAP_OR_INPUT_INVALID')
    live = [y for y in range(so.shape[1]) if (so[:, y] >= 0).all()]
    if live != [0, 1] or not (so[:, 2:] == -1).all():
        raise RuntimeError('LIVE_WARPS_UNEXPECTED')
    NB = so.shape[0] * len(live)
    finals = {}
    for ch in wiring['chains']['chains']:
        C = np.zeros((NB, 16, 8), np.float16)
        for st in ch['steps']:
            e, j, q, breg = st['epoch'], st['a_read_slot'], st['b_load_slot'], st['b_reg_index']
            if breg[1] != breg[0] + 1:
                raise RuntimeError('B_REGISTERS_NOT_ADJACENT')
            s_ = src[:, live, e, j, :]
            if (s_ < 0).any():
                raise RuntimeError('LIVE_A_READ_NOT_FROM_INPUT')
            a_l = inp[s_[..., None] + np.arange(16)].reshape(NB, 32, 16)
            b_l = W[wo[:, live, e, q, :][..., None] + 4 * breg[0] + np.arange(8)].reshape(NB, 32, 8)
            C = LANE.qmma_step(LANE.a_matrix(a_l), LANE.b_matrix(b_l), C, stats=stats)
        finals[tuple(ch['final'])] = C
    out = np.zeros(262144, np.uint8)
    cover = np.zeros(262144, np.int32)
    codes = {k: LANE.gate_codes(LANE.d_to_lanes(v)).reshape(so.shape[0], len(live), 32, 4) for k, v in finals.items()}
    for row in wiring['store_table']:
        s_idx = row['slot']
        for b, prov in enumerate(row['bytes']):
            val = codes[(prov['mma_epoch'], prov['mma_slot'])][..., 2 * prov['d_reg'] + prov['half']]
            dst = so[:, live, s_idx, :] + b
            out[dst.ravel()] = val.ravel()
            np.add.at(cover, dst.ravel(), 1)
    if not (cover == 1).all():
        raise RuntimeError('OUTPUT_COVERAGE_NOT_EXACTLY_ONCE')
    return {'output': out.tobytes()}
