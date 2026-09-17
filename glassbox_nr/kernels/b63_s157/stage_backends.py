"""Thin per-node CPU adapters for B63 -> S157: bytes in, bytes out, reusing already accepted numeric modules unchanged.

node   module (imported read-only)                                         input codec          output codec
B63    D22 BranchedWindowSm120(63, 2 heads, 80, origin (-4,-4))             D17 TIN64            D17 TIN64
B64    D22 BranchedWindowSm120(64, 2 heads, 80, origin (-4, 0))             D17 TIN64            D17 TIN64
B65    D22 BranchedWindowSm120(65, 2 heads, 80, origin ( 0,-4))             D17 TIN64            outview PLANAR16 S80 C64 map
B66    Opus B66 b66_backend_h4.B66CandidateH4 (H4: FFN seed = unquantised merge half), main PLANAR16 + skip TIN32 -> TIN32
B67    Opus b6x SingleWindowB6x(67, origin (-4,-4)) over D13 arithmetic      TIN32 map            TIN32 map
B68    Opus b6x SingleWindowB6x(68, origin (-4, 0))                          TIN32 map            TIN32 map
B69    Opus b6x SingleWindowB6x(69, origin ( 0,-4))                          TIN32 map            OUTVIEW-1H S160 C32 map
B70    b70_adapter over D16 r08 CAND_NF1_HPACK: feature = B69 physical PLANAR16 bytes in place, native B0 skip, id1
S157   D15 pp_exec.Machine, D15 P2 r2 settings (tex_lanes ba_rg, store_round rz, INF_MUL ieee, ex2neg evidence 152618):
       textures {params word4 handle: id9 = B70 output, params word8 handle: id10}
Every adapter checks the params words it relies on (extent, origin, pointers / handles) against SPEC before computing.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SYS = {"D22": "glassbox_nr/kernels/encoder_windows",
       "B66": "glassbox_nr/kernels/b66",
       "B6X": "glassbox_nr/kernels/b6x",
       "D15": "experiments/parity-astra-d15-output-tail-cpu-20260912",  # not migrated: SASS-dependent (S157), see DESIGN_ZH.md 3.5
       "OWN": "experiments/parity-astra-opus-b63-s157-local-chain-20260913/scripts"}  # not migrated: SASS-dependent (B70), see DESIGN_ZH.md 3.5
MAPS = {"PLANAR16_S80": "data/maps/OUTVIEW-PLANAR16-S80-C64.npy",
        "TIN32_S160": "data/maps/TIN32-S160-physical-to-logical.npy",
        "OUTVIEW1H_S160": "data/maps/OUTVIEW-1H-S160-C32.npy"}
WEIGHTS_LOGICAL = "weights/dlssnr-weights-logical.safetensors"
D15_EX2NEG = "experiments/parity-astra-d15-output-tail-cpu-20260912/derive/ex2neg-152618.json"  # not migrated: only reachable via the S157 branch, see above
D15_LG2 = "experiments/parity-astra-opus-b63-s157-local-chain-20260913/tables/lg2-postprocess-domain-r2.npz"  # not migrated: same
SPEC = {
    "B63": dict(kind="D22", block=63, origin=(-4, -4), grid=(11, 11), plen=88, extent=(80, 80), out="TIN64"),
    "B64": dict(kind="D22", block=64, origin=(-4, 0), grid=(11, 10), plen=88, extent=(80, 80), out="TIN64"),
    "B65": dict(kind="D22", block=65, origin=(0, -4), grid=(10, 11), plen=88, extent=(80, 80), out="PLANAR16_S80"),
    "B66": dict(kind="B66H4", block=66, origin=(0, 0), grid=(20, 20), plen=96, extent=(160, 160)),
    "B67": dict(kind="B6X", block=67, origin=(-4, -4), grid=(21, 21), plen=96, extent=(160, 160), out="TIN32_S160"),
    "B68": dict(kind="B6X", block=68, origin=(-4, 0), grid=(21, 20), plen=96, extent=(160, 160), out="TIN32_S160"),
    "B69": dict(kind="B6X", block=69, origin=(0, -4), grid=(20, 21), plen=96, extent=(160, 160), out="OUTVIEW1H_S160"),
    "B70": dict(kind="B70", plen=184),
    "S157": dict(kind="S157", plen=376, extent=(320, 240)),
}


class AdapterError(Exception):
    pass


def add_path(key):
    p = str(ROOT / SYS[key])
    if p not in sys.path:
        sys.path.insert(0, p)


def _s32(v):
    return v - (1 << 32) if v & (1 << 31) else v


def check_params(node, params):
    sp = SPEC[node]
    if len(params) != sp["plen"]:
        raise AdapterError("%s params length %d != %d" % (node, len(params), sp["plen"]))
    w = struct.unpack("<%dQ" % (len(params) // 8), params)
    if sp["kind"] == "D22":
        got = dict(extent=(w[4] & 0xFFFFFFFF, w[4] >> 32), origin=(_s32(w[5] & 0xFFFFFFFF), _s32(w[5] >> 32)))
    elif sp["kind"] in ("B66H4", "B6X"):
        got = dict(extent=(w[3] & 0xFFFFFFFF, w[3] >> 32), origin=(_s32(w[4] & 0xFFFFFFFF), _s32(w[4] >> 32)))
    elif sp["kind"] == "S157":
        got = dict(extent=(w[36] >> 32, w[37] & 0xFFFFFFFF))
    else:
        return w
    want = {k: sp[k] for k in got}
    if got != want:
        raise AdapterError("%s params %r != frozen spec %r" % (node, got, want))
    return w


def _bijection(mp, n):
    mp = np.asarray(mp, np.int64)
    if mp.shape != (n,) or not np.array_equal(np.sort(mp), np.arange(n)):
        raise AdapterError("map is not a bijection of %d" % n)
    return mp


def load_weights():
    from safetensors.numpy import load_file
    return load_file(str(ROOT / WEIGHTS_LOGICAL))


def run_node(node, inputs, params):
    """inputs: dict name -> bytes (main, skip, colour); returns (output bytes, describe dict, log dict)"""
    sp = SPEC[node]
    w = check_params(node, params)
    kind = sp["kind"]
    if kind == "D22":
        add_path("D22")
        import d22_backend as D22
        D17 = D22.B5
        x = inputs["main"]
        if len(x) != 409600:
            raise AdapterError("%s main input length %d" % (node, len(x)))
        tables = D22.S.Sm120Tables.load()
        obj = D22.BranchedWindowSm120(sp["block"], 2, 80, sp["origin"], load_weights(), tables)
        obj.validate_invocation(block_index=sp["block"], head_count=2, origin_xy=sp["origin"], grid_xy=sp["grid"], shape=(80, 80, 64))
        y = np.asarray(obj.run_codes(D17.tin64_decode(x))["Y"], np.uint8)
        if y.shape != (80, 80, 64):
            raise AdapterError("%s Y shape %s" % (node, y.shape))
        if sp["out"] == "TIN64":
            out = D17.tin64_encode(y)
        else:
            mp = _bijection(np.load(ROOT / MAPS[sp["out"]]), 409600)
            out = y.reshape(-1)[mp].tobytes()
        return out, obj.describe(), {}
    if kind == "B66H4":
        add_path("B66")
        import b66_backend_h4 as H4
        if len(inputs["main"]) != 409600 or len(inputs["skip"]) != 819200:
            raise AdapterError("B66 input lengths")
        if w[10] != 0x18E56C00 or w[0] != 0x18F1EC00:
            raise AdapterError("B66 main/skip pointer words differ from the S0 interface")
        planar16 = _bijection(np.load(ROOT / MAPS["PLANAR16_S80"]), 409600)
        cand = H4.B66CandidateH4(load_weights(), H4.S.Sm120Tables.load(), planar16)
        out, _stages = cand.run_raw(inputs["main"], inputs["skip"])
        return out, cand.describe(), {}
    if kind == "B6X":
        add_path("B6X")
        import b6x_backend as BK
        if tuple(BK.SPECS[sp["block"]]["origin_xy"]) != sp["origin"]:
            raise AdapterError("b6x spec origin differs from params")
        weights = load_weights()
        tables = BK.S.Sm120Tables.load()
        gate = BK.gate_equivalence(weights, tables)
        if not all(gate.values()):
            raise AdapterError("b6x constructor equivalence gate failed: %r" % gate)
        x = inputs["main"]
        if len(x) != 819200:
            raise AdapterError("%s main input length" % node)
        in_map = _bijection(np.load(ROOT / MAPS["TIN32_S160"]), 819200)
        out_map = _bijection(np.load(ROOT / MAPS[sp["out"]]), 819200)
        logical = np.zeros(819200, np.uint8)
        logical[in_map] = np.frombuffer(x, np.uint8)
        cand = BK.load_candidate(sp["block"], weights, tables)
        y = np.asarray(cand.run_codes(logical.reshape(160, 160, 32))["Y"], np.uint8)
        if y.shape != (160, 160, 32):
            raise AdapterError("%s Y shape" % node)
        return y.reshape(-1)[out_map].tobytes(), cand.describe(), dict(equivalence_gate=gate)
    if kind == "B70":
        add_path("OWN")
        import b70_adapter as A
        st = A.prepare_static()
        out, log = A.run_b70(st, inputs["main"], inputs["skip"], params, inputs["colour"])
        return out, A.describe(), log
    if kind == "S157":
        add_path("D15")
        import pp_exec as X
        id9, id10 = inputs["main"], inputs["colour"]
        if len(id9) != 614400 or len(id10) != 614400:
            raise AdapterError("S157 texture lengths")
        if w[4] != 0x7FA00008804 or w[8] != 0x7FA00008807 or w[0] != 0x8806:
            raise AdapterError("S157 handle words differ from the S0 interface")
        W, H = sp["extent"]
        X.INF_MUL["mode"] = "ieee"
        mufu = X.Mufu(census=False, ex2neg_evidence=ROOT / D15_EX2NEG)
        lg2_r2 = np.load(ROOT / D15_LG2)
        mufu._lg2 = lg2_r2["native_float_bits"]
        mufu._lg2_blocks = [int(v) for v in lg2_r2["block_indices"]]
        m = X.Machine(params, {w[4]: np.frombuffer(id9, "<f2").reshape(H, W, 4), w[8]: np.frombuffer(id10, "<f2").reshape(H, W, 4)},
                      W, H, mufu=mufu, tex_lanes="ba_rg", store_round="rz")
        res = m.run()
        img = res[w[0]]
        if not (img["count"] == 1).all():
            raise AdapterError("S157 output not written exactly once per pixel")
        st = {k: (sorted(v) if isinstance(v, set) and k != "visited" else v) for k, v in m.stats.items()}
        st["visited"] = len(m.stats["visited"])
        desc = dict(executor=X.__file__, tex_lanes="ba_rg", store_round="rz", inf_mul="ieee", ex2neg_evidence=D15_EX2NEG,
                    lg2_evidence=D15_LG2,
                    textures={hex(w[4]): "id9 (CPU B70 output)", hex(w[8]): "id10 (declared colour)"})
        return img["img"].astype("<f2").tobytes(), desc, json.loads(json.dumps(dict(stats=st, mufu_demand=mufu.demand_report()), default=str))
    raise AdapterError("unknown node " + node)


def probe_static():
    """touch every static dependency of every node without computing (used only by the freeze builder's probe)"""
    add_path("D22")
    import d22_backend as D22
    D22.S.Sm120Tables.load()
    add_path("B66")
    import b66_backend_h4  # noqa: F401
    add_path("B6X")
    import b6x_backend  # noqa: F401
    add_path("OWN")
    import b70_adapter as A
    A.prepare_static()
    add_path("D15")
    import pp_exec as X
    X.parse()
    X.Mufu(census=False, ex2neg_evidence=ROOT / D15_EX2NEG)
    np.load(ROOT / D15_LG2)
    for k in MAPS.values():
        np.load(ROOT / k)
    load_weights()
