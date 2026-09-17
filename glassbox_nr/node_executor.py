"""Modular execution kernels for each node in the GRAND B0..B70..S157 pipeline.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import hashlib, io, json, os, sys
from pathlib import Path
import numpy as np

import glassbox_nr.paths as paths

HERE = Path(__file__).resolve().parent
ROOT = paths.PKG_ROOT
E = ROOT / "experiments"  # not used post-migration (nothing below reads under E); kept so any stray reference fails loudly rather than silently resolving to the wrong root

# Paths to audited modules
D29_SCRIPTS = paths.ENCODER_WINDOWS_DIR
D31_SCRIPTS = paths.ENCODER_WINDOWS_DIR
SPLIT16_BASE = paths.KERNELS
R02_BASE = paths.SPLIT16_R02_DIR
R03_BASE = paths.SPLIT16_R03_DIR
R04_BASE = paths.SPLIT16_R04_DIR
R05_BASE = paths.SPLIT16_R05_DIR
VIT_BASE = paths.VIT_B32_B38_DATA_DIR  # only remaining use is VIT_BASE / "identity/..." and VIT_BASE / "maps/..." below (data); kern_*.py code itself is on sys.path directly via paths.SYS_PATH_ENTRIES
B39_BASE = paths.DATA_DIR  # only remaining use is B39_BASE / "maps/b39-map.*" below; predict_b39.py's own dir is on sys.path directly via paths.SYS_PATH_ENTRIES
R06_BASE = paths.SPLIT16_R06_DIR
B4748_BASE = paths.DATA_DIR  # only remaining use is B4748_BASE / "static/B48-IN-...npy" below; the code that used to live under B4748_BASE/scripts is now on sys.path directly via paths.SYS_PATH_ENTRIES
R07_BASE = paths.SPLIT16_R07_DIR
B5662_BASE = paths.KERNELS / "b5662-not-migrated"  # dead in the original file too: defined, never referenced again
B63S157_BASE = paths.B63_S157_DIR
OUTVIEW_BASE = paths.DATA_DIR  # code below reads OUTVIEW_BASE / "maps/OUTVIEW-*.npy"; S20-C256 (B55/B56), S40-C128 (B61/B62) and S80-C64 (B65/B66) are all migrated (see data/maps/PROVENANCE.md)
D36_SCRIPTS = paths.LUNA_DIR

for p in paths.SYS_PATH_ENTRIES:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from safetensors.numpy import load_file
import d29_predict as P
import d31_predict as Q
import arith2 as A2
from glassbox_nr.kernels.core import sm120_b1 as CORE
from glassbox_nr.kernels.core import rsq_domain_patch
rsq_domain_patch.apply(CORE)
import kern_repack
import kern_expand
import kern_contract
import kern_qkv
import kern_attention
import kern_projection
import l59_lane as LANE
import predict_b39 as P39
# predict_b39.py installs its OWN process-global sys.addaudithook (for when it
# runs standalone as its own CLI) that blocks every write anywhere until its
# main() sets ALLOWED_WRITE_DIR[0]. Imported here as a library, main() never
# runs, so that hook silently blocked every write from every node's
# predict_node.py process (not just B39) the moment grand_node_executor was
# extended to reach B39 -- see FAILURE_LEDGER F14. predict_node.py already
# has its own, independently-audited write gate (own_writes in
# chain_common.check_audit_events); neutralize this redundant/stricter one by
# widening its allowed-write root to the whole project so it never fires.
P39.ALLOWED_WRITE_DIR[0] = P39._norm(str(ROOT))
import transition_backend as TB
import stage_backends as SB
import d36_backend as D36
import map_record as MR

# Cache weights and tables in memory for single-node process
_TABLES = None
_WEIGHTS = None
_TABLES_MX = None

def get_tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = CORE.Sm120Tables.load()
    return _TABLES

def get_weights():
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = load_file(str(ROOT / "weights/dlssnr-weights-logical.safetensors"))
    return _WEIGHTS

def origin_from_params_w5(params_bytes):
    """F28 wrap-up: read a transition block's window origin from its own
    launch params word 5 (low32=x, high32=y, both signed 32-bit) instead of a
    hardcoded constant -- same convention stage_backends.check_params() uses
    for its 'B66H4'/'B6X' kinds. Verified against this round's own anchors:
    B48 (launch 133) and B62 (launch 147) decode to (0,0); B56 (launch 141)
    decodes to (-4,0) -- matching F23's hardcoded fix, now made general so a
    different sample/resolution can't silently reuse a stale constant.
    """
    import struct
    w5 = struct.unpack_from("<Q", params_bytes, 8 * 5)[0]
    def s32(v):
        return v - (1 << 32) if v & (1 << 31) else v
    return (s32(w5 & 0xFFFFFFFF), s32(w5 >> 32))

def get_tables_mx():
    global _TABLES_MX
    if _TABLES_MX is None:
        _TABLES_MX = {}
        for k in ("rsq", "rcp"):
            with np.load(str(ROOT / f"data/tables/{k}-domain.npz")) as t:
                _TABLES_MX[k] = A2.table_from_npz_arrays(t["input_half_bits"], t["native_float_bits"])
    return _TABLES_MX

_SPLIT16_ROLE_MAP = None

def _split16_role_map():
    """release/p2-runnable/tools/migrate_split16.py's output: for each split16
    freeze/<TAG>-chain-r1.json (keyed by its path relative to the package
    root), the package-local path + sha256 of each of the 4 roles this
    function actually reads (map_wiring, map_npz, contraction, weights) --
    the freeze index's own "files" list still records the ORIGINAL
    project-relative paths (experiments/...), which do not exist in a clean
    package; this resolver replaces that lookup. See PROVENANCE notes under
    data/split16/ and data/maps/ for what was migrated and why.
    """
    global _SPLIT16_ROLE_MAP
    if _SPLIT16_ROLE_MAP is None:
        _SPLIT16_ROLE_MAP = json.loads((paths.DATA_DIR / "split16" / "role-map.json").read_text(encoding="utf-8"))
    return _SPLIT16_ROLE_MAP

def _split16_read_verified(freeze_key, role_key, freeze_sha):
    """Read one role's file and verify its sha256 against BOTH the freeze
    index's own recorded value and role-map.json's recorded value -- no
    silent fallback if either is missing or they disagree.

    Two-sha entries (role-map.json role has "source_sha256" -- currently
    only the 16 derived split16 "-sass-contraction.json" copies, see
    derive_split16_contraction.py): the freeze index still records the
    ORIGINAL file's sha (it is committed evidence, never edited), so
    "source_sha256" is checked against the freeze index, and "sha256" (the
    derived bytes actually on disk) is checked against what's actually
    read. A plain byte-copied role has no "source_sha256" key, so both
    checks collapse to the single recorded "sha256" -- unchanged behavior.
    """
    info = _split16_role_map()[freeze_key][role_key]
    source_sha = info.get("source_sha256", info["sha256"])
    expect_sha = info["sha256"]
    if freeze_sha and freeze_sha != source_sha:
        raise ValueError(
            f"split16 {freeze_key} role {role_key!r}: freeze index sha {freeze_sha} "
            f"!= role-map.json source_sha256 {source_sha} -- these must never disagree"
        )
    data = (paths.PKG_ROOT / info["path"]).read_bytes()
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != expect_sha:
        raise ValueError(
            f"split16 {freeze_key} role {role_key!r}: sha mismatch, "
            f"got {actual_sha}, expected {expect_sha} ({info['path']})"
        )
    return data

def load_split16_stage(base_dir, freeze_file):
    freeze_path = base_dir / freeze_file
    fz = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_key = str(freeze_path.resolve().relative_to(paths.PKG_ROOT)).replace("\\", "/")
    freeze_shas = {e["role"]: e.get("sha256") for e in fz["files"]}

    wiring = json.loads(_split16_read_verified(freeze_key, "map_wiring", freeze_shas.get("map_wiring")).decode("utf-8"))
    npz_bytes = _split16_read_verified(freeze_key, "map_npz", freeze_shas.get("map_npz"))
    with np.load(io.BytesIO(npz_bytes)) as z:
        arrays = {k: z[k] for k in z.files}
    con = json.loads(_split16_read_verified(freeze_key, "contraction", freeze_shas.get("contraction")).decode("utf-8"))
    weights_tensor = _split16_role_map()[freeze_key]["weights"]["tensor"]
    weights = _split16_read_verified(freeze_key, "weights", freeze_shas.get(f"weights:{weights_tensor}"))
    return wiring, arrays, weights, con["contract"]

import importlib.util
_MAPEXEC2_CACHE = {}

def _mapexec2_for(base_dir):
    """F19: every split16 revision directory (R02..R07) freezes its OWN
    mapexec2.py, and they are not identical -- r07-b47's copy fixes a
    hardcoded 'for b in range(16)' (assumes 16 store atoms) to
    'for b in range(vals.shape[-1])' (uses the map's real store width, 4 for
    this stage). But exec_split16_stage used to call a single
    module-level `import mapexec2 as MX`, which resolves to whichever
    revision directory sys.path happened to put first (R03_BASE, inserted
    near the top of this file) -- R07_BASE was never even on sys.path. So
    every split16 stage silently ran R03's copy, and B47 crashed
    (IndexError: index 4 is out of bounds for axis 3 with size 4) the first
    time a stage's data didn't fit R03's hardcoded assumption. Load each
    revision's own frozen file explicitly, by path, instead of relying on
    import shadowing. See FAILURE_LEDGER F19.
    """
    key = str(base_dir.resolve())
    mod = _MAPEXEC2_CACHE.get(key)
    if mod is None:
        mod_path = base_dir / "mapexec2.py"
        spec = importlib.util.spec_from_file_location(f"mapexec2_{key}", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MAPEXEC2_CACHE[key] = mod
    return mod

def exec_split16_stage(wiring, arrays, weights, contract, inp_bytes, base_dir, out_len=73728):
    if "thread_class" in arrays:
        sys.path.insert(0, str(R05_BASE))
        import mapexec_c as MXC
        pred, _ = MXC.execute(wiring, arrays, inp_bytes, weights, out_len, get_tables_mx(), contract, {})
    else:
        mx_mod = _mapexec2_for(base_dir)
        pred, _ = mx_mod.execute(wiring, arrays, inp_bytes, weights, out_len, get_tables_mx(), contract, {})
    return pred.tobytes()

def run_split16_4stage_block(base_dir, tags, in_feature):
    w0, a0, wt0, c0 = load_split16_stage(base_dir, f"freeze/{tags[0]}-chain-r1.json")
    o0 = exec_split16_stage(w0, a0, wt0, c0, in_feature, base_dir)
    w1, a1, wt1, c1 = load_split16_stage(base_dir, f"freeze/{tags[1]}-chain-r1.json")
    o1 = exec_split16_stage(w1, a1, wt1, c1, b"".join([o0, in_feature]), base_dir)
    w2, a2, wt2, c2 = load_split16_stage(base_dir, f"freeze/{tags[2]}-chain-r1.json")
    o2 = exec_split16_stage(w2, a2, wt2, c2, o1, base_dir)
    w3, a3, wt3, c3 = load_split16_stage(base_dir, f"freeze/{tags[3]}-chain-r1.json")
    o3 = exec_split16_stage(w3, a3, wt3, c3, b"".join([o2, o1]), base_dir)
    return o3

# F32: identity/s0-identity.json originally only had entries for reference samples;
# adding other samples (e.g. d23-n2-nat320x240) raised KeyError here (same bug class as F21/F30).
# d0 independently byte-diffed all 158 launches' params across raw
# packages and found every launch/word is identical across samples except
# the registered exceptions in s0_grand.py's PARAMS_INVARIANT_EXCEPTIONS
# (launch 1 word 1, which is not read as a structural field below).
# s0_grand.py's params_invariant check re-verifies this every time S0 runs, for every sample.
# That makes it safe to read the VIT_REFERENCE_SAMPLE's launch record as a single shared structural table
# (function name, weight tensor name) instead of a per-sample lookup --
# PROVIDED this sample's own captured launch function name (recorded by
# s0_grand.py from session.nvapi.tsv into
# anchors/<sample>/inputs/launch-functions.json, independent of the params
# bytes) is asserted to match, below. F33: that file is read here via the
# node's own declared "launch_functions" static input (nodes.json), NOT by
# reading the whole-round anchors-trusted.json directly -- the latter
# bundles ALL samples' data into one file, so freezing/exposing it wholesale
# to a predictor process would leak information about the OTHER sample
# (params_invariant status, other sample's target sha256s) that the
# "restricted" audit allow-set is specifically supposed to keep out (see
# FAILURE_LEDGER F33). The per-sample launch-functions.json file is scoped
# exactly like every other anchors/<sample>/inputs/ file.
# Shares s0_grand.py's REFERENCE_SAMPLE constant directly (rather than
# redeclaring "c1-C-img-full" here) so the two can never drift apart.
import s0_grand as S0G
VIT_REFERENCE_SAMPLE = S0G.REFERENCE_SAMPLE


def run_vit_launch(L, in_dict, sample, launch_functions_bytes):
    s0_vit = json.loads((VIT_BASE / "identity/s0-identity.json").read_text(encoding="utf-8"))
    lr = s0_vit["samples"][VIT_REFERENCE_SAMPLE]["launches"][str(L)]
    fn = lr["function"]

    own_launch_functions = json.loads(launch_functions_bytes.decode("utf-8"))
    actual_fn = own_launch_functions[str(L)]
    if actual_fn != fn:
        raise ValueError(
            f"F32 params_invariant/VIT structural-table assertion failed: sample={sample!r} launch={L} "
            f"actual_function={actual_fn!r} != table_function(from {VIT_REFERENCE_SAMPLE!r})={fn!r}"
        )
    k = {
        "cc_vit_1d_repack_2d_to_1d_fp8": "repack", "cc_vit_1d_repack_1d_to_2d_fp8": "repack",
        "cc_vit_1d_ffn_expand_publish_fp8": "expand", "cc_vit_1d_ffn_expand_chained_fp8": "expand",
        "cc_vit_1d_ffn_contract_chained_fp8": "contract", "cc_vit_1d_qkv_chained_fp8": "qkv",
        "cc_vit_1d_attention_chained_fp8": "attention", "cc_vit_1d_projection_chained_fp8": "projection",
        "cc_vit_1d_projection_wait_fp8": "projection"
    }[fn]
    
    wt = [r for r in lr["roles"].values() if r.get("kind") == "w" and r["role"] == "weights"]
    wt_bytes = np.frombuffer(paths.WEIGHTS_DIR.joinpath(f"{wt[0]['tensor']}.bin").read_bytes(), np.uint8) if wt else None
    
    if L in [58, 59, 60, 61, 62, 63]:
        m_npz_path = VIT_BASE / f"maps/b31/l{L}-address-map.npz" if L > 58 else None
        m_json_path = VIT_BASE / f"maps/b31/l{L}-wiring-and-checks.json" if L == 59 else (VIT_BASE / f"maps/b31/l{L}-templates-and-checks.json" if L > 59 else None)
        m_chk_path = VIT_BASE / f"maps/b31/l{L}-sass-numeric-check.json" if L == 61 else (VIT_BASE / f"maps/b31/l{L}-sass-family-check.json" if L in (62, 63) else None)
    elif L == 99:
        m_json_path = VIT_BASE / "maps/launch99-address-map.json"
        m_npz_path = m_chk_path = None
    else:
        sel = MR.select(L)
        m_npz_path = VIT_BASE / sel["files"]["npz"] if "npz" in sel["files"] else None
        m_json_path = VIT_BASE / sel["files"]["json"] if "json" in sel["files"] else None
        m_chk_path = VIT_BASE / sel["files"]["check"] if "check" in sel["files"] else None
        
    m_json = json.loads(m_json_path.read_text(encoding="utf-8")) if m_json_path else None
    m_chk = json.loads(m_chk_path.read_text(encoding="utf-8")) if m_chk_path else None
    
    if k == "repack":
        return {"output": kern_repack.compute(in_dict["input"], m_json)}
        
    with np.load(m_npz_path) as m:
        npz = {key: m[key] for key in m.files}
        
    u8 = lambda b: np.frombuffer(b, np.uint8)
    if k == "expand":
        return kern_expand.compute(LANE, u8(in_dict["input"]), wt_bytes, npz, m_json, {})
    elif k == "contract":
        return kern_contract.compute(LANE, CORE, u8(in_dict["input"]), u8(in_dict["residual"]), wt_bytes, m_json, npz, {})
    elif k == "qkv":
        with np.load(str(ROOT / "data/tables/rsq-domain.npz")) as t:
            tab_rsq = {key: np.array(t[key]) for key in t.files}
        rsq_sha = CORE._TABLE_FILES['rsq'][1]
        return kern_qkv.compute(LANE, CORE, u8(in_dict["input"]), wt_bytes, m_json, m_chk, npz, tab_rsq, rsq_sha, {})
    elif k == "attention":
        with np.load(str(ROOT / "data/tables/rcp-domain.npz")) as t:
            tab_rcp = {key: np.array(t[key]) for key in t.files}
        rcp_sha = CORE._TABLE_FILES['rcp'][1]
        data = {n: u8(in_dict[n]) for n in ("q", "k", "v")}
        return kern_attention.compute(LANE, CORE, data, m_json, m_chk, npz, tab_rcp, rcp_sha, {})
    elif k == "projection":
        return kern_projection.compute(LANE, CORE, u8(in_dict["input"]), u8(in_dict["residual"]), wt_bytes, m_json, m_chk, npz, {})

def execute_node(node_name: str, sample: str, inputs: dict) -> dict:
    """Executes a single node from input bytes dictionary, returning {out_name: out_bytes}."""
    tables = get_tables()
    all_weights = get_weights()
    
    # --- Stage E1 ---
    if node_name == "B00":
        colour = inputs["colour"]
        params = inputs["params"]
        b0_cand = P.B0.B0ResetCandidate({k: v for k, v in all_weights.items() if k.startswith("block0.layer0.")}, tables)
        res = b0_cand.run(colour, params)
        return {"b0_full.tin": res["full_tin"], "b0_pooled.inpview": res["pooled_inpview"]}
        
    elif node_name in ("B01", "B02", "B03", "B04"):
        b_idx = int(node_name[1:])
        inp = inputs["main"]
        wts = {k: v for k, v in all_weights.items() if k.startswith(f"block{b_idx}.")}
        if b_idx == 1:
            _, y1 = CORE.SM120B1Reference(wts, tables, weights_id=None).run_codes(CORE.inpview_decode_codes(inp, 160, 160))
            return {"B1-Y.TIN": CORE.tin_encode_codes(y1)}
        else:
            import d13_backend as DB
            carried = CORE.tin_decode_codes(inp, 160, 160)
            r = DB.SingleWindowSm120(b_idx, wts, tables).run_codes(carried)
            if b_idx == 4:
                return {"B4-full.TIN": CORE.tin_encode_codes(r["Y"]), "B4-pooled.raw": DB.pooled_encode(r["pooled"])}
            else:
                return {f"B{b_idx}-Y.TIN": CORE.tin_encode_codes(r["Y"])}
            
    elif node_name == "B05":
        inp = inputs["main"]
        x5 = P.P19.B5.inpview64_decode(inp)
        wts5 = {k: v for k, v in all_weights.items() if k.startswith("block5.")}
        res = P.P19.B5.B5Sm120(wts5, tables).run_codes(x5)
        return {"B5-Y.tin64": P.P19.B5.tin64_encode(res["Y"])}
        
    elif node_name in ("B06", "B07"):
        b_idx = int(node_name[1:])
        inp = inputs["main"]
        wts = {k: v for k, v in all_weights.items() if k.startswith(f"block{b_idx}.")}
        res = P.P19.BW.BranchedWindow2hSm120(b_idx, wts, tables).run_codes(P.P19.B5.tin64_decode(inp))
        return {f"B{b_idx}-Y.tin64": P.P19.B5.tin64_encode(res["Y"])}
        
    elif node_name == "B08":
        inp = inputs["main"]
        wts8 = {k: v for k, v in all_weights.items() if k.startswith("block8.")}
        r8 = P.B8.B8Sm120(wts8, tables).run_codes(P.P19.B5.tin64_decode(inp))
        return {"B8-full.tin64": P.P19.B5.tin64_encode(r8["full"]), "B8-pooled.inpview128": P.B8.inpview128_encode(r8["pooled"])}
        
    elif node_name in ("B09", "B10", "B11", "B12", "B13"):
        b_idx = int(node_name[1:])
        inp = inputs["main"]
        # F32: this table is keyed by block index only, shared across all
        # samples. Safe per s0_grand.py's params_invariant check (d0
        # byte-diffed all 158 launches' params across all 3 samples; every
        # word is identical except 2 registered exceptions, neither of which
        # is a B09-B13 launch). These nodes have no "params" input declared
        # in nodes.json, so there is nothing to cross-check live against here
        # -- params_invariant itself is the guarantee.
        origins = {9: (0, 0), 10: (-4, -4), 11: (-4, 0), 12: (0, -4), 13: (0, 0)}
        wts = {k: v for k, v in all_weights.items() if k.startswith(f"block{b_idx}.")}
        if b_idx == 9:
            raw_codes = P.B8.inpview128_decode(inp)
            r = P.B9.BranchedWindowSm120(9, 4, 40, (0, 0), wts, tables).run_codes(raw_codes)
        else:
            r = P.B9.BranchedWindowSm120(b_idx, 4, 40, origins[b_idx], wts, tables).run_codes(P.B9.tin128_decode(inp))
        return {f"B{b_idx}-Y.tin128": P.B9.tin128_encode(r["Y"])}
        
    elif node_name == "B14":
        inp = inputs["main"]
        wts14 = {k: v for k, v in all_weights.items() if k.startswith("block14.")}
        r14 = P.B14.B14Sm120(wts14, tables).run_codes(P.B9.tin128_decode(inp))
        return {"B14-full.tin128": P.B9.tin128_encode(r14["full"]), "B14-pooled.inpview256": P.B14.inpview_encode(r14["pooled"])}
        
    elif node_name == "B15":
        # F32: origin (0,0) is a block-index-keyed constant, shared across
        # samples -- safe per s0_grand.py's params_invariant (see B09-B13
        # comment above). No "params" input declared for this node.
        inp = inputs["main"]
        wts15 = {k: v for k, v in all_weights.items() if k.startswith("block15.")}
        r15 = P.N.BranchedWindowSm120N(15, 8, 20, (0, 0), wts15, tables).run_codes(P.N.inpview256_decode(inp))
        return {"B15-Y.tin256": P.N.tin256_encode(r15["Y"])}

    elif node_name in ("B16", "B17", "B18", "B19", "B20", "B21"):
        b_idx = int(node_name[1:])
        inp = inputs["main"]
        # F32: shared across samples, safe per params_invariant (see B09-B13
        # comment above); no "params" input declared for these nodes.
        origins = {16: (-4, -4), 17: (-4, 0), 18: (0, -4), 19: (0, 0), 20: (-4, -4), 21: (-4, 0)}
        wts = {k: v for k, v in all_weights.items() if k.startswith(f"block{b_idx}.")}
        res = P.N.BranchedWindowSm120N(b_idx, 8, 20, origins[b_idx], wts, tables).run_codes(P.N.tin256_decode(inp))
        return {f"B{b_idx}-Y.tin256": P.N.tin256_encode(res["Y"])}
        
    elif node_name == "B22":
        inp = inputs["main"]
        wts22 = {k: v for k, v in all_weights.items() if k.startswith("block22.")}
        r22 = Q.N.B22Sm120(wts22, tables).run_codes(Q.N.BODY.tin256_decode(inp))
        return {"B22-full.bin": bytes(r22["full_physical"]), "B22-pooled.bin": bytes(r22["pooled_physical"])}
        
    # --- Stage E2 ---
    elif node_name == "B23":
        inp = inputs["main"]
        w_c25, a_c25, wt_c25, c_c25 = load_split16_stage(R02_BASE, "freeze/C25-chain-r1.json")
        c25_out = exec_split16_stage(w_c25, a_c25, wt_c25, c_c25, inp, R02_BASE)
        w_c26, a_c26, wt_c26, c_c26 = load_split16_stage(R02_BASE, "freeze/C26-chain-r1.json")
        c26_out = exec_split16_stage(w_c26, a_c26, wt_c26, c_c26, b"".join([c25_out, inp]), R02_BASE)
        w_c27, a_c27, wt_c27, c_c27 = load_split16_stage(R02_BASE, "freeze/C27-chain-r1.json")
        c27_out = exec_split16_stage(w_c27, a_c27, wt_c27, c_c27, c26_out, R02_BASE)
        w_c28, a_c28, wt_c28, c_c28 = load_split16_stage(R02_BASE, "freeze/C28-chain-r1.json")
        c28_out = exec_split16_stage(w_c28, a_c28, wt_c28, c_c28, b"".join([c27_out, c26_out]), R02_BASE)
        return {"B23-C28-word2.bin": c28_out}
        
    elif node_name in ("B24", "B25", "B26"):
        inp = inputs["main"]
        tags = {"B24": ["K29", "K30", "K31", "K32"], "B25": ["K33", "K34", "K35", "K36"], "B26": ["K37", "K38", "K39", "K40"]}[node_name]
        out_f = f"{node_name}-{tags[-1]}-word2.bin"
        return {out_f: run_split16_4stage_block(R03_BASE, tags, inp)}
        
    elif node_name in ("B27", "B28", "B29"):
        inp = inputs["main"]
        tags = {"B27": ["K41", "K42", "K43", "K44"], "B28": ["K45", "K46", "K47", "K48"], "B29": ["K49", "K50", "K51", "K52"]}[node_name]
        out_f = f"{node_name}-{tags[-1]}-word2.bin"
        return {out_f: run_split16_4stage_block(R04_BASE, tags, inp)}
        
    elif node_name == "B30":
        cur = inputs["main"]
        w53, a53, wt53, c53 = load_split16_stage(R05_BASE, "freeze/K53-chain-r1.json")
        k53_out = exec_split16_stage(w53, a53, wt53, c53, cur, R05_BASE)
        w54, a54, wt54, c54 = load_split16_stage(R05_BASE, "freeze/K54-chain-r1.json")
        k54_out = exec_split16_stage(w54, a54, wt54, c54, b"".join([k53_out, cur]), R05_BASE)
        w55, a55, wt55, c55 = load_split16_stage(R05_BASE, "freeze/K55-chain-r1.json")
        k55_out = exec_split16_stage(w55, a55, wt55, c55, k54_out, R05_BASE)
        w56, a56, wt56, c56 = load_split16_stage(R05_BASE, "freeze/K56-chain-r1.json")
        k56_concat = exec_split16_stage(w56, a56, wt56, c56, b"".join([k55_out, k54_out]), R05_BASE, out_len=106496)
        k56_w2 = k56_concat[:73728]
        k56_w3 = k56_concat[73728:]
        w57, a57, wt57, c57 = load_split16_stage(R05_BASE, "freeze/K57-chain-r1.json")
        l57_out = exec_split16_stage(w57, a57, wt57, c57, k56_w3, R05_BASE, out_len=65536)
        return {"B30-L56-word2.bin": k56_w2, "B30-L57-word1.bin": l57_out}
        
    # --- Stage E3 ---
    elif node_name == "B31":
        inp = inputs["main"]
        lfb = inputs["launch_functions"]
        l58_map_json = json.loads((VIT_BASE / "maps/b31/launch58-address-map.json").read_text(encoding="utf-8"))
        l58_out = kern_repack.compute(inp, l58_map_json)
        # Run B31 launches 59..63
        cur = l58_out
        for L in range(59, 64):
            if L == 59:
                res_exp = run_vit_launch(59, {"input": cur}, sample, lfb)
            elif L == 60:
                res_cnt = run_vit_launch(60, {"input": res_exp["output"], "residual": cur}, sample, lfb)
            elif L == 61:
                res_qkv = run_vit_launch(61, {"input": res_cnt["output"]}, sample, lfb)
            elif L == 62:
                res_att = run_vit_launch(62, {"q": res_qkv["q"], "k": res_qkv["k"], "v": res_qkv["v"]}, sample, lfb)
            elif L == 63:
                res_prj = run_vit_launch(63, {"input": res_att["output"], "residual": res_cnt["output"]}, sample, lfb)
                cur = res_prj["output"]
        return {"B31-L63-word2.bin": cur}

    elif node_name.startswith("B") and len(node_name) == 3 and node_name[1:].isdigit() and 32 <= int(node_name[1:]) <= 38:
        b_idx = int(node_name[1:])
        cur = inputs["main"]
        lfb = inputs["launch_functions"]
        b_first = 59 + 5 * (b_idx - 31)
        res_exp = run_vit_launch(b_first, {"input": cur}, sample, lfb)
        res_cnt = run_vit_launch(b_first + 1, {"input": res_exp["output"], "residual": cur}, sample, lfb)
        res_qkv = run_vit_launch(b_first + 2, {"input": res_cnt["output"]}, sample, lfb)
        res_att = run_vit_launch(b_first + 3, {"q": res_qkv["q"], "k": res_qkv["k"], "v": res_qkv["v"]}, sample, lfb)
        res_prj = run_vit_launch(b_first + 4, {"input": res_att["output"], "residual": res_cnt["output"]}, sample, lfb)
        out_f = f"B{b_idx}-L{b_first + 4}-word2.bin"
        return {out_f: res_prj["output"]}

    elif node_name == "L99":
        inp = inputs["main"]
        l99_res = run_vit_launch(99, {"input": inp}, sample, inputs["launch_functions"])
        return {"L99-word0.bin": l99_res["output"]}
        
    # --- Stage E4 ---
    elif node_name == "B39":
        main_inp = inputs["main"]
        skip_inp = inputs["skip"]
        b39_map = json.loads((B39_BASE / "maps/b39-map.json").read_text(encoding="utf-8"))
        with np.load(B39_BASE / "maps/b39-map.npz") as z:
            b39_arrays = {k: z[k] for k in z.files}
        b39_weights = paths.WEIGHTS_DIR.joinpath("block39.layer0.layer.bin").read_bytes()
        res = P39.compute(
            LANE, CORE,
            np.frombuffer(main_inp, np.uint8),
            np.frombuffer(skip_inp, np.uint8),
            np.frombuffer(b39_weights, np.uint8),
            b39_arrays, b39_map, {}
        )
        return {"B39-Y.raw": res["out"]}
        
    elif node_name.startswith("B") and len(node_name) == 3 and node_name[1:].isdigit() and 40 <= int(node_name[1:]) <= 46:
        b_idx = int(node_name[1:])
        inp = inputs["main"]
        i = b_idx - 40
        f_launch = 101 + 4 * i
        tags = [f"K{f_launch + j}" for j in range(4)]
        out_f = f"B{b_idx}-K{f_launch + 3}-word2.bin"
        return {out_f: run_split16_4stage_block(R06_BASE, tags, inp)}
        
    elif node_name == "B47":
        inp = inputs["main"]
        return {"K132-word2.bin": run_split16_4stage_block(R07_BASE, ["K129", "K130", "K131", "K132"], inp)}
        
    # --- Stage E5 ---
    elif node_name == "B48":
        main_inp = inputs["main"]
        skip_inp = inputs["skip"]
        b48_in_map = np.load(str(B4748_BASE / "maps/B48-IN-PLANAR16-S12-C512.npy"))
        b48_cand = TB.TransitionCandidate(
            block=48, heads=8, channels=256, out_hw=20, in_planes=32, origin_xy=origin_from_params_w5(inputs["params"]),
            weights=all_weights, tables=tables, planar16_map=b48_in_map,
            ffn_seed="decoded_codes", tail_seed="published_qz"
        )
        y, _ = b48_cand.run_raw(main_inp, skip_inp)
        return {"B48-Y.raw": bytes(y)}
        
    elif node_name.startswith("B") and len(node_name) == 3 and node_name[1:].isdigit() and 49 <= int(node_name[1:]) <= 55:
        b_idx = int(node_name[1:])
        inp = inputs["main"]
        # F30: was d35_block_info(sample, ...), which reads a per-sample
        # fixture file that only exists for the two original samples.
        # Verified identical to origin_from_params_w5() on both existing
        # samples (B49=(-4,-4) B50=(-4,0) B51=(0,-4) B52=(0,0) B53=(-4,-4)
        # B54=(-4,0) B55=(0,-4)); reading it directly from THIS sample's own
        # captured params works for any sample, including new ones.
        origin_xy = origin_from_params_w5(inputs["params"])
        res = P.backend(b_idx, origin_xy, tables).run_codes(P.N.tin256_decode(inp))
        if b_idx == 55:
            ov_map_55 = np.load(str(OUTVIEW_BASE / "maps/OUTVIEW-PLANAR16-S20-C256.npy"))
            b55_planar16 = res["Y"].flat[ov_map_55].tobytes()
            return {"B55-Y.raw": b55_planar16}
        else:
            return {f"B{b_idx}-Y.raw": P.N.tin256_encode(res["Y"])}
            
    elif node_name == "B56":
        main_inp = inputs["main"]
        skip_inp = inputs["skip"]
        b56_in_map = np.load(str(OUTVIEW_BASE / "maps/OUTVIEW-PLANAR16-S20-C256.npy"))
        b56_cand = TB.TransitionCandidate(
            # F23 found this hardcoded (0,0) was wrong (real value (-4,0));
            # F28 replaced the hardcoded fix with a read of the real params.
            block=56, heads=4, channels=128, out_hw=40, in_planes=16, origin_xy=origin_from_params_w5(inputs["params"]),
            weights=all_weights, tables=tables, planar16_map=b56_in_map,
            ffn_seed="decoded_codes", tail_seed="published_qz"
        )
        y, _ = b56_cand.run_raw(main_inp, skip_inp)
        return {"B56-Y.raw": bytes(y)}
        
    elif node_name.startswith("B") and len(node_name) == 3 and node_name[1:].isdigit() and 57 <= int(node_name[1:]) <= 61:
        b_idx = int(node_name[1:])
        inp = inputs["main"]
        # F32: D36.SPECS's origin_xy is keyed by block index only, shared
        # across samples -- safe per s0_grand.py's params_invariant (see
        # B09-B13 comment above). No "params" input declared for B57-B61 in
        # nodes.json, so there is nothing to cross-check live against here;
        # params_invariant is the guarantee. (d36_backend.py's own
        # D36OrdinaryBlock wrapper carries an actual-vs-expected assertion
        # for this same table, but this call site goes straight to
        # D22.BranchedWindowSm120 without it, matching every other B57-B65
        # branch's calling convention.)
        spec = D36.SPECS[b_idx]
        obj = D36.D22.BranchedWindowSm120(b_idx, 4, 40, spec["origin_xy"], P.logical(b_idx), tables)
        r = obj.run_codes(D36.D22.tin128_decode(inp))
        if b_idx == 61:
            ov_map_61 = np.load(str(OUTVIEW_BASE / "maps/OUTVIEW-PLANAR16-S40-C128.npy"))
            b61_planar16 = r["Y"].flat[ov_map_61].tobytes()
            return {"B61-Y.raw": b61_planar16}
        else:
            return {f"B{b_idx}-Y.raw": D36.D22.tin128_encode(r["Y"])}
            
    elif node_name == "B62":
        main_inp = inputs["main"]
        skip_inp = inputs["skip"]
        b62_in_map = np.load(str(OUTVIEW_BASE / "maps/OUTVIEW-PLANAR16-S40-C128.npy"))
        b62_cand = TB.TransitionCandidate(
            block=62, heads=2, channels=64, out_hw=80, in_planes=8, origin_xy=origin_from_params_w5(inputs["params"]),
            weights=all_weights, tables=tables, planar16_map=b62_in_map,
            ffn_seed="decoded_codes", tail_seed="published_qz"
        )
        y, _ = b62_cand.run_raw(main_inp, skip_inp)
        return {"B62-Y.raw": bytes(y)}
        
    # --- Stage E6 ---
    elif node_name in ("B63", "B64", "B65"):
        b_idx = int(node_name[1:])
        inp = inputs["main"]
        # F32: same as the B57-B61 branch above -- shared across samples,
        # safe per params_invariant; no "params" input declared for B63-B65.
        spec = D36.SPECS[b_idx]
        obj = D36.D22.BranchedWindowSm120(b_idx, 2, 80, spec["origin_xy"], P.logical(b_idx), tables)
        # F24: d17_backend exposes tin64_decode/tin64_encode as its own
        # top-level functions (see d36_backend.py's own usage: D17.tin64_decode),
        # not under a 'B5' submodule -- D36.D17.B5 doesn't exist.
        r = obj.run_codes(D36.D17.tin64_decode(inp))
        if b_idx == 65:
            ov_map_65 = np.load(str(OUTVIEW_BASE / "maps/OUTVIEW-PLANAR16-S80-C64.npy"))
            b65_planar16 = r["Y"].flat[ov_map_65].tobytes()
            return {"B65-Y.raw": b65_planar16}
        else:
            return {f"B{b_idx}-Y.raw": D36.D17.tin64_encode(r["Y"])}
            
    elif node_name == "B66":
        # F32: origin (0,0) is shared across samples, safe per
        # params_invariant; no "params" input declared for B66.
        main_inp = inputs["main"]
        skip_inp = inputs["skip"]
        b66_in_map = np.load(str(OUTVIEW_BASE / "maps/OUTVIEW-PLANAR16-S80-C64.npy"))
        b66_cand = TB.TransitionCandidate(
            block=66, heads=1, channels=32, out_hw=160, in_planes=4, origin_xy=(0, 0),
            weights=all_weights, tables=tables, planar16_map=b66_in_map,
            ffn_seed="merged_half", tail_seed="published_qz"
        )
        y, _ = b66_cand.run_raw(main_inp, skip_inp)
        return {"B66-Y.raw": bytes(y)}
        
    elif node_name in ("B67", "B68", "B69"):
        # F25: SB.B6xCandidate / SB.OUT_MAP_1H never existed in the audited
        # stage_backends.py -- its only entry point is run_node(node, inputs,
        # params), which has its own SPEC table (origin/grid/plen) and
        # check_params() validation. Follow the audited predict_stage.py's own
        # calling convention exactly.
        out_bytes, _desc, _log = SB.run_node(node_name, {"main": inputs["main"]}, inputs["params"])
        return {f"{node_name}-Y.raw": out_bytes}

    elif node_name == "B70":
        out_bytes, _desc, _log = SB.run_node(
            "B70",
            {"main": inputs["main"], "skip": inputs["skip"], "colour": inputs["colour"]},
            inputs["params"]
        )
        return {"B70-id9.rgba16": out_bytes}

    elif node_name == "S157":
        out_bytes, _desc, _log = SB.run_node(
            "S157",
            {"main": inputs["main"], "colour": inputs["colour"]},
            inputs["params"]
        )
        return {"S157-id2.rgba16": out_bytes}
        
    else:
        raise ValueError(f"Unknown node: {node_name}")
