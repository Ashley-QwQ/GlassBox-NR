"""D31 CPU prediction stages.  Native B22 oracle files are never opened here."""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

OPENED = {}
HASHING = [False]


def audit(event, args):
    if event != "open" or HASHING[0] or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        path = str(Path(os.fsdecode(args[0])).resolve())
        mode = args[1] if len(args) > 1 and args[1] is not None else (args[2] if len(args) > 2 else None)
        OPENED.setdefault(path, []).append(mode)
    except Exception:
        pass


sys.addaudithook(audit)

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
ROOT = ROUND.parents[1]
D26S = HERE
D25S = HERE
sys.path.insert(0, str(HERE))
import d31_backend as N  # noqa: E402
import d26_predict as P26  # noqa: E402
import d25_backend as TAIL  # noqa: E402

torch.set_num_threads(1)
SAMPLES = ("c1-C-img-full",)
FREEZE = ROUND / "dependency-freeze.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    HASHING[0] = True
    try:
        return sha(path.read_bytes())
    finally:
        HASHING[0] = False


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def verify_freeze():
    fz = json.loads(FREEZE.read_text(encoding="utf-8"))
    drift = {k: v for k, v in fz["files"].items() if sha_file(ROOT / k) != v}
    if drift:
        raise SystemExit("INVALID: freeze drift %s" % drift)
    return sha_file(FREEZE), fz


def denied_reads():
    denied = []
    root_low = str(ROOT.resolve()).lower().replace("/", "\\")
    for path, modes in OPENED.items():
        low = path.lower().replace("/", "\\")
        read_mode = any(m is None or m == "r" or m == "rb" or m == "r+" or m == "rb+" or (isinstance(m, int) and not (m & os.O_WRONLY)) for m in modes)
        if not read_mode:
            continue
        if "rental-transfer" in low or "\\oracle\\" in low or (low.startswith(root_low) and low.endswith(".zip")):
            denied.append(path)
    return sorted(set(denied))


def finish(outdir, manifest):
    denied = denied_reads()
    manifest.update(invalid=bool(denied), denied_opens=denied, finished_utc=utc())
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    (outdir / "opened.json").write_text(json.dumps({k: [str(x) for x in v] for k, v in sorted(OPENED.items())}, indent=0), encoding="utf-8")
    if denied:
        raise SystemExit("INVALID: forbidden native/oracle reads %s" % denied)


def load_tables_and_weights(index):
    tables = N.S.Sm120Tables.load()
    weights = P26.P19.logical(index)
    return tables, weights


def stage_gates():
    freeze_sha, freeze = verify_freeze()
    out = ROUND / "gates"
    if out.exists():
        raise SystemExit("INVALID: gates exists")
    out.mkdir()
    checks = {}
    interfaces = {}
    for sample in SAMPLES:
        doc = json.loads((ROUND / "interfaces" / sample / "interface.json").read_text(encoding="utf-8"))
        interfaces[sample] = doc
    checks["G0_ASSETS_VALID"] = {
        "passed": all(all(c["ok"] for c in d["checks"]) for d in interfaces.values()) and
                  interfaces[SAMPLES[0]]["sources"]["runtime_dll_sha256"] == "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e" and
                  (len(SAMPLES) == 1 or interfaces[SAMPLES[0]]["sha256"]["inputs/B22-X.bin"] != interfaces[SAMPLES[1]]["sha256"]["inputs/B22-X.bin"]),
        "detail": {s: {"input_sha256": d["sha256"]["inputs/B22-X.bin"], "checks": d["n_checks"]} for s, d in interfaces.items()},
    }
    checks["G1_INTERFACE_VALID"] = {
        "passed": all(d["b22"]["derived_bytes"]["full"] == 102400 and d["b22"]["derived_bytes"]["pooled"] == 73728 and
                       d["b22"]["consumer"]["params_extent_wh"] == [12, 12] and
                       d["b22"]["physical_layout_candidates"] == {"X": "TIN256", "full": "TIN256", "pooled": "INPVIEW512"}
                       for d in interfaces.values()),
        "detail": {s: d["b22"] for s, d in interfaces.items()},
    }
    static = json.loads((ROUND / "static-chains/static-chains.json").read_text(encoding="utf-8"))
    compat = {}
    tables = N.S.Sm120Tables.load()
    for sample in SAMPLES:
        # Same D26 body at B15: compare to D26's frozen CPU prediction, not a native oracle.
        raw15 = (ROOT / "experiments/parity-astra-d26-b15-8h-cpu-chain-20260912/interfaces" / sample / "inputs/B15-X.bin").read_bytes()
        x15 = N.inpview_decode(raw15, 20, 20, 256)
        r15 = N.BODY.BranchedWindowSm120N(15, 8, 20, (0, 0), P26.P19.logical(15), tables).run_codes(x15)
        got15 = N.BODY.tin256_encode(r15["Y"])
        old15 = json.loads((ROOT / "experiments/parity-astra-d26-b15-8h-cpu-chain-20260912/predictions" / sample / "B15/manifest.json").read_text(encoding="utf-8"))["predictions"]["B15-Y.tin256"]["sha256"]
        # Same D25 tail at B14: compare to D25's frozen pooled prediction, not a native oracle.
        raw14 = (ROOT / "experiments/parity-astra-d25-b14-downsample-cpu-20260912/interfaces" / sample / "inputs/B14-X.bin").read_bytes()
        x14 = N.BODY.B9.tin128_decode(raw14)
        w14 = P26.P19.logical(14)
        r14 = N.BODY.BranchedWindowSm120N(14, 4, 40, (-4, -4), w14, tables).run_codes(x14)
        pooled14 = TAIL.pool_tail(r14["Y_half"], w14["block14.layer0.weight0"], grouping="row_pairs", source="unpublished")[1]
        got14 = N.inpview_encode(pooled14)
        old14 = json.loads((ROOT / "experiments/parity-astra-d25-b14-downsample-cpu-20260912/predictions" / sample / "B14/manifest.json").read_text(encoding="utf-8"))["predictions"]["B14-pooled.inpview256"]["sha256"]
        compat[sample] = {"B15_body_matches_D26": sha(got15) == old15, "B14_tail_matches_D25": sha(got14) == old14,
                          "B15_sha": sha(got15), "D26_sha": old15, "B14_sha": sha(got14), "D25_sha": old14}
    checks["G2_CANDIDATE_READY"] = {
        "passed": bool(static["calibration"]["passed"]) and all(v["B15_body_matches_D26"] and v["B14_tail_matches_D25"] for v in compat.values()) and
                  freeze_sha == sha_file(FREEZE),
        "detail": {"static_calibration": static["calibration"], "compatibility": compat,
                    "candidate": "D26 body + D25 row_pairs tail + B22 end-pad + weight0[256,512]"},
    }
    checks["G3_PREDICTION_VALID"] = {"passed": None, "detail": "pending two independent prediction processes"}
    checks["G4_COMPARATOR_VALID"] = {"passed": None, "detail": "pending fixed 4-item comparator and 12+ negtests"}
    checks["G5_BYTE_MATCH"] = {"passed": None, "detail": "pending final comparison"}
    checks["G6_CHAIN_VALID"] = {"passed": True, "detail": "D31 is a single native-B22-input block; no B21 CPU chain is claimed or injected."}
    manifest = {"stage": "gates", "valid": bool(checks["G0_ASSETS_VALID"]["passed"] and checks["G1_INTERFACE_VALID"]["passed"] and checks["G2_CANDIDATE_READY"]["passed"]),
                "match": None, "status": "READY" if all(checks[k]["passed"] for k in ("G0_ASSETS_VALID", "G1_INTERFACE_VALID", "G2_CANDIDATE_READY")) else "INVALID",
                "freeze_sha256": freeze_sha, "checks": checks, "opened_before_finish": len(OPENED), "gpu_used": False}
    finish(out, manifest)
    print(json.dumps(manifest, indent=1, default=str))
    if not manifest["valid"]:
        raise SystemExit("GATES FAILED")


def stage_b22(sample):
    if sample not in SAMPLES:
        raise SystemExit("sample required")
    gates = json.loads((ROUND / "gates/manifest.json").read_text(encoding="utf-8"))
    if not gates.get("valid") or gates.get("invalid"):
        raise SystemExit("prediction refused: gates not ready")
    freeze_sha, _ = verify_freeze()
    out = ROUND / "predictions" / sample / "B22"
    if out.exists():
        raise SystemExit("INVALID: prediction output exists")
    out.mkdir(parents=True)
    started = utc()
    iface = json.loads((ROUND / "interfaces" / sample / "interface.json").read_text(encoding="utf-8"))
    raw = (ROUND / "interfaces" / sample / "inputs/B22-X.bin").read_bytes()
    x = N.BODY.tin256_decode(raw)
    tables, weights = load_tables_and_weights(22)
    candidate = N.B22Sm120(weights, tables)
    candidate.validate_invocation(block_index=22, head_count=8, origin_xy=tuple(iface["b22"]["origin_xy"]),
                                  grid_xy=tuple(iface["b22"]["grid_xyz"][:2]), shape=(20, 20, 256))
    t0 = time.perf_counter()
    result = candidate.run_codes(x)
    seconds = time.perf_counter() - t0
    files = {"B22-full.tin256": result["full_physical"], "B22-pooled.inpview512": result["pooled_physical"]}
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = {
        "stage": "B22", "sample": sample, "valid": True, "match": None, "status": "PREDICTED",
        "started_utc": started, "predictions_written_utc": utc(), "freeze_sha256": freeze_sha,
        "input_sha256": sha(raw), "input_layout": "TIN256", "candidate": candidate.describe(),
        "predictions": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in files.items()},
        "logical_shapes": {"full": [20, 20, 256], "pooled": [12, 12, 512]},
        "pool_input_unpublished_half": True, "native_intermediate_used": False, "native_oracle_opened": False,
        "gpu_used": False, "seconds": seconds, "interface_sha256": sha_file(ROUND / "interfaces" / sample / "interface.json"),
        "all_expected_outputs_written": set(files) == {"B22-full.tin256", "B22-pooled.inpview512"},
    }
    finish(out, manifest)
    print(json.dumps({"sample": sample, "predictions": manifest["predictions"], "seconds": seconds}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gates", "B22"))
    ap.add_argument("--sample", choices=SAMPLES)
    args = ap.parse_args()
    if args.stage == "gates":
        stage_gates()
    else:
        stage_b22(args.sample)


if __name__ == "__main__":
    main()
