"""Fast Pipeline Runner (P2-A): Single-Process End-to-End Execution of DLSS-NR GRAND Chain.

Adapted from the performance-track single-process runner (see
release/_private/diffs/run_pipeline.py.diff for the exact changes against its
origin). Wiring-only changes: calls the packaged glassbox_nr.node_executor
instead of the internal r01 grand_node_executor, resolves every path through
glassbox_nr.paths instead of hardcoded project-absolute defaults, inlines
chain_common.resolve_inputs (the only chain_common function this runner
actually calls) instead of importing the whole r01 audit-gate module, and
(added after the first smoke-test round) writes run_manifest.json
incrementally -- after every node, and on failure -- instead of only once at
the very end, since a crash must still leave real on-disk evidence of how far
the run got rather than only a console log. The "in-memory transfer, missing
input raises Chain discontinuity" execution logic is unchanged.

Features:
- Single-process execution: eliminates per-node sub-process launches.
- In-memory transfer: passes intermediate tensors via a RAM dictionary.
- Checkpoint/export support: saves final images and optionally per-stage
  intermediate files.
- Optional --audit-opens: records every file path opened during the run, to
  catch runtime-only dependencies (dynamic imports, lazy data loads) that
  static import analysis and the clean-room import test both miss -- see the
  noise_reference.py/mufu_tables.py lazy-load gap this found in group (b).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import os
import sys
import time
import json
import hashlib
import argparse
import gc
from pathlib import Path
import numpy as np

try:
    import ctypes
    ctypes.cdll.msvcrt._setmaxstdio(2048)
except Exception:
    pass

# Enforcement: single-threaded CPU execution and unbuffered logs. Also works
# around a known, not-yet-root-caused file-handle leak (some map/weight file
# is opened without being closed under load) -- see FAILURE_LEDGER F01's
# note on the MSVCRT default 512-handle limit; do not remove _setmaxstdio or
# this gc.collect() call without first finding and fixing that leak.
os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONUNBUFFERED="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
)
sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glassbox_nr.paths as paths  # noqa: E402
from glassbox_nr import node_executor as GNE  # noqa: E402


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# Recorded in every manifest so a run always identifies exactly which code
# produced it -- a run started while this file or node_executor.py is being
# edited (see FAILURE_LEDGER F03's addendum: editing a file a running
# process already loaded doesn't affect that process, since Python reads
# the whole module into memory at import time, but it left two manifests
# whose code version wasn't otherwise recorded anywhere) is now traceable
# after the fact instead of only by cross-referencing chat history.
RUNNER_SHA256 = _sha256_file(Path(__file__).resolve())
NODE_EXECUTOR_SHA256 = _sha256_file(Path(GNE.__file__).resolve())

# The two published samples. Only samples with captured frame inputs and
# complete published maps are offered here; intermediate-injected samples
# requiring non-shipped original intermediate tensors are excluded.
PUBLISHED_SAMPLES = ("c1-C-img-full", "d23-n2-nat320x240")

# B70 and S157 are the two nodes whose backend (b70_adapter.py / pp_exec.py)
# parses raw native SASS disassembly text at runtime (see
# release/p2-runnable/DESIGN_ZH.md section 3.5). That text is never published
# (user decision: replace those two backends with a pure-Python
# reimplementation, tracked as a separate follow-up card -- not this one).
# Until that rewrite lands, this runner defaults to stopping at B69; passing
# --stop-after S157 or B70 fails loudly instead of silently trying to read a
# SASS file that isn't there.
SASS_DEPENDENT_NODES = ("B70", "S157")
DEFAULT_STOP_AFTER = "B69"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest_atomic(manifest_path: Path, manifest: dict) -> None:
    """Write to a temp file in the same directory, then os.replace (a rename
    on the same filesystem) -- a reader never sees a half-written file even
    if the process is killed mid-write.
    """
    manifest["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp_path = manifest_path.with_name(manifest_path.name + f".tmp{os.getpid()}")
    tmp_path.write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
    os.replace(tmp_path, manifest_path)


class OpenAudit:
    """Opt-in (--audit-opens) record of every file path opened during a run,
    via sys.addaudithook's "open" event -- this fires for plain io.open/
    pathlib opens AND for importlib.util.spec_from_file_location-style
    dynamic loads (both go through the same C-level open), which is exactly
    the class of dependency that static import analysis misses.
    """

    def __init__(self, log_path: Path):
        self._fh = open(log_path, "w", encoding="utf-8")
        self.count = 0
        self._active = True

    def install(self):
        def hook(event, args):
            # sys.addaudithook installs are permanent for the process (there is
            # no removehook); once this OpenAudit is closed, the hook must
            # become a no-op instead of writing to a closed file handle --
            # e.g. main()'s own summary-file write, after audit.close(), would
            # otherwise crash the whole run right after a successful result.
            if not self._active or event != "open":
                return
            try:
                path = args[0]
            except Exception:
                return
            if not isinstance(path, (str, bytes, os.PathLike)):
                return
            self._fh.write(json.dumps({"path": str(path), "mode": args[1] if len(args) > 1 else None}) + "\n")
            self.count += 1

        sys.addaudithook(hook)

    def close(self):
        self._active = False
        self._fh.close()


def resolve_inputs(node_info, sample):
    """Inlined from experiments/parity-grand-e2e-r01-20260914/scripts/chain_common.py
    (verbatim, mechanical copy) -- the only function of that module this
    runner needs. F29: the single place that decides which input list
    applies for a given (node, sample); a node's plain 'inputs' is the
    default for every sample, an optional 'sample_inputs' dict wholesale-
    replaces it for the named sample only.
    """
    overrides = node_info.get("sample_inputs")
    if overrides and sample in overrides:
        return overrides[sample]
    return node_info["inputs"]


def run_sample(sample: str, args):
    t_start_wall = time.perf_counter()

    weights_dir = Path(args.weights_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    test_input_dir = Path(args.test_input_dir).resolve()
    output_base = Path(args.output_dir).resolve()
    sample_out_dir = output_base / args.run_tag / sample
    manifest_path = sample_out_dir / "run_manifest.json"
    # FAILURE_LEDGER F04: a prior throwaway verification run reused a proof
    # run's own --sample/--output-dir, silently overwrote its manifest, and
    # was then rm -rf'd, permanently destroying real evidence. --run-tag is
    # now mandatory (no default) and this refuses outright rather than
    # overwriting -- --resume is the only sanctioned way to write into an
    # existing run-tag directory, and only for continuing an incomplete run.
    if manifest_path.exists() and not args.resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        raise SystemExit(
            f"refusing to overwrite existing evidence at {manifest_path} "
            f"(status={existing.get('status')!r}, node_count={existing.get('node_count')!r}, "
            f"match_count={existing.get('match_count')!r}). Pick a different --run-tag, or pass "
            f"--resume to continue this same run-tag's incomplete run. Never rm -rf this "
            f"directory to work around this -- move it to <output-dir>/_superseded/ instead if "
            f"it genuinely needs to be discarded."
        )
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    custom_weights_file = weights_dir / "dlssnr-weights-logical.safetensors"
    if custom_weights_file.exists():
        from safetensors.numpy import load_file
        GNE._WEIGHTS = load_file(str(custom_weights_file))

    GNE.get_tables()

    with open(args.nodes_json, "r", encoding="utf-8") as f:
        nodes_list = json.load(f)

    all_node_names = [n["node"] for n in nodes_list]
    requested_stop_after = args.stop_after
    if requested_stop_after in SASS_DEPENDENT_NODES:
        raise SystemExit(
            f"--stop-after {requested_stop_after}: B70/S157 pure-Python backend not yet available "
            f"(the package ships no SASS text; see DESIGN_ZH.md section 3.5). "
            f"Use --stop-after B69 or earlier."
        )
    if requested_stop_after not in all_node_names:
        raise SystemExit(f"--stop-after {requested_stop_after!r}: not a known node name")
    stop_idx = all_node_names.index(requested_stop_after)
    nodes_list = nodes_list[: stop_idx + 1]

    truncated_by_limit_nodes = bool(args.limit_nodes and args.limit_nodes < len(nodes_list))
    if args.limit_nodes:
        nodes_list = nodes_list[: args.limit_nodes]

    planned_node_names = [n["node"] for n in nodes_list]
    full_chain_last_node = all_node_names[-1]

    print("========================================================")
    print(f"[RUN PIPELINE] Starting sample: {sample}")
    print(f"Total nodes to execute: {len(nodes_list)}")
    print(f"Weights: {custom_weights_file}")
    print(f"Data dir: {data_dir}")
    print(f"Test input dir: {test_input_dir}")
    print(f"Output dir: {sample_out_dir}")
    print("========================================================")

    memory_store = {}  # (node_name, filename) -> bytes
    node_metrics = {}
    comparison_count = 0
    match_count = 0
    mismatch_count = 0
    total_calc_seconds = 0.0
    final_image_info = None
    last_completed_node = None

    def not_run_nodes():
        # Chain-relative, not plan-relative: every node after the last
        # completed one in the FULL node graph (all_node_names), not just
        # within whatever --stop-after/--limit-nodes truncated the plan to.
        # A run that stops at L99 by request is still partial -- B39..S157
        # were never run -- even though nothing "left to do" within this
        # invocation's own plan.
        if last_completed_node in all_node_names:
            return all_node_names[all_node_names.index(last_completed_node) + 1:]
        return list(all_node_names)

    def stop_reason_for(status: str) -> str:
        if status == "failed":
            return "failed"
        if last_completed_node == full_chain_last_node:
            return "completed"
        if truncated_by_limit_nodes and planned_node_names and last_completed_node == planned_node_names[-1]:
            return "limit_nodes"
        return "stop_after"

    def build_manifest(status: str, **extra) -> dict:
        m = {
            "sample": sample,
            "mode": "run-pipeline",
            "status": status,
            "stop_reason": stop_reason_for(status),
            "runner_sha256": RUNNER_SHA256,
            "node_executor_sha256": NODE_EXECUTOR_SHA256,
            "resolved_paths": {
                "weights_dir": str(weights_dir),
                "data_dir": str(data_dir),
                "test_input_dir": str(test_input_dir),
                "compare_dir": str(Path(args.compare_dir).resolve()) if args.compare_dir else None,
                "output_dir": str(sample_out_dir),
            },
            "requested_stop_after": requested_stop_after,
            "planned_node_count": len(planned_node_names),
            "stopped_after": last_completed_node,
            "last_completed_node": last_completed_node,
            "not_run_nodes": not_run_nodes(),
            "partial": bool(not_run_nodes()),
            "total_wall_seconds": round(time.perf_counter() - t_start_wall, 3),
            "total_calc_seconds": round(total_calc_seconds, 3),
            "node_count": len(node_metrics),
            "comparison_count": comparison_count,
            "match_count": match_count,
            "mismatch_count": mismatch_count,
            "final_image": final_image_info,
            "nodes": node_metrics,
        }
        m.update(extra)
        return m

    for idx, node_info in enumerate(nodes_list, 1):
        node_name = node_info["node"]
        stage = node_info["stage"]

        try:
            resolved_inps = resolve_inputs(node_info, sample)
            inputs_dict = {}
            for inp in resolved_inps:
                role = inp["role"]
                source = inp["source"]
                if source == "cpu":
                    src_node = inp["node"]
                    src_file = inp["file"]
                    key = (src_node, src_file)
                    if key in memory_store:
                        inputs_dict[role] = memory_store[key]
                    elif args.resume:
                        local_path = sample_out_dir / src_node / src_file
                        if local_path.exists():
                            inputs_dict[role] = local_path.read_bytes()
                        else:
                            raise FileNotFoundError(f"Missing resume checkpoint input: {local_path} for node {node_name}")
                    else:
                        raise RuntimeError(
                            f"Chain discontinuity: missing in-memory tensor {key} for node {node_name}. "
                            f"Strict mode requires in-memory transfer from upstream nodes."
                        )
                elif source == "static":
                    rel_path = inp["path"].format(sample=sample)
                    sub_rel = rel_path[len("anchors/"):] if rel_path.startswith("anchors/") else rel_path
                    if role in ("params", "launch_functions"):
                        # nodes.json declares these under anchors/{sample}/inputs/...,
                        # but per-launch param buffers and the launch-function-name
                        # metadata are published package data (DESIGN_ZH.md:
                        # data/params/320x240/, an approved category) -- unlike
                        # "colour", which is never published and is locally
                        # regenerated into the anchors dir. sub_rel is
                        # "<sample>/inputs/..." (the {sample} substitution already
                        # happened above); drop that leading "<sample>/" segment
                        # since the destination tree supplies its own <sample>/.
                        after_sample = sub_rel[len(sample) + 1:] if sub_rel.startswith(sample + "/") else sub_rel
                        static_file = paths.DATA_DIR / "params" / "320x240" / sample / after_sample
                        assert test_input_dir not in static_file.parents, (
                            f"{role!r} role must never resolve under the anchors dir: {static_file}"
                        )
                    else:
                        static_file = test_input_dir / sub_rel
                    if not static_file.exists():
                        raise FileNotFoundError(f"Missing static input: {static_file} for node {node_name}")
                    inputs_dict[role] = static_file.read_bytes()
                elif source == "declared_injection":
                    inj_path = data_dir / inp["path"]
                    if not inj_path.exists():
                        raise FileNotFoundError(
                            f"Missing declared injection: {inj_path} for node {node_name}. "
                            f"This input source is not provisioned by the published package."
                        )
                    inputs_dict[role] = inj_path.read_bytes()
                else:
                    raise ValueError(f"Unknown input source {source}")

            resumed = False
            node_dir = sample_out_dir / node_name
            node_calc_s = 0.0
            if args.resume and node_dir.exists():
                all_exist = True
                cached_outputs = {}
                for out_spec in node_info["outputs"]:
                    f_p = node_dir / out_spec["name"]
                    if not f_p.exists() or f_p.stat().st_size != out_spec["bytes"]:
                        all_exist = False
                        break
                    cached_outputs[out_spec["name"]] = f_p.read_bytes()
                if all_exist:
                    outputs = cached_outputs
                    resumed = True

            if not resumed:
                t0 = time.perf_counter()
                outputs = GNE.execute_node(node_name, sample, inputs_dict)
                t1 = time.perf_counter()
                node_calc_s = t1 - t0
                total_calc_seconds += node_calc_s

            node_record = {
                "node": node_name,
                "stage": stage,
                "calc_seconds": round(node_calc_s, 4),
                "resumed": resumed,
                "outputs": {},
            }

            if args.save_all_nodes:
                node_dir.mkdir(parents=True, exist_ok=True)

            for out_name, out_bytes in outputs.items():
                out_sha = sha256_bytes(out_bytes)
                node_record["outputs"][out_name] = {"bytes": len(out_bytes), "sha256": out_sha}
                memory_store[(node_name, out_name)] = out_bytes
                if args.save_all_nodes and not resumed:
                    (node_dir / out_name).write_bytes(out_bytes)
                if node_name in ("B70", "S157"):
                    (sample_out_dir / out_name).write_bytes(out_bytes)

            if args.compare_dir:
                ref_mf = Path(args.compare_dir) / sample / node_name / "manifest.json"
                if ref_mf.exists():
                    with open(ref_mf, "r", encoding="utf-8") as f:
                        exp_data = json.load(f)
                    for out_name, out_meta in node_record["outputs"].items():
                        comparison_count += 1
                        exp_sha = exp_data["outputs"][out_name]["sha256"]
                        is_match = out_meta["sha256"] == exp_sha
                        out_meta["reference_sha256"] = exp_sha
                        out_meta["match"] = is_match
                        if is_match:
                            match_count += 1
                        else:
                            mismatch_count += 1
                            print(f"  [MISMATCH] {node_name}/{out_name}: got {out_meta['sha256']}, expected {exp_sha}")
                        if node_name == "S157":
                            final_image_info = {
                                "file": out_name,
                                "bytes": out_meta["bytes"],
                                "sha256": out_meta["sha256"],
                                "reference_sha256": exp_sha,
                                "match": is_match,
                            }

            node_metrics[node_name] = node_record
            last_completed_node = node_name
            gc.collect()
            write_manifest_atomic(manifest_path, build_manifest("running"))
            if idx % 10 == 0 or idx == len(nodes_list):
                resumed_flag = " (cached)" if resumed else ""
                print(f"  Progress: [{idx:2d}/{len(nodes_list):2d}] {stage} {node_name:4s} ({node_calc_s:.2f}s{resumed_flag}) | CumCalc: {total_calc_seconds:.1f}s")

        except Exception as exc:
            write_manifest_atomic(
                manifest_path,
                build_manifest(
                    "failed",
                    failed_node=node_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ),
            )
            raise

    # "completed" means this invocation's own planned loop finished without
    # exception -- independent of whether the full 73-node chain was
    # covered. Whether more of the chain remains is exactly what
    # partial/not_run_nodes/stop_reason communicate; do not conflate the two.
    final_status = "completed"
    out_manifest = build_manifest(final_status)
    write_manifest_atomic(manifest_path, out_manifest)

    print(f"\n[RUN SUMMARY: {sample}]")
    print(f"  Status                : {final_status}")
    print(f"  Total Wall Clock Time : {out_manifest['total_wall_seconds']:.2f} s")
    print(f"  Total Pure Calc Time  : {total_calc_seconds:.2f} s")
    if args.compare_dir:
        print(f"  Outputs Compared/Match: {match_count} / {comparison_count} (mismatches: {mismatch_count})")
    if out_manifest["not_run_nodes"]:
        print(f"  PARTIAL RUN: stopped after {last_completed_node}; not run: {', '.join(out_manifest['not_run_nodes'])}")
        print("  No final-image hash is produced or claimed by a partial run.")
    elif final_image_info:
        print(f"  Final Image Output    : {final_image_info['file']} ({final_image_info['bytes']} bytes)")
        print(f"  Final Image SHA-256   : {final_image_info['sha256']}")
    print(f"  Manifest written to   : {manifest_path}")

    return out_manifest


def main():
    ap = argparse.ArgumentParser(description="DLSS-NR GlassBox-NR single-process pipeline runner")
    ap.add_argument("--sample", default=PUBLISHED_SAMPLES[0], choices=[*PUBLISHED_SAMPLES, "all"])
    ap.add_argument("--nodes-json", default=str(paths.NODES_JSON))
    ap.add_argument("--weights-dir", default=str(paths.WEIGHTS_DIR))
    ap.add_argument("--data-dir", default=str(paths.DATA_DIR))
    ap.add_argument("--test-input-dir", default=str(paths.ANCHORS_DIR))
    ap.add_argument("--output-dir", default=str(paths.PKG_ROOT / "runs"))
    ap.add_argument("--run-tag", required=True, help="Identifies this run's own subdirectory, <output-dir>/<run-tag>/<sample>/ -- required, no default, so a throwaway run can never silently reuse and overwrite a real proof/acceptance run's evidence (see FAILURE_LEDGER F04)")
    ap.add_argument("--save-all-nodes", action="store_true", help="Save intermediate binaries for all nodes")
    ap.add_argument("--compare-dir", default=None, help="Optional: directory of reference per-node manifest.json files to compare outputs against")
    ap.add_argument("--stop-after", default=DEFAULT_STOP_AFTER, help="Last node to execute (inclusive); defaults to B69 since B70/S157 need a pure-Python rewrite not yet in this package (see DESIGN_ZH.md 3.5)")
    ap.add_argument("--limit-nodes", type=int, default=None, help="Further limit number of nodes to run (for testing)")
    ap.add_argument("--resume", action="store_true", help="Resume from cached node outputs in output_dir")
    ap.add_argument("--audit-opens", action="store_true", help="Record every file path opened during the run to <output-dir>/<run-tag>/<sample>/open_audit.jsonl")
    args = ap.parse_args()

    samples = list(PUBLISHED_SAMPLES) if args.sample == "all" else [args.sample]

    all_summaries = {}
    for s in samples:
        audit = None
        if args.audit_opens:
            sample_out_dir = Path(args.output_dir).resolve() / args.run_tag / s
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            audit = OpenAudit(sample_out_dir / "open_audit.jsonl")
            audit.install()  # sys.addaudithook cannot be removed; fine for one run_sample call
        try:
            res = run_sample(s, args)
        finally:
            if audit is not None:
                audit.close()
                print(f"  [audit-opens] {audit.count} opens recorded to {sample_out_dir / 'open_audit.jsonl'}")
        all_summaries[s] = res

    summary_path = Path(args.output_dir) / args.run_tag / "all_samples_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                s: {
                    "status": m["status"],
                    "wall_s": m["total_wall_seconds"],
                    "calc_s": m["total_calc_seconds"],
                    "node_count": m["node_count"],
                    "comparison_count": m["comparison_count"],
                    "match_count": m["match_count"],
                    "mismatches": m["mismatch_count"],
                    "not_run_nodes": m["not_run_nodes"],
                    "final_image": m["final_image"],
                }
                for s, m in all_summaries.items()
            },
            f,
            indent=2,
        )
    print(f"\nAll samples completed. Global summary saved to {summary_path}")


if __name__ == "__main__":
    main()
