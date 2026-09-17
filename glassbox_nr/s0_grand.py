"""S0: Identity, Interface and External-Input Gate for B00 -> S157 grand pipeline.

Reads ONLY from rental-transfer/astra-d12-20260912/*.zip (min and full) and receipt files.
Extracts:
  anchors/<sample>/inputs/colour.rgba16
  anchors/<sample>/inputs/pre.params
  anchors/<sample>/inputs/params/launch-{L:04d}.params
  anchors/<sample>/targets/<target_name> (FOR COMPARATOR ONLY; NEVER read by predictors)
Writes:
  anchors-trusted.json

CRITICAL MANDATE:
  DO NOT print any target sha or target bytes to stdout (only count and PASS/FAIL).
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, os, struct, time, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
ROOT = ROUND.parents[1]
CAP = ROOT / "rental-transfer/astra-d12-20260912"
DLL = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
WS_BASE = 0x18800000
WS_LEN = 15711232
W, H = 320, 240
SAMPLES = ("c1-C-img-full", "d23-n2-nat320x240")

# Phase C: this sample's package lives in a different rental-transfer
# directory, under a different filename prefix than its round-internal
# sample id. {sample: (dir, filename_prefix)}; falls back to (CAP, sample)
# for the original two samples.
SAMPLE_PACKAGE = {
    "d23-n2-nat320x240": (ROOT / "rental-transfer/astra-d23-20260912/net", "n2-C-d23-nat320x240-full-any"),
}

def package_paths(sample):
    pkg_dir, prefix = SAMPLE_PACKAGE.get(sample, (CAP, sample))
    return pkg_dir / f"{prefix}.min.zip", pkg_dir / f"{prefix}.full.zip", pkg_dir / f"{prefix}.receipt.json"

# F32: d0 independently byte-diffed all 158 launches' params across the
# three raw packages and found the ONLY differences anywhere are: launch 1
# word 1 (some run-to-run counter/id, differs even between the two original
# samples, not sample-content-dependent)  Every other launch/word is byte-identical
# across all three samples. This makes every launch/block-indexed static
# table in the codebase (VIT function/weight names, D36.SPECS origins, the
# B09-B21/B66 origins dicts, etc.) safe to keep as a single shared table --
# PROVIDED this invariant itself is checked every time S0 runs, since a
# future 4th sample is not guaranteed to uphold it. REFERENCE_SAMPLE is an
# explicit constant (not SAMPLES[0]) so reordering SAMPLES can never silently
# change which sample the whole invariant is checked against; asserted below
# to actually be a member of SAMPLES.
REFERENCE_SAMPLE = "c1-C-img-full"
assert REFERENCE_SAMPLE in SAMPLES, f"REFERENCE_SAMPLE {REFERENCE_SAMPLE!r} must be one of SAMPLES {SAMPLES!r}"
PARAMS_INVARIANT_EXCEPTIONS = {
    (1, 1): set(SAMPLES),
}

def params_invariant_violations(sample, params_words, reference_params_words):
    """Pure function (no I/O, no global state) so it's directly unit-testable:
    returns a list of human-readable violation strings, empty if `sample`'s
    params_words agree word-for-word with reference_params_words except for
    PARAMS_INVARIANT_EXCEPTIONS. Used by run_s0() below and by
    tests/test_cli_rejections.py's params_invariant negative tests.
    """
    violations = []
    sample_launches = set(params_words)
    ref_launches = set(reference_params_words)
    if sample_launches != ref_launches:
        violations.append(
            f"launch_set_mismatch sample_only={sorted(sample_launches - ref_launches)} "
            f"reference_only={sorted(ref_launches - sample_launches)}"
        )
    for L in sorted(sample_launches & ref_launches):
        words = params_words[L]
        ref_words = reference_params_words[L]
        if len(words) != len(ref_words):
            violations.append(f"launch={L} word_count sample={len(words)} reference={len(ref_words)}")
            continue
        for idx, (w, rw) in enumerate(zip(words, ref_words)):
            if w == rw:
                continue
            allowed = PARAMS_INVARIANT_EXCEPTIONS.get((L, idx))
            if allowed is not None and sample in allowed:
                continue
            violations.append(
                f"launch={L} word={idx} sample_value={hex(w)} reference_value={hex(rw)}; "
                f"not in registered PARAMS_INVARIANT_EXCEPTIONS"
            )
    return violations

CHECKS = []

def check(name, ok, detail=None):
    CHECKS.append(dict(check=name, ok=bool(ok), detail=str(detail) if detail is not None else None))
    if not ok:
        raise SystemExit(f"INVALID {name}: {detail}")

def find_texture(zf, prefix):
    """F30: the D23 package names its texture snapshots with a format/size
    suffix (e.g. '...id1.fmt10-w320-h240-pitch2560.tex') instead of the
    plain '...id1.rgba16' used by the D12 package -- same bytes, same
    614,400-byte RGBA16F content (verified), different capture-tool naming.
    Resolve by prefix instead of hardcoding the extension, so this keeps
    working regardless of which naming convention a given package uses.
    """
    matches = [n for n in zf.namelist() if n.startswith(prefix)]
    if len(matches) != 1:
        raise SystemExit(f"INVALID find_texture: expected exactly 1 match for prefix {prefix!r}, got {matches}")
    return zf.read(matches[0])

def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha_f(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()

# Registered exceptions to the F17 hard rule "exit == next node's native input".
# In this standalone release package, no injection overrides are active.
DECLARED_INJECTIONS = {}

# Targets definition for comparator: (launch, word_index, size_bytes, filename)
TARGETS_SPEC = (
    (2, 27, 3276800, "b0_full.tin"),
    (2, 31, 819200, "b0_pooled.inpview"),
    (3, 1, 819200, "B1-Y.TIN"),
    (4, 1, 819200, "B2-Y.TIN"),
    (5, 1, 819200, "B3-Y.TIN"),
    (6, 1, 819200, "B4-full.TIN"),
    (6, 8, 409600, "B4-pooled.raw"),
    (7, 1, 409600, "B5-Y.tin64"),
    (8, 1, 409600, "B6-Y.tin64"),
    (9, 1, 409600, "B7-Y.tin64"),
    (10, 1, 409600, "B8-full.tin64"),
    (10, 9, 204800, "B8-pooled.inpview128"),
    (11, 1, 204800, "B9-Y.tin128"),
    (12, 1, 204800, "B10-Y.tin128"),
    (13, 1, 204800, "B11-Y.tin128"),
    (14, 1, 204800, "B12-Y.tin128"),
    (15, 1, 204800, "B13-Y.tin128"),
    (16, 1, 204800, "B14-full.tin128"),
    (16, 9, 102400, "B14-pooled.inpview256"),
    (17, 1, 102400, "B15-Y.tin256"),
    (18, 1, 102400, "B16-Y.tin256"),
    (19, 1, 102400, "B17-Y.tin256"),
    (20, 1, 102400, "B18-Y.tin256"),
    (21, 1, 102400, "B19-Y.tin256"),
    (22, 1, 102400, "B20-Y.tin256"),
    (23, 1, 102400, "B21-Y.tin256"),
    (24, 1, 102400, "B22-full.bin"),
    (24, 9, 73728, "B22-pooled.bin"),
    (28, 2, 73728, "B23-C28-word2.bin"),
    (32, 2, 73728, "B24-K32-word2.bin"),
    (36, 2, 73728, "B25-K36-word2.bin"),
    (40, 2, 73728, "B26-K40-word2.bin"),
    (44, 2, 73728, "B27-K44-word2.bin"),
    (48, 2, 73728, "B28-K48-word2.bin"),
    (52, 2, 73728, "B29-K52-word2.bin"),
    (56, 2, 73728, "B30-L56-word2.bin"),
    (57, 1, 65536, "B30-L57-word1.bin"),
    # F17: these were w1 (the launch's own before/after-unchanged in-block
    # residual, written by an EARLIER launch -- launch60/65/70/75/80/85/90/95
    # respectively) not w2 (the actual block exit, consumed as w0 by the next
    # launch: 64/69/74/79/84/89/94/99). See FAILURE_LEDGER F17 and
    # experiments/parity-claude-grand-e1-audit-20260914/triage-b31.json.
    (63, 2, 65536, "B31-L63-word2.bin"),
    (68, 2, 65536, "B32-L68-word2.bin"),
    (73, 2, 65536, "B33-L73-word2.bin"),
    (78, 2, 65536, "B34-L78-word2.bin"),
    (83, 2, 65536, "B35-L83-word2.bin"),
    (88, 2, 65536, "B36-L88-word2.bin"),
    (93, 2, 65536, "B37-L93-word2.bin"),
    (98, 2, 65536, "B38-L98-word2.bin"),
    (99, 1, 65536, "L99-word0.bin"),
    (100, 2, 73728, "B39-Y.raw"),
    (104, 2, 73728, "B40-K104-word2.bin"),
    (108, 2, 73728, "B41-K108-word2.bin"),
    (112, 2, 73728, "B42-K112-word2.bin"),
    (116, 2, 73728, "B43-K116-word2.bin"),
    (120, 2, 73728, "B44-K120-word2.bin"),
    (124, 2, 73728, "B45-K124-word2.bin"),
    (128, 2, 73728, "B46-K128-word2.bin"),
    (132, 2, 73728, "K132-word2.bin"),
    (133, 1, 102400, "B48-Y.raw"),
    (134, 1, 102400, "B49-Y.raw"),
    (135, 1, 102400, "B50-Y.raw"),
    (136, 1, 102400, "B51-Y.raw"),
    (137, 1, 102400, "B52-Y.raw"),
    (138, 1, 102400, "B53-Y.raw"),
    (139, 1, 102400, "B54-Y.raw"),
    (140, 1, 102400, "B55-Y.raw"),
    (141, 1, 204800, "B56-Y.raw"),
    (142, 1, 204800, "B57-Y.raw"),
    (143, 1, 204800, "B58-Y.raw"),
    (144, 1, 204800, "B59-Y.raw"),
    (145, 1, 204800, "B60-Y.raw"),
    (146, 1, 204800, "B61-Y.raw"),
    (147, 1, 409600, "B62-Y.raw"),
    (148, 1, 409600, "B63-Y.raw"),
    (149, 1, 409600, "B64-Y.raw"),
    (150, 1, 409600, "B65-Y.raw"),
    (151, 1, 819200, "B66-Y.raw"),
    (152, 1, 819200, "B67-Y.raw"),
    (153, 1, 819200, "B68-Y.raw"),
    (154, 1, 819200, "B69-Y.raw"),
)

def run_s0(allow_update=False):
    t_start = time.time()
    anchors_dir = ROUND / "anchors"
    if (ROUND / "anchors-trusted.json").exists() and not allow_update:
        check("anchors_not_already_extracted", False, "anchors-trusted.json already exists; refusing to overwrite without --allow-update")
    
    trusted = {
        "format": "grand-e2e-anchors-trusted/1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "specimen": "C",
        "runtime_dll_sha256": DLL,
        "samples": {}
    }
    
    target_by_launch = {}
    for L, w_idx, sz, name in TARGETS_SPEC:
        target_by_launch.setdefault(L, []).append((w_idx, sz, name))

    # F17 hard rule: an exit target must be provably the same bytes the
    # consumer actually reads, not just "a name that sounds right" -- the
    # B31 bug (word1 picked instead of word2) passed the old by-name self
    # check because that check only compared a target against ITS OWN
    # producing node's inputs, never against what the downstream consumer
    # actually loads. Two checks, built once from nodes.json (launch
    # numbers/roles are unaffected by the F17 filename bug, so this is safe
    # to build from the nodes.json already on disk):
    #   A. "main"-role exits that feed forward as w0 of the very next launch:
    #      verify the extracted bytes equal cur[w0_ptr] at that launch, using
    #      the SAME cur we already have (extraction launch L, consumer's
    #      first launch == L+1).
    #   B. "skip"-role exits: verify the bytes at their own fixed offset are
    #      still there, unmodified, immediately before the consumer's first
    #      launch runs (checked using 'cur' as it stands right before that
    #      launch's delta entry is applied).
    nodes_json_path = ROUND / "nodes.json"
    downstream_w0_check = {}   # output name -> consumer's first launch (only when == producing launch + 1)
    skip_trigger = {}          # consumer's first launch -> [output names to re-check as "skip"]
    if nodes_json_path.exists():
        nodes_data_global = json.loads(nodes_json_path.read_text(encoding="utf-8"))

        # F32 step 2: PARAMS_INVARIANT_EXCEPTIONS carves out launch 1 (word 1)
        # across samples. That's only safe if no node actually reads
        # launch-0000 or launch-0001 params as a "params" input anywhere --
        # otherwise a node could silently be fed a value that legitimately
        # varies per sample without any of the per-node cross-checks below
        # ever looking at it. Assert that hasn't happened.
        for n_info in nodes_data_global:
            for inp in n_info.get("inputs", []):
                if inp.get("role") == "params" and inp.get("source") == "static":
                    p = inp.get("path", "")
                    check(f"no_node_reads_launch0000_or_0001_params_via_{n_info.get('node')}",
                          "launch-0000.params" not in p and "launch-0001.params" not in p, p)

        outputs_role = {out["name"]: out.get("role", "") for n in nodes_data_global for out in n.get("outputs", [])}
        producing_launch = {L: name for L, w_idx, sz, name in TARGETS_SPEC}
        producing_launch_by_name = {name: L for L, w_idx, sz, name in TARGETS_SPEC}
        for n_info in nodes_data_global:
            launches = n_info.get("launches") or []
            if not launches:
                continue
            for inp in n_info.get("inputs", []):
                if inp.get("source") != "cpu":
                    continue
                up_file = inp.get("file")
                if inp.get("role") == "main" and outputs_role.get(up_file, "").startswith("main_to"):
                    prod_L = producing_launch_by_name.get(up_file)
                    if prod_L is not None and launches[0] == prod_L + 1:
                        downstream_w0_check[up_file] = launches[0]
                elif inp.get("role") == "skip":
                    skip_trigger.setdefault(launches[0], []).append(up_file)
    else:
        nodes_data_global = []

    reference_params_words = {}  # filled when REFERENCE_SAMPLE is processed; SAMPLES[0] runs first
    reference_launch_functions = {}  # ditto, for the F32 step-3 cross-check below

    for sample in SAMPLES:
        print(f"=== S0: Extracting sample {sample} ===")
        mzp, fzp, rcp = package_paths(sample)
        
        check(f"{sample}_receipt_exists", rcp.exists(), str(rcp))
        check(f"{sample}_min_zip_exists", mzp.exists(), str(mzp))
        check(f"{sample}_full_zip_exists", fzp.exists(), str(fzp))
        
        rec = json.loads(rcp.read_text(encoding="utf-8"))
        min_sha = sha_f(mzp)
        full_sha = sha_f(fzp)
        check(f"{sample}_min_sha", min_sha == rec["min"]["sha256"], min_sha)
        check(f"{sample}_full_sha", full_sha == rec["full"]["sha256"], full_sha)
        
        with zipfile.ZipFile(mzp) as mz, zipfile.ZipFile(fzp) as fz:
            check(f"{sample}_min_crc", mz.testzip() is None)
            check(f"{sample}_full_crc", fz.testzip() is None)
            
            run_json = json.loads(mz.read("run.json"))
            check(f"{sample}_dll", run_json.get("runtime_dll_sha256") == DLL, run_json.get("runtime_dll_sha256"))
            job = run_json["job"]
            check(f"{sample}_dims", (job["width"], job["height"]) == (W, H), (job["width"], job["height"]))
            check(f"{sample}_reset_0", len(job["frames"]) == 1 and job["frames"][0]["reset"] == 1)
            
            # Colour Texture Gate verification
            upload = mz.read("frame-00/session.upload.rgba16")
            check(f"{sample}_upload_len", len(upload) == 614400, len(upload))
            # Trailing '.' matters: 'id1' is itself a prefix of 'id10'.
            k2_id1 = find_texture(fz, "frame-00/session.texture-f0-k2-before-pre-id1.")
            k155_id1 = find_texture(fz, "frame-00/session.texture-f0-k155-after-id1.")
            k157_id10 = find_texture(fz, "frame-00/session.texture-f0-k157-after-id10.")
            
            colour_gate_pass = (upload == k2_id1 == k155_id1 == k157_id10)
            check(f"{sample}_colour_texture_gate_equal", colour_gate_pass, "ID1/ID10 do not equal upload colour")
            
            # F32 step 3: record each launch's real function name from this
            # sample's OWN capture (session.nvapi.tsv, one 'launch' row per
            # launch index, format: launch<TAB>index<TAB>ptr<TAB>function<TAB>...).
            # Only adds data -- does not touch any arithmetic/decoding path.
            # Used by grand_node_executor.py's run_vit_launch to assert this
            # sample's actual launch function matches the shared structural
            # table (which is keyed by launch number only, sourced from
            # REFERENCE_SAMPLE, and safe per params_invariant above).
            nvapi_text = mz.read("session.nvapi.tsv").decode("utf-8", errors="replace")
            launch_functions = {}
            for line in nvapi_text.splitlines():
                cols = line.split("\t")
                if cols and cols[0] == "launch":
                    launch_functions[int(cols[1])] = cols[3]
            check(f"{sample}_launch_functions_count", len(launch_functions) == 158, len(launch_functions))
            check(f"{sample}_launch_functions_indices_complete", sorted(launch_functions) == list(range(158)))
            if sample == REFERENCE_SAMPLE:
                reference_launch_functions = dict(launch_functions)
            else:
                # Independent cross-check: nvapi.tsv is a separate capture
                # stream from the params words, so agreeing with them
                # corroborates the params_invariant conclusion rather than
                # just re-deriving it. This is exactly what run_vit_launch
                # relies on to use REFERENCE_SAMPLE's static VIT table safely.
                for L, fn in launch_functions.items():
                    check(f"{sample}_launch_function_matches_reference_launch{L}",
                          fn == reference_launch_functions.get(L),
                          (L, fn, reference_launch_functions.get(L)))

            # Read all params words
            params_raw = {}
            params_words = {}
            for L in range(158):
                pname = f"frame-00/session.launch-{L:04d}.params"
                if pname in mz.namelist():
                    raw = mz.read(pname)
                    params_raw[L] = raw
                    params_words[L] = struct.unpack(f"<{len(raw)//8}Q", raw)
            check(f"{sample}_params_count", len(params_raw) == 158, len(params_raw))

            # F32: params_invariant hard check. Every launch's params must be
            # word-for-word identical to REFERENCE_SAMPLE's, except the
            # registered exceptions in PARAMS_INVARIANT_EXCEPTIONS. This is
            # what makes every launch/block-indexed static table elsewhere in
            # the codebase (VIT function/weight names, D36.SPECS origins, the
            # B09-B21/B66 origin dicts) safe to share across samples without
            # a per-node fixture lookup. Runs for every sample including the
            # reference itself (trivially all-equal), so a bug in the
            # exception table can't hide by only ever being checked against
            # itself.
            if sample == REFERENCE_SAMPLE:
                reference_params_words = dict(params_words)
            else:
                check(f"{sample}_params_invariant_reference_available", bool(reference_params_words),
                      f"REFERENCE_SAMPLE {REFERENCE_SAMPLE!r} must be processed before {sample!r}; check SAMPLES order")
                violations = params_invariant_violations(sample, params_words, reference_params_words)
                check(f"{sample}_params_invariant", not violations,
                      f"(reference={REFERENCE_SAMPLE!r}) " + "; ".join(violations[:5]) + (f" ... and {len(violations)-5} more" if len(violations) > 5 else ""))

            # Reject sample-2 if requested
            if sample == "c1-C-b1-full":
                raise ValueError("sample-2 ('c1-C-b1-full') is excluded from publication; configuration removed per release decision")
            injected_extraction_meta = None
            injection_spec = DECLARED_INJECTIONS.get(sample)

            # Rebuild checkpoints and extract target slices
            man = json.loads(fz.read("frame-00/delta/manifest.json"))
            check(f"{sample}_delta_format", man["format"] == "dlssnr-workspace-delta/1")
            page = man["page_size"]
            pages = fz.read("frame-00/delta/pages.bin")
            
            cur = None
            extracted_targets = {}
            extracted_provenance = {}
            for e in man["entries"]:
                L = e["launch"]

                # F17 hard rule B: right before launch L's own delta entry is
                # applied, 'cur' is exactly the "before" state of launch L --
                # the same moment the consumer's launch would read its skip
                # input. Verify any already-extracted skip exit is still
                # byte-identical at its own recorded offset.
                if cur is not None and L in skip_trigger:
                    for sk_name in skip_trigger[L]:
                        if sk_name in extracted_targets and sk_name in extracted_provenance:
                            sk_off = extracted_provenance[sk_name]["offset"]
                            sk_bytes = extracted_targets[sk_name]
                            sk_sz = len(sk_bytes)
                            in_bounds = 0 <= sk_off and sk_off + sk_sz <= len(cur)
                            seg = bytes(cur[sk_off : sk_off + sk_sz]) if in_bounds else None
                            check(f"{sample}_{sk_name}_skip_persists_to_launch{L}", in_bounds and seg == sk_bytes)

                if e["kind"] == "base":
                    cur = np.frombuffer(fz.read("frame-00/delta/" + e["base_file"]), np.uint8).copy()
                else:
                    n = e["total_pages"]
                    pad = n * page - e["size"]
                    buf = np.concatenate([cur, np.zeros(pad, np.uint8)]) if pad else cur.copy()
                    view = buf.reshape(n, page)
                    lo = e["page_offset"] * page
                    blob = np.frombuffer(pages[lo : lo + e["dirty_pages"] * page], np.uint8)
                    view[np.asarray(e["pages"], np.int64)] = blob.reshape(e["dirty_pages"], page)
                    cur = buf[: e["size"]].copy()

                check(f"{sample}_cp_{L}_sha", sha_b(cur.tobytes()) == e["sha256"])

                if L in target_by_launch:
                    w = params_words[L]
                    for w_idx, sz, name in target_by_launch[L]:
                        ptr = w[w_idx]
                        off = ptr - WS_BASE
                        check(f"{sample}_{name}_bounds", 0 <= off and off + sz <= len(cur), (hex(ptr), sz))
                        extracted_targets[name] = bytes(cur[off : off + sz])
                        # Real capture timing: this slice is read from 'cur' as it
                        # stands immediately AFTER launch L's delta entry has been
                        # applied above -- i.e. genuinely "after", not asserted.
                        extracted_provenance[name] = {
                            "launch": L, "word": w_idx, "pointer": hex(ptr),
                            "offset": off, "timing": "after"
                        }
                        # F17 hard rule A: for a "main"-role exit that feeds
                        # forward as w0 of the immediate next launch, verify
                        # against that launch's own w0 pointer NOW, while
                        # 'cur' is still exactly this launch's "after" state
                        # (== the consumer's "before" state, since nothing
                        # else touches 'cur' in between).
                        injected = DECLARED_INJECTIONS.get(sample)
                        is_registered_exception = injected is not None and injected["replaces_output"] == name
                        if name in downstream_w0_check and not is_registered_exception:
                            next_L = downstream_w0_check[name]
                            w0_ptr = params_words[next_L][0]
                            off0 = w0_ptr - WS_BASE
                            in_bounds0 = 0 <= off0 and off0 + sz <= len(cur)
                            seg0 = bytes(cur[off0 : off0 + sz]) if in_bounds0 else None
                            check(f"{sample}_{name}_matches_downstream_w0_launch{next_L}", in_bounds0 and seg0 == extracted_targets[name])
                        elif is_registered_exception:
                            # Cross-check the DECLARED natural pointer (what B01 would
                            # have read absent the injection) against this output's own
                            # extracted offset, so the exception table itself can't drift
                            # silently out of sync with the real pointer.
                            nat_ptr = int(injected_extraction_meta["natural_input_pointer"], 16) if injected_extraction_meta else None
                            if nat_ptr is not None:
                                check(f"{sample}_{name}_declared_natural_pointer_matches_own_offset", (nat_ptr - WS_BASE) == off)

            # Tail texture targets: read directly from the session capture
            # files (not the checkpoint delta chain), so "launch"/"word" are
            # not applicable; timing is "after" because these are the
            # post-launch texture/output snapshots the capture tool wrote.
            extracted_targets["B70-id9.rgba16"] = find_texture(fz, "frame-00/session.texture-f0-k155-after-id9.")
            extracted_provenance["B70-id9.rgba16"] = {"launch": 155, "word": None, "pointer": None, "offset": None, "timing": "after"}
            extracted_targets["S157-id2.rgba16"] = mz.read("frame-00/session.output.rgba16")
            extracted_provenance["S157-id2.rgba16"] = {"launch": 157, "word": None, "pointer": None, "offset": None, "timing": "after"}
            check(f"{sample}_total_targets_extracted", len(extracted_targets) == len(TARGETS_SPEC) + 2, len(extracted_targets))
            
            # Write to disk
            s_in_dir = anchors_dir / sample / "inputs"
            s_par_dir = s_in_dir / "params"
            s_tgt_dir = anchors_dir / sample / "targets"
            s_par_dir.mkdir(parents=True, exist_ok=True)
            s_tgt_dir.mkdir(parents=True, exist_ok=True)
            
            (s_in_dir / "colour.rgba16").write_bytes(upload)
            (s_in_dir / "pre.params").write_bytes(params_raw[2])
            for L, raw in params_raw.items():
                (s_par_dir / f"launch-{L:04d}.params").write_bytes(raw)
            # F32/F33: this sample's own launch_functions, as a proper
            # per-sample static input (anchors/<sample>/inputs/..., same
            # scoping as the params files above) instead of a predictor
            # reading the whole-round anchors-trusted.json directly -- that
            # file bundles ALL samples' data together, so freezing/exposing
            # it wholesale to a predictor process would leak information
            # about the OTHER sample (params_invariant status, target
            # sha256s) that "restricted" is specifically supposed to keep
            # out. This file is properly sample-scoped like every other
            # anchors/<sample>/inputs/ file.
            (s_in_dir / "launch-functions.json").write_text(
                json.dumps({str(L): fn for L, fn in launch_functions.items()}), encoding="utf-8"
            )
                
            # Self-check: none of the extracted targets can be identical to any input of its producing node.
            # Match by role (exact), not by substring-guessing the static path string.
            nodes_json_path = ROUND / "nodes.json"
            if nodes_json_path.exists():
                nodes_data = json.loads(nodes_json_path.read_text(encoding="utf-8"))
                for n_info in nodes_data:
                    n_name = n_info["node"]
                    for out in n_info.get("outputs", []):
                        out_name = out["name"]
                        if out_name in extracted_targets:
                            tgt_b = extracted_targets[out_name]
                            for inp in n_info.get("inputs", []):
                                role = inp.get("role")
                                if inp["source"] == "static":
                                    if role == "colour":
                                        check(f"{sample}_{n_name}_{out_name}_differs_from_input_colour", tgt_b != upload)
                                    elif role == "params":
                                        check(f"{sample}_{n_name}_{out_name}_differs_from_input_params", tgt_b != params_raw[2])
                                elif inp["source"] == "cpu":
                                    up_f = inp.get("file")
                                    if up_f and up_f in extracted_targets:
                                        check(f"{sample}_{n_name}_{out_name}_differs_from_input_{up_f}", tgt_b != extracted_targets[up_f])

            targets_dict = {}
            for k, v in extracted_targets.items():
                prov = extracted_provenance.get(k, {})
                targets_dict[k] = {
                    "bytes": len(v),
                    "sha256": sha_b(v),
                    "launch": prov.get("launch"),
                    "word": prov.get("word"),
                    "pointer": prov.get("pointer"),
                    "offset": prov.get("offset"),
                    "timing": prov.get("timing", "after")
                }

            sample_meta = {
                "label": f"MAIN_CHAIN_{sample.upper().replace('-', '_')}",
                "upload_colour_sha256": sha_b(upload),
                "colour_texture_gate_verified": True,
                "pre_params_sha256": sha_b(params_raw[2]),
                "targets_count": len(extracted_targets),
                "targets": targets_dict,
                "launch_functions": {str(L): fn for L, fn in launch_functions.items()},
                "params_invariant": {
                    "reference_sample": REFERENCE_SAMPLE,
                    "verified": True,
                    "registered_exceptions_applicable": sorted(
                        f"launch{L}_word{idx}" for (L, idx), allowed in PARAMS_INVARIANT_EXCEPTIONS.items() if sample in allowed
                    ),
                }
            }
            if injection_spec is not None:
                sample_meta["declared_injection"] = {}
            trusted["samples"][sample] = sample_meta
            
            for k, v in extracted_targets.items():
                (s_tgt_dir / k).write_bytes(v)
                
    trusted["total_checks"] = len(CHECKS)
    trusted["all_checks_passed"] = True
    (ROUND / "anchors-trusted.json").write_text(json.dumps(trusted, indent=2), encoding="utf-8")
    
    # Print summary: ONLY counts and PASS/FAIL, NO target SHAs or target bytes!
    print("--------------------------------------------------")
    print(f"S0 EXTRACTION COMPLETE: ALL {len(CHECKS)} CHECKS PASS")
    print(f"Samples processed: {len(SAMPLES)}")
    print(f"Inputs extracted per sample: colour.rgba16, pre.params, 158 launch params")
    print(f"Targets extracted per sample: {len(TARGETS_SPEC) + 2} targets (stored for comparator ONLY)")
    print(f"Total time elapsed: {time.time()-t_start:.2f}s")
    print("--------------------------------------------------")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-update", action="store_true", help="Allow updating targets and overwriting anchors-trusted.json")
    args = ap.parse_args()
    run_s0(allow_update=args.allow_update)
