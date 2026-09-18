# GlassBox-NR (runnable preview package)

[中文版](README_ZH.md)

**Byte-exact CPU reconstruction preview of a neural-rendering denoiser
inference chain: B00–B69 (71 / 73 nodes), 77 compared exits × 2 published
samples, zero byte mismatches against the audited reference captures.
B70 / S157 and final-image output are not included in this public preview.**

An independently reconstructed CPU implementation of the inference
execution path. The inference kernels and execution semantics are
independently reconstructed; weight extraction and unpacking use the
vendored MLX-DLSS tooling under its original Apache-2.0 license. No vendor
source, headers, or SDK were used.

## Scope and requirements

This package needs:

- Your own, lawfully obtained copy of the proprietary runtime DLL. It is
  never shipped, downloaded, or bundled by this package.
- Python 3.13 and the dependencies in `requirements.txt` (numpy, torch,
  safetensors; a CPU-only torch build is sufficient).

It runs entirely on CPU. No GPU, no network access, no telemetry.

**Runnable range**: nodes B00 through B69 of the 73-node chain, for the two
published samples (`c1-C-img-full`, `d23-n2-nat320x240`; public names are
listed in [`docs/SAMPLE_NAMES.md`](docs/SAMPLE_NAMES.md)). Every one of the
77 compared exits in that range is byte-identical to the reference output
for both samples. The reference sha256 of all 154 exits is published in
[`reference/reference_hashes.json`](reference/reference_hashes.json), so you
can check this yourself (see "How to verify" below).

**Not runnable yet**: nodes B70 and S157, the final output-tail stage. **This
package does not currently produce the pipeline's final image.** Every run
stops at B69. Why, and what is planned, is in
[`docs/B70_S157_STATUS.md`](docs/B70_S157_STATUS.md).

This is a preview, not a finished or "official" release. No claim here
means every code path has been audited to the same depth, and no claim here
is "100%" anything. The scope above is exactly what has been verified;
[`docs/SCOPE.md`](docs/SCOPE.md) lists what is and is not covered, and
[`docs/NUMERICAL_EXACTNESS.md`](docs/NUMERICAL_EXACTNESS.md) defines what
"byte-exact" means here.

## How to run

```
python -B tools/extract_weights.py /path/to/your/nvngx_dlssnr.dll
python -B tools/make_inputs.py
python -B tools/run_pipeline.py --sample all --run-tag my-run
python -B tools/verify_reference_hashes.py runs/my-run
```

`extract_weights.py` refuses to proceed if your DLL's weight resource
doesn't match the sha256 this package's kernels were verified against (see
its own docstring).

`make_inputs.py` generates each published sample's colour input from a
closed-form procedural formula (no external asset, nothing taken from a
native capture) into `anchors/<sample>/inputs/colour.rgba16`, and refuses to
write it unless its sha256 equals the input the reference results were
produced from.

`run_pipeline.py` runs the chain and writes
`runs/my-run/<sample>/run_manifest.json`: every node's output byte count
and sha256, and whether the run was partial. `--run-tag` is required and
must not already exist under `--output-dir` (pass `--resume` to continue an
unfinished run under the same tag). This package never silently overwrites
a previous run's evidence.

## How to verify

1. `make_inputs.py` (above) checks that the generated colour inputs are
   byte-identical to the ones the reference results were produced from. Use
   `--check` to verify without writing anything.
2. `verify_reference_hashes.py` compares every exit in your run manifests
   against `reference/reference_hashes.json` and prints `N/154 MATCH`. It
   exits non-zero unless all 154 exits are present and match.
3. Optionally run `run_pipeline.py` with `--audit-opens` and inspect
   `runs/<tag>/<sample>/open_audit.jsonl` to confirm the run only touched
   this package's own directory tree.

`reference/reference_hashes.json` contains hashes only, no reference tensor
data. Its header records the reference identity: runtime DLL version and
sha256, `WEIGHTS_HT` sha256, and the capture GPU and architecture. Note that
154 compared outputs are not 154 graph nodes: several nodes expose more than
one compared exit (71 nodes → 77 exits per sample, × 2 samples = 154).

The canonical environment and full reproduction steps are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Code commitment and derived files

A complete cryptographic commitment of 997 research codebase files (`freeze_code_sha256 = 83d1a5c2...`) is published in `commitment/freeze-code-manifest.json`.
For a detailed explanation and per-file audit cross-referencing published files with their committed sources, see [`docs/COMMITMENT_AND_DERIVED_FILES.md`](docs/COMMITMENT_AND_DERIVED_FILES.md) and [`docs/DERIVED_FILES_MAP_PUBLIC.json`](docs/DERIVED_FILES_MAP_PUBLIC.json).

## Priority and timestamps

What this project can and cannot prove about dates, and which third-party
timestamps it relies on, is in [`PRIORITY.md`](PRIORITY.md).

## License

- Source code written by this project: **AGPL-3.0-or-later** (`LICENSE`).
- Documentation and data files written or derived by this project:
  **CC-BY-4.0** (`LICENSE-docs`).
- `third_party/mlxdlss/`: MLX-DLSS, its own **Apache-2.0** license (see
  `third_party/mlxdlss/LICENSE`, `NOTICE`, `MODIFICATIONS.md`).

See `NOTICE` for the full breakdown.

## What isn't published, and why

See `NOTICE` and each directory's own `PROVENANCE.md` for exactly what is
and isn't included and why (weights, native captures, and raw disassembly
text are never published; several data files that originally mixed
needed computation data with disassembly-derived diagnostic content are
published here as derived copies with that content removed -- each such
file's `PROVENANCE.md` entry names the rule applied and the resulting
sha256, so the exact transformation is checkable without needing anything
outside this distribution).
