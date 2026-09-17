# GlassBox-NR (runnable preview package)

[中文版](README_ZH.md)

A from-scratch CPU reimplementation of a proprietary neural-rendering
inference pipeline, reverse-engineered independently (no vendor source,
headers, or SDK), reproducing the pipeline's intermediate and final outputs
byte-for-byte against the audited reference results -- for the part of the
pipeline this package currently covers.

## Scope and requirements

This package needs:

- Your own, lawfully obtained copy of the proprietary runtime DLL. It is
  never shipped, downloaded, or bundled by this package.
- Python 3.13 and the dependencies in `requirements.txt` (numpy, torch-cpu,
  safetensors).

It runs entirely on CPU. No GPU, no network access, no telemetry.

**Runnable range**: nodes B00 through B69 of the 73-node chain, for the two
published samples (`c1-C-img-full`, `d23-n2-nat320x240`; public names are
listed in `docs/SAMPLE_NAMES.md`). Every one of the 77 compared exits in
that range is byte-identical to the project's own audited reference results
for both samples, verified through this package's own tooling (see "How to
verify" below).

**Not runnable yet**: nodes B70 and S157 -- the final output-tail stage --
because their backend parses raw native SASS disassembly text at runtime,
and that text is not published (see `DESIGN_ZH.md` section 3.5). **This
package does not currently produce the pipeline's final image.** A
pure-Python reimplementation of B70/S157 is a separate, tracked follow-up;
until it lands, every run stops at B69.

This is a preview, not a finished or "official" release. No claim here
means every code path has been audited to the same depth, and no claim here
is "100%" anything -- the scope above is exactly what has been verified,
stated as plainly as it can be.

## How to run it

```
python tools/extract_weights.py /path/to/your/nvngx_dlssnr.dll
python -B tools/run_pipeline.py --sample c1-C-img-full --run-tag my-run --stop-after B69
```

`extract_weights.py` refuses to proceed if your DLL's weight resource
doesn't match the sha256 this package's kernels were verified against (see
its own docstring). `run_pipeline.py` generates its own local test input
(`--sample`'s colour image) into `anchors/`, runs the chain, and writes
`runs/my-run/<sample>/run_manifest.json` -- every node's output byte count
and sha256, whether the run was partial, and (if you pass `--compare-dir`)
a per-output match/mismatch against a reference manifest set in the same
layout.

`--run-tag` is required and must not already exist under `--output-dir`
(pass `--resume` to continue an unfinished run under the same tag) -- this
package never silently overwrites a previous run's evidence.

## How to verify

1. Run `python -B release/p2-runnable/generators/verify_inputs.py` (if
   present in your copy) to confirm the published colour-image generator
   reproduces the exact bytes this package's own reference manifests were
   produced against.
2. Run the command above with `--audit-opens`; inspect
   `runs/<tag>/<sample>/open_audit.jsonl` to confirm the run only touched
   this package's own directory tree (plus whatever `--compare-dir` you
   supplied, if any) -- nothing outside it.
3. Compare `run_manifest.json`'s per-node output sha256 values against
   whatever independent reference you trust. This package does not ship
   its own "golden" hash list in this preview; that is a planned addition.

## Code commitment and derived files

A complete cryptographic commitment of 997 research codebase files (`freeze_code_sha256 = 83d1a5c2...`) is published in `commitment/freeze-code-manifest.json`.
For a detailed explanation and per-file audit cross-referencing published files with their committed sources, see [`docs/COMMITMENT_AND_DERIVED_FILES.md`](docs/COMMITMENT_AND_DERIVED_FILES.md) and [`docs/DERIVED_FILES_MAP_PUBLIC.json`](docs/DERIVED_FILES_MAP_PUBLIC.json).

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
