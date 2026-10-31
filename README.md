# GlassBox-NR v2.0.1-preview

[中文版](README_ZH.md)

**Aligned to `nvngx_dlssnr.dll` 310.8.0.0.** This is an independent **CPU reference implementation** (C++) of the inference chain of that one
fixed target library (not of any other version, including 310.9.16). Licence: Apache-2.0 (see `NOTICE`).

The claim of this release is **measured bit-exact (byte-for-byte) equivalence** on the listed inputs, configurations and hardware: on
1080p / 1440p / 4K frame 0 (158 launches each), plus the multi-frame chains **1080p f0-f5**, **1440p f0-f3** and **4K f0-f2** in both workspace
fills, the own-CPU chain reproduces the final image of the GPU capture byte for byte. A formal
equivalence proof is **NOT_ESTABLISHED**; nothing here says "proved".

You must supply your own, lawfully obtained copy of `nvngx_dlssnr.dll`. The DLL, the weights and anything derived from the weight
bytes are never part of this repository or of the release assets.

## What is in this repository (git HEAD)

| Directory | Content |
|---|---|
| `cpp/` | C++ sources of the chain modules (10 modules, 57 files) |
| `runtime/`, `third_party/mlxdlss/tools/` | Python runtime glue (loaders, ABI, pins) and the weight extractor (MLX-DLSS, Apache-2.0, verbatim) |
| `devkit/` | user-facing run driver for the 1080p frame-0 chain, plus `payload_layout.py` |
| `params/` | per-launch parameter files (329) |
| `modules/` | compiled Python extension modules (10, `cpython-313-x86_64-linux-gnu`) |
| `docs/FILES_MAP_V2.json` | repo path -> release-whitelist path, used by `payload_layout.py` |

Everything else (about 32 GB when counted per path, 16.6 GB unique) is the **payload**: one compressed volume,
`payload.tar.zst.001` (462,661,132 B), a GitHub Release asset. The v1 Python toolchain (`v0.1.0-preview`) is no longer in HEAD;
it stays reachable through the git tag `v0.1.0-preview` (commit `8f951866`).

## Quickstart (Linux x86-64, Python 3.13, numpy, safetensors, zstd >= 1.4)

1. From the GitHub Release [`v2.0.1-preview`](https://github.com/Ashley-QwQ/GlassBox-NR/releases/tag/v2.0.1-preview) download `payload.tar.zst.001` into the root of this checkout. The manifest is in git at `devkit/data/PAYLOAD_MANIFEST.json`.
2. Verify the volume:
   ```
   echo "c86119619d0638e7bfa94b6ab84f6dd465bba132aaf9d9b4130c648082455b8c  payload.tar.zst.001" | sha256sum -c -
   ```
3. Extract (the `--long=27` flag is required; about 16.6 GB of disk):
   ```
   mkdir extract && cd extract && zstd -d --long=27 < ../payload.tar.zst.001 | tar -x && cd ..
   ```
4. Lay the files out where the driver expects them (hard links; the git-resident files come from this checkout):
   ```
   python devkit/payload_layout.py --manifest devkit/data/PAYLOAD_MANIFEST.json --extracted extract --repo . --out PAYLOAD_DIR --tables TABLES_DIR
   ```
5. Make all five weight files from your own DLL (`weights.bin`, `device_weights.bin`, `weights-device.bin`, `weights_gpu.bin`, `weights-packed.safetensors`; standard library only, a few seconds; the script prints OK/FAIL per file against the manifest size/sha256 and refuses to overwrite existing files):
   ```
   python devkit/make_weights.py /path/to/nvngx_dlssnr.dll WEIGHTS
   ```
   `WEIGHTS` is the `--weights` directory of the next step. The DLL and the generated weights are not part of this release; do not upload them.
6. Run the 1080p frame-0 chain (158 launches) and compare with the reference table. The two frame-0 input files come from the payload (laid out by step 4 under the `--inputs` path below):
   ```
   python devkit/run_1080_f0.py --root WORK --payload PAYLOAD_DIR --weights WEIGHTS --tables TABLES_DIR \
       --inputs PAYLOAD_DIR/kit/data/video-multiframe-r01-20261006-codex/resume2-20261007/chain_port/chain-input-only-v2/frame-000 --threads 4
   ```
   Add `--verify-only` for the relocation and file checks without running. Exit code 0 means every launch matches the reference
   table and the final image sha256 is `764e7dd0d8f440b5afeb36e72fea3d8bc137bef891e8f09dc3476b6fef8d34ef`; 2 = a launch differs,
   3 = the final image differs, 4 = the receipt is incomplete (it must hold launches 0..stop-after exactly once, in order).
   `python devkit/test_run_1080_f0.py` self-tests this comparison without a payload.

## What is claimed, by tier

Release seal: **M2**, the release-clean sub-seal of the unified seal M (content base `9d0a4593...`): no data derived from analysis listings of the vendor binary,
`repack` rewritten as a closed-form address map, derived JSON stripped, path guard literals XOR-encoded.
v2.0.1-preview is M2 plus the HMMA zero-sign restore (see `CHANGELOG.md`): three modules and six C++ sources changed and the payload was re-pinned. On v2.0.1 only 1080p f0 was re-run (user driver, end-to-end from the new volume: 158/158, final `764e7dd0...`); every other row below was measured on the v2.0.0 modules.

| Tier | Where | What |
|---|---|---|
| Numeric gates run on **M2** | 1080p f0, 1440p f0, 4K L0-L99 | 1080 f0: 158 launches byte-equal, final `764e7dd0...`. 1440 f0: 158 launches MATCH, final `20b4f026...`. 4K L0-L99: full workspace plus six POST textures, all segments PASS |
| User driver, payload only | 1080p f0 | end-to-end from the unpacked payload plus the git files: 158/158 launches match the reference table, final `764e7dd0...` |
| **Configuration census only** on M2 (no numeric gate) | 4K L100-L157, 1080 f1-f5, 1440 f3 | 1,738 census records identical to M (up to the mapped paths / shas); no numeric re-run on M2 |
| Evidence carried from **M** (or its parent seal 854a8724), not re-measured on M2 | 4K f0 L0-L157 final `152d6f07...` (with the limits listed in the release notes), 1440p f0-f3, 1080p f0-f5, MSVC build (1080 f0), non-finite line (144/147 held-out finals identical, 3 refused by guard), scene cut f0-f15, 3/3 pre-registered predictions (registered by git commit time, not a third-party timestamp) | |
| Measured on **M** after the v2.0.0-preview release (2026-10-22) | 4K f0-f2, both fills | one process per fill from reset through 474 launches, own CPU state only: full workspace and six POST match at every observed point, finals f0 `152d6f07...`, f1 `9aa7d914...`, f2 `cc32bdf4...`; f0/f1 reproduced a second run exactly; WR B(i) not evaluated at f0 L0/L1 only (native snapshot gap) |
| Sealed-exam test (2026-10-24): frozen code, predictions committed before the answers were opened | 1080p: a sealed 32-frame sequence never seen during development, plus a sealed single frame | 32-frame sequence: 32/32 finals byte-identical (frame 0 shares its input with a known sequence, so the unseen part is frames 1-31: 31/31). Single frame: refused by the non-finite guard on seal M; MATCH on the A1 child seal that adds +Inf semantics. Overall 32 MATCH / 1 REFUSED / 0 DIFF. Parameters fixed within the sequence; parameter-change paths and 1440p/4K sealed exams not yet run |
| Not covered | 4K f3 and later (no per-launch native capture), 1440p f4/f5, held-out 1440p/4K inputs, other DLL versions, multi-parameter streams beyond the listed ones, formal timing | NOT_RUN |
| Formal proof of equivalence | | **NOT_ESTABLISHED** |

Speed and memory figures in the release notes are small samples (one or two runs each), not statistics. This release contains no
showcase renders.

## Measured on

One 1080p frame 0 (158 launches); every run below ended with the final image `764e7dd0...`. Each figure is a single run with
its own threads, load and build, so this is not a benchmark and the rows are not comparable to each other.

| Platform | CPU | Threads | Build | One frame |
|---|---|---|---|---|
| X79 | Xeon E5-2667 v2 (8C/16T) | 4 | g++, seal M2 | 2,822 s |
| Desktop | Core i5-12400F | 12 | MSVC, older seal `ea731776` | 691 s (mean of 8 runs; WSL/GCC 899 s) |
| Mini PC | Core i5-10600T (WSL) | 6 | g++ | 2,566 s |
| Laptop | Core i7-1280P (P-cores only) | 12 | MSVC, seal M | 2,522 s, peak 0.83 GiB |
| Special guest | **Raspberry Pi 3B+** (Cortex-A53, **1 GB RAM**) | - | native g++ 14.2, seal M | **10,283 s (~2.9 h)**, peak RSS **677 MiB**; final image checked, no per-launch gate |

## Provenance of the output-tail stage

The B70 / S157 tail logic was derived by analysing the NVIDIA binary; the code is an independent rewrite and was verified byte-exact
against GPU measurements. See `docs/B70_S157_STATUS.md` and `NOTICE`.

## Priority and history

`PRIORITY.md` explains what can and cannot be proved about dates, and how to verify the v1 commitment against the tag
`v0.1.0-preview`.

## Licence

Code and data: **Apache-2.0** (`LICENSE`). Documentation: **CC-BY-4.0** (`LICENSE-docs`). `third_party/mlxdlss/`: MLX-DLSS, its own
Apache-2.0 licence and notice. Details in `NOTICE`.
