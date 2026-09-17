# Code Commitment and Derived Files Cross-Reference

## 1. The 997-File Code Commitment

At the conclusion of the end-to-end parity milestone, the entire research codebase was frozen as an append-only commitment:

- **Freeze Code Root SHA-256**: `83d1a5c274652471487dcc770cfbcba1627e072452cb70551a0b3484c3502c5a`
- **Manifest Location**: `commitment/freeze-code-manifest.json` (997 files)
- **Purpose**: Cryptographically seals the exact files and dependencies that demonstrated end-to-end bit-exact parity against RTX 5090D hardware captures.
- **Note on Sample-2 in Commitment Manifest**: 承诺清单中的 sample-2 路径只是承诺记录，对应文件不随包发布。Research paths referencing sample-2 (`c1-C-b1-full`) recorded in `commitment/freeze-code-manifest.json` exist solely as historical verification evidence and are excluded from publication per user release decisions.

## 2. Relationship to This Runnable Package

The runnable release package (`pkg/`) is a standalone, self-contained subset of the committed codebase designed to run entirely on standard CPU environments without requiring access to internal research scratchpads or proprietary binary captures.

- **Direct Copies**: Unmodified runtime code, maps, and tables are verbatim byte copies matching the hashes in `freeze-code-manifest.json`.
- **Derived Copies (Option B Policy)**: Per user decision on publication boundaries, files that originally intermingled runtime data with non-essential disassembly text fragments, offline test scaffolding, or unreleased sample data were derived using automated, auditable scripts.
- **Total Derived Files**: 241 files (0 commitment mismatches against `freeze-code-manifest.json`).

## 3. Derivation Categories and Invariance Guarantees

| Category | Count | Rule ID | Invariance Guarantee |
|---|---|---|---|
| `sass-check-whitelist-certificate` | 24 | `R1-sass-whitelist` | Keeps format/valid/status fields; runtime only evaluates `.status` |
| `vit-wiring-templates-derived` | 40 | `R2-keep6-strip-pc` | Keeps 6 top-level keys used by kernels; removes nested disassembly labels; bit-exact execution |
| `vit-map-record-derived` | 35 | `R3-record-whitelist-7-keys` | Keeps 7 fields required by `map_record.py`; removes unread trace logs; hashes updated to match derived interps |
| `vit-shim-derived` | 40 | `R4-strip-sample2-key` | Removes unreleased sample-2 parameter keys; sample-1 runtime unchanged |
| `vit-ptx-interp-derived-copy` | 5 | `R5-interp-spdx-and-drop-loop-head` | Adds SPDX AGPL-3.0-or-later; replaces hardcoded loop labels with required CLI arguments; strips sample-2 check |
| `vit-identity-derived` | 1 | `R6-identity-keep-function-and-roles` | Retains runtime roles for published sample-1; strips unreleased sample-2 and registry metadata |
| `split16-contraction-derived` | 16 | `R7-contraction-keep-contract-only` | Retains numeric `contract` table; strips unread SASS site disassembly listings |
| `b39-map-derived` | 1 | `R8-b39-drop-interp` | Drops unread `interp` diagnostic subtree; calculation unchanged |
| `code-disassembly-comments-cleaned` | 3 | `R9-disassembly-comments-to-neutral` | Replaces disassembly comments with neutral docstrings; AST matches; zero regex hits (d20 follows two-step chain: Step 1 import migration, Step 2 comment clean) |
| `split16-chain-strip-sample2` | 65 | `R10-split16-chain-strip-sample2` | Strips unreleased sample-2 intermediate tensor lines and sample entry from split16 freeze indices |
| `s0-grand-strip-sample2` | 1 | `R11-s0-grand-strip-sample2` | Strips unreleased sample-2 injection configuration and pointer rebind exceptions from s0_grand.py |

## 4. Verification and Audit

The full machine-readable mapping is available in [`docs/DERIVED_FILES_MAP_PUBLIC.json`](DERIVED_FILES_MAP_PUBLIC.json).
Each entry records: published path, committed source path, committed source SHA-256, output SHA-256, derivation script SHA-256, and rule ID.

Numerical verification: both published samples (`c1-C-img-full` and `d23-n2-nat320x240`) execute through all 71 pipeline nodes from B00 through B69, producing 77/77 bit-exact output matches against reference hardware outputs.
