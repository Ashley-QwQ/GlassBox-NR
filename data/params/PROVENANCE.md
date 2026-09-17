# Per-launch parameter data provenance (320x240)

`data/params/320x240/<sample>/inputs/...` -- byte copies of the 17 static
inputs `nodes.json` declares with role `params` or `launch_functions`, for
each of the two published samples (`c1-C-img-full`, `d23-n2-nat320x240`).
Per-step launch parameters are one of the four categories the user
explicitly approved for publication (task card line 19: "CPU 代码；320×240
每步 params；地址映射...；硬件实测数学表"); `DESIGN_ZH.md` plans this exact
`data/params/320x240/` location. This is the one static-input category that
is *not* left to local user regeneration the way `colour.rgba16` is -- the
per-launch params encode fixed geometry/workspace-layout data (grid,
block_dim, workspace addresses -- already-approved "address map" content),
not raw captured pixels, and are near-identical across samples (see the
project's own PARAMS_INVARIANT checks).

Enumerated from `nodes.json` itself (all 73 nodes' `inputs`, not a hand
list): 18 distinct static paths total, of which 17 fall in this category
(`pre.params`; 15 `params/launch-NNNN.params` files for B48/B49/B50/B51/
B52/B53/B54/B55/B56/B62/B67/B68/B69/B70/S157; `launch-functions.json`) and
1 (`colour.rgba16`) does not -- that one alone stays under the locally
generated `anchors/` dir. No sample-2 (`c1-C-b1-full`) file exists in this
set; `nodes.json`'s sample_inputs overrides for sample-2 were already
stripped by `tools/filter_nodes_sample2.py`.

Content-swept before copying (`release/_private/CONTENT_SWEEP_2B_PARAMS.json`):
`launch-functions.json` -- 158 entries per sample, zero disassembly-signal
regex hits, zero sample-2 markers (the native kernel function names
themselves are already published throughout this project, e.g. in the
derived `identity/s0-identity.json`). The 15+15 `.params` files (raw
u64/float parameter words, not JSON -- the disassembly regexes don't apply
to binary content the same way) -- checked byte-for-byte for embedded
6+-byte ASCII runs and the sample-2 markers: zero hits across all 30 files.

Every file's sha256 is checked against its r01 anchors source
(`release/_private/vit_derivation_manifests/migrated-params-manifest.json`)
and against `commitment/freeze-code-manifest.json` where a matching entry
exists (none currently do -- the commitment manifest covers code and static
dependencies, not per-sample captured anchor data, so these come back
`not-in-commitment-manifest`, which is expected, not a discrepancy).

`glassbox_nr/tools/run_pipeline.py`'s static-input resolver routes the
`params` and `launch_functions` roles here instead of the `--test-input-dir`
anchors directory (the `colour` role is the only one that stays there),
with a runtime assertion that a `params`/`launch_functions` path never
resolves under the anchors directory.
