# VIT (B31-B38, L99) map data provenance

Source round: `experiments/parity-claude-b32-b38-vit-20260913/` (committed,
see `commitment/freeze-code-manifest.json`). Full per-file source/output/
script sha accounting is in `release/_private/DERIVED_FILES_MAP.json`
(145 rows); this file is the public-facing summary.

User decision (option B): publish derived copies of the files that mixed
load-bearing computation data with disassembly-derived content or sample-2
data, instead of shipping the originals verbatim.

## What's here

- `maps/L64/` .. `maps/L98/`, `maps/b31/`: per-launch address-map `.npz`
  files (byte-identical copies -- pure int32/int64 offset arrays, verified
  to carry no strings, addresses, or sample-2 content) and derived
  `*-wiring-and-checks.json` / `*-templates-and-checks.json` files (keep
  only the top-level keys `{chains, gather_sources, nodes, status,
  store_table, templates}` -- the only fields the VIT kernels actually
  read, confirmed by a live access-trace running the full B00->L99 chain --
  then every nested key literally named `pc` is recursively removed).
- `maps/L64/maps.json` .. `maps/L98/maps.json`: derived per-launch records,
  keeping only the 7 fields `glassbox_nr/kernels/vit/map_record.py`'s
  `verify_record()`/`select()` ever read (`status`, `files`, `file_sha256`,
  `function`, `interp_sha256`, `launch`, `shim_sha256`). The originals'
  `row` field carried real SASS branch-label data and, for 7 of the 35
  records (the "expand"-launch family), a `state_slot_polls` subtree keyed
  by both samples including `c1-C-b1-full` (sample-2) -- neither survives.
- `maps/b31/launch58-address-map.json`, `maps/launch99-address-map.json`:
  byte-identical copies (pure offset-permutation tables, zero disassembly-
  signal hits on inspection).
- `maps/{b31,L66,L71,L76,L81,L86,L91,L96,gate-b31}/.../*-sass-numeric-check.json`
  and the 14 `*-sass-family-check.json` equivalents: derived whitelist
  certificates (keep `format`/`valid`/`status`/`match`/`*_sha256` plus plain
  int/bool scalars only; the runtime gate only ever reads `.status`).
- `interp/ptx_interp60.py`, `ptx_interp61.py`, `ptx_interp62.py`,
  `ptx_interp63.py`, `ptx_interp_expand.py`: derived copies of the offline
  PTX/SASS-text interpreter tools (never imported or executed by the
  package -- `map_record.py` only hashes them). The one hardcoded `$L__BB`
  default value for each file's `--loop-head` CLI argument is removed
  (the argument is now required, which `_arg()` already enforces); one
  docstring mention of a concrete label is rephrased neutrally. **Known
  consequence**: `ptx_interp_expand.py:438` still contains
  `s0['params']['c1-C-b1-full']['hex']`, a self-check comparing sample-2's
  captured params against sample-1's when this tool is run standalone
  against an S0 shim. Running this tool against the *published* (sample-2-
  stripped) shims under `interp/s0/` will raise `KeyError` on that line --
  expected and harmless, since this tool is a hash subject only and is
  never runnable from within the package itself.
- `interp/s0/s0-launch59.json` .. `s0-launch98.json`: derived shims
  (every original held `params["c1-C-b1-full"]`, i.e. sample-2's launch
  params; stripped, nothing else changed).
- `identity/s0-identity.json`: derived, 524,349 -> 42,706 bytes. Keeps only
  `samples["c1-C-img-full"].launches[*].{function, roles.*.{kind,role,tensor}}`
  -- confirmed by static reading of `node_executor.py`'s `run_vit_launch()`
  to be the entire read surface. Drops the sample-2 subtree entirely, the
  2323-entry `checks` dict, the top-level `weights` registry, and every
  per-role field beyond `kind`/`role`/`tensor` (word, value, bytes,
  workspace_offset, params_hex, params_sha256, etc.).

## Not packaged

`maps/gate-b31/` and `maps/neg/`: never opened by `map_record.records()`/
`select()` (which only look under `maps/L<NN>/`) or by the B31 hardcoded
paths in `node_executor.py`.

## Build scripts (release/p2-runnable/tools/, not part of the published package)

`migrate_vit_npz.py`, `derive_vit_maps.py`, `generate_sass_whitelist_certs.py`,
`derive_vit_records.py`, `derive_vit_shims.py`, `derive_ptx_interp.py`,
`derive_identity.py`, `build_derived_files_map.py`. Every script's own sha256
and every source/output sha256 pair is recorded in
`release/_private/DERIVED_FILES_MAP.json`.
