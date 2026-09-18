# B70 and S157: why this preview stops at B69

The pipeline has 73 nodes. This preview runs 71 of them (B00 through B69,
including L99). The last two, B70 and S157, form the output-tail stage that
produces the final image. They are not included, so **this package does not
produce the pipeline's final image**.

## Why they are not included

For B00 through B69, every node has its own numerical implementation in
this package: Python code that computes the node's output from its inputs
and the weights.

B70 and S157 were implemented differently in the project's research
codebase. Instead of a per-node numerical implementation, their backends
read a text disassembly of the corresponding native GPU kernels at runtime
and execute it instruction by instruction through a general-purpose
interpreter. The data that backend consumes is disassembly-derived
instruction content.

This project does not publish disassembly-derived content. Its publication
policy covers four categories (code, per-launch parameters, address maps
and measured hardware tables), and a disassembly listing is none of them. Shipping those two backends would mean shipping that text, so they
are left out, and the preview ends where the per-node numerical
implementations end.

## What the package does instead

- `tools/run_pipeline.py` defaults to `--stop-after B69`.
- Passing `--stop-after B70` or `--stop-after S157` fails immediately with
  an explicit message. It does not try to read a file that isn't there.
- Every `run_manifest.json` records the run as partial and lists B70 and
  S157 under `not_run_nodes`. No final-image hash is produced or claimed.
- `reference/reference_hashes.json` covers B00 through B69 only.

## What is planned

B70 and S157 are to be reimplemented as ordinary numerical code that reads
no disassembly text, and checked byte for byte against the existing
results, the same way every other node was. That work is tracked
separately and is not part of this preview. Until it is published and
listed in `CHANGELOG.md`, the scope of this package is B00 through B69.
