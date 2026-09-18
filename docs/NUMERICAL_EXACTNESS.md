# What "byte-exact" means here

## Definition

For each compared exit, "byte-exact" means:

> The output buffer this package computes on CPU has exactly the same length
> and exactly the same SHA-256 as the buffer the reference runtime produced
> at the same point in the pipeline, given the same weights and the same
> input.

There is no tolerance, no error metric and no "close enough". Two buffers
either have the same SHA-256 or they do not. A single differing bit anywhere
in a multi-megabyte buffer counts as a mismatch.

## What is compared

- **Exits.** Each node's declared outputs in `data/nodes.json` (77 per
  sample across B00–B69). These are the raw buffers passed between nodes,
  in the runtime's own storage layout and number formats, not a decoded or
  re-normalised view of them.
- **Reference.** The corresponding buffers produced by the native runtime
  on the capture GPU. Their SHA-256 values, together with the runtime DLL,
  weights and GPU identity, are published in
  `reference/reference_hashes.json`. The reference data itself is not
  published.
- **This package's side.** `tools/run_pipeline.py` records the byte count
  and SHA-256 of every output in `run_manifest.json`;
  `tools/verify_reference_hashes.py` compares the two.

## Why this is a strong condition

Floating-point results depend on the order of operations, on where values
are rounded, and on how intermediate precisions are converted. Two correct
implementations of the same network usually agree only approximately.
Matching byte for byte means reproducing the reference's arithmetic
exactly, not just its mathematics.

## What it does not mean

- It does not extend beyond the scope in [`SCOPE.md`](SCOPE.md): the two
  published samples, nodes B00–B69, one resolution, one frame, one set of
  weight bytes.
- It is relative to the weight bytes identified by the `WEIGHTS_HT`
  sha256, not to a product name or version label.
- It does not by itself say anything about environments other than the one
  documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
- It says nothing about the final image, which this preview does not
  produce.
