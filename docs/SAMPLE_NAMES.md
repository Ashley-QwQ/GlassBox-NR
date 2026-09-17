# Public sample names

This package's data paths and code use the project's internal sample
identifiers (`c1-C-img-full`, `d23-n2-nat320x240`) -- unchanged, so that
`--sample` on the command line and every file under `data/params/320x240/`
matches exactly what the acceptance run manifests and this package's own
tests reference. Renaming them in the tree itself is a later docs-only
step, not done in this round.

Public-facing documentation (this README, the P0 priority note, and
anything written for people outside the project) refers to the same two
samples by these names instead:

| Internal name | Public name |
|---|---|
| `c1-C-img-full` | `sample-1-reference` |
| `d23-n2-nat320x240` | `sample-3-procedural-scene` |

`c1-C-b1-full` (internally "sample-2", the B1-injected-input sample) is not
published in any form -- see `tools/filter_nodes_sample2.py` and
`data/maps/PROVENANCE.md`.
