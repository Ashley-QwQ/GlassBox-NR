# Public sample names

This package's data paths and code use the project's internal sample
identifiers (`c1-C-img-full`, `d23-n2-nat320x240`) -- unchanged, so that
`--sample` on the command line and every file under `data/params/320x240/`
matches exactly what the acceptance run manifests and
`reference/reference_hashes.json` reference.

Public-facing documentation refers to the same two samples by these names:

| Internal name | Public name |
|---|---|
| `c1-C-img-full` | `sample-1-reference` |
| `d23-n2-nat320x240` | `sample-3-procedural-scene` |

The numbering skips sample-2 on purpose. The project's second sample
(`c1-C-b1-full`, a variant whose block-1 input was injected from a native
capture rather than computed from a colour image) is not published in any
form: its data files, map entries and node-input overrides were removed
from this package before release.
