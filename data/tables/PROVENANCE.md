# Hardware-measured table provenance

Both tables in this directory are this project's own data, produced by
exhaustively probing specific GPU instructions across their measured input
domain (not derived from, or contained in, any NVIDIA binary or weight file).
Classification: this project's own hardware-measurement data, CC-BY-4.0.

| File | sha256 | Instruction measured | Produced by |
|---|---|---|---|
| `rsq-domain.npz` | `06ffffcad0796e18eb51cce1166712d57b33a2138857bfa791d0e36f35210e62` | RSQ (reciprocal square root) | this project's own RSQ hardware-probe round (`rsq_oracle.py`) |
| `rcp-domain.npz` | `055e26f1b2324491e47b7faf02f3b82c4da71fd6d7cb4843b1275f61f95b2b59` | RCP (reciprocal) | this project's own RCP hardware-probe round (`verify_rcp.py`) |

Each file stores three arrays: `input_half_bits` (the half-precision input
codes probed), `native_float_bits` (the hardware's float32 output for each
input, bit pattern), and `native_half_bits` (the same, rounded to half). These
are consumed by `glassbox_nr/kernels/core/sm120_b1.py`'s `Sm120Tables.load()`,
which verifies each file's sha256 against the constants above before use.

These tables are unrelated to (and were confirmed by hash to be identical to,
not merely similarly named as) the LG2/EX2 post-processing domain tables used
elsewhere in this project (`lg2-postprocess-domain.npz`,
`ex2-negative-sliver.npz`) -- those measure different MUFU instructions and
are not part of this package's data set unless a later kernels group needs
them, in which case they will get their own provenance entry here.

## B00 noise-generation MUFU tables (found late: a smoke-test run, not the
static import test, is what surfaces this class of gap -- `mufu_tables.py`
and `noise_reference.py` were never migrated in group (b); `d20_backend.py`'s
`FRONTEND` constant pointed at `experiments/parity-block0-frontend-20260911`
until fixed to the sibling directory these two files now live in,
`glassbox_nr/kernels/encoder_windows/`)

| File | sha256 | Instruction measured | Consumed by |
|---|---|---|---|
| `lg2-boxmuller-domain.npz` | `fef83eafe83f1bba00aecdf0da40210a485a20ab2471aa79aba6a7813230d127` | MUFU.LG2 | `mufu_tables.py`'s `Table("LG2")`, via `noise_reference.py`'s `gaussians()` |
| `sqrt-boxmuller-domain.npz` | `dee50a83274fb0a33d373544116ca856ad6e092b86639bb916251413b53f02ed` | MUFU.SQRT | `mufu_tables.py`'s `Table("SQRT")` |
| `cos-boxmuller-preduction.npz` | `bbb94623c04259bd2df9219646d276871de831269bec59f84ce656f483e3a2a0` | COS (pre-reduction argument) | `mufu_tables.py`'s `Table("COS")` |
| `sin-boxmuller-preduction.npz` | `792b344a022b5febd625b17d681bded03e56d9b34ab2366e643a117dba0204fa` | SIN (pre-reduction argument) | `mufu_tables.py`'s `Table("SIN")` |

Each stores `input_float_bits` / `native_float_bits` bit-pattern pairs plus a
`meta` JSON blob; lookup is an exact membership test that raises
`OutsideMeasuredDomain` for any unmeasured input (see `mufu_tables.py`).
`cos-boxmuller-domain.npz` and `sin-boxmuller-domain.npz` exist in the
original `tables/` directory but are not referenced by `mufu_tables.py`'s
`FILES` dict (dead for this round's actual lookup path) and were not copied.
