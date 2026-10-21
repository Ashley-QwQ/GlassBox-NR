# B70 and S157 (output-tail stage): provenance and where it lives

The inference chain has 73 nodes; the last two, B70 and S157, form the output-tail stage that produces the final image.

## Provenance of the B70 / S157 logic

- The logic of B70 and S157 was derived by analysing the NVIDIA binary.
- The code that implements it is an independent rewrite, written as ordinary numerical code. It does not read, embed or execute any
  listing of the vendor binary.
- The rewrite was verified byte-for-byte against GPU measurements.

This statement is the whole provenance claim for the output-tail stage. No program counters, instruction forms or register names from
the analysis are published, in code, comments, data files or documentation.

## Where the output-tail stage lives

v2.0.0-preview carries B70 and S157 as independently written C++ source, in `cpp/chain_outtail_optm/` (with the associated compiled
module in `modules/`). Together with the other chain modules it produces the final image; the final-image hashes are listed in the
README. The v1 Python reference package, which stopped at B69, is no longer in HEAD; it stays reachable through the tag
`v0.1.0-preview`.
