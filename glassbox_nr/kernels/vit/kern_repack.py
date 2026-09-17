"""repack kernels (launch 58 cc_vit_1d_repack_2d_to_1d_fp8, launch 99 cc_vit_1d_repack_1d_to_2d_fp8).

compute() body copied line by line from experiments/parity-astra-d34-r01-repack-20260913/predict_launch58.py (main, the
mapping loop); the map file is a parameter.  No file I/O here.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors


def compute(raw, mapping_doc):
    mapping = mapping_doc["output_to_input_byte_offsets"]
    if len(raw) != 65536 or len(mapping) != 16384:
        raise SystemExit("input/map size invalid")
    output = bytearray(65536)
    for output_chunk, input_offset in enumerate(mapping):
        if input_offset % 4 or not (0 <= input_offset <= 65532):
            raise SystemExit(f"invalid map entry {output_chunk}: {input_offset}")
        dst = output_chunk * 4
        output[dst:dst + 4] = raw[input_offset:input_offset + 4]
    return bytes(output)
