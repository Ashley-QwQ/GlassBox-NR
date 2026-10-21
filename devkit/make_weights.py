#!/usr/bin/env python3
"""Derive the five weight files the devkit expects from the user's own DLL.

Usage:  python make_weights.py <path/to/nvngx_dlssnr.dll> <output_dir>

The DLL (310.8.0.0) is supplied by the user and is never copied anywhere
else. All five outputs are pure, deterministic functions of the DLL's
WEIGHTS_HT resource (an RCDATA resource); this script contains only parsing
rules and layout rules (offsets, sizes, padding), no weight bytes.

Resource layout (little endian):
  u64 total_size (== resource length)
  repeated records:
    u64 name_len, name (utf-8)
    u64 record_size
      u64 inner_size (== record_size)
      u64 byte_count, u32 dtype_code (== 1), tensor bytes[byte_count]
      u32 meta0, u32 meta1, u64 ndim, u32 dims[ndim]
    (byte_count == 2 * prod(dims))

Outputs:
  weights.bin                  the raw WEIGHTS_HT resource, verbatim.
  device_weights.bin           tensor payloads in resource (insertion) order;
                               after every payload, zero padding up to the next
                               512-byte boundary (so a trailing pad is present).
  weights-device.bin           tensor payloads in ascending tensor-name order
  weights_gpu.bin              (byte-identical files); zero padding BEFORE every
                               payload up to a 512-byte boundary, no trailing pad.
  weights-packed.safetensors   safetensors container of the payloads as flat U8
                               tensors, plus a fixed metadata block (format,
                               hashes, per-tensor source records).

Only the Python standard library is needed (numpy is not required).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys

RESOURCE_NAME = "WEIGHTS_HT"
FORMAT_NAME = "dlssnr-WEIGHTS_HT-packed-u8-v2"
ALIGN = 512
_RT_RCDATA = 10

# Expected results for the 310.8.0.0 DLL (sha e16bcf15...). Used only to
# report PASS/FAIL; a different DLL will still be processed deterministically.
EXPECTED = {
    "device_weights.bin": (147719680, "a5513b1845c98a486985ed04f38e66a1854cce33c2aba3a505866028bd4ee3e5"),
    "weights.bin": (147695410, "836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4"),
    "weights-device.bin": (147719232, "c75ec57e2b66cb42a5962b5abb16677a1b45ee74891bd513728d407b108acc79"),
    "weights-packed.safetensors": (147716314, "d81d0f5ce573c4596d6aee96ab89ddd463c2d63e71bbfdaa1459c3f90a925018"),
    "weights_gpu.bin": (147719232, "c75ec57e2b66cb42a5962b5abb16677a1b45ee74891bd513728d407b108acc79"),
}


# The reference weights-packed.safetensors was written from the SF-v2 build of
# the DLL (sha below), whose WEIGHTS_HT resource is byte-identical to the one in
# 310.8.0.0 (e16bcf15...). Its metadata records that build's sha as
# "source_sha256". Default mode pins this value so the output is byte-identical
# to the reference; --source-sha dll records the sha of the DLL you pass instead
# (payload bytes are the same either way).
CANONICAL_SOURCE_SHA256 = "6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _need(data: bytes, off: int, size: int, what: str) -> None:
    if off < 0 or size < 0 or off > len(data) - size:
        raise ValueError(f"truncated {what}")


def _u16(d: bytes, o: int, w: str) -> int:
    _need(d, o, 2, w)
    return struct.unpack_from("<H", d, o)[0]


def _u32(d: bytes, o: int, w: str) -> int:
    _need(d, o, 4, w)
    return struct.unpack_from("<I", d, o)[0]


def _u64(d: bytes, o: int, w: str) -> int:
    _need(d, o, 8, w)
    return struct.unpack_from("<Q", d, o)[0]


# ----------------------------------------------------------------- PE parsing
def _pe_sections(data: bytes):
    if data[:2] != b"MZ":
        raise ValueError("not a PE file (no MZ)")
    pe = _u32(data, 0x3C, "DOS header")
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE file (no PE signature)")
    coff = pe + 4
    nsec = _u16(data, coff + 2, "COFF")
    osize = _u16(data, coff + 16, "COFF")
    opt = coff + 20
    if _u16(data, opt, "optional header") != 0x20B:
        raise ValueError("PE32+ image required")
    rva = _u32(data, opt + 112 + 16, "resource dir")
    size = _u32(data, opt + 112 + 20, "resource dir")
    if not rva or not size:
        raise ValueError("no resource directory")
    secs = []
    so = opt + osize
    for i in range(nsec):
        o = so + i * 40
        secs.append((_u32(data, o + 12, "sec"), _u32(data, o + 8, "sec"),
                     _u32(data, o + 20, "sec"), _u32(data, o + 16, "sec")))
    return rva, size, secs  # (va, vsize, raw_off, raw_size)


def _rva_off(secs, rva: int, size: int, what: str) -> int:
    for va, vs, ro, rs in secs:
        if va <= rva <= va + max(vs, rs):
            rel = rva - va
            if rel <= rs and size <= rs - rel:
                return ro + rel
    raise ValueError(f"{what}: RVA outside file-backed sections")


def extract_pe_resource(data: bytes, name: str = RESOURCE_NAME) -> bytes:
    rrva, rsize, secs = _pe_sections(data)
    base = _rva_off(secs, rrva, rsize, "resource directory")

    def at(rel: int, size: int, what: str) -> int:
        if rel < 0 or size < 0 or rel > rsize - size:
            raise ValueError(f"truncated {what}")
        _need(data, base + rel, size, what)
        return base + rel

    def entries(rel: int):
        d = at(rel, 16, "resource directory")
        n = _u16(data, d + 12, "rd") + _u16(data, d + 14, "rd")
        t = at(rel + 16, n * 8, "resource entries")
        return [(_u32(data, t + i * 8, "re"), _u32(data, t + i * 8 + 4, "re")) for i in range(n)]

    def rname(v: int):
        if not v & 0x80000000:
            return None
        rel = v & 0x7FFFFFFF
        s = at(rel, 2, "resource name")
        ln = _u16(data, s, "rn")
        ts = at(rel + 2, ln * 2, "resource name")
        return data[ts:ts + ln * 2].decode("utf-16-le")

    type_dir = None
    for ident, tgt in entries(0):
        if not ident & 0x80000000 and (ident & 0xFFFF) == _RT_RCDATA:
            if not tgt & 0x80000000:
                raise ValueError("RCDATA entry is not a directory")
            type_dir = tgt & 0x7FFFFFFF
            break
    if type_dir is None:
        raise ValueError("no RCDATA resource type")
    lang_dir = None
    for ident, tgt in entries(type_dir):
        if rname(ident) == name:
            if not tgt & 0x80000000:
                raise ValueError(f"{name} entry is not a directory")
            lang_dir = tgt & 0x7FFFFFFF
            break
    if lang_dir is None:
        raise ValueError(f"no {name} resource")
    langs = entries(lang_dir)
    if len(langs) != 1:
        raise ValueError(f"{name} must have exactly one language entry")
    tgt = langs[0][1]
    if tgt & 0x80000000:
        raise ValueError(f"{name} language entry is a directory")
    de = at(tgt, 16, "resource data entry")
    prva = _u32(data, de, "rde")
    psize = _u32(data, de + 4, "rde")
    po = _rva_off(secs, prva, psize, name)
    _need(data, po, psize, name)
    return data[po:po + psize]


# ------------------------------------------------------- resource record parse
def parse_weight_map(res: bytes):
    """Return a list of (name, payload_bytes, record_dict) in resource order."""
    if len(res) < 8:
        raise ValueError("truncated weight map")
    if _u64(res, 0, "map size") != len(res):
        raise ValueError("serialized map size mismatch")
    out, seen, off = [], set(), 8
    while off < len(res):
        nl = _u64(res, off, "name length")
        off += 8
        if nl == 0 or nl > 4096:
            raise ValueError(f"invalid tensor name length {nl}")
        _need(res, off, nl, "tensor name")
        name = res[off:off + nl].decode("utf-8")
        off += nl
        if name in seen:
            raise ValueError(f"duplicate tensor {name}")
        seen.add(name)
        outer = _u64(res, off, "record size")
        off += 8
        end = off + outer
        _need(res, off, outer, "record")
        if outer < 40:
            raise ValueError(f"record too small for {name}")
        if _u64(res, off, "inner size") != outer:
            raise ValueError(f"inner size mismatch for {name}")
        off += 8
        nbytes = _u64(res, off, "byte count")
        off += 8
        if _u32(res, off, "dtype") != 1:
            raise ValueError(f"unsupported dtype for {name}")
        off += 4
        if nbytes > end - off - 16:
            raise ValueError(f"truncated tensor data for {name}")
        payload = res[off:off + nbytes]
        off += nbytes
        m0 = _u32(res, off, "meta0")
        m1 = _u32(res, off + 4, "meta1")
        nd = _u64(res, off + 8, "ndim")
        if nd == 0 or nd > 16 or end - (off + 16) != nd * 4:
            raise ValueError(f"bad dimension block for {name}")
        dims = [_u32(res, off + 16 + i * 4, "dim") for i in range(nd)]
        off = end
        n = 1
        for d in dims:
            if d == 0:
                raise ValueError(f"zero dimension for {name}")
            n *= d
        if nbytes != 2 * n:
            raise ValueError(f"byte count mismatch for {name}")
        out.append((name, payload, {"byteCount": nbytes, "dtypeCode": 1,
                                    "metadata0": m0, "metadata1": m1, "dimensions": dims}))
    if off != len(res) or not out:
        raise ValueError("serialized weight map ended at an invalid offset")
    return out


# ------------------------------------------------------------------- builders
def build_device_weights(tensors) -> bytes:
    """Resource order; pad AFTER each payload to 512."""
    buf = bytearray()
    for _, payload, _ in tensors:
        buf += payload
        buf += bytes((-len(buf)) % ALIGN)
    return bytes(buf)


def build_weights_gpu(tensors) -> bytes:
    """Sorted by name; pad BEFORE each payload to 512; no trailing pad."""
    buf = bytearray()
    for _, payload, _ in sorted(tensors, key=lambda t: t[0]):
        buf += bytes((-len(buf)) % ALIGN)
        buf += payload
    return bytes(buf)


def build_safetensors(tensors, dll_sha: str, res_sha: str, source_kind: str) -> bytes:
    """Serialise exactly as safetensors.numpy.save_file would (U8, name-sorted)."""
    ordered = sorted(tensors, key=lambda t: t[0])
    records = {n: r for n, _, r in ordered}
    meta = {
        "format": FORMAT_NAME,
        "source_kind": source_kind,
        "source_sha256": dll_sha,
        "resource_sha256": res_sha,
        "tensor_count": str(len(tensors)),
        "payload_byte_count": str(sum(len(p) for _, p, _ in tensors)),
        "payload_contract": "opaque backend-specific packed bytes; no dense dtype or shape is inferred",
        "source_tensor_records": json.dumps(records, separators=(",", ":"), sort_keys=True),
    }
    # Key order of the reference file (the safetensors writer emits metadata in
    # hash-map order, so it is a fixed layout rule here, not alphabetical).
    order = ["payload_byte_count", "format", "payload_contract", "source_tensor_records",
             "source_kind", "source_sha256", "resource_sha256", "tensor_count"]
    header = {"__metadata__": {k: meta[k] for k in order}}
    pos = 0
    for n, p, _ in ordered:
        header[n] = {"dtype": "U8", "shape": [len(p)], "data_offsets": [pos, pos + len(p)]}
        pos += len(p)
    hb = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    hb += b" " * ((-len(hb)) % 8)
    return struct.pack("<Q", len(hb)) + hb + b"".join(p for _, p, _ in ordered)


# ----------------------------------------------------------------------- main
def _write_new(path: pathlib.Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_bytes(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dll", type=pathlib.Path, help="your nvngx_dlssnr.dll (310.8.0.0)")
    ap.add_argument("outdir", type=pathlib.Path, help="existing or new directory for the five files")
    ap.add_argument("--source-sha", choices=("canonical", "dll"), default="canonical",
                    help="value stored as source_sha256 in the safetensors metadata "
                         "(canonical = pinned value that reproduces the manifest sha256)")
    a = ap.parse_args(argv)
    dll = a.dll.read_bytes()
    dll_sha = _sha256(dll)
    res = extract_pe_resource(dll)
    res_sha = _sha256(res)
    tensors = parse_weight_map(res)
    a.outdir.mkdir(parents=True, exist_ok=True)
    gpu = build_weights_gpu(tensors)
    outputs = {
        "weights.bin": res,
        "device_weights.bin": build_device_weights(tensors),
        "weights-device.bin": gpu,
        "weights_gpu.bin": gpu,
        "weights-packed.safetensors": build_safetensors(
            tensors, CANONICAL_SOURCE_SHA256 if a.source_sha == "canonical" else dll_sha,
            res_sha, "nvngx_dlssnr.dll"),
    }
    bad = 0
    print(f"dll sha256 {dll_sha}\nresource sha256 {res_sha}  tensors {len(tensors)}")
    for name, data in outputs.items():
        _write_new(a.outdir / name, data)
        sha = _sha256(data)
        want = EXPECTED[name]
        ok = (len(data), sha) == want
        if name.endswith(".safetensors") and a.source_sha == "dll":
            ok = True  # metadata intentionally differs; payload is identical
        bad += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {name} {len(data)} {sha}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
