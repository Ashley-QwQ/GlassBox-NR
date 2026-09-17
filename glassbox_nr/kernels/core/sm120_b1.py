"""Block-1 reference backend that models the sm_120 native arithmetic.

WHAT THIS IS
------------
A CPU reference for ONE block -- block 1, the single-window 1-head
``cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8`` kernel -- computed the
way the sm_120 code path computes it, rather than the way ``mlxdlss.model``'s
ordinary torch path computes it.

``mlxdlss.model``'s window_block is an approximate float path. It is NOT the
validated sm_75 exact path and it is NOT this. Nothing here changes it; this
module is selected explicitly or not used at all.

TARGET ARCHITECTURE IS A CHOICE, NOT A DETECTION
------------------------------------------------
"sm_120" names the NATIVE architecture being modelled. It is never inferred
from the machine this runs on, and this module never creates a CUDA context,
never imports a GPU probe, and never patches ``torch.cuda`` or anything else.
It is pure CPU numpy/torch-on-CPU.

WHAT IT REPRODUCES, AND ON WHAT EVIDENCE
----------------------------------------
Chain structure (experiments/parity-astra-d06-b1-static-mma-chains-20260912 and
its revisions/r01, read off the disassembly of the module blob that was dumped
from GPU memory on an RTX 5090 D):

    ffn_w1      1 x QMMA K=32                 C = RZ
    ffn_w2      4 x QMMA K=32, hidden rows
                0-31, 32-63, 64-95, 96-127    C = half(ffn_cos_skip * decode(X))
    qkv_q/k/v   1 x QMMA K=32 each            C = RZ
    score       1 x QMMA K=32                 C = attn_bias
    av          2 x QMMA K=32, window keys
                0-31 then 32-63               C = RZ
    projection  1 x QMMA K=32                 C = half(attn_cos_skip * z),
                                                  z = the UNQUANTISED ffn_w2 half

Accumulator model ``W27_trunc``
(experiments/parity-astra-d04-mma-cancellation-cpu-20260912/revisions/r02): the
non-zero exact E4M3 products together with the non-zero C seed are placed on
q = 2**(floor(log2 max|addend|) - 26) by truncation toward zero, summed
exactly, and rounded once to binary16 with subnormals preserved.

    W = 27 IS A PARAMETER OF THAT MODEL FAMILY. It is not a hardware
    significand width, and this module does not turn it into one.

VERIFIED SCOPE -- ANYTHING ELSE IS REFUSED, NOT APPROXIMATED
------------------------------------------------------------
    block index      1
    head count       1
    window           8 x 8, 64 tokens
    channels         32
    spatial extent   160 x 160
    weights          the 8 block1.layer0 logical tensors, exact shapes below
    reference DLL    nvngx_dlssnr 310.8.SF-v2,
                     sha256 6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927
                     weight resource WEIGHTS_HT sha256
                     836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4

Byte-exactness was established for four (X, weight package) pairs on one card
and one driver. It is not a proof for every possible block-1 input, and this
module never claims one. Out-of-scope calls raise :class:`UnsupportedByBackend`.

DEPLOYMENT DEPENDENCY -- NOT A PURE-PACKAGE MODULE
--------------------------------------------------
Two measured hardware lookup tables are required and are NOT generated at
import time: ``rsq-domain.npz`` and ``rcp-domain.npz``. One copy of each ships
under ``mlxdlss/resources/``; :class:`Sm120Tables` verifies their sha256 and
accepts an explicit directory so a deployment can relocate them. If they are
absent, this backend does not start -- it does not fall back to a float
approximation.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 GlassBox-NR contributors

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from third_party.mlxdlss import model as _model

__all__ = [
    "UnsupportedByBackend",
    "Sm120Tables",
    "SM120B1Reference",
    "TARGET_ARCHITECTURE",
    "ACCUMULATOR_MODEL",
    "SUPPORTED",
    "inpview_decode_codes",
    "inpview_encode_codes",
    "tin_decode_codes",
    "tin_encode_codes",
]

TARGET_ARCHITECTURE = "sm_120"
ACCUMULATOR_MODEL = "W27_trunc"
_W = 27
_UNIT_EXP = -24
_E4M3_UNIT_EXP = -9
_PROD_SCALE = 1 << (_E4M3_UNIT_EXP * 2 - _UNIT_EXP)

SUPPORTED = {
    "block_index": 1,
    "head_count": 1,
    "window_size": 8,
    "channels": 32,
    "height": 160,
    "width": 160,
    # windows are partitioned at origin (0, 0) inside this backend; that is
    # block 1's recovered origin and the only one it was verified at
    "window_origin": (0, 0),
    # consumes and returns PUBLISHED E4M3; an unpublished request is unmodelled
    "publish": True,
    "weight_shapes": {
        "weight1": (32, 128),
        "weight2": (128, 32),
        "ffn_cos_skip": (32,),
        "qkv_weight": (32, 96),
        "attn_bias": (1, 64, 64),
        "attn_scale": (1,),
        "projection_weight": (32, 32),
        "attn_cos_skip": (32,),
    },
}

_RESOURCES = Path(__file__).resolve().parents[3] / "data" / "tables"  # moved out of mlxdlss/resources/ during repackaging; see MODIFICATIONS.md
_TABLE_FILES = {
    "rsq": ("rsq-domain.npz",
            "06ffffcad0796e18eb51cce1166712d57b33a2138857bfa791d0e36f35210e62"),
    "rcp": ("rcp-domain.npz",
            "055e26f1b2324491e47b7faf02f3b82c4da71fd6d7cb4843b1275f61f95b2b59"),
}
_RSQ_FIRST_CODE = 0x0410
_RCP_FIRST_CODE = 0x0410


# One exception class shared with mlxdlss.model, so the model hook and this
# backend refuse with the same type and a caller needs one except clause.
UnsupportedByBackend = _model.UnsupportedByBackend


def _sha256(path: os.PathLike | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------- tables
@dataclass(frozen=True)
class Sm120Tables:
    """The two measured MUFU lookup tables, loaded from an explicit location.

    The tables are hardware measurements, not formulas, so they are a data
    dependency of this backend. ``expected_sha256`` may be relaxed to ``None``
    only by a caller that knowingly supplies its own verified copies; the
    recorded hashes are reported either way.
    """

    rsq_half_bits: np.ndarray
    rcp_half_bits: np.ndarray
    directory: str
    hashes: Mapping[str, str]

    @classmethod
    def load(cls, directory: os.PathLike | str | None = None,
             *, expected_sha256: Mapping[str, str] | None = "default") -> "Sm120Tables":
        folder = Path(directory) if directory is not None else _RESOURCES
        want = ({name: h for name, (_f, h) in _TABLE_FILES.items()}
                if expected_sha256 == "default" else expected_sha256)
        arrays, hashes = {}, {}
        for key, (filename, _default_hash) in _TABLE_FILES.items():
            path = folder / filename
            if not path.exists():
                raise UnsupportedByBackend(
                    "the %s lookup table is missing at %s. This backend is a measured "
                    "reference and has no formula to fall back on; install the resource "
                    "or pass an explicit directory." % (key, path))
            digest = _sha256(path)
            hashes[key] = digest
            if want is not None and key in want and want[key] != digest:
                raise UnsupportedByBackend(
                    "%s has sha256 %s, expected %s -- refusing to compute with an "
                    "unverified table" % (path, digest, want[key]))
            with np.load(path) as data:
                table = np.array(data["native_half_bits"], dtype=np.uint16, copy=True)
            # measured data, shared by every backend built from this object:
            # read-only, so no caller can edit it in place underneath them
            table.setflags(write=False)
            arrays[key] = table
        return cls(rsq_half_bits=arrays["rsq"], rcp_half_bits=arrays["rcp"],
                   directory=str(folder), hashes=hashes)

    def describe(self) -> dict:
        return dict(directory=self.directory, sha256=dict(self.hashes),
                    rsq_entries=int(self.rsq_half_bits.size),
                    rcp_entries=int(self.rcp_half_bits.size),
                    note="measured MUFU tables; a deployment dependency of this backend")


# ---------------------------------------------------------------- E4M3 domain
def _finite_e4m3_units() -> np.ndarray:
    """Every representable finite E4M3 value, in integer units of 2**-9.

    Derived from torch's own float8_e4m3fn decode of all 256 codes, so the
    admission set does not depend on a table transcribed by hand.
    """
    codes = np.arange(256, dtype=np.uint8)
    values = torch.from_numpy(codes).view(torch.float8_e4m3fn).float().numpy()
    finite = values[(codes & 0x7F) != 0x7F].astype(np.float64)
    units = np.rint(finite * 512.0).astype(np.int64)
    if not np.array_equal(units.astype(np.float64) / 512.0, finite):
        raise RuntimeError("the E4M3 decode is not on the 2**-9 grid")
    return np.array(sorted(set(units.tolist())), dtype=np.int64)


_LEGAL_UNITS = _finite_e4m3_units()


def _e4m3_units(value, what: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise UnsupportedByBackend(
            "%s contains NaN or Inf; the measured accumulator model defines nothing "
            "for them and this backend will not guess" % what)
    scaled = array * 512.0
    units = np.rint(scaled).astype(np.int64)
    if not np.array_equal(units.astype(np.float64), scaled):
        raise UnsupportedByBackend("%s is not on the 2**-9 grid" % what)
    legal = np.isin(units, _LEGAL_UNITS)
    if not legal.all():
        bad = array[~legal]
        raise UnsupportedByBackend(
            "%s holds %d value(s) on the 2**-9 grid that are not representable finite "
            "E4M3, e.g. %r" % (what, int((~legal).sum()), float(bad.ravel()[0])))
    return units


def _f16_units(value, what: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise UnsupportedByBackend("%s contains NaN or Inf" % what)
    if not np.array_equal(array.astype(np.float16).astype(np.float64), array):
        raise UnsupportedByBackend(
            "%s holds value(s) that are not exactly representable in binary16" % what)
    scaled = array * np.float64(2.0) ** 24
    units = np.rint(scaled).astype(np.int64)
    if not np.array_equal(units.astype(np.float64), scaled):
        raise UnsupportedByBackend("%s is not on the 2**-24 grid" % what)
    return units


# ------------------------------------------------------------------ the model
def _place_and_sum(terms: np.ndarray, seed: np.ndarray | None) -> np.ndarray:
    magnitude = np.abs(terms)
    largest = magnitude.max(axis=-1)
    if seed is not None:
        largest = np.maximum(largest, np.abs(seed))
    _, bits = np.frexp(largest.astype(np.float64))
    shift = np.maximum((bits - _W).astype(np.int64), 0)
    wide = shift[..., None]
    placed = np.sign(terms) * ((magnitude >> wide) << wide)
    total = placed.sum(axis=-1)
    if seed is not None:
        total = total + np.sign(seed) * ((np.abs(seed) >> shift) << shift)
    return np.where(largest == 0, 0, total)


def _to_f16(total: np.ndarray) -> np.ndarray:
    if np.any(np.abs(total) >= (1 << 52)):
        raise UnsupportedByBackend("accumulator magnitude outside the exact int64 range")
    with np.errstate(over="ignore"):
        return (total.astype(np.float64) * np.float64(2.0) ** _UNIT_EXP).astype(np.float16)


def qmma_rowwise(a, b, c=None, *, target_elements: int = 8_000_000) -> np.ndarray:
    """One W27_trunc accumulation per output element of ``[rows, K] @ [K, N]``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rows, inner = a.shape
    outputs = b.shape[1]
    au = _e4m3_units(a, "A operand")
    bu = _e4m3_units(b, "B operand")
    cu = None if c is None else _f16_units(np.broadcast_to(np.asarray(c), (rows, outputs)),
                                          "C seed")
    out = np.empty((rows, outputs), np.float16)
    chunk = max(1, int(target_elements // max(1, outputs * inner)))
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        terms = (au[start:stop, None, :] * bu.T[None, :, :]) * _PROD_SCALE
        out[start:stop] = _to_f16(_place_and_sum(
            terms, None if cu is None else cu[start:stop]))
    return out


def qmma_batched(a, b, c=None, *, target_elements: int = 8_000_000) -> np.ndarray:
    """Same model for batched ``[..., M, K] @ [..., K, N]`` with a ``[..., M, N]`` seed."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("inner dimension mismatch %s %s" % (a.shape, b.shape))
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    lead = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    au = np.broadcast_to(_e4m3_units(a, "A operand"), lead + (m, k)).reshape(-1, m, k)
    bu = np.broadcast_to(_e4m3_units(b, "B operand"), lead + (k, n)).reshape(-1, k, n)
    cu = (None if c is None else
          np.broadcast_to(_f16_units(c, "C seed"), lead + (m, n)).reshape(-1, m, n))
    out = np.empty(lead + (m, n), np.float16).reshape(-1, m, n)
    chunk = max(1, int(target_elements // max(1, m * n * k)))
    for start in range(0, au.shape[0], chunk):
        stop = min(start + chunk, au.shape[0])
        terms = (au[start:stop, :, None, :]
                 * bu[start:stop].transpose(0, 2, 1)[:, None, :, :]) * _PROD_SCALE
        out[start:stop] = _to_f16(_place_and_sum(
            terms, None if cu is None else cu[start:stop]))
    return out.reshape(lead + (m, n))


# ----------------------------------------------------- non-MMA stages, reused
def _half(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).astype(np.float16)


def quantise_codes(value) -> np.ndarray:
    """Publish to E4M3: clamp to +-448, round to nearest even, return the codes."""
    tensor = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
    return tensor.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def decode_codes(codes) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(codes, dtype=np.uint8)
                            ).view(torch.float8_e4m3fn).float().numpy()


def feed_forward_gate(expansion: np.ndarray) -> np.ndarray:
    """Clamp, two HFMA2 steps, one HMUL2, then publish. SASS-derived candidate.

    Ported unchanged from experiments/parity-ffn-sm75/candidate.py ``gate``,
    where it was validated inside the sm_75 FFN closure. It has not been
    independently verified against sm_120 hardware.
    """
    clamped = np.clip(np.asarray(expansion, dtype=np.float64).astype(np.float16),
                      -4, 4).astype(np.float16)
    linear = (np.abs(clamped).astype(np.float64) * -0.055908203125
              + 0.447265625).astype(np.float16)
    factor = (clamped.astype(np.float64) * linear.astype(np.float64)
              + 0.89453125).astype(np.float16)
    product = (np.asarray(expansion, dtype=np.float64)
               * factor.astype(np.float64)).astype(np.float16)
    return quantise_codes(product)


def _norm_squared(value: np.ndarray) -> np.ndarray:
    """The 32-channel reduction tree. Ported from parity-qkv-norm-history."""
    x = np.asarray(value, dtype=np.float16)
    if x.shape[-1] != 32 or not np.isfinite(x).all():
        raise UnsupportedByBackend("only finite 32-channel half vectors are supported")
    f = x.astype(np.float64).reshape(*x.shape[:-1], 4, 8)
    a = (f[..., 2, :] * f[..., 2, :]).astype(np.float16).astype(np.float64)
    b = (f[..., 3, :] * f[..., 3, :]).astype(np.float16).astype(np.float64)
    a = (f[..., 0, :] * f[..., 0, :] + a).astype(np.float16)
    b = (f[..., 1, :] * f[..., 1, :] + b).astype(np.float16)
    t = (a.astype(np.float64) + b).astype(np.float16).reshape(*x.shape[:-1], 4, 2)
    t = (t.astype(np.float64) + t[..., np.arange(4) ^ 2, :]).astype(np.float16)
    t = (t.astype(np.float64) + t[..., np.arange(4) ^ 1, :]).astype(np.float16)
    n = (t[..., 0, 0].astype(np.float64) + t[..., 0, 1]).astype(np.float16)
    return np.maximum(n, np.array(_RSQ_FIRST_CODE, np.uint16).view(np.float16))


def _rsq(norm: np.ndarray, tables: Sm120Tables) -> np.ndarray:
    bits = np.asarray(norm, dtype=np.float16).view(np.uint16)
    if np.any(bits < _RSQ_FIRST_CODE) or np.any(bits >= 0x7C00):
        raise UnsupportedByBackend("RSQ table domain is positive half [0x0410, 0x7c00)")
    return tables.rsq_half_bits[bits - _RSQ_FIRST_CODE].view(np.float16)


def normalise(value: np.ndarray, tables: Sm120Tables, scale=None):
    n = _norm_squared(value)
    r = _rsq(n, tables)
    out = (np.asarray(value, dtype=np.float16).astype(np.float64)
           * r[..., None]).astype(np.float16)
    if scale is not None:
        out = (out.astype(np.float64) * np.asarray(scale, dtype=np.float16)).astype(np.float16)
    return out, n


def _denominator(w: np.ndarray) -> np.ndarray:
    def add(left, right):
        return (left.astype(np.float32) + right.astype(np.float32)).astype(np.float16)

    def leg(b):
        x = add(add(w[..., b], w[..., b + 16]), add(w[..., b + 4], w[..., b + 20]))
        x = add(x, add(w[..., b + 32], w[..., b + 48]))
        return add(x, add(w[..., b + 36], w[..., b + 52]))

    return add(add(add(add(leg(0), leg(2)), leg(8)), leg(10)),
               add(add(add(leg(1), leg(3)), leg(9)), leg(11)))


def published_attention_weights(score: np.ndarray, tables: Sm120Tables) -> np.ndarray:
    """Score -> normalised half weights. Ported from parity-attention-temporal-literal."""
    s = np.asarray(score, dtype=np.float16)
    if s.shape[-1] != 64 or not np.isfinite(s).all():
        raise UnsupportedByBackend("finite half score with 64 keys is required")
    affine = (s.astype(np.float32) * np.float32(0.044921875)
              + np.float32(1.30078125)).astype(np.float16)
    clipped = np.clip(affine, np.float16(1.03125), np.float16(1.5693359375))
    bits = clipped.view(np.uint16).astype(np.uint32)
    w = (((bits << 5) + 0x8000) & 0xFFFF).astype(np.uint16).view(np.float16)
    d = _denominator(w)
    if np.any(d.view(np.uint16) < 0x1C00) or np.any(d.view(np.uint16) > 0x60E0):
        raise UnsupportedByBackend("denominator outside the proven normal-domain bounds")
    bits_d = d.view(np.uint16)
    if np.any(bits_d < _RCP_FIRST_CODE) or np.any(bits_d >= 0x7C00):
        raise UnsupportedByBackend("RCP table domain is positive half [0x0410, 0x7c00)")
    r = tables.rcp_half_bits[bits_d - _RCP_FIRST_CODE].view(np.float16)
    return (w.astype(np.float32) * r[..., None].astype(np.float32)).astype(np.float16)


# --------------------------------------------------------------------- layout
def inpview_decode_codes(raw: bytes, height: int, width: int) -> np.ndarray:
    """INPVIEW bytes -> [H, W, 32] E4M3 codes. Layout from parity-b01 label probes."""
    data = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray)) \
        else np.asarray(raw, dtype=np.uint8)
    if data.size != height * width * 32:
        raise UnsupportedByBackend("inpview buffer is %d bytes, expected %d"
                                   % (data.size, height * width * 32))
    physical = data.reshape(2, height, width, 16).transpose(1, 2, 0, 3).reshape(height, width, 32)
    channel = np.arange(32)
    order = (channel // 16) * 16 + ((channel % 16) // 4) * 2 + (channel % 2) \
        + ((channel % 4) // 2) * 8
    out = np.empty_like(physical)
    out[..., order] = physical
    return out


def inpview_encode_codes(codes: np.ndarray) -> bytes:
    codes = np.asarray(codes, dtype=np.uint8)
    height, width, _ = codes.shape
    channel = np.arange(32)
    order = (channel // 16) * 16 + ((channel % 16) // 4) * 2 + (channel % 2) \
        + ((channel % 4) // 2) * 8
    return codes[..., order].reshape(height, width, 2, 16).transpose(2, 0, 1, 3).copy().tobytes()


def _tin_tables():
    lanes, bytes_ = np.arange(32), np.arange(16)
    lane, byte = np.meshgrid(lanes, bytes_, indexing="ij")
    x = (lane >> 2) & 3
    y = (lane >> 4) + ((byte >> 2) & 1) * 2
    c = (lane & 3) * 2 + (byte & 1) + ((byte >> 1) & 1) * 8 + (byte >> 3) * 16
    return x, y, c


def tin_decode_codes(raw: bytes, height: int, width: int) -> np.ndarray:
    """TIN bytes -> [H, W, 32] codes. 4x4 spatial tiles, 32 lanes, 16 bytes/lane."""
    data = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray)) \
        else np.asarray(raw, dtype=np.uint8)
    if data.size != height * width * 32 or height % 4 or width % 4:
        raise UnsupportedByBackend("TIN buffer/extent not supported")
    tiles = data.reshape(height // 4, width // 4, 32, 16)
    out = np.empty((height // 4, width // 4, 4, 4, 32), np.uint8)
    x, y, c = _tin_tables()
    out[:, :, y, x, c] = tiles[:, :, np.arange(32)[:, None], np.arange(16)[None, :]]
    return out.transpose(0, 2, 1, 3, 4).reshape(height, width, 32)


def tin_encode_codes(codes: np.ndarray) -> bytes:
    codes = np.asarray(codes, dtype=np.uint8)
    height, width, _ = codes.shape
    if height % 4 or width % 4:
        raise UnsupportedByBackend("TIN extent must be a multiple of 4")
    tiles = codes.reshape(height // 4, 4, width // 4, 4, 32).transpose(0, 2, 1, 3, 4)
    x, y, c = _tin_tables()
    physical = tiles[:, :, y, x, c]
    return np.ascontiguousarray(physical, dtype=np.uint8).tobytes()


# -------------------------------------------------------------------- backend
class SM120B1Reference:
    """Explicitly-selected sm_120 block-1 reference.

    WEIGHT CONTRACT -- IMMUTABLE SNAPSHOT
    The constructor COPIES every logical weight into a private float64 array and
    marks it read-only. Writing to the caller's arrays or tensors afterwards has
    no effect on the instance; two instances never share memory with each other
    or with the source; ``weight()`` hands out a read-only view, so an in-place
    write raises instead of silently changing later results. There is no update
    method: different weights mean a new instance. Because the snapshot cannot
    change, the cached attention-bias layout cannot go stale.

    CALL CONTRACT
    ``validate_invocation`` states exactly what this backend was verified for.
    ``mlxdlss.model`` checks the declared block and window origin when the
    backend is registered, and calls ``validate_invocation`` before every use,
    so an unsupported block, head count, window size, window origin,
    publication mode or shape is refused before any arithmetic runs.
    """

    architecture = TARGET_ARCHITECTURE
    accumulator_model = ACCUMULATOR_MODEL
    block_index = SUPPORTED["block_index"]
    head_count = SUPPORTED["head_count"]
    window_size = SUPPORTED["window_size"]
    window_origin = SUPPORTED["window_origin"]

    def __init__(self, weights: Mapping[str, np.ndarray], tables: Sm120Tables,
                 *, prefix: str = "block1.layer0.", weights_id: str | None = None,
                 height: int = 160, width: int = 160):
        if height != SUPPORTED["height"] or width != SUPPORTED["width"]:
            raise UnsupportedByBackend(
                "this backend is verified only at %dx%d; %dx%d was requested"
                % (SUPPORTED["height"], SUPPORTED["width"], height, width))
        self.tables = tables
        self.height, self.width = height, width
        self.weights_id = weights_id
        self._w: dict[str, np.ndarray] = {}
        for name, shape in SUPPORTED["weight_shapes"].items():
            key = prefix + name
            if key not in weights:
                raise UnsupportedByBackend("missing logical weight %r" % key)
            # copy=True: np.asarray would keep an alias to a float64 caller array
            # (or to a CPU tensor's storage) and let outside writes reach in
            array = np.array(_as_numpy(weights[key]), dtype=np.float64, copy=True)
            if tuple(array.shape) != shape:
                raise UnsupportedByBackend(
                    "%s has shape %s, expected %s" % (key, array.shape, shape))
            array.setflags(write=False)
            self._w[name] = array
        self._bias_qk = None

    # ---- the call contract
    def validate_invocation(self, *, block_index: int, head_count: int,
                            window_size: int, window_origin, publish: bool,
                            shape) -> None:
        """Refuse, before any arithmetic, every call outside the verified scope."""
        want = dict(block_index=self.block_index, head_count=self.head_count,
                    window_size=self.window_size,
                    window_origin=tuple(self.window_origin), publish=True,
                    shape=(self.height, self.width, SUPPORTED["channels"]))
        got = dict(block_index=block_index, head_count=head_count,
                   window_size=window_size,
                   window_origin=None if window_origin is None else tuple(window_origin),
                   publish=publish,
                   shape=None if shape is None else tuple(shape))
        wrong = {key: got[key] for key in want if got[key] != want[key]}
        if wrong:
            raise UnsupportedByBackend(
                "SM120B1Reference was verified only for %s; refusing %s" % (want, wrong))

    # ---- weights and the bias layout
    def weight(self, name: str) -> np.ndarray:
        """A read-only view of the weight snapshot taken at construction."""
        view = self._w[name].view()
        view.setflags(write=False)
        return view

    def _bias(self) -> np.ndarray:
        # derived from an immutable snapshot, so computing it once is safe
        if self._bias_qk is None:
            bias = torch.from_numpy(np.ascontiguousarray(self._w["attn_bias"],
                                                         dtype=np.float32))
            laid = _model.recover_attention_bias_layout(bias).reshape(64, 64)
            cached = laid.numpy().astype(np.float16)
            cached.setflags(write=False)
            self._bias_qk = cached
        return self._bias_qk

    # ---- the six MMA stages
    def _ffn(self, x2d: np.ndarray):
        expansion = qmma_rowwise(x2d, self._w["weight1"], None)
        gate_codes = feed_forward_gate(expansion)
        gated = decode_codes(gate_codes).astype(np.float64).reshape(-1, 128)
        seed = _half(x2d * self._w["ffn_cos_skip"])
        acc, segments = seed, []
        w2 = self._w["weight2"]
        for step in range(4):
            rows = slice(32 * step, 32 * (step + 1))
            acc = qmma_rowwise(gated[:, rows], w2[rows, :], acc)
            segments.append(acc)
        return expansion, gate_codes, seed, acc, segments

    def _attention(self, published_x: np.ndarray):
        qkv = qmma_rowwise(published_x.reshape(-1, 32), self._w["qkv_weight"], None)
        shape = (self.height, self.width, 32)
        q, nq = normalise(qkv[..., 0:32].reshape(shape), self.tables,
                          float(self._w["attn_scale"].ravel()[0]))
        k, nk = normalise(qkv[..., 32:64].reshape(shape), self.tables)
        v = qkv[..., 64:96].reshape(shape)
        parts = []
        for tensor in (q, k, v):
            published = decode_codes(quantise_codes(tensor))
            windows = _model.partition_windows(
                torch.from_numpy(np.ascontiguousarray(published, np.float32)).unsqueeze(0), 8)
            parts.append(np.asarray(windows.numpy(), dtype=np.float64))
        qw, kw, vw = parts
        score = qmma_batched(qw, np.ascontiguousarray(np.transpose(kw, (0, 2, 1))),
                             np.broadcast_to(self._bias(), (qw.shape[0], 64, 64)))
        weights = published_attention_weights(score, self.tables)
        published_weights = decode_codes(quantise_codes(weights)).astype(np.float64)
        acc, segments = None, []
        for keys in (slice(0, 32), slice(32, 64)):
            acc = qmma_batched(np.ascontiguousarray(published_weights[:, :, keys]),
                               np.ascontiguousarray(vw[:, keys, :]), acc)
            segments.append(acc)
        av_published = _model.e4m3_round_trip(
            torch.from_numpy(np.ascontiguousarray(acc, np.float32)))
        av = _model.reverse_windows(av_published, batch_count=1, height=self.height,
                                    width=self.width, window_size=8)[0].numpy()
        return qkv, nq, nk, score, weights, acc, segments, av

    # ---- the block
    def run_codes(self, x_codes: np.ndarray, *, with_stages: bool = False):
        """[H, W, 32] E4M3 codes in, published (AV codes, Y codes) out."""
        # codes, not values: np.asarray(float_array, dtype=uint8) would turn 449.0
        # or 1.5 into some code instead of refusing it
        if not isinstance(x_codes, np.ndarray) or x_codes.dtype != np.uint8:
            raise UnsupportedByBackend(
                "run_codes takes a uint8 numpy array of E4M3 codes, got %s"
                % (getattr(x_codes, "dtype", type(x_codes).__name__),))
        codes = x_codes
        if codes.shape != (self.height, self.width, 32):
            raise UnsupportedByBackend(
                "input shape %s is outside the verified %s"
                % (codes.shape, (self.height, self.width, 32)))
        if np.any((codes & 0x7F) == 0x7F):
            raise UnsupportedByBackend("input holds E4M3 NaN codes, which this backend "
                                       "has no measured behaviour for")
        x = decode_codes(codes).astype(np.float64)
        x2d = x.reshape(-1, 32)
        expansion, gate_codes, ffn_seed, z2d, w2_segments = self._ffn(x2d)
        z = z2d.reshape(self.height, self.width, 32)
        published_x = decode_codes(quantise_codes(z)).astype(np.float64)
        qkv, nq, nk, score, weights, av_half, av_segments, av = self._attention(published_x)
        projection_seed = _half(z.reshape(-1, 32).astype(np.float64)
                                * self._w["attn_cos_skip"])
        y2d = qmma_rowwise(np.ascontiguousarray(av, np.float64).reshape(-1, 32),
                           self._w["projection_weight"], projection_seed)
        y = y2d.reshape(self.height, self.width, 32)
        av_codes, y_codes = quantise_codes(av), quantise_codes(y)
        if not with_stages:
            return av_codes, y_codes
        return av_codes, y_codes, dict(
            E_half=expansion, G_published=gate_codes, ffn_seed_half=ffn_seed,
            W2_segment_halves=w2_segments, Z_half=z, QKV_half=qkv,
            Q_norm_squared=nq, K_norm_squared=nk, score=score,
            weight_normalized_half=weights, AV_segment_halves=av_segments,
            AV_half=av_half, AV_reversed=av, projection_seed_half=projection_seed,
            Y_half=y)

    def run_inpview(self, raw: bytes, *, height: int | None = None,
                    width: int | None = None) -> bytes:
        """INPVIEW bytes in, TIN bytes out -- the native block interface."""
        h = self.height if height is None else height
        w = self.width if width is None else width
        if (h, w) != (self.height, self.width):
            raise UnsupportedByBackend("extent %dx%d is outside this instance" % (h, w))
        _av, y_codes = self.run_codes(inpview_decode_codes(raw, h, w))
        return tin_encode_codes(y_codes)

    def describe(self) -> dict:
        return dict(
            backend="mlxdlss.sm120_b1.SM120B1Reference",
            target_architecture=self.architecture,
            accumulator_model=self.accumulator_model,
            W=_W, placement="truncate toward zero",
            selected_explicitly=True, gpu_detection_used=False,
            supported=dict(SUPPORTED), extent=[self.height, self.width],
            weights_id=self.weights_id, tables=self.tables.describe(),
            chain_structure=dict(
                ffn_w1="1 x K=32, C = RZ",
                ffn_w2="4 x K=32, hidden rows 0-31/32-63/64-95/96-127, C = half(ffn_cos_skip * X)",
                qkv="1 x K=32 each, C = RZ",
                score="1 x K=32, C = attn_bias",
                av="2 x K=32, window keys 0-31 then 32-63, C = RZ",
                projection="1 x K=32, C = half(attn_cos_skip * unquantised z)"),
            not_claimed=[
                "W is not a hardware significand width",
                "byte-exactness is established for four (X, weight package) pairs on one "
                "card and one driver, not for every block-1 input",
                "the gate constants, the reduction order, the softmax constants and the "
                "E4M3 publication semantics are sm_75-derived and were not independently "
                "verified on sm_120",
                "this is block 1 only; nothing here generalises to the single_window family",
            ])


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value
