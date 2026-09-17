# Modifications to upstream MLX-DLSS (Apache-2.0 License, section 4)

This directory vendors four files from MLX-DLSS:
- Modified files: `model.py` and `__init__.py`. This document records every change, as Apache-2.0 §4 requires for modified redistributed files.
- Unmodified files: `tools/extract_dlssnr_weights.py` and `tools/unpack_dlssnr_weights.py` (verbatim copies from upstream).

- Upstream project: MLX-DLSS (`github.com/iamwavecut/MLX-DLSS`).
- Pinned upstream commit used for this comparison: `6499d59c900f5e525d800e951f0000880d85c9f9` (2026-09-08).
- Original files in that commit:
  - `python/mlxdlss/model.py` (modified; see diff below)
  - `python/mlxdlss/__init__.py` (modified; see diff below)
  - `python/mlxdlss/tools/extract_dlssnr_weights.py` (unmodified verbatim copy)
  - `python/mlxdlss/tools/unpack_dlssnr_weights.py` (unmodified verbatim copy)

Everything else that used to live alongside these two files in this
project's own working tree (an SM120 B1 reference implementation and two
hardware-measured lookup tables) is **not** part of upstream MLX-DLSS and is
**not** covered by this notice or by the Apache-2.0 license — it is this
project's own code and data, packaged separately under `glassbox_nr/kernels/`
and `data/tables/`.

## `__init__.py`

Modified 2026-09-09 (by this project) to omit the upstream package's eager
imports of its video, frame-generation and UI modules, none of which this
project uses. Diff against the pinned upstream commit:

```diff
--- upstream/mlxdlss/__init__.py
+++ this project's mlxdlss/__init__.py
@@ -1,18 +1,5 @@
-"""MLX-DLSS: PyTorch inference for the recovered neural-rendering transformer."""
-
-from .features import PROFILES, AutomaticMask, NetworkGeometry, deterministic_noise, make_features
-from .composition import compose_detail, compose_head, resample
-from .pipeline import EnhanceResult, NeuralRenderingPipeline, NeuralRenderingSession, PreparedFrame, load_weights, resolve_device
-from .temporal import BLEND_SCALE, TemporalOptions, TemporalSession, compose_temporal, make_temporal_features, normalize_pixel_motion
-from .video import ConvertOptions, ConvertResult, VideoInfo, convert, probe
-from .framegen import FrameGenerator
-
-__version__ = "0.1.0"
-__all__ = [
-    "FrameGenerator",
-    "PROFILES", "AutomaticMask", "NetworkGeometry", "deterministic_noise", "make_features",
-    "compose_detail", "compose_head", "resample",
-    "EnhanceResult", "NeuralRenderingPipeline", "NeuralRenderingSession", "PreparedFrame", "load_weights", "resolve_device",
-    "ConvertOptions", "ConvertResult", "VideoInfo", "convert", "probe",
-    "BLEND_SCALE", "TemporalOptions", "TemporalSession", "compose_temporal", "make_temporal_features", "normalize_pixel_motion",
-]
+"""Minimal CPU still-image subset of MLX-DLSS.
+
+Modified 2026-09-09: omit upstream eager imports of video, frame generation and
+UI modules. Core inference modules are preserved verbatim; see HANDOFF_ZH.md.
+"""
```

**Second modification, made while repackaging this file for publication
(2026-09-15)**: the docstring line "Core inference modules are preserved
verbatim" was misleading as originally written — it was true only of
`model.py`, not of everything that used to sit alongside it in this
project's own copy of this directory (see the note above). Comment-only
change, no code affected:

```diff
-Modified 2026-09-09: omit upstream eager imports of video, frame generation and
-UI modules. Core inference modules are preserved verbatim; see HANDOFF_ZH.md.
+Modified 2026-09-09: omit upstream eager imports of video, frame generation and
+UI modules. This file and model.py in this directory are modified from
+upstream MLX-DLSS (Apache-2.0); see MODIFICATIONS.md in this directory for the
+full diff against the pinned upstream commit. The rest of this project's SM120
+B1 reference implementation and hardware-measured tables are NOT part of
+upstream MLX-DLSS -- they are this project's own code and data, and live under
+glassbox_nr/kernels/ and data/tables/ respectively, not in this directory.
```

## `model.py`

Modified 2026-09-12 (by this project) to add an opt-in mechanism for routing
one block's computation to an explicitly-registered alternate backend
(`set_block_backend`/`block_backend`/`_backend_window`), plus a supporting
E4M3-code validation helper (`UnsupportedByBackend`, `exact_e4m3_codes`).
None of the upstream code paths are altered — with no backend registered
(the default), every block still runs the original upstream torch path
unchanged. This is what lets this project's own block implementations
(published separately, not part of MLX-DLSS) be substituted in for specific
blocks without forking the rest of the upstream model graph.

Full diff against the pinned upstream commit (148 lines changed, additions only):

```diff
--- upstream/mlxdlss/model.py
+++ this project's mlxdlss/model.py
@@ -16,6 +16,58 @@
 COSINE_NORM_FLOOR = 0.00006198883056640625


+class UnsupportedByBackend(Exception):
+    """Raised instead of silently computing something a backend never verified."""
+
+
+# dtypes a backend block may receive. Each widens to float64 without loss, so
+# the E4M3 check below always runs on the caller's exact values.
+BACKEND_INPUT_DTYPES = (torch.float8_e4m3fn, torch.float16, torch.bfloat16,
+                        torch.float32, torch.float64)
+
+
+def exact_e4m3_codes(value: torch.Tensor) -> torch.Tensor:
+    """Return the uint8 E4M3 codes of a tensor whose values ARE E4M3, or refuse.
+
+    Nothing is clamped or rounded: every element must already be a finite,
+    exactly representable E4M3 value, checked in the input's own precision
+    (float64 1+2**-30 is refused, it is not first narrowed to 1.0). The sign of
+    zero is preserved (-0 -> 0x80). Only CPU tensors are accepted; any other
+    device is refused explicitly rather than by whatever ``.numpy()`` raises.
+    """
+    if not isinstance(value, torch.Tensor):
+        raise UnsupportedByBackend("backend input must be a torch.Tensor, got %s"
+                                   % type(value).__name__)
+    if value.device.type != "cpu":
+        raise UnsupportedByBackend(
+            "backend input is on device %r; only CPU tensors are supported, no "
+            "device path is modelled" % str(value.device))
+    if value.dtype not in BACKEND_INPUT_DTYPES:
+        raise UnsupportedByBackend("backend input dtype %s is not one of %s"
+                                   % (value.dtype, BACKEND_INPUT_DTYPES))
+    if value.dtype == torch.float8_e4m3fn:
+        codes = value.view(torch.uint8)
+        if bool(((codes & 0x7F) == 0x7F).any()):
+            raise UnsupportedByBackend("backend input holds E4M3 NaN codes")
+        return codes.contiguous()
+    wide = value.to(torch.float64)
+    if not bool(torch.isfinite(wide).all()):
+        raise UnsupportedByBackend("backend input contains NaN or Inf")
+    if bool((wide.abs() > 448.0).any()):
+        raise UnsupportedByBackend(
+            "backend input exceeds the E4M3 range, e.g. %r"
+            % float(wide[wide.abs() > 448.0].reshape(-1)[0]))
+    codes = wide.to(torch.float8_e4m3fn).view(torch.uint8)
+    back = codes.view(torch.float8_e4m3fn).to(torch.float64)
+    exact = (back == wide) & (torch.signbit(back) == torch.signbit(wide))
+    if not bool(exact.all()):
+        bad = wide[~exact].reshape(-1)
+        raise UnsupportedByBackend(
+            "backend input holds %d value(s) that are not exactly representable "
+            "E4M3, e.g. %r" % (int(bad.numel()), float(bad[0])))
+    return codes.contiguous()
+
+
 def recovered_window_origin(block_index: int) -> tuple[int, int]:
     """Return the vendor window origin as ``(y, x)`` for one graph block."""
     if block_index == 0:
@@ -858,6 +910,56 @@
         except KeyError as error:
             raise ValueError(f"missing logical weight: {name}") from error

+    def set_block_backend(self, index: int, backend) -> None:
+        """Route one block index to an explicitly chosen reference backend.
+
+        Opt-in only. With no backend registered -- the default -- every block
+        keeps the ordinary torch path in this module unchanged. A backend is a
+        choice about which NATIVE architecture to model; it is never selected
+        from what hardware happens to be present. Pass ``None`` to unregister.
+
+        SUPPORT CONTRACT. A backend must expose
+
+        * ``block_index`` -- the one block it models, which must equal ``index``;
+        * ``window_origin`` -- its window origin, which must equal this graph's
+          ``recovered_window_origin(index)``;
+        * ``validate_invocation(block_index=, head_count=, window_size=,
+          window_origin=, publish=, shape=)`` -- raising ``UnsupportedByBackend``
+          for anything outside its verified scope; it is called before every use;
+        * ``run_codes(codes) -> (av_codes, y_codes)`` on uint8 E4M3 codes.
+
+        Registration is refused, with nothing recorded, when any of these is
+        missing or disagrees; see ``mlxdlss.sm120_b1.SM120B1Reference``.
+        """
+        backends = getattr(self, "_block_backends", None)
+        if backends is None:
+            backends = {}
+            object.__setattr__(self, "_block_backends", backends)
+        if backend is None:
+            backends.pop(index, None)
+            return
+        missing = [name for name in ("run_codes", "validate_invocation")
+                   if not callable(getattr(backend, name, None))]
+        missing += [name for name in ("block_index", "window_origin")
+                    if not hasattr(backend, name)]
+        if missing:
+            raise UnsupportedByBackend(
+                "backend %s does not implement the support contract (missing %s); "
+                "refusing to register it at block %d"
+                % (type(backend).__name__, missing, index))
+        if backend.block_index != index:
+            raise UnsupportedByBackend(
+                "backend models block %r; refusing to register it at block %d"
+                % (backend.block_index, index))
+        if tuple(backend.window_origin) != recovered_window_origin(index):
+            raise UnsupportedByBackend(
+                "backend window origin %r differs from block %d's recovered origin %r"
+                % (tuple(backend.window_origin), index, recovered_window_origin(index)))
+        backends[index] = backend
+
+    def block_backend(self, index: int):
+        return getattr(self, "_block_backends", {}).get(index)
+
     def _window(
         self,
         value: torch.Tensor,
@@ -866,6 +968,17 @@
         head_count: int,
         publish: bool = True,
     ) -> torch.Tensor:
+        backend = self.block_backend(index)
+        if backend is not None:
+            # The backend consumes and returns PUBLISHED E4M3, so the block's
+            # own publication step is the backend's, not this one's. A backend
+            # is expected to raise on anything outside its verified scope; this
+            # path never falls back to the float approximation below, because a
+            # silent fallback would keep the byte-exactness label while quietly
+            # computing something else.
+            return self._backend_window(
+                value, index, backend, head_count=head_count, window_size=8,
+                window_origin=recovered_window_origin(index), publish=publish)
         prefix = f"block{index}.layer0"
         attention_bias = self.weight(f"{prefix}.attn_bias")
         if uses_fragment_swizzle(index, head_count):
@@ -909,6 +1022,41 @@
         # ``publish=False`` hands the half-precision output to a fused transition.
         return output if index in (0, 70) or not publish else e4m3_round_trip(output)

+    def _backend_window(self, value: torch.Tensor, index: int, backend, *,
+                        head_count: int, window_size: int, window_origin,
+                        publish: bool = True) -> torch.Tensor:
+        """Hand one block to a registered backend, in published E4M3 codes.
+
+        Every check happens before the backend computes anything: the backend
+        must be the one registered at this block, the call must satisfy its
+        ``validate_invocation`` (block, heads, window, origin, publication,
+        shape), and the input must already be exact E4M3 -- it is never clamped
+        or rounded into range here.
+        """
+        if self.block_backend(index) is not backend:
+            raise UnsupportedByBackend(
+                f"the backend passed for block {index} is not the one registered there")
+        if getattr(backend, "block_index", None) != index:
+            raise UnsupportedByBackend(
+                f"backend models block {getattr(backend, 'block_index', None)!r}, "
+                f"not block {index}")
+        if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4):
+            raise UnsupportedByBackend(
+                "backend blocks take a rank-3 HWC or rank-4 NHWC tensor, got rank %s"
+                % getattr(value, "ndim", None))
+        tensor = value if value.ndim == 4 else value.unsqueeze(0)
+        if tensor.shape[0] != 1:
+            raise UnsupportedByBackend("backend blocks take one image at a time, got "
+                                       f"batch {tensor.shape[0]}")
+        backend.validate_invocation(block_index=index, head_count=head_count,
+                                    window_size=window_size,
+                                    window_origin=window_origin, publish=publish,
+                                    shape=tuple(tensor.shape[1:]))
+        codes = exact_e4m3_codes(tensor[0]).numpy()
+        _av_codes, y_codes = backend.run_codes(codes)
+        published = torch.from_numpy(y_codes).view(torch.float8_e4m3fn).float()
+        return published if value.ndim == 3 else published.unsqueeze(0)
+
     def _split_window(self, value: torch.Tensor, index: int) -> torch.Tensor:
         prefix = f"block{index}"
         return e4m3_round_trip(
```

## Is `model.py` actually used at runtime?

Yes. This project's SM120 B1 reference implementation (`glassbox_nr/kernels/core/sm120_b1.py`)
imports `model.py` and calls several of its unmodified upstream functions directly
(`recover_attention_bias_layout`, `partition_windows`, `e4m3_round_trip`,
`reverse_windows`) plus the `UnsupportedByBackend` exception added in the diff
above. It is not a dead import.

## `tools/extract_dlssnr_weights.py` and `tools/unpack_dlssnr_weights.py`

These two helper scripts under `third_party/mlxdlss/tools/` are vendored from upstream `python/mlxdlss/tools/` at the pinned commit `6499d59c900f5e525d800e951f0000880d85c9f9`.
Both are unmodified, byte-for-byte verbatim copies of their respective upstream files (0-diff against upstream).

