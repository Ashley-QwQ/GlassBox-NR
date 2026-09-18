# Reproducing the preview result

## Steps

From a fresh clone of this repository:

```
python -m pip install -r requirements.txt
python -B tools/extract_weights.py /path/to/your/nvngx_dlssnr.dll
python -B tools/make_inputs.py
python -B tools/run_pipeline.py --sample all --run-tag my-run
python -B tools/verify_reference_hashes.py runs/my-run
```

The expected last line is `154/154 MATCH`, with exit status 0.

1. `extract_weights.py` reads the weight resource out of your DLL's bytes
   (it never loads or executes the DLL) and writes the weights under
   `weights/`. It refuses to continue unless the resource's sha256 is
   `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`.
2. `make_inputs.py` writes each sample's colour input to
   `anchors/<sample>/inputs/colour.rgba16`, after checking its sha256.
3. `run_pipeline.py` runs B00 through B69 for both samples and writes
   `runs/my-run/<sample>/run_manifest.json`.
4. `verify_reference_hashes.py` compares all 154 exits with
   `reference/reference_hashes.json`.

`weights/`, `anchors/` and `runs/` are local only and are ignored by git.

## Execution settings the runner enforces

`tools/run_pipeline.py` sets these at start-up:

| Variable | Value |
|---|---|
| `CUDA_VISIBLE_DEVICES` | empty |
| `OMP_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `PYTHONDONTWRITEBYTECODE` | `1` |

The runner imports NumPy before it sets them, so also set them in your shell
before running, so that NumPy's own BLAS picks them up as well. The verified
runs below were made with all five set in the environment of the process.

```
# bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
```

```
# PowerShell
$env:CUDA_VISIBLE_DEVICES=""; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:PYTHONDONTWRITEBYTECODE="1"
```

Single-threaded execution is a correctness setting, not a speed setting: a
multi-threaded reduction can change the order of floating-point
operations, and with it the output bytes.

## Verified environment

The `154/154 MATCH` result for this revision was obtained, from a fresh
clone, in this environment:

| Component | Version |
|---|---|
| OS | Windows Server 2025 Standard, 10.0.26100, x86-64 |
| CPU | 12th Gen Intel Core i5-12400F (PyTorch CPU capability: AVX2) |
| Python | 3.13.14 |
| NumPy | 2.5.3 |
| PyTorch | 2.14.0+cpu |
| safetensors | 0.8.0 |

`requirements.txt` gives version ranges. The byte-exact claim refers to the
environment above. Other environments may work but are not part of the
current claim unless listed here as tested.

## Timing

A full two-sample run (B00 to B69, 71 nodes per sample) takes on the order
of 20 minutes per sample on the CPU above. This is a correctness reference,
not a real-time implementation.
