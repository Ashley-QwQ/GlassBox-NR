M2 run driver: 1080p frame 0 on the CPU runtime
================================================

What it does
  Runs the 158 launches of frame 0 (1080p) through the released runtime and compares the workspace and the six textures after
  every launch, and the final image, with a reference table (data/reference_f0.json). Exit code 0 means every launch matches.

What you need
  1. The payload: the directory that holds the released files under kit/data/... and kit/home/...  (the release package
     unpacks to this layout; the path names inside the configuration files are relocated to your directory by the driver).
  2. The tables: produced by payload_layout.py from payload.tar.zst.001 (see the top-level README quickstart, steps 1-4).
     Pass the directory that contains tables/ as --tables.
  3. The weight files (names and sizes in data/manifest.json, role "weights"). Make them from your own copy of the vendor
     library with:  python devkit/make_weights.py <nvngx_dlssnr.dll> WEIGHTS   (WEIGHTS = --weights). They are not part of the
     release; the driver checks every file by size and sha256 before it starts.
  4. The two frame-0 input files (role "inputs"): caller-id1.bin and caller-id3.bin. They are unpacked with the payload, under
     PAYLOAD/kit/data/video-multiframe-r01-20261006-codex/resume2-20261007/chain_port/chain-input-only-v2/frame-000 (pass as --inputs).

Run
  python run_1080_f0.py --root WORK --payload PAYLOAD --weights WEIGHTS --tables TABLES --inputs INPUTS --threads 4
  python run_1080_f0.py ... --verify-only     # relocation and input checks only, no launch

Notes
  * WORK, the weights and the tables should be on one file system (they are linked, not copied, when possible).
  * Python 3.13 with numpy and safetensors; the compiled modules are in the payload build directory.
  * The path guard of the loaders stays on; the driver never reads outside PAYLOAD, WORK and the directories you pass.
