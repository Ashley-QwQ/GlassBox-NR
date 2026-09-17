"""Minimal CPU still-image subset of MLX-DLSS.

Modified 2026-09-09: omit upstream eager imports of video, frame generation and
UI modules. This file and model.py in this directory are modified from
upstream MLX-DLSS (Apache-2.0); see MODIFICATIONS.md in this directory for the
full diff against the pinned upstream commit. The rest of this project's SM120
B1 reference implementation and hardware-measured tables are NOT part of
upstream MLX-DLSS -- they are this project's own code and data, and live under
glassbox_nr/kernels/ and data/tables/ respectively, not in this directory.
"""
