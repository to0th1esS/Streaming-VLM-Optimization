# Turbo-ViT-v1 Experiments

This directory contains lightweight, local-first experiments for validating
front-end ViT frame reuse before running large-scale Video-LLM evaluation.

Current scope:

- Dense per-frame ViT baseline.
- Layer-wise hidden-state capture.
- Adjacent-frame redundancy analysis.
- Local JSON/CSV/SVG outputs under `results/turbovit_v1/`.

The initial implementation uses a small self-contained ViT-like encoder so the
local loop does not depend on downloaded model weights.
