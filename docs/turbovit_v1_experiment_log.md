# Turbo-ViT-v1 Experiment Log

This document records each local experiment iteration so results remain
traceable before remote large-scale runs.

## v0 Dense Baseline And Redundancy Scaffold

Goal:

- Build the first local-only Turbo-ViT-v1 experiment scaffold.
- Validate dense per-frame ViT encoding latency measurement.
- Capture per-layer hidden states and compute adjacent-frame redundancy.

Implementation:

- Added a self-contained TinyViT encoder under `experiments/turbovit_v1/models/`.
- Added a synthetic redundant video stream generator.
- Added dense stream encoding with per-frame latency logging.
- Added layer-wise adjacent-frame cosine similarity analysis.
- Added JSON, CSV, and SVG output helpers.

Expected outputs:

```text
results/turbovit_v1/v0_dense_baseline/dense_summary.json
results/turbovit_v1/v0_dense_baseline/dense_latency.csv
results/turbovit_v1/v0_dense_baseline/layer_redundancy.csv
results/turbovit_v1/v0_dense_baseline/layer_redundancy.svg
```

Status:

- Completed locally on CPU with the certification conda environment.

Command:

```powershell
.\scripts\run_turbovit_v1_dense_local.ps1
```

Observed result:

```text
mean latency/frame: 1.543 ms
total latency/video: 37.041 ms for 24 frames
layer-0 adjacent cosine: 0.999740
last-layer adjacent cosine: 0.999933
```

Interpretation:

- The synthetic stream has strong frame-to-frame redundancy by construction.
- Redundancy increases with depth in this TinyViT scaffold, matching the
  expected direction for the first Turbo-ViT-v1 motivation.
- This is not yet a real speedup result; it is the dense baseline and
  measurement scaffold needed before implementing selective recomputation.

Next step:

- Implement Turbo-ViT-v1 functional baseline with `refresh_interval` and
  `dynamic_ratio`, then compare latency and final-feature fidelity against this
  dense baseline.

## v1 Functional Turbo-ViT Baseline

Goal:

- Implement the first functional Turbo-ViT-v1 mechanism.
- Compare dense per-frame encoding with reference-frame reuse and dynamic token
  recomputation.
- Produce a small local trade-off table over `refresh_interval` and
  `dynamic_ratio`.

Implementation:

- Added Q/K/V projection helpers to the TinyViT block.
- Added reference full encoding caches with per-layer key and output states.
- Added sparse non-reference frame encoding:
  - compute current per-layer key projection;
  - compare current key with reference key using cosine similarity;
  - select low-similarity dynamic tokens by top-ratio;
  - compute attention and MLP only for selected tokens;
  - scatter selected token outputs into cached reference layer output.
- Added dense-vs-turbo fidelity metrics: final output cosine and MSE.
- Added local ablation runner over refresh interval and dynamic ratio.

Commands:

```powershell
.\scripts\run_turbovit_v1_local.ps1 -RefreshInterval 4 -DynamicRatio 0.5
.\scripts\run_turbovit_v1_ablation_local.ps1
```

Single configuration result (`refresh_interval=4`, `dynamic_ratio=0.5`):

```text
speedup: 0.697x
mean output cosine: 0.999954
mean output mse: 0.00009158
mean selector time/frame: 0.759 ms
mean sparse compute time/frame: 1.226 ms
```

Ablation summary:

```text
best speed: N=2, r=0.5
speedup: 1.449x
mean output cosine: 0.999985
mean output mse: 0.0000301

best fidelity: N=2, r=0.75
speedup: 1.135x
mean output cosine: 0.999993
mean output mse: 0.0000144
```

Observed trade-off:

```text
N=2, r=0.25 -> 1.359x, cosine 0.999977
N=2, r=0.50 -> 1.449x, cosine 0.999985
N=2, r=0.75 -> 1.135x, cosine 0.999993
N=4, r=0.25 -> 1.060x, cosine 0.999914
N=4, r=0.50 -> 0.870x, cosine 0.999954
N=4, r=0.75 -> 1.248x, cosine 0.999983
N=8, r=0.25 -> 0.904x, cosine 0.999710
N=8, r=0.50 -> 1.025x, cosine 0.999880
N=8, r=0.75 -> 1.052x, cosine 0.999970
```

Interpretation:

- The functional mechanism works and preserves final features well on the
  redundant synthetic stream.
- The local CPU microbench is noisy, but it already shows usable speedup in some
  settings and clear slowdowns in others.
- Larger refresh intervals increase stale-reference drift, especially at low
  dynamic ratios.
- Selector and sparse compute overhead are large enough to erase gains in some
  settings, which directly supports the next research question: reduce
  per-layer decision overhead and avoid stale single-anchor reuse.

Next step:

- Add cleaner time breakdown and drift curves.
- Then implement a v2 candidate: dual-anchor or segment-level decision to reduce
  stale-reference drift and selector overhead.

## v1 Real Video Sanity Check

Goal:

- Move beyond synthetic redundant streams and test Turbo-ViT-v1 on a small real
  video clip.
- Keep the test local and lightweight while using the same dense/turbo metrics.

Implementation:

- Added `experiments/turbovit_v1/data/real_video.py`.
- Default real clip:
  `https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/big_buck_bunny.mp4`
- The video is downloaded to `data/turbovit_v1/big_buck_bunny.mp4`, which is
  ignored by git.
- Decoding uses `imageio + imageio-ffmpeg`, so system `ffmpeg` is not required.
- Added `--video-source real` support to the single-run and ablation scripts.

Commands:

```powershell
.\scripts\run_turbovit_v1_local.ps1 `
  -VideoSource real `
  -OutputDir results\turbovit_v1\v1_real_turbo_baseline `
  -RefreshInterval 4 `
  -DynamicRatio 0.5

.\scripts\run_turbovit_v1_ablation_local.ps1 `
  -VideoSource real `
  -OutputDir results\turbovit_v1\v1_real_ablation
```

Single configuration result (`N=4`, `r=0.5`):

```text
speedup: 1.026x
mean output cosine: 0.922687
mean output mse: 0.154624
mean selector time/frame: 0.612 ms
mean sparse compute time/frame: 0.998 ms
```

Real-video ablation summary:

```text
best speed: N=4, r=0.75
speedup: 1.347x
mean output cosine: 0.967529
mean output mse: 0.064941

best fidelity: N=2, r=0.75
speedup: 1.055x
mean output cosine: 0.977259
mean output mse: 0.045482
```

Observed trade-off on real video:

```text
N=2, r=0.25 -> 1.021x, cosine 0.953017
N=2, r=0.50 -> 1.189x, cosine 0.961417
N=2, r=0.75 -> 1.055x, cosine 0.977259
N=4, r=0.25 -> 1.132x, cosine 0.876226
N=4, r=0.50 -> 1.034x, cosine 0.922687
N=4, r=0.75 -> 1.347x, cosine 0.967529
N=8, r=0.25 -> 1.148x, cosine 0.846362
N=8, r=0.50 -> 1.197x, cosine 0.912387
N=8, r=0.75 -> 1.110x, cosine 0.964135
```

Interpretation:

- Real video is much less forgiving than the synthetic stream.
- Turbo-ViT-v1 still finds speedup settings, but stale-reference drift becomes
  visible in final feature fidelity.
- Low dynamic ratios are risky on real motion; `r=0.75` is consistently safer.
- The result strengthens the case for v2: adaptive refresh or dual-anchor reuse
  is needed before moving to real Video-LLM quality evaluation.

## v1 Drift And Time Breakdown

Goal:

- Make stale-reference drift and per-frame overhead explicit.
- Produce reusable analysis artifacts for every ablation run.

Implementation:

- `run_ablation.py` now writes:
  - `drift_by_distance.csv`
  - `time_breakdown.csv`
  - `best_speed_drift.svg`
  - `best_speed_time_breakdown.svg`
- Drift is grouped by `distance_from_reference = frame_idx % refresh_interval`.
- Time breakdown separates:
  - dense baseline average latency;
  - selector time;
  - sparse compute time;
  - other/reference/scatter overhead.

Synthetic rerun:

```text
best speed config: N=4, r=0.5
speedup: 0.951x
mean output cosine: 0.999954
mean selector time/frame: 0.673 ms
mean sparse compute time/frame: 1.151 ms
mean other/reference overhead/frame: 0.693 ms
```

Real-video rerun:

```text
best speed config: N=4, r=0.75
speedup: 0.921x
mean output cosine: 0.967529
mean selector time/frame: 0.691 ms
mean sparse compute time/frame: 1.197 ms
mean other/reference overhead/frame: 0.649 ms
```

Important caveat:

- CPU microbench latency is noisy at this tiny model scale. Earlier runs showed
  speedup for the same mechanism, while this rerun showed slowdown after adding
  extra analysis outputs. Treat local CPU latency as a sanity signal, not a
  final performance claim.
- The stable conclusions are qualitative:
  - feature drift increases with distance from the reference;
  - real video has much larger drift than synthetic video;
  - selector + sparse compute overhead is large enough to erase gains;
  - v1 needs reduced decision overhead and better reference management.

Next step:

- Run the same local experiment on remote GPU for a cleaner latency signal, or
  implement v2 with lower-overhead segment-level decisions before remote GPU
  scaling.

## v2 Segment-Level Decision Prototype

Goal:

- Reduce v1 per-layer token selector overhead.
- Add a cheap frame-level drift decision before entering per-layer sparse
  recomputation.

Implementation:

- Added `encode_stream_turbovit_v2`.
- For each non-reference frame, compute patch-embedding MSE against the current
  rolling anchor.
- Decision policy:
  - `dense`: forced refresh or frame drift above `dense_threshold`;
  - `skip`: frame drift below `skip_threshold`, directly reuse anchor output;
  - `sparse`: medium drift, run v1-style per-layer sparse update.
- Sparse frames update the rolling anchor state, so subsequent frame drift is
  measured against the latest approximate visual state.

Default real-video setting:

```text
refresh_interval: 4
dynamic_ratio: 0.75
skip_threshold: 0.0005
dense_threshold: 0.006
```

Real-video result after state-consistency fix:

```text
speedup: 0.940x
mean output cosine: 0.990327
mean output mse: 0.019346
dense frames: 7
sparse frames: 10
skip frames: 7
mean selector time/frame: 0.420 ms
mean sparse compute time/frame: 0.724 ms
```

More conservative skip setting:

```text
skip_threshold: 0.0001
dense_threshold: 0.006
speedup: 0.769x
mean output cosine: 0.995885
dense frames: 7
sparse frames: 15
skip frames: 2
```

Interpretation:

- Segment-level decision successfully reduces selector calls when more frames
  are skipped.
- The skip threshold controls a clear speed/fidelity trade-off.
- CPU latency is still noisy and not final, but the mechanism now exposes a
  useful research knob: cheap frame-level routing before expensive per-layer
  sparse updates.
- Current frame-drift MSE is too crude; it can mistakenly skip frames whose
  downstream feature changes are still meaningful. Next versions should use a
  better routing signal, such as low-layer feature drift, motion-aware drift, or
  dual-anchor interpolation.
