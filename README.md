# DoubleBee

DoubleBee is a two-wheel balancing robot project built on NVIDIA Isaac Sim / Isaac Lab, focused on simulation, training, and evaluation workflows for control research.

## Demo

- **Video**: [`visual/doubleBee.mp4`](visual/doubleBee.mp4) (≈59s, 1280×720)

> GitHub doesn’t reliably inline-play `.mp4` in `README.md` across all views, but the link above will open/download the video.

## Repo layout

- **`lab/`**: Isaac Lab tasks, configs, training/play entrypoints
- **`visual/`**: media assets (demo videos, renders)

## Setup

This project assumes a working Isaac Sim + Isaac Lab install.

- **Isaac Sim install**: `https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html`
- **Isaac Lab**: `https://github.com/isaac-sim/IsaacLab`

From the repo root:

```bash
conda activate <your_isaaclab_env>
pip install -e .
```

## Run

Look for DoubleBee task definitions under `lab/` and the corresponding entry scripts under `scripts/` (train/play).
