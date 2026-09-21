# Handover

Written 2026-09-21, immediately after the ICRA 2027 submission, on the last day
of work on `airlab-desktop`. The machine will not be used again, so anything
that mattered and lived only on its disk is either captured here or listed below
as still at risk.

## Status

- **Paper submitted.** "Learning Energy-Efficient Air–Ground Actuation for
  Hybrid Robots on Stair-Like Terrain"
- **Video submitted.** `dB-6000.mp4`, 115.3 s, 1920x1080 at 60 fps, 11.8 MB,
  inside ICRA's limits of 180 s and 20 MB
- **Tag `icra2027-submission`** marks the submitted state of this code

## Restoring context on a new machine

Memory files are read from `~/.claude/projects/<slug>/memory/`, where `<slug>`
is derived from the project's absolute path. On this machine it was
`-home-airlab-doublebee-PID-JAI`. On a new machine the path differs, so the slug
differs too.

```bash
git clone git@github.com:Louis2099/doubleBee.git
cd doubleBee && git checkout ish/hybrid_mode
# then, from inside the project, copy .claude-memory/* into
# ~/.claude/projects/<slug-for-your-new-path>/memory/
```

Easiest way to find the right slug is to start Claude once inside the new
project directory. It creates the correctly named folder itself, and the files
can then be dropped in. Those 21 files load automatically at the start of every
session and carry the project's accumulated decisions, metric definitions and
hardware quirks.

The raw session transcript is deliberately **not** in this repo. It ran to
987 MB, and even with image data stripped and gzipped it was 92.7 MB. Committing
an opaque blob that size would bloat every future clone permanently, and a
verbatim log is worth far less than the memory files above.

## What is in this repo

| path | contents |
| --- | --- |
| `scripts/deploy/` | `db_inference.py` and `RUNNING.md`, the hardware policy node and how to start everything |
| `scripts/figures/` | generators for every paper figure and video asset, plus the Fig. 4 data as `fig4_*.csv` |
| `.claude-memory/` | the 21 project memory files described above |

Two files in `scripts/figures/` end in `.session.py` and `.session.sh`. Those
are edited versions of `pin6.py` and `final_video.sh` that diverged from the
committed copies. They were kept separately rather than overwriting, so compare
before assuming either is canonical.

## What is NOT under version control

Still on `airlab-desktop` only, and lost when that machine goes:

- **Custom MAVROS additions** the policy node depends on, so `scripts/deploy/db_inference.py` is a record rather than something runnable on its own
  - `mavros/src/plugins/jai_out_plugin.cpp` and `jai_raw_plugin.cpp` with their headers
  - `mavros_msgs/msg/JAIOut.msg`, `JaiRaw.msg`, `mavros_msgs/srv/JAISET.srv`
- **`hw_final/`** including `FROZEN_COMMAND.sh`, `terrain_commands/*.sh` and the trial CSVs
- **`roboclaw_velocity_test.py`** and the runbooks at project root
- Local edits to `natnet_ros.launch` in the vendored `castacks` clone

Roughly 1.5 MB of code against 250 MB of trial logs. Worth rescuing the code if
the machine is still reachable.

Model checkpoints are excluded by `.gitignore` (`*.pt`), so `--model_path` in
any recorded command points outside the repo.

## Gotchas

**Author history was rewritten on 2026-09-21.** 160 commits from 2026-08-21
onward were wrongly attributed to the machine's global git identity and have
been corrected to `Ishaan Bhimwal <ibhimwal@andrew.cmu.edu>`, author and
committer. File contents were unchanged, verified by tree hash. Any clone made
before that date has incompatible history and needs

```bash
git fetch origin
git reset --hard origin/ish/hybrid_mode
git fetch --tags --force     # the tag moved too
```

**Git identity was set repo-local, not global**, because `airlab-desktop` was
shared and its global config belonged to someone else. On a new machine set it
normally.
