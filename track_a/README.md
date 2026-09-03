# Sim2Real Competition Evaluator

Organizer-side tooling for a sim-to-real image translation competition. Participants receive
simulated (rendered) in-car scenes and submit a translated ("generated") version of each one that
should look real. This repo scores those submissions.

## What it measures

Every submission is scored on three things:

1. **Realism** — how close the generated images are to the real reference distribution.
   FID, KID, Inception Score, plus paired SSIM/PSNR/MSE/cosine similarity/etc. against the
   reference images. Reported for `generated vs real` and, as a baseline for comparison,
   `simulated vs real`.
2. **Semantic preservation** — does the generated image still show the same scene content
   as the simulated source? Class-agnostic SAM segmentation is run on both images and compared
   with boundary F1, symmetric region covering, matched IoU, ARI, and NMI. Also run for
   `real vs generated`, since a submission may instead aim to reproduce the real image directly.
3. **Qwen2.5-VL agreement** — the Qwen2.5-VL-3B-Instruct VLM
   is asked the same categorical questions (driver emotion, phone use, seatbelt, etc., drawn from
   each scene's label JSON) about the generated image and the real reference image. A scene
   "fails" if any answer disagrees.

A submission is only counted as valid if it passes *either* the simulated-fidelity route
(reasonable SAM agreement with the simulated source) *or* the real-fidelity route (near-perfect
SAM agreement with the real reference) — a team can target either domain. Final ranking uses a
Pareto front over failure rate, realism, feature diversity, and (if timing data is provided)
efficiency, with a normalized weighted score as a tie-break.

## Layout

```
dataset/data/            7 example scenes: <stem>.jpg (real), <stem>_sim.png (simulated), <stem>.json (label)
submissions/<team>/      simulated/, generated/, reference/, labels/ — one folder per contestant
image_metrics_utils.py   metric implementations (SAM, Qwen2.5-VL, FID/KID/SSIM/etc.)
track_a_metrics.py       the official Track A pipeline: validation, realism, failure rate, ranking
track_a_challenge_evaluator.ipynb   run this to score everything in submissions/ → track_a_evaluation_output/
sim2real_challenge_evaluator.ipynb  earlier/parallel notebook with a free-form Qwen2.5-VL captioning comparison → evaluation_output/
participant_starter.ipynb           template participants use to build their submissions/<team>/ folder
README_CONTESTANTS.md    submission format rules handed to participants
```

## Running it

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Open `track_a_challenge_evaluator.ipynb` with the `.venv` kernel and run all cells. It discovers
every folder under `submissions/` that has populated `simulated/`, `generated/`, `reference/`, and
`labels/` subfolders and writes one JSON report per team plus `leaderboard.json` to
`track_a_evaluation_output/`. Re-running only some cells recomputes only those sections in memory —
since each cell overwrites the team's JSON file with whatever is currently in memory, do a full
top-to-bottom run before trusting `leaderboard.json`, or a partial rerun will silently drop the
stages you didn't re-execute from the saved report.

SAM (`facebook/sam-vit-base`) and Qwen2.5-VL-3B-Instruct are downloaded from Hugging Face on
first use and cached locally.

## Baseline

`submissions/provided_data_baseline/` is a degenerate sanity-check submission where `generated`
is byte-identical to `reference` — it exists to confirm the pipeline reports near-perfect scores
when there's nothing left to measure, not as an example of a real submission.
