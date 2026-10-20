# Track B: In-Car Scene Understanding

Implement an in-car scene understanding system that predicts categorical
features visible in an RGB image. The evaluator sends only the image to your
system; labels are used separately for scoring and are not passed to the
model.

## Requirements

- Implement the `ISUSystem` interface with `predict(image)` and `name()` methods.
- Accept a PIL RGB image and return a dictionary of feature names and values.
- Return only values from the answer options supplied in the evaluator prompt. I.e. predict predefined [features](https://isu-challenge.github.io/feature-schema.html)
- Do not use ground-truth labels or label-derived data during inference.
- Keep system code separate from evaluation and scoring logic.

The provided `QwenISUSystem` in `evaluator_cli.py` and
`example_custom_system.py` show the expected system interface.

## Jupyter Notebooks

To get familiar with the task, run the two notebooks in this order:

1. [`participant_starter.ipynb`](participant_starter.ipynb) downloads the
   dataset, lets you browse the scenes, implements `isu_system(image)`, runs it
   on every scene, and saves the predictions to
   `submissions/track_b_predictions.json`.
2. [`track_b_evaluator.ipynb`](track_b_evaluator.ipynb) scores every
   prediction file in `submissions/` against the label files and writes the
   reports to `track_b_evaluation_output/`.

## Evaluate a System

From the `track_b` directory:

```bash
python evaluator_cli.py evaluate \
    --data-dir ../data \
    --output report.json \
    --model qwen \
    --device cuda
```

Options:

- `--data-dir`: data directory with `images/`, `real/`, and `labels/` (default: `data`).
- `--image-kind`: `real`, `synthetic`, or `both` (default: `both`).
- `--model`: `qwen` or `moondream` (default: `qwen`).
- `--device`: `cuda`, `cpu`, or `mps` (default: `cuda`).
- `--output`: report output path (default: `track_b_evaluation_output/report.json`).
- `--ignore-feature`: additional feature to exclude; repeatable.

The report contains the evaluated feature vocabulary, per-scene predictions,
feature matches, scene exact-match accuracy, aggregate feature accuracy, and
inference latency. Features containing `env` or `light` in their name are
excluded from scoring.

## Dataset Structure

```
data/
├── images/                 Synthetic RGB images, named `<stem>_sim.png`
├── real/                   Real RGB images, named `<stem>.jpg` when available
├── labels/                 Scene labels, named `<stem>_label.json`
├── depth/                  Depth channels, when available
│   ├── png/                PNG maps, named `<stem>_sim_depth.png`
│   └── exr/                EXR maps, named `<stem>_sim_depth.exr`
├── canny/                  Canny edge maps, named `<stem>_canny.png`
└── instance_seg/           Instance maps, named `<stem>_instance_seg.png`
```

Depth, Canny, and instance-segmentation files are auxiliary channels for the
same simulated camera view. They may be used as additional input by a custom
system, while the built-in evaluator systems receive only the RGB image. The
instance-segmentation color mapping is defined in
`instance_seg/instance_seg_legend.json` when provided.

## Evaluation Order

1. Implement or configure your `ISUSystem`.
2. Run it against the starter dataset with `evaluator_cli.py`.
3. Inspect the JSON report for prediction format, feature matches, and latency.
4. Test again with the image kind and device appropriate for your system.

The evaluator builds the feature vocabulary from labels, creates the prompt,
sends each RGB image to the system, and compares returned predictions with the
applicable labels.

## Quick Qwen Smoke Test

From the `track_b/test` directory, run the existing Qwen system against the
starter dataset:

```bash
python test_evaluator_qwen.py \
    --device cpu \
    --team-name sample_team
```

The smoke test writes its report to:

```text
test/test_output/sample_team/sample_team_report.json
```

Use `--image-kind synthetic`, `real`, or `both` to choose which dataset images
to evaluate. The first run may download Qwen2.5-VL-3B-Instruct from Hugging
Face and requires the model dependencies installed from `requirements.txt`.

The smoke test verifies that scenes, evaluated features, predictions, the team
name, and the JSON report are all present. It does not imply that the model
achieved a high accuracy score.

## Model Requirements

The first Qwen run may download `Qwen/Qwen2.5-VL-3B-Instruct` from Hugging
Face. Install dependencies from `requirements.txt`. CPU execution is
supported for a functional smoke test but is considerably slower.

## Official Submission

The official evaluation runs your system on a withheld set of real and synthetic
images. Prepare it in a folder of its own, for example `submissions/<team_name>/` in
this directory (ignored by Git); [`sample_submission/track_b/`](../sample_submission/track_b/)
is a template:

```text
system.py          defines create_system(device), which returns your ISUSystem
models.txt         optional: public Hugging Face models your system loads, <repo_id>[@<revision>]
requirements.txt   optional: extra PyPI packages, one per line, e.g. timm==1.0.15
```

Add any other files your system needs, such as weights. The evaluation calls
`create_system(device)` once, `set_prompt(prompt)` once if your system defines it,
and then `predict(image)` for every image. The `track_b/` folder is importable, so
the reference baseline is this `system.py` with `Qwen/Qwen2.5-VL-3B-Instruct` in
`models.txt`:

```python
from evaluator_cli import QwenISUSystem


def create_system(device):
    return QwenISUSystem(device=device)
```

Your system runs with:

- an NVIDIA GPU with at least 24 GB of memory, 14 GB of RAM, and 4 CPU cores;
- no network access, and 2 hours for all images, including model loading;
- a read-only file system except `/tmp`; the working directory is your submission folder;
- Python 3.11, `torch` 2.6.0 (CUDA 12.4), `transformers` 4.x, `accelerate`,
  `torchvision`, `numpy`, `Pillow`, `scipy`, `scikit-learn`, `scikit-image`, `pandas`,
  and `einops`;
- the models from `models.txt` in the default Hugging Face cache: load them with
  `from_pretrained("<repo_id>")`, with the same `revision` if you pinned one.

Packages from `requirements.txt` are installed from PyPI as binary wheels; pip
options, URLs, and local paths are not accepted, and `torch` stays at 2.6.0. Latency
is measured again on the predefined `g6e.2xlarge` instance for the final evaluation.

### Test your submission locally

Run your system offline from its folder, as the official evaluation does. Download the
models from `models.txt` once beforehand, and define `set_prompt(prompt)` even if
your system ignores the prompt:

```bash
cd track_b/submissions/<team_name>
HF_HUB_OFFLINE=1 python - <<'EOF'
import sys
sys.path[:0] = [".", "../.."]
from evaluator_cli import EvaluatorTrackB
from system import create_system

EvaluatorTrackB(data_dir="../../../data", isu_system=create_system("cuda"),
                output_path="../../track_b_evaluation_output/report.json").evaluate()
EOF
```

Then, from the repository root, encrypt the folder and commit the result as described
in [Submitting via pull request](../README.md#submitting-via-pull-request):

```bash
python tools/encrypt_submission.py --source track_b/submissions/<team_name> --output submissions/<team_id>/track_b
```
