# Track B: In-Car Scene Understanding

Implement an in-car scene understanding system that predicts categorical
features visible in an RGB image. The evaluator sends only the image to your
system; labels are used separately for scoring and are not passed to the
model.

## Requirements

- Implement the `ISUSystem` interface with `predict(image)` and `name()` methods.
- Accept a PIL RGB image and return a dictionary of feature names and values.
- Return only values from the answer options supplied in the evaluator prompt.
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
