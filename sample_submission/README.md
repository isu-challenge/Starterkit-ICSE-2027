# Sample submission

Templates for an official submission. Copy the folder of your track to `private/` (ignored
by Git), replace the `TODO` parts, and test locally. Then encrypt it into your team folder
`submissions/<team_id>/` (you receive the team id with your registration) and open a pull
request against `main`.

```text
sample_submission/
├── track_a/
│   ├── generated/             your generated images go here (the folder ships empty)
│   ├── generation_times.json  generation time and hardware
│   └── code/generate.py       template Translator and script that write generated/ and generation_times.json
└── track_b/
    ├── system.py              template ISU system with create_system(device)
    ├── models.txt             Hugging Face models the system loads (optional)
    └── requirements.txt       extra PyPI packages (optional)
```

General rules:

- Pull requests are public, so only the encrypted submission is committed. With the
  [age](https://github.com/FiloSottile/age#installation) tool installed, run from the
  repository root:

  ```bash
  python tools/encrypt_submission.py --source private/track_b --output submissions/<team_id>/track_b
  ```

  It writes `submission.tar.age.000`, `.001`, ... into the team folder; commit only these.
  Run it again after every change to your submission.
- One pull request per track. It may only change files in `submissions/<team_id>/track_a/`
  or `submissions/<team_id>/track_b/`.
- A submission may contain up to 2 GB and 20,000 files before encryption.
- Every evaluation that starts counts toward the daily limit per track, including one
  cancelled by a newer push.
- The result is posted as a comment on the pull request and published on the leaderboard.

## Track A

1. Copy the template: `mkdir -p private && cp -r sample_submission/track_a private/track_a`.
2. Implement `MyTranslator.translate()` in `code/generate.py`. It subclasses `Translator`
   from [`track_a/evaluator_cli.py`](../track_a/evaluator_cli.py) and receives one simulated
   RGB image plus `auxiliary_data` with the scene label and, when downloaded, its Canny,
   instance-segmentation, and depth images. It must return an image of the same size that
   keeps the scene content and geometry (see the [Track A README](../track_a/README.md)).
   Working examples are in [`track_a/test/example_custom_translator.py`](../track_a/test/example_custom_translator.py);
   `create_translator()` shows how to use `AppearanceTranslator` directly.
3. Run it on the simulated images of the evaluation set, in the starter-kit data layout
   (`data/` at the repository root, as downloaded by the starter notebooks):

   ```bash
   python private/track_a/code/generate.py --data-dir data \
       --output private/track_a --hardware "1x NVIDIA RTX 4090"
   ```

   This writes `generated/<stem>.png` for every `images/<stem>_sim.png` and
   `generation_times.json` with the time spent in `translate()`. Add `--limit 5` for a
   quick test.
4. Encrypt it: `python tools/encrypt_submission.py --source private/track_a --output submissions/<team_id>/track_a`.

The evaluation checks that `generated/` contains exactly one image per simulated scene,
with the same width and height, named without `_sim` (`sample_00218.png` for
`sample_00218_sim.png`), and no sub-folders or other files. The `code/` folder is not run
by the evaluation; it documents how the images were produced.

To score your images locally, the evaluator needs a folder that also contains the
simulated images, labels, and real reference images. Assemble it in
`track_a/submissions/` (ignored by Git), as the
[starter notebook](../track_a/participant_starter.ipynb) does:

```bash
L=track_a/submissions/local && mkdir -p $L/simulated $L/labels $L/reference
cp -r private/track_a/generated private/track_a/generation_times.json $L/
for f in $L/generated/*.png; do s=$(basename "$f" .png)
  cp data/images/${s}_sim.png $L/simulated/; cp data/labels/${s}.json data/labels/${s}_label.json $L/labels/ 2>/dev/null
done
cp data/real/*_real.jpg $L/reference/
cd track_a && python evaluator_cli.py evaluate --submission-path submissions/local --output report.json --failure-model qwen
```

[`track_a/test/test_evaluator_example.py`](../track_a/test/test_evaluator_example.py) runs
the same pipeline end to end with `AppearanceTranslator`.

## Track B

1. Copy the template: `mkdir -p private && cp -r sample_submission/track_b private/track_b`.
2. Implement `MyISUSystem` in `system.py`, a subclass of `ISUSystem` from
   [`track_b/evaluator_cli.py`](../track_b/evaluator_cli.py): load your model in `__init__`
   and return one answer per feature in `predict(image)`. The simplest valid submission is
   the reference baseline shown at the top of `system.py`; more examples are `QwenISUSystem`
   in `evaluator_cli.py` and the CLIP and ensemble systems in
   [`track_b/example_custom_system.py`](../track_b/example_custom_system.py).
3. List every Hugging Face model your system loads in `models.txt`, and extra PyPI packages
   in `requirements.txt`. Remove lines you do not need. Your own weights can be files in the
   folder; they stay private because the folder is encrypted.
4. Encrypt it: `python tools/encrypt_submission.py --source private/track_b --output submissions/<team_id>/track_b`.

Your system runs in a container with an NVIDIA GPU (at least 24 GB), 14 GB of RAM, 4 CPU
cores, no network access, and a 2-hour limit for all evaluation images. The file system is
read-only except `/tmp`; the working directory is your submission folder. Python 3.11,
`torch` 2.6.0 (CUDA 12.4), `transformers` 4.x, `accelerate`, `torchvision`, `numpy`,
`Pillow`, `scipy`, `scikit-learn`, `scikit-image`, `pandas`, and `einops` are installed.

`predict(image)` receives a `PIL.Image` in RGB and returns a dictionary with one answer per
feature in `TRACK_B_FEATURES` (see [`track_b/track_b_evaluator.py`](../track_b/track_b_evaluator.py)).
Answers are compared as text, ignoring case and surrounding spaces. Features whose label is
`null` are not scored. `set_prompt(prompt)` receives the questions and allowed answers of
the evaluation set before the first image.

Test the system offline, as the official evaluation runs it, on the dataset in `data/` at
the repository root (download the models from `models.txt` once beforehand):

```bash
cd private/track_b
HF_HUB_OFFLINE=1 python - <<'EOF'
import sys
sys.path[:0] = [".", "../../track_b"]
from evaluator_cli import EvaluatorTrackB
from system import create_system

EvaluatorTrackB(data_dir="../../data", isu_system=create_system("cuda"),
                output_path="../../track_b/track_b_evaluation_output/report.json").evaluate()
EOF
```

[`track_b/test/test_evaluator_qwen.py`](../track_b/test/test_evaluator_qwen.py) runs the
reference baseline the same way.
