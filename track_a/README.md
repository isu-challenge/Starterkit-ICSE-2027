# Track A: In-Car Scene Translation

Translate each provided simulated in-car image into a realistic-looking image
while preserving the original scene content and geometry.

## Requirements

- Return one generated image for every simulated image.
- Keep the original width, height, framing, and camera viewpoint.
- Do not crop, rotate, mirror, add, remove, or move scene objects.
- Preserve predefined [features](https://isu-challenge.github.io/feature-schema.html)
- Return RGB images in PNG, JPEG, BMP, or TIFF format. PNG is recommended.
- Keep filename stems unique and consistent across paired simulated and generated images.
- Do not write semantic masks or unrelated metadata into the generated images.

## Jupyter Notebooks

To get familiar with the task, run the two notebooks in this order:

1. [`participant_starter.ipynb`](participant_starter.ipynb) downloads the
   dataset, lets you browse the scenes and their auxiliary channels,
   implements `translate(...)`, runs it on every scene, and writes the
   submission folder `submissions/<team_name>/` including
   `generation_times.json`.
2. [`track_a_evaluator.ipynb`](track_a_evaluator.ipynb) scores every
   submission in `submissions/` (SAM structure, realism, failure detection,
   diversity, and efficiency) and writes the reports to
   `track_a_evaluation_output/`.

## Submission Structure

```text
submissions/your_team_name/
├── simulated/       Provided simulated source images
├── generated/       Your translated output images
├── reference/       Provided real target-domain images
├── labels/          Provided scene-label JSON files
└── generation_times.json   Generation timing (see below)
```

Example pairing:

```text
simulated/sample_0001_sim.png
generated/sample_0001.png
reference/sample_0001.jpg
labels/sample_0001_label.json
```

Simulated and generated images pair by filename stem. Real images are a target
distribution and do not need one-to-one filename pairing. Do not rename or
omit the provided simulated scenes, and do not modify the simulated or real
folders.

Record generation timing in `generation_times.json` at the top level of your
submission folder (next to `generated/`, not inside it):

```json
{
  "total_seconds": 12.0,
  "hardware": "cpu"
}
```

## Generate Images

Use `participant_starter.ipynb` or implement a `Translator` in
`test/example_custom_translator.py`. The translator receives a simulated RGB image
and must return one RGB generated image with the same scene structure.

## Evaluate a Submission

From the `track_a` directory:

```bash
python evaluator_cli.py evaluate \
    --submission-path submissions/your_team_name \
    --output report.json \
    --failure-model qwen \
    --device cpu
```

Options:

- `--submission-path` (required): completed submission directory.
- `--output`: report path; defaults to `track_a_evaluation_output/report.json`.
- `--failure-model`: `moondream` or `qwen`; defaults to `moondream`.
- `--device`: `cpu`, `cuda`, or `mps`; defaults to `cpu`.

The report includes structural validity, realism, failure detection, feature
diversity, CLIP visual diversity, and generation efficiency metrics.

Evaluation uses public image metrics, class-agnostic SAM segmentation, and the
selected VLM. Model weights may be downloaded from Hugging Face on first use.

## Evaluation Order

1. Generate translated images with your `Translator` implementation.
2. Store the generated images and timing metadata in the submission folders.
3. Run `evaluator_cli.py` from the `track_a` directory.
4. Inspect the JSON report before submitting your results.

The evaluator does not generate submission images; it evaluates the files
already present in your submission. The first run may download model weights.
SAM is used for structural validity, and the selected failure model is used
for feature-based failure detection. `qwen` generally requires more memory
than `moondream`.

## Quick Translation and Evaluation Smoke Test

From the `track_a` directory, run the example test to generate a translated
submission with `AppearanceTranslator` and evaluate it with the full pipeline:

```bash
python -m pytest test/test_evaluator_example.py -v -s
```

The test writes the generated example submission and report under
`test/test_output/example/`. It loads SAM, VGG16, CLIP, and Qwen2.5-VL, so it
requires the model dependencies and may download weights on the first run.

## Official Submission

You submit your translator as code, and the official evaluation runs it on withheld
simulated scenes. Prepare a folder, for example `submissions/<team_name>/` in this
directory (ignored by Git); [`sample_submission/track_a/`](../sample_submission/track_a/)
is a template:

```text
translator.py      defines create_translator(device), which returns your Translator
models.txt         optional: public Hugging Face models your translator loads, <repo_id>[@<revision>]
requirements.txt   optional: extra PyPI packages, one per line, e.g. diffusers==0.35.1
```

Add any other files your translator needs, such as weights. The evaluation calls
`create_translator(device)` once and then `translate(image, auxiliary_data)` for every
simulated scene, exactly as [`translate_cli.py`](translate_cli.py) does. `auxiliary_data`
holds the scene `stem`, its `label`, and its `canny`, `segmentation`, and `depth` images.
Each returned image must keep the size and scene content of its input (see Requirements).

Your translator runs with the same environment as Track B systems: an NVIDIA GPU with at
least 24 GB of memory, 14 GB of RAM, 4 CPU cores, no network access, 2 hours for all
scenes including model loading, and a read-only file system except `/tmp` with your
submission folder as working directory. The pre-installed packages and the rules for
`models.txt` and `requirements.txt` are listed in the
[Track B README](../track_b/README.md#official-submission). Generation time is measured
by the evaluation for the whole run, including model loading.

### Test your submission locally

Run your translator offline with `translate_cli.py`, then evaluate the result. The
evaluation needs `simulated/`, `reference/`, and `labels/` in the folder, as created by
the starter notebook. From the repository root:

```bash
HF_HUB_OFFLINE=1 python track_a/translate_cli.py --translator-dir track_a/submissions/<team_name> \
    --data-dir data --output track_a/submissions/<team_name> --limit 10
cd track_a
python evaluator_cli.py evaluate --submission-path submissions/<team_name> --output report.json --failure-model qwen
```

Then, from the repository root, encrypt the folder and commit the result as described in
[Submitting via pull request](../README.md#submitting-via-pull-request). The tool leaves
out `simulated/`, `reference/`, `labels/`, `generated/`, and `generation_times.json`:

```bash
python tools/encrypt_submission.py --source track_a/submissions/<team_name> --output submissions/<team_id>/track_a
```
