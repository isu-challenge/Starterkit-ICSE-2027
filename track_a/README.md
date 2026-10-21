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

You submit your translator as code, together with a Docker image that it runs in, and the
official evaluation runs it on withheld simulated scenes. Prepare a folder, for example
`submissions/<team_name>/` in this directory (ignored by Git);
[`sample_submission/track_a/`](../sample_submission/track_a/) is a template:

```text
translator.py      defines create_translator(device), which returns your Translator
Dockerfile         builds the image your translator runs in, starting with FROM isu-challenge-base
```

Add any other files your translator needs. The evaluation calls `create_translator(device)`
once and then `translate(image, auxiliary_data)` for every simulated scene, exactly as
[`translate_cli.py`](translate_cli.py) does. `auxiliary_data` holds the scene `stem`, its
`label`, and its `canny`, `segmentation`, and `depth` images. Each returned image must keep
the size and scene content of its input (see Requirements).

The withheld scenes include scenes with a real reference image, which are used for the
realism metrics, and synthetic scenes without one, which are used for failure detection and
diversity: the ISU system answers the feature questions for your generated image, and a
feature fails when its answer differs from the scene label. SAM structure is evaluated on
all scenes.

The rules for the `Dockerfile` and the runtime environment are the same as for Track B, see
[Docker image](../track_b/README.md#docker-image) and [Runtime](../track_b/README.md#runtime).
Generation time is measured by the evaluation for the whole run, including model loading.

### Test your submission locally

From the repository root, build your image and translate the first 80 scenes of the public
dataset with the restrictions of the official evaluation: 60 scenes with a real reference
image and 20 without. Without an NVIDIA GPU, leave out `--gpus all` and add `--device cpu`:

```bash
TEAM=track_a/submissions/<team_name>
docker build -t isu-track-a "$TEAM"
docker run --rm --gpus all --network none --read-only --tmpfs /tmp --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -v "$PWD:/starterkit:ro" -v "$PWD/data:/data:ro" \
    -v "$PWD/$TEAM:/submission:ro" -v "$PWD/$TEAM:/out" \
    --workdir /submission --entrypoint python isu-track-a \
    /starterkit/track_a/translate_cli.py --translator-dir /submission --data-dir /data --output /out --limit 80
```

This writes `generated/` and `generation_times.json` into your folder. Add the matching
simulated images, labels, and the real images, then evaluate:

```bash
mkdir -p "$TEAM"/simulated "$TEAM"/labels "$TEAM"/reference && cp data/real/*_real.jpg "$TEAM"/reference/
for image in "$TEAM"/generated/*.png; do
    stem=$(basename "$image" .png)
    cp "data/images/${stem}_sim.png" "$TEAM"/simulated/ && cp "data/labels/${stem}.json" "$TEAM"/labels/
done
cd track_a
python evaluator_cli.py evaluate --submission-path "../$TEAM" --output report.json --failure-model qwen
```

Then, from the repository root, encrypt the folder and commit the result as described in
[Submitting via pull request](../README.md#submitting-via-pull-request). The tool leaves
out `simulated/`, `reference/`, `labels/`, `generated/`, and `generation_times.json`:

```bash
python tools/encrypt_submission.py --source track_a/submissions/<team_name> --output submissions/<team_id>/track_a
```
