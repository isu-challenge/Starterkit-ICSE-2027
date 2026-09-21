# Track A: In-Car Scene Translation

Translate each provided simulated in-car image into a realistic-looking image
while preserving the original scene content and geometry.

## Requirements

- Submit one generated image for every simulated image.
- Keep the original width, height, framing, and camera viewpoint.
- Do not crop, rotate, mirror, add, remove, or move scene objects.
- Return RGB images in PNG, JPEG, BMP, or TIFF format. PNG is recommended.
- Keep filename stems unique and consistent across paired simulated and generated images.
- Do not include semantic masks, model files, or unrelated metadata in the submission folder.

## Submission Structure

```text
submissions/your_team_name/
├── simulated/       Provided simulated source images
├── generated/       Your translated output images
├── reference/       Provided real target-domain images
└── labels/          Provided scene-label JSON files
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

Record generation timing in `generated/generation_times.json`:

```json
{
  "total_seconds": 12.0,
  "hardware": "cpu"
}
```

## Generate Images

Use `participant_starter.ipynb` or implement a `Translator` in
`example_custom_translator.py`. The translator receives a simulated RGB image
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
