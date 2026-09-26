"""Full, unmocked run of the Track A evaluation pipeline.

Unlike test_evaluator_cli.py (which mocks every metric), this test mirrors
the real participant workflow: first generate a submission's generated/
scenes with the shipped example translator (AppearanceTranslator from
example_custom_translator.py), then run the real evaluators end to end on
that output: SAM structural validity, public realism metrics, Qwen2.5-VL
failure detection (the configured SUT proxy), feature diversity, CLIP visual
diversity, and efficiency.

This downloads/loads SAM, Qwen2.5-VL-3B-Instruct, CLIP and VGG16, so it is
slow and network-dependent. Run it explicitly, separate from the fast mocked
suite:

    python -m pytest test/test_evaluator_example.py -v -s
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import evaluator_cli
from example_custom_translator import AppearanceTranslator

SAMPLE_SUBMISSION = Path(__file__).parent / "test_output" / "example" / "submission"

pytestmark = pytest.mark.integration


def find_sample_submission() -> Path:
    """Return the checked-in example submission fixture."""
    required = ("simulated", "generated", "reference", "labels")
    if not SAMPLE_SUBMISSION.exists() or not all(
        (SAMPLE_SUBMISSION / folder).is_dir() for folder in required
    ):
        pytest.skip(f"Example submission fixture not found at {SAMPLE_SUBMISSION}")
    return SAMPLE_SUBMISSION


def generate_submission(tmp_path: Path) -> Path:
    """Step 1: run the example custom translator to (re)generate generated/."""
    submission = tmp_path / "generated_team"
    shutil.copytree(find_sample_submission(), submission)

    np.random.seed(0)
    translator = AppearanceTranslator(intensity=0.5)
    generated_dir = submission / "generated"
    for path in sorted((submission / "simulated").glob("*")):
        transformed = translator.translate(Image.open(path).convert("RGB"))
        stem = path.stem.removesuffix("_sim")
        transformed.save(generated_dir / f"{stem}.png")

    (submission / "generation_times.json").write_text(
        json.dumps({"total_seconds": 1.0, "hardware": "test-harness"})
    )
    return submission


def test_full_pipeline_with_appearance_translator_and_qwen(tmp_path):
    # Save outputs to test folder for inspection
    test_output_dir = Path(__file__).parent / "test_output" / "example"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: generate the submission with the example custom translator.
    submission = generate_submission(tmp_path)

    # Copy submission to test output for inspection
    submission_copy = test_output_dir / "submission"
    if submission_copy.exists():
        shutil.rmtree(submission_copy)
    shutil.copytree(submission, submission_copy)
    print(f"submission generated and saved to {submission_copy}")

    # Step 2: evaluate the generated submission with the real CLI pipeline.
    output = test_output_dir / "report.json"
    report = evaluator_cli.ParticipantEvaluator(
        submission, output, failure_model="qwen", device="cpu"
    ).evaluate()

    assert output.exists()

    # Structural validity ran through real SAM segmentation.
    assert "semantic_segmentation_score" in report["sam"]["simulated_vs_generated"]["summary"]

    # Public realism metrics ran for both generated and simulated vs. reference.
    assert report["realism_raw"]["generated_vs_real"]["summary"]
    assert report["realism_raw"]["simulated_vs_real"]["summary"]

    # Failure detection actually queried the Qwen2.5-VL SUT proxy.
    assert report["failure_raw"]["executed"] >= 1
    assert report["final"]["failure"]["executed"] == report["failure_raw"]["executed"]

    # Feature diversity and CLIP visual diversity were both computed.
    assert "mean_pairwise_hamming" in report["final"]["feature_diversity"]
    assert "images" in report["final"]["clip_visual_diversity"]

    # Efficiency read the generation_times.json fixture.
    assert report["efficiency"]["available"] is True
    assert report["efficiency"]["seconds_per_scene"] == 1.0

