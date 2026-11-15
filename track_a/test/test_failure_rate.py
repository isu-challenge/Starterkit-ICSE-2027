"""Failure definition of evaluate_failure_rate, with a fake SUT (no model download).

A feature fails when the SUT answers it correctly on the real photo and wrongly on the
generated image of the same scene. Run from track_a: python -m pytest test/test_failure_rate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track_a_metrics import evaluate_failure_rate, feature_diversity  # noqa: E402

LABEL = {"driver_phone": "YES", "driver_safety_belt": True, "suitcase": "NO"}
CORRECT = {"driver_phone": "YES", "driver_safety_belt": "YES", "suitcase": "NO"}
ALL_WRONG = {"driver_phone": "NO", "driver_safety_belt": "NO", "suitcase": "YES"}
# Images are uniform; the fake SUT answers by pixel value.
ANSWERS = {
    100: CORRECT, 101: {**CORRECT, "driver_safety_belt": "NO"},
    110: {**CORRECT, "driver_phone": "NO"}, 111: CORRECT,
    120: CORRECT,
    130: ALL_WRONG, 131: CORRECT,
    140: CORRECT, 141: ALL_WRONG,
}
# scene: (real photo pixel, generated image pixel)
SCENES = {
    "s1": (100, 101),  # right on real, wrong on generated -> failure (1 feature)
    "s2": (110, 111),  # phone wrong on real, right on generated -> no failure, phone not testable
    "s3": (120, 120),  # identical images -> no failure
    "s4": (130, 131),  # every feature wrong on real -> not testable
    "s5": (140, 141),  # right on real, all wrong on generated -> failure (3 features)
}


class FakeSUT:
    revision = "test"
    seed = 0
    DETERMINISTIC_SETTINGS = {}

    def classify_features(self, image, vocabulary):
        return dict(ANSWERS[int(np.asarray(image)[0, 0, 0])]), {}


def _write_image(path, value):
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(path)


def test_failures_are_correct_on_real_and_wrong_on_generated(tmp_path):
    folders = {name: tmp_path / name for name in ("generated", "reference", "labels")}
    for folder in folders.values():
        folder.mkdir()
    for stem, (real, generated) in SCENES.items():
        (folders["labels"] / f"{stem}.json").write_text(json.dumps(LABEL))
        _write_image(folders["reference"] / f"{stem}_real.png", real)
        _write_image(folders["generated"] / f"{stem}.png", generated)

    result = evaluate_failure_rate(folders, FakeSUT())
    rows = {row["image"]: row for row in result["images"]}

    assert {stem for stem, row in rows.items() if row["failure"]} == {"s1", "s5"}
    assert rows["s1"]["feature_failures"] == {"driver_phone": False, "driver_safety_belt": True,
                                              "suitcase": False}
    assert "driver_phone" not in rows["s2"]["feature_failures"]
    assert rows["s4"]["feature_failures"] == {}
    # The boolean label True is compared as YES.
    assert rows["s3"]["real_correct"]["driver_safety_belt"]

    assert result["executed"] == 5
    assert result["testable_scenes"] == 4
    assert result["failures"] == 2
    assert result["failure_rate"] == 2 / 4
    assert (result["failing_features"], result["testable_features"]) == (4, 11)
    assert result["feature_failure_rate"] == 4 / 11
    assert feature_diversity(result, {stem: LABEL for stem in SCENES})["failing_scenes"] == 2
