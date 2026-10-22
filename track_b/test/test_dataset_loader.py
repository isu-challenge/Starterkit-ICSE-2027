"""Check that the Track B loader finds real photos and synthetic renders in the dataset layout.

Run from track_b/ (no model is loaded):

    python test/test_dataset_loader.py
    python -m pytest test/test_dataset_loader.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

TRACK_B_DIR = Path(__file__).resolve().parents[1]
if str(TRACK_B_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_B_DIR))

from track_b_evaluator import load_dataset  # noqa: E402


def make_dataset(root: Path, real_name: str) -> Path:
    """Two paired scenes with a real photo and one synthetic-only scene, as on Hugging Face."""
    for folder in ("images", "real", "labels"):
        (root / folder).mkdir(parents=True)
    for stem in ("sample_00000", "sample_00001", "sample_00500"):
        (root / "labels" / f"{stem}.json").write_text(json.dumps({"driver_phone": "NO"}))
        (root / "images" / f"{stem}_sim.png").touch()
    for stem in ("sample_00000", "sample_00001"):
        (root / "real" / real_name.format(stem=stem)).touch()
    return root


def scene_ids(data_dir: Path, image_kind: str) -> list[str]:
    return sorted(scene for scene, _, _ in load_dataset(data_dir, image_kind))


def check_layout(real_name: str) -> None:
    with tempfile.TemporaryDirectory() as temp:
        data_dir = make_dataset(Path(temp), real_name)
        assert scene_ids(data_dir, "real") == ["sample_00000", "sample_00001"]
        assert scene_ids(data_dir, "synthetic") == ["sample_00000_sim", "sample_00001_sim", "sample_00500_sim"]
        assert scene_ids(data_dir, "both") == [
            "sample_00000", "sample_00000_sim", "sample_00001", "sample_00001_sim", "sample_00500_sim"]


def test_real_photos_named_as_on_hugging_face():
    check_layout("{stem}_real.jpg")


def test_real_photos_without_suffix():
    check_layout("{stem}.jpg")


def test_labels_with_label_suffix():
    with tempfile.TemporaryDirectory() as temp:
        data_dir = make_dataset(Path(temp), "{stem}_real.jpg")
        for path in (data_dir / "labels").glob("*.json"):
            path.rename(path.with_name(f"{path.stem}_label.json"))
        assert len(scene_ids(data_dir, "both")) == 5


if __name__ == "__main__":
    test_real_photos_named_as_on_hugging_face()
    test_real_photos_without_suffix()
    test_labels_with_label_suffix()
    print("Track B dataset loader: all checks passed.")
