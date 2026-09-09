"""Run the existing Qwen ISU system against the Track B starter dataset.

Run from track_b/:

    python test/test_evaluator_qwen.py --device cpu

This is a real-model smoke test. It downloads Qwen2.5-VL-3B-Instruct on the
    first run and writes the report to test/test_output/<team_name>/qwen_report.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TRACK_B_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRACK_B_DIR.parent
DATA_DIR = REPO_ROOT / "data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

if str(TRACK_B_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_B_DIR))

from evaluator_cli import EvaluatorTrackB, QwenISUSystem  # noqa: E402


def run_smoke_test(
    data_dir: Path = DATA_DIR,
    output_path: Path | None = None,
    device: str = "cpu",
    image_kind: str = "synthetic",
    team_name: str = "sample_team",
) -> dict:
    """Evaluate the existing Qwen ISU system on the starter dataset."""
    if output_path is None:
        output_path = TEST_OUTPUT_DIR / team_name / f"{team_name}_report.json"
    system = QwenISUSystem(device=device)
    evaluator = EvaluatorTrackB(
        data_dir=data_dir,
        isu_system=system,
        image_kind=image_kind,
        output_path=output_path,
        team_name=team_name,
    )
    report = evaluator.evaluate()

    metrics = report["metrics"]
    assert report["team_name"] == team_name
    assert metrics["scenes"] > 0
    assert metrics["evaluated_features"] > 0
    assert len(report["predictions"]) == metrics["scenes"]
    assert output_path.exists()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--team-name", default="sample_team")
    parser.add_argument("--device", default="cuda", help="Torch device: cpu, cuda, or mps")
    parser.add_argument(
        "--image-kind",
        choices=("real", "synthetic", "both"),
        default="synthetic",
    )
    args = parser.parse_args()
    report = run_smoke_test(
        args.data_dir, args.output, args.device, args.image_kind, args.team_name
    )
    print("Qwen Track B smoke test passed")
    print(report["metrics"])


if __name__ == "__main__":
    main()
