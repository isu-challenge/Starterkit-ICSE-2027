"""Track B participant evaluation CLI for perception robustness.

Usage:
    python evaluator_cli.py evaluate --data-dir ../data --output report.json
    python evaluator_cli.py evaluate --data-dir ../data --model qwen --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from track_b_evaluator import (
    DEFAULT_IGNORED_FEATURES,
    build_prompt,
    build_vocabulary,
    comparable_features,
    feature_metrics,
    load_dataset,
    normalize,
    parse_json_answer,
    query_model,
)


def _load_dataset_with_real_folder(data_dir: Path, image_kind: str) -> list[tuple[str, Path, dict]]:
    """Load dataset supporting real/ folder for real images.

    Supports both:
    - New structure: data/real/, data/images/, data/labels/
    - Old structure: data/images/, data/labels/
    """
    data_dir = Path(data_dir)
    labels_dir = data_dir / "labels" if (data_dir / "labels").is_dir() else data_dir
    images_dir = data_dir / "images" if (data_dir / "images").is_dir() else data_dir
    real_dir = data_dir / "real" if (data_dir / "real").is_dir() else None

    scenes = []
    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem
        if stem.endswith("_label"):
            stem = stem[: -len("_label")]

        # Load real images (prioritize real/ folder if it exists)
        if image_kind in ("real", "both"):
            if real_dir:
                image_path = real_dir / f"{stem}.jpg"
            else:
                image_path = images_dir / f"{stem}.jpg"
            if image_path.exists():
                scenes.append((stem, image_path, json.loads(label_path.read_text())))

        # Load synthetic images
        if image_kind in ("synthetic", "both"):
            image_path = images_dir / f"{stem}_sim.png"
            if image_path.exists():
                scenes.append((f"{stem}_sim", image_path, json.loads(label_path.read_text())))

    if not scenes:
        raise FileNotFoundError(
            f"No matching images and labels found in {data_dir}. Expected "
            "real images in real/ or images/, synthetic in images/ (as *_sim.png), "
            "and labels in labels/ (as *.json)."
        )
    return scenes


class DataLoader(ABC):
    """Abstract base class for loading scene data."""

    @abstractmethod
    def load(self, data_dir: Path, image_kind: str) -> list[tuple[str, Path, dict]]:
        """Load scenes with matching images and labels."""
        pass


class TrackBDataLoader(DataLoader):
    """Load Track B dataset with images and labels.

    Supports both directory structures:
    - New: data/real/ (real images), data/images/ (synthetic), data/labels/
    - Old: data/images/ (all images), data/labels/
    """

    def load(self, data_dir: Path, image_kind: str) -> list[tuple[str, Path, dict]]:
        """Load dataset with support for real/ folder and legacy structure."""
        return _load_dataset_with_real_folder(Path(data_dir), image_kind)


class ISUSystem(ABC):
    """Abstract base class for in-car scene understanding systems.

    Participants must implement this interface with their own ISU model.
    """

    @abstractmethod
    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Classify scene features from RGB image.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Dictionary mapping feature names to predicted values (strings)
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the system name for reporting."""
        pass


class QwenISUSystem(ISUSystem):
    """Qwen2.5-VL-3B-Instruct based ISU system."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self.prompt = None
        self._load_model()

    def _load_model(self):
        """Load the Qwen model."""
        print(f"Loading {self.model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device,
        ).eval()

    def set_prompt(self, prompt: str) -> None:
        """Set the evaluation prompt."""
        self.prompt = prompt

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Query Qwen model for feature predictions."""
        if self.prompt is None:
            raise RuntimeError("Prompt not set. Call set_prompt() first.")
        raw_answer = query_model(self.model, self.processor, image, self.prompt, self.device)
        prediction = parse_json_answer(raw_answer) or {}
        return prediction

    def name(self) -> str:
        """Return system name."""
        return self.model_name


class MoondreamISUSystem(ISUSystem):
    """Moondream-based ISU system for lightweight evaluation.

    This is a stub implementation; actual Moondream integration would require
    the moondream library and appropriate prompting logic.
    """

    def __init__(self, revision: str = "2025-06-21", device: str = "cpu"):
        self.revision = revision
        self.device = device
        self.prompt = None

    def set_prompt(self, prompt: str) -> None:
        """Set the evaluation prompt."""
        self.prompt = prompt

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Query Moondream model for feature predictions.

        Note: This is a placeholder. Implement actual Moondream integration.
        """
        raise NotImplementedError("Moondream integration not yet implemented in this CLI.")

    def name(self) -> str:
        """Return system name."""
        return f"moondream@{self.revision}"


class ReportWriter:
    """Write evaluation reports to JSON files."""

    @staticmethod
    def write(report: dict[str, Any], output_path: Path) -> None:
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
        print(f"💾 Saved report to {output_path}")

    @staticmethod
    def write_readable_summary(report: dict[str, Any]) -> None:
        """Print a human-readable summary."""
        metrics = report.get("metrics", {})
        print("\n📊 Evaluation Summary:")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.3%}")
        print(f"  Scene exact match: {metrics.get('scene_exact_match', 'N/A'):.3%}")
        print(f"  Correct features: {metrics.get('correct_features', 0)}/{metrics.get('evaluated_features', 0)}")
        print(f"  Latency: {metrics.get('latency_seconds_mean', 'N/A'):.3f} s/scene")
        print(f"  Total time: {metrics.get('latency_seconds_total', 'N/A'):.1f} s")


class EvaluatorTrackB:
    """Orchestrate Track B evaluation pipeline."""

    def __init__(
        self,
        data_dir: Path,
        isu_system: ISUSystem,
        image_kind: str = "both",
        ignored_features: Optional[set[str]] = None,
        output_path: Path = None,
        team_name: str = "unnamed_team",
    ):
        self.data_dir = Path(data_dir)
        self.isu_system = isu_system
        self.image_kind = image_kind
        self.ignored_features = ignored_features or DEFAULT_IGNORED_FEATURES.copy()
        self.output_path = Path(output_path) if output_path else Path("track_b_evaluation_output/report.json")
        self.team_name = team_name
        self.data_loader = TrackBDataLoader()

    def evaluate(self) -> dict[str, Any]:
        """Run complete Track B evaluation pipeline."""
        print("📂 Loading scenes and labels...")
        scenes = self.data_loader.load(self.data_dir, self.image_kind)
        print(f"   Found {len(scenes)} scene-image pairs ({self.image_kind})")

        print("🧾 Building vocabulary and prompt...")
        vocabulary = build_vocabulary(scenes, self.ignored_features)
        prompt = build_prompt(vocabulary)
        self.isu_system.set_prompt(prompt)

        print("🔎 Running predictions...")
        rows = self._run_predictions(scenes, vocabulary)

        print("📊 Aggregating metrics...")
        report = self._build_report(rows, vocabulary, prompt)

        print("💾 Saving report...")
        ReportWriter.write(report, self.output_path)
        ReportWriter.write_readable_summary(report)

        return report

    def _run_predictions(self, scenes: list, vocabulary: dict) -> list[dict]:
        """Run predictions on all scenes."""
        rows = []
        for index, (stem, image_path, label) in enumerate(scenes, 1):
            image = Image.open(image_path).convert("RGB")
            started = time.perf_counter()
            prediction = self.isu_system.predict(image)
            print("prediction:", prediction)
            expected = comparable_features(label, self.ignored_features)
            print("expected:", expected)
            feature_matches = {
                key: normalize(prediction.get(key)) == normalize(value)
                for key, value in expected.items()
            }
            rows.append({
                "image": stem,
                "image_path": str(image_path),
                "expected": expected,
                "prediction": prediction,
                "feature_matches": feature_matches,
                "all_features_match": all(feature_matches.values()),
                "inference_seconds": time.perf_counter() - started,
            })
            print(f"[{index}/{len(scenes)}] {stem}", flush=True)
        return rows

    def _build_report(self, rows: list[dict], vocabulary: dict, prompt: str) -> dict[str, Any]:
        """Assemble complete evaluation report."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "team_name": self.team_name,
            "track": "B",
            "model": self.isu_system.name(),
            "device": str(device),
            "image_kind": self.image_kind,
            "prompt": prompt,
            "features": vocabulary,
            "metrics": {
                **feature_metrics(rows),
                "scene_exact_match": (
                    sum(row["all_features_match"] for row in rows) / len(rows)
                    if rows else None
                ),
                "latency_seconds_total": sum(row["inference_seconds"] for row in rows),
                "latency_seconds_mean": (
                    sum(row["inference_seconds"] for row in rows) / len(rows)
                    if rows else None
                ),
                "scenes": len(rows),
            },
            "predictions": rows,
        }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Evaluation command")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an ISU system")
    eval_parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing images and labels",
    )
    eval_parser.add_argument(
        "--image-kind",
        choices=("real", "synthetic", "both"),
        default="both",
        help="Image types to evaluate",
    )
    eval_parser.add_argument(
        "--model",
        choices=("qwen", "moondream"),
        default="qwen",
        help="ISU model to use",
    )
    eval_parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (cuda, cpu, mps)",
    )
    eval_parser.add_argument(
        "--output",
        default="track_b_evaluation_output/report.json",
        help="Output report path",
    )
    eval_parser.add_argument(
        "--ignore-feature",
        action="append",
        default=[],
        help="Additional features to ignore",
    )
    eval_parser.add_argument(
        "--team-name",
        default="unnamed_team",
        help="Team name stored in the evaluation report",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "evaluate":
        if args.model == "qwen":
            isu_system = QwenISUSystem(device=args.device)
        elif args.model == "moondream":
            isu_system = MoondreamISUSystem(device=args.device)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        ignored_features = DEFAULT_IGNORED_FEATURES | set(args.ignore_feature)

        evaluator = EvaluatorTrackB(
            args.data_dir,
            isu_system,
            args.image_kind,
            ignored_features,
            args.output,
            args.team_name,
        )
        evaluator.evaluate()


if __name__ == "__main__":
    main()
