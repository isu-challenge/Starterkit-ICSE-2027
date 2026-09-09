"""Track A participant evaluation CLI for test generation.

Usage:
    python evaluator_cli.py evaluate --submission-path ../submissions/your_team
    python evaluator_cli.py evaluate --submission-path submissions/my_generator --failure-model qwen
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from image_metrics_utils import (
    MoondreamAnalyzer,
    Qwen25VLAnalyzer,
    TransformersSamSegmenter,
    save_report,
)
from track_a_metrics import (
    build_final_report,
    clip_visual_diversity,
    discover_contestants,
    efficiency,
    evaluate_failure_rate,
    evaluate_public_realism,
    evaluate_sam,
    load_labels,
)


class DataLoader(ABC):
    """Abstract base class for loading submission data."""

    @abstractmethod
    def load(self, submission_path: Path) -> dict[str, Path]:
        """Load submission data and return paths to data directories."""
        pass

    @abstractmethod
    def validate(self, data: dict[str, Path]) -> bool:
        """Validate that required data directories exist."""
        pass


class TrackADataLoader(DataLoader):
    """Load Track A submission data with structured folders."""

    def load(self, submission_path: Path) -> dict[str, Path]:
        """Return paths to required Track A folders."""
        return {
            "simulated": submission_path / "simulated",
            "generated": submission_path / "generated",
            "reference": submission_path / "reference",
            "labels": submission_path / "labels",
        }

    def validate(self, data: dict[str, Path]) -> bool:
        """Check that all required folders exist."""
        return all(path.is_dir() for path in data.values())


class Translator(ABC):
    """Abstract base class for scene translation/transformation.

    Participants implement custom translation logic in subclasses.
    """

    @abstractmethod
    def translate(self, image: Image.Image, auxiliary_data: Optional[dict] = None) -> Image.Image:
        """Transform an image while preserving semantic structure."""
        pass


class EvaluatorTrackA:
    """Static evaluation methods for Track A submissions."""

    @staticmethod
    def evaluate_validity(folders: dict[str, Path], output_dir: Path) -> dict[str, Any]:
        """Run SAM-based structural validity evaluation."""
        print("🧩 Evaluating structural validity with SAM...")
        sam = TransformersSamSegmenter("facebook/sam-vit-base")
        result = evaluate_sam(folders, output_dir, sam)
        del sam
        gc.collect()
        return result

    @staticmethod
    def evaluate_realism(folders: dict[str, Path]) -> dict[str, Any]:
        """Run public image-quality metrics."""
        print("📊 Evaluating realism with public metrics...")
        return evaluate_public_realism(folders)

    @staticmethod
    def evaluate_failures(
        folders: dict[str, Path],
        failure_model: str = "moondream",
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Run failure detection evaluation."""
        print(f"🔍 Evaluating failure rate with {failure_model}...")
        if failure_model == "moondream":
            analyzer = MoondreamAnalyzer(revision="2025-06-21", device=device, seed=0)
        elif failure_model == "qwen":
            analyzer = Qwen25VLAnalyzer(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
        else:
            raise ValueError(f"Unknown failure model: {failure_model}")

        result = evaluate_failure_rate(folders, analyzer)
        del analyzer
        gc.collect()
        return result

    @staticmethod
    def evaluate_efficiency(
        submission_path: Path,
        executed_scenes: int,
    ) -> dict[str, Any]:
        """Evaluate generation efficiency from timestamps."""
        print("⏱️ Evaluating efficiency...")
        return efficiency(submission_path / "generated", executed_scenes)

    @staticmethod
    def build_report(
        evaluation_results: dict[str, Any],
        labels: dict[str, dict],
    ) -> dict[str, Any]:
        """Assemble final report from evaluation sections."""
        return build_final_report(evaluation_results, labels)


class ReportWriter:
    """Write evaluation reports to JSON files."""

    @staticmethod
    def write(report: dict[str, Any], output_path: Path) -> None:
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
        print(f"💾 Saved report to {output_path}")

    @staticmethod
    def write_summary(reports: dict[str, dict], output_dir: Path) -> None:
        """Write a summary of all contestant reports."""
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        summary = {}
        for name, report in reports.items():
            final = report.get("final")
            if final is None:
                continue
            failure = final["failure"]
            summary[name] = {
                "failure_rate": failure.get("failure_rate"),
                "failures": failure.get("failures"),
                "executed": failure.get("executed"),
                "ssim_generated": final["realism"]["generated_vs_real"]["summary"].get("ssim"),
                "ssim_simulated": final["realism"]["simulated_vs_real"]["summary"].get("ssim"),
                "feature_diversity": final["feature_diversity"]["mean_pairwise_hamming"],
                "seconds_per_scene": report.get("efficiency", {}).get("seconds_per_scene"),
            }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
        print(f"📄 Saved summary to {summary_path}")


class ParticipantEvaluator:
    """Evaluate a single Track A submission - participant facing."""

    def __init__(
        self,
        submission_path: Path,
        output_path: Path,
        failure_model: str = "moondream",
        device: str = "cpu",
    ):
        self.submission_path = Path(submission_path)
        self.output_path = Path(output_path)
        self.failure_model = failure_model
        self.device = device
        self.data_loader = TrackADataLoader()

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the submission."""
        # Load and validate submission
        folders = self.data_loader.load(self.submission_path)
        if not self.data_loader.validate(folders):
            raise ValueError(f"Invalid submission structure at {self.submission_path}")

        name = self.submission_path.name
        print(f"\n🧪 Evaluating Track A submission: {name}")
        print(f"{'='*60}")

        report = {"contestant": name}
        output_subdir = self.output_path.parent / name

        # Structural validity
        print("🧩 Evaluating structural validity with SAM...")
        report["sam"] = EvaluatorTrackA.evaluate_validity(folders, output_subdir / "sam")
        ReportWriter.write(report, self.output_path)

        # Realism
        print("📊 Evaluating realism with public metrics...")
        report["realism_raw"] = EvaluatorTrackA.evaluate_realism(folders)
        ReportWriter.write(report, self.output_path)

        # Failure detection
        report["failure_raw"] = EvaluatorTrackA.evaluate_failures(
            folders, self.failure_model, self.device
        )
        failure_rate = report["failure_raw"].get("failure_rate")
        if failure_rate is not None:
            print(f"  Failure rate: {failure_rate:.3f} ({report['failure_raw']['failures']}/{report['failure_raw']['executed']})")
        ReportWriter.write(report, self.output_path)

        # Diversity
        print("📋 Calculating feature diversity...")
        labels = load_labels(folders["labels"])
        report["final"] = EvaluatorTrackA.build_report(report, labels)
        failing_stems = [
            row["image"]
            for row in report["final"]["failure"]["images"]
            if row["failure"]
        ]
        report["final"]["clip_visual_diversity"] = clip_visual_diversity(
            folders["generated"], failing_stems
        )
        ReportWriter.write(report, self.output_path)

        # Efficiency
        print("⏱️ Evaluating efficiency...")
        report["efficiency"] = EvaluatorTrackA.evaluate_efficiency(
            folders["generated"].parent,
            report["final"]["failure"]["executed"],
        )
        timing = report["efficiency"].get("seconds_per_scene")
        if timing is not None:
            print(f"  Generation time: {timing:.3f} s/scene")
        ReportWriter.write(report, self.output_path)

        print(f"✅ Evaluation complete: {self.output_path}")
        return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Evaluation command")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate your submission")
    eval_parser.add_argument(
        "--submission-path",
        required=True,
        help="Path to your submission folder (contains simulated/, generated/, reference/, labels/)",
    )
    eval_parser.add_argument(
        "--output",
        default="track_a_evaluation_output/report.json",
        help="Output report path (default: track_a_evaluation_output/report.json)",
    )
    eval_parser.add_argument(
        "--failure-model",
        choices=["moondream", "qwen"],
        default="moondream",
        help="VLM to use for failure detection",
    )
    eval_parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu, cuda, mps)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "evaluate":
        evaluator = ParticipantEvaluator(
            args.submission_path,
            args.output,
            args.failure_model,
            args.device,
        )
        evaluator.evaluate()


if __name__ == "__main__":
    main()
