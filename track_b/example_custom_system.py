"""Example Track B ISU system implementation using the evaluator CLI.

This example shows how to:
1. Implement a custom ISUSystem subclass
2. Integrate with the evaluator CLI
3. Run evaluation on your system

Participants can use this as a template for their own implementations.
"""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from evaluator_cli import EvaluatorTrackB, ISUSystem


class CLIPBasedISUSystem(ISUSystem):
    """Example ISU system based on CLIP embeddings.

    This is a simple baseline that uses CLIP image embeddings to classify
    scene features through similarity matching.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.prompt = None
        self.feature_templates = {}
        self._load_model()

    def _load_model(self):
        """Load CLIP model."""
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def set_prompt(self, prompt: str) -> None:
        """Set evaluation prompt and extract feature templates."""
        self.prompt = prompt
        # Parse feature vocabulary from prompt for use in classification
        self._build_feature_templates()

    def _build_feature_templates(self) -> None:
        """Build text templates for each feature's possible values."""
        # This is a simplified example; actual implementation would parse prompt
        self.feature_templates = {
            "driver_phone": ["driver using phone", "no phone in use"],
            "baby_seat": ["baby seat present", "no baby seat"],
            "driver_safety_belt": ["driver wearing seatbelt", "seatbelt unfastened"],
            "suitcase": ["suitcase in vehicle", "no suitcase"],
        }

    def predict(self, image: Image.Image) -> dict[str, str]:
        """Predict features using CLIP similarity.

        This is a baseline approach. Production systems would use:
        - Fine-tuned vision-language models
        - Multi-stage classification pipelines
        - Ensemble methods
        - Scene context modeling
        """
        predictions = {}

        # Encode image
        with self.processor.image_processor.image_mean:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)

        # Classify each feature
        for feature_name, templates in self.feature_templates.items():
            if not templates:
                continue

            # Encode text descriptions
            text_inputs = self.processor(text=templates, padding=True, return_tensors="pt").to(self.device)
            text_features = self.model.get_text_features(**text_inputs)

            # Compute similarity
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            best_idx = similarities[0].argmax().item()
            predictions[feature_name] = templates[best_idx].split()[-1].upper()

        return predictions

    def name(self) -> str:
        """Return system name."""
        return "CLIPBasedISU-v1"


class EnsembleISUSystem(ISUSystem):
    """Example ensemble ISU system combining multiple models.

    This shows how to combine predictions from different models
    for improved robustness.
    """

    def __init__(self, models: list[ISUSystem]):
        self.models = models
        self.prompt = None

    def set_prompt(self, prompt: str) -> None:
        """Set prompt for all models."""
        self.prompt = prompt
        for model in self.models:
            model.set_prompt(prompt)

    def predict(self, image: Image.Image) -> dict[str, str]:
        """Get predictions from all models and ensemble them."""
        all_predictions = []
        for model in self.models:
            pred = model.predict(image)
            all_predictions.append(pred)

        # Ensemble by voting
        ensemble_pred = {}
        feature_names = set()
        for pred in all_predictions:
            feature_names.update(pred.keys())

        for feature in feature_names:
            values = [pred.get(feature, "UNKNOWN") for pred in all_predictions]
            # Simple majority voting
            most_common = max(set(values), key=values.count)
            ensemble_pred[feature] = most_common

        return ensemble_pred

    def name(self) -> str:
        """Return system name."""
        model_names = ", ".join(m.name() for m in self.models)
        return f"Ensemble[{model_names}]"


def example_evaluate_custom_system():
    """Example: Evaluate a custom ISU system."""
    # Create your system
    system = CLIPBasedISUSystem(device="cuda")

    # Create evaluator
    evaluator = EvaluatorTrackB(
        data_dir=Path("data"),
        isu_system=system,
        image_kind="synthetic",  # Test on synthetic first
        output_path=Path("track_b_evaluation_output") / "custom_system.json",
    )

    # Run evaluation
    report = evaluator.evaluate()

    # Access metrics
    metrics = report["metrics"]
    print(f"\n✅ Evaluation complete!")
    print(f"   Accuracy: {metrics['accuracy']:.1%}")
    print(f"   Scene exact-match: {metrics['scene_exact_match']:.1%}")
    print(f"   Mean latency: {metrics['latency_seconds_mean']:.3f} s")


def example_ensemble_evaluation():
    """Example: Evaluate an ensemble system."""
    from evaluator_cli import QwenISUSystem

    # Create base systems
    qwen = QwenISUSystem(device="cuda")
    # clip_system = CLIPBasedISUSystem(device="cuda")

    # Create ensemble
    ensemble = EnsembleISUSystem([qwen])  # Add more systems here

    # Evaluate ensemble
    evaluator = EvaluatorTrackB(
        data_dir=Path("data"),
        isu_system=ensemble,
        output_path=Path("track_b_evaluation_output") / "ensemble.json",
    )

    report = evaluator.evaluate()
    return report


if __name__ == "__main__":
    # Run example evaluation
    example_evaluate_custom_system()

    # Uncomment to test ensemble
    # example_ensemble_evaluation()
