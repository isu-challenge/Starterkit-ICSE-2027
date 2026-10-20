"""Template Track B system; see "Official Submission" in track_b/README.md.

Examples: QwenISUSystem in track_b/evaluator_cli.py and track_b/example_custom_system.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from evaluator_cli import ISUSystem
from track_b_evaluator import TRACK_B_FEATURES

HERE = Path(__file__).resolve().parent


class MyISUSystem(ISUSystem):
    def __init__(self, device: str):
        self.device = device
        self.prompt: str | None = None
        # TODO: load your model once, without network access. For example, a model listed in models.txt:
        #     from transformers import AutoModelForImageTextToText, AutoProcessor
        #     self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        #     self.model = AutoModelForImageTextToText.from_pretrained(
        #         "Qwen/Qwen2.5-VL-3B-Instruct", dtype="auto", device_map=device).eval()
        # and your own files relative to this folder:
        #     self.head = torch.load(HERE / "weights" / "head.pt", map_location=device)

    def set_prompt(self, prompt: str) -> None:
        """Questions and allowed answers of the evaluation set; using them is optional."""
        self.prompt = prompt

    def name(self) -> str:
        return "my-isu-system"  # TODO: your system name

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Return one answer per feature for an RGB image, e.g. {"driver_phone": "NO", ...}."""
        # TODO: replace with your model's inference. This placeholder answers nothing.
        return {feature: None for feature in TRACK_B_FEATURES}


def create_system(device: str) -> MyISUSystem:
    return MyISUSystem(device)
