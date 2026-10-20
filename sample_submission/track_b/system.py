"""Template Track B system.

The evaluation imports this file and calls create_system(device) once, then
set_prompt(prompt) once, then predict(image) for every evaluation image.

The simplest valid submission is the reference baseline; replace this whole file with:

    from evaluator_cli import QwenISUSystem

    def create_system(device):
        return QwenISUSystem(device=device)

and list Qwen/Qwen2.5-VL-3B-Instruct in models.txt. More examples: QwenISUSystem in
track_b/evaluator_cli.py and the CLIP and ensemble systems in track_b/example_custom_system.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

# The starter-kit track_b/ folder is importable during the evaluation.
from evaluator_cli import ISUSystem
from track_b_evaluator import TRACK_B_FEATURES

# Your submission folder; also the working directory during the evaluation (read-only).
HERE = Path(__file__).resolve().parent


class MyISUSystem(ISUSystem):
    def __init__(self, device: str):
        self.device = device
        self.prompt: str | None = None
        # TODO: load your model here, once. There is no network access during the evaluation.
        #
        # A Hugging Face model listed in models.txt is available from the default cache:
        #     from transformers import AutoModelForImageTextToText, AutoProcessor
        #     self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        #     self.model = AutoModelForImageTextToText.from_pretrained(
        #         "Qwen/Qwen2.5-VL-3B-Instruct", dtype="auto", device_map=device).eval()
        #
        # Your own files are loaded relative to this folder (each file below 100 MB):
        #     import torch
        #     self.head = torch.load(HERE / "weights" / "head.pt", map_location=device)

    def set_prompt(self, prompt: str) -> None:
        """Receives the questions and allowed answers of the evaluation set. Using it is optional."""
        self.prompt = prompt

    def name(self) -> str:
        return "my-isu-system"  # TODO: your system name

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Answer every feature for one RGB image, e.g. {"driver_phone": "NO", ...}.

        Answers are compared as text, ignoring case and surrounding spaces. Use the
        allowed answers from the prompt. Features whose label is null are not scored.
        """
        # TODO: replace with your model's inference. This placeholder answers nothing.
        return {feature: None for feature in TRACK_B_FEATURES}


def create_system(device: str) -> MyISUSystem:
    return MyISUSystem(device)
