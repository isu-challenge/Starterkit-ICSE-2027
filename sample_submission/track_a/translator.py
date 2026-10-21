"""Template Track A translator; see "Official Submission" in track_a/README.md.

Examples: track_a/test/example_custom_translator.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from evaluator_cli import Translator

HERE = Path(__file__).resolve().parent


class MyTranslator(Translator):
    def __init__(self, device: str):
        self.device = device
        # TODO: load your model once, without network access: models listed in models.txt,
        # or your own files relative to this folder, e.g. HERE / "weights" / "model.pt".

    def translate(self, image: Image.Image, auxiliary_data: Optional[dict] = None) -> Image.Image:
        """Return a realistic version of the simulated RGB image with the same size and scene content.

        auxiliary_data holds the scene "stem", its "label", and, when available, its "canny",
        "segmentation", and "depth" images.
        """
        # TODO: replace with your generator. This placeholder returns the image unchanged.
        return image.copy()


def create_translator(device: str) -> MyTranslator:
    return MyTranslator(device)
