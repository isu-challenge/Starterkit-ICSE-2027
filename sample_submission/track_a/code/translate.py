"""Your Track A generation code. It is submitted for reproducibility; the evaluation does not run it."""

from typing import Optional

from PIL import Image

from evaluator_cli import Translator  # track_a/evaluator_cli.py


class MyTranslator(Translator):
    def translate(self, image: Image.Image, auxiliary_data: Optional[dict] = None) -> Image.Image:
        # TODO: return a realistic version of the simulated image with the same size, framing,
        # and scene content. Examples: track_a/test/example_custom_translator.py.
        return image.copy()
