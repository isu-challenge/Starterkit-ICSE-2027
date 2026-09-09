"""Example Track A test generator implementation using the evaluator CLI.

This example shows how to:
1. Implement a custom Translator subclass for scene transformation
2. Integrate with the evaluator CLI
3. Generate test scenes while preserving semantic structure

Participants can use this as a template for their own test generators.
"""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from evaluator_cli import Translator
from image_metrics_utils import evaluate_validity


class AppearanceTranslator(Translator):
    """Example translator that modifies visual appearance.

    This baseline modifies lighting, colors, and textures while
    preserving object geometry and semantic structure.

    Key constraint: Must NOT modify scene structure that ground-truth
    labels describe (object positions, occupancy, semantic properties).
    """

    def __init__(self, intensity: float = 0.5):
        """Initialize translator.

        Args:
            intensity: Scale for appearance modifications (0.0-1.0)
        """
        self.intensity = intensity

    def translate(self, image: Image.Image, auxiliary_data: Optional[dict] = None) -> Image.Image:
        """Transform image appearance while preserving scene structure.

        Args:
            image: PIL Image (RGB) of the source scene
            auxiliary_data: dict with optional segmentation, depth maps, etc.
                           Must not be used to add/remove/move objects

        Returns:
            Transformed image with same scene structure
        """
        # Start with a copy
        result = image.copy()

        # Example 1: Modify lighting (brightness/contrast)
        result = self._adjust_lighting(result)

        # Example 2: Modify colors (hue/saturation)
        result = self._adjust_colors(result)

        # Example 3: Add texture/noise (simulates weather, aging, wear)
        result = self._add_texture(result)

        # Example 4: Adjust exposure (simulates camera settings)
        result = self._adjust_exposure(result)

        return result

    def _adjust_lighting(self, image: Image.Image) -> Image.Image:
        """Adjust brightness and contrast (simulates lighting conditions)."""
        # Brightness: [-0.3, 0.3] range based on intensity
        brightness_factor = 1.0 + (np.random.uniform(-0.3, 0.3) * self.intensity)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Contrast: [0.7, 1.3] range
        contrast_factor = 0.7 + (0.6 * (1 + np.random.uniform(-1, 1) * self.intensity))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        return image

    def _adjust_colors(self, image: Image.Image) -> Image.Image:
        """Adjust hue and saturation (simulates color variations)."""
        # Hue: simulate different lighting temperatures
        hue_factor = 1.0 + (np.random.uniform(-0.1, 0.1) * self.intensity)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(hue_factor)

        return image

    def _add_texture(self, image: Image.Image) -> Image.Image:
        """Add subtle texture/noise (simulates surface properties, artifacts)."""
        if self.intensity < 0.1:
            return image

        # Convert to numpy for processing
        img_array = np.array(image, dtype=np.float32)

        # Add Gaussian noise proportional to intensity
        noise = np.random.normal(0, 5 * self.intensity, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)

        # Optionally add texture via filtering
        if self.intensity > 0.3:
            # Simulate weathering/wear with slight blur variations
            img_array = img_array * 0.95 + np.roll(img_array, 1, axis=0) * 0.05

        return Image.fromarray(np.uint8(img_array))

    def _adjust_exposure(self, image: Image.Image) -> Image.Image:
        """Adjust exposure compensation (camera settings)."""
        # Exposure: [-0.5, 0.5] EV range
        exposure_factor = 1.0 + (np.random.uniform(-0.5, 0.5) * self.intensity)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(exposure_factor)

        return image


class SaturationTranslator(Translator):
    """Minimal translator that only rescales color saturation.

    Useful as a lightweight stand-in for AppearanceTranslator in tests: it
    trivially preserves scene structure (no geometry/texture changes) while
    still producing a visibly different "generated" image.
    """

    def __init__(self, saturation: float = 1.5):
        self.saturation = saturation

    def translate(self, image: Image.Image, auxiliary_data: Optional[dict] = None) -> Image.Image:
        return ImageEnhance.Color(image.copy()).enhance(self.saturation)


class DiffusionTranslator(Translator):
    """Example translator using diffusion models for appearance transfer.

    This is a more advanced example showing integration with generative models.
    The key constraint is that transformations must be appearance-only and
    must preserve the segmentation and spatial structure.

    Note: This is a placeholder. Real implementation would require:
    - Diffusion model setup (e.g., Stable Diffusion, SDXL)
    - Segmentation-guided inpainting
    - Prompt engineering for specific appearance changes
    - Careful validation that structure is preserved
    """

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2"):
        self.model_id = model_id
        self.pipeline = None
        # Would load diffusion pipeline here
        # from diffusers import StableDiffusionImg2ImgPipeline
        # self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(...)

    def translate(self, image: Image.Image, auxiliary_data: Optional[dict] = None) -> Image.Image:
        """Transform using diffusion model with structure preservation.

        Args:
            image: Source scene image
            auxiliary_data: Must include segmentation mask for structure preservation

        Returns:
            Transformed image
        """
        if auxiliary_data is None or "segmentation" not in auxiliary_data:
            raise ValueError("Segmentation required for structure-preserving translation")

        segmentation = auxiliary_data["segmentation"]

        # Prompt for appearance modification (NOT structural change)
        prompt = self._generate_prompt()

        # Use segmentation mask to ensure objects maintain their boundaries
        # This is a conceptual example; actual implementation is complex
        transformed = self._diffusion_transform(image, prompt, segmentation)

        # Critical: Validate that segmentation is preserved
        if not self._validate_structure(image, transformed, segmentation):
            raise RuntimeError("Structure validation failed; transformation not valid for Track A")

        return transformed

    def _generate_prompt(self) -> str:
        """Generate appearance modification prompt."""
        # Examples: "in heavy rain", "at sunset", "in snow", "with camera glare"
        conditions = [
            "in natural daylight",
            "in evening light",
            "in overcast conditions",
            "with soft shadows",
        ]
        return np.random.choice(conditions)

    def _diffusion_transform(self, image: Image.Image, prompt: str, mask) -> Image.Image:
        """Apply diffusion model with mask guidance."""
        # Placeholder: actual implementation would use pipeline
        # output = self.pipeline(
        #     prompt=prompt,
        #     image=image,
        #     mask_image=mask,
        #     strength=0.7,
        #     guidance_scale=7.5,
        # ).images[0]
        return image

    def _validate_structure(self, original: Image.Image, transformed: Image.Image, seg_mask) -> bool:
        """Validate that scene structure is preserved.

        This is a critical validation step for Track A.
        Returns False if structural changes detected.
        """
        return BoundingBoxValidator.validate_with_sam(original, transformed, self._segmenter())

    def _segmenter(self):
        """Lazily create and cache the SAM segmenter used for validation."""
        if getattr(self, "_sam", None) is None:
            from image_metrics_utils import TransformersSamSegmenter

            self._sam = TransformersSamSegmenter()
        return self._sam


class BoundingBoxValidator:
    """Utility to validate that test generators preserve object locations."""

    @staticmethod
    def validate_positions(
        original_segmentation: np.ndarray,
        transformed_segmentation: np.ndarray,
        max_drift: float = 5.0,  # pixels
    ) -> bool:
        """Check that object centroids haven't moved significantly.

        Args:
            original_segmentation: Segmentation mask of original scene
            transformed_segmentation: Segmentation mask after transformation
            max_drift: Maximum allowed pixel movement for object center

        Returns:
            True if all objects within allowed drift, False otherwise
        """
        from scipy import ndimage

        # Get unique object IDs
        original_ids = set(np.unique(original_segmentation)) - {0}
        transformed_ids = set(np.unique(transformed_segmentation)) - {0}

        if original_ids != transformed_ids:
            return False  # Objects added/removed

        # Check centroid movement
        for obj_id in original_ids:
            orig_mask = original_segmentation == obj_id
            trans_mask = transformed_segmentation == obj_id

            orig_center = np.array(ndimage.center_of_mass(orig_mask))
            trans_center = np.array(ndimage.center_of_mass(trans_mask))

            drift = np.linalg.norm(orig_center - trans_center)
            if drift > max_drift:
                return False

        return True

    @staticmethod
    def validate_with_sam(
        original: Image.Image,
        transformed: Image.Image,
        segmenter,
        max_semantic_drift: float = 0.1,
    ) -> bool:
        """Validate structure preservation using SAM-based segmentation validity.

        Reuses the official `evaluate_validity` metric (simulated vs. generated
        segmentation agreement) so translators are checked with the same
        structural-validity criterion as the Track A evaluator.

        Args:
            original: Source scene image before translation.
            transformed: Scene image after translation.
            segmenter: Segmenter instance (e.g. TransformersSamSegmenter).
            max_semantic_drift: Maximum allowed semantic_segmentation_score
                (0 = identical partitions, 1 = fully mismatched).

        Returns:
            True if segmentation agreement is within max_semantic_drift.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_dir, transformed_dir = root / "original", root / "transformed"
            original_dir.mkdir()
            transformed_dir.mkdir()
            original.convert("RGB").save(original_dir / "scene.png")
            transformed.convert("RGB").save(transformed_dir / "scene.png")

            result = evaluate_validity(
                original_dir, transformed_dir, segmenter,
                first_name="original", second_name="transformed",
            )
        score = result["summary"]["semantic_segmentation_score"]
        return score <= max_semantic_drift


def example_generate_scenes():
    """Example: Generate test scenes using a custom translator."""
    translator = AppearanceTranslator(intensity=0.8)

    # Load source scene
    source_image = Image.open("data/images/sample_0001_sim.png")

    # Generate multiple transformed versions
    for i in range(5):
        transformed = translator.translate(source_image)
        output_path = Path("submissions/test_generator/generated") / f"sample_0001_variant_{i}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transformed.save(output_path)
        print(f"Generated {output_path}")


def example_validate_generator():
    """Example: Validate that a generator preserves structure."""
    import json
    from pathlib import Path

    translator = AppearanceTranslator(intensity=0.5)

    # Load source and reference segmentation
    source = Image.open("data/images/sample_0001_sim.png")
    with open("data/sample_0001_instance_seg.json") as f:
        orig_seg = np.array(json.load(f))

    # Generate transformed scene
    auxiliary = {"segmentation": orig_seg}
    transformed = translator.translate(source, auxiliary)

    # Validate structure (would need transformed segmentation)
    # In practice, you'd generate new segmentation and validate
    print("✅ Validation complete (placeholder)")


if __name__ == "__main__":
    # Run example generation
    example_generate_scenes()

    # Run example validation
    example_validate_generator()
