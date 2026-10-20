"""Template Track A generator.

Runs your Translator on every simulated scene of a dataset folder and writes the files of
an official submission:
    <output>/generated/<stem>.png   one image per images/<stem>_sim.png
    <output>/generation_times.json  total translation time and hardware

The dataset folder has the starter-kit layout (data/ at the repository root): images/
with <stem>_sim.png, labels/, and optionally canny/, instance_seg/, and depth/png/.

Usage, from the repository root, with the template copied to private/track_a:
    python private/track_a/code/generate.py --data-dir data \
        --output private/track_a --hardware "1x NVIDIA RTX 4090"

Then encrypt the folder for the pull request:
    python tools/encrypt_submission.py --source private/track_a --output submissions/<team_id>/track_a
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image


def find_track_a_dir() -> Path:
    """The starter kit's track_a/ folder, which provides the Translator interface."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "track_a" / "evaluator_cli.py").is_file():
            return parent / "track_a"
    raise SystemExit("Run this script inside a clone of the ISU-Challenge starter kit.")


TRACK_A_DIR = find_track_a_dir()
sys.path[:0] = [str(TRACK_A_DIR), str(TRACK_A_DIR / "test")]

from evaluator_cli import Translator  # noqa: E402


class MyTranslator(Translator):
    def __init__(self) -> None:
        # TODO: load your model here, once. Loading is not counted as generation time.
        pass

    def translate(self, image: Image.Image, auxiliary_data: Optional[dict] = None) -> Image.Image:
        """Return a realistic version of one simulated RGB scene.

        auxiliary_data holds the scene "stem", its "label" (dict), and, when present in the
        dataset folder, the "canny", "segmentation", and "depth" images of the same view.

        TODO: replace with your generator. Keep the width, height, framing, and viewpoint,
        and do not add, remove, move, or reshape annotated objects or people; the labels
        of the source scene must stay valid. This placeholder returns the image unchanged.
        """
        return image.copy()


def create_translator() -> Translator:
    # To try the shipped example first (track_a/test/example_custom_translator.py):
    #     from example_custom_translator import AppearanceTranslator
    #     return AppearanceTranslator(intensity=0.5)
    return MyTranslator()


def first_file(*paths: Path) -> Optional[Path]:
    return next((path for path in paths if path.is_file()), None)


def load_auxiliary(data_dir: Path, stem: str) -> dict:
    auxiliary: dict = {"stem": stem}
    label = first_file(data_dir / "labels" / f"{stem}.json", data_dir / "labels" / f"{stem}_label.json")
    if label:
        auxiliary["label"] = json.loads(label.read_text(encoding="utf-8"))
    channels = {
        "canny": (data_dir / "canny" / f"{stem}_canny.png",),
        "segmentation": (data_dir / "instance_seg" / f"{stem}_instance_seg.png",),
        "depth": (data_dir / "depth" / "png" / f"{stem}_sim_depth.png",
                  data_dir / "depth" / "png" / f"{stem}_depth.png"),
    }
    for key, candidates in channels.items():
        path = first_file(*candidates)
        if path:
            with Image.open(path) as channel:
                auxiliary[key] = channel.copy()
    return auxiliary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset folder with images/<stem>_sim.png.")
    parser.add_argument("--output", type=Path, required=True, help="Your submissions/<team_id>/track_a folder.")
    parser.add_argument("--hardware", default="", help="Short description of the hardware used.")
    parser.add_argument("--limit", type=int, help="Translate only the first N scenes, for a quick test.")
    args = parser.parse_args()

    images = args.data_dir / "images" if (args.data_dir / "images").is_dir() else args.data_dir
    sources = sorted(images.glob("*_sim.png"))[: args.limit]
    if not sources:
        raise SystemExit(f"No *_sim.png images found in {images}")
    generated = args.output / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    translator = create_translator()
    total = 0.0
    for index, path in enumerate(sources, 1):
        stem = path.name[: -len("_sim.png")]
        with Image.open(path) as image:
            source = image.convert("RGB")
        auxiliary = load_auxiliary(args.data_dir, stem)
        started = time.perf_counter()
        result = translator.translate(source, auxiliary).convert("RGB")
        total += time.perf_counter() - started
        if result.size != source.size:
            raise SystemExit(f"{path.name}: translate() returned {result.size}, expected {source.size}")
        result.save(generated / f"{stem}.png")
        print(f"[{index}/{len(sources)}] {stem}", flush=True)

    (args.output / "generation_times.json").write_text(
        json.dumps({"total_seconds": round(total, 3), "scenes": len(sources), "hardware": args.hardware},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Translated {len(sources)} scene(s) in {total:.1f} s")


if __name__ == "__main__":
    main()
