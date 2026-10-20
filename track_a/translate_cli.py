"""Run a Track A translator on simulated scenes, as the official evaluation does.

The translator folder contains translator.py with create_translator(device), which returns a
Translator (see evaluator_cli.py). For every simulated image, translate(image, auxiliary_data)
is called; auxiliary_data holds the scene "stem", its "label", and, when available, its
"canny", "segmentation", and "depth" images. The results are written to
<output>/generated/<stem>.png and the translation time to <output>/generation_times.json.

Example, from the repository root:
    python track_a/translate_cli.py --translator-dir track_a/submissions/<team_name> \
        --data-dir data --output track_a/submissions/<team_name> --limit 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator_cli import Translator  # noqa: E402
from image_metrics_utils import files_by_stem  # noqa: E402


def load_translator(folder: Path, device: str) -> Translator:
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("participant_translator", folder / "translator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["participant_translator"] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "create_translator", None)):
        raise SystemExit("translator.py must define create_translator(device).")
    return module.create_translator(device)


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
        "depth": (data_dir / "depth" / "png" / f"{stem}_sim_depth.png", data_dir / "depth" / "png" / f"{stem}_depth.png"),
    }
    for key, candidates in channels.items():
        path = first_file(*candidates)
        if path:
            with Image.open(path) as channel:
                auxiliary[key] = channel.copy()
    return auxiliary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--translator-dir", type=Path, required=True, help="Folder with translator.py.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with simulated/ or images/ (<stem>_sim.png), labels/, and optional channels.")
    parser.add_argument("--output", type=Path, required=True, help="Receives generated/ and generation_times.json.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="Translate only the first N scenes, for a quick test.")
    args = parser.parse_args()

    simulated = next((path for path in (args.data_dir / "simulated", args.data_dir / "images") if path.is_dir()),
                     args.data_dir)
    scenes = sorted(files_by_stem(simulated, normalize_simulated=True).items())[: args.limit]
    if not scenes:
        raise SystemExit(f"No simulated images found in {simulated}")
    generated = args.output / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    translator = load_translator(args.translator_dir.resolve(), args.device)
    total, errors = 0.0, 0
    for index, (stem, path) in enumerate(scenes, 1):
        with Image.open(path) as image:
            source = image.convert("RGB")
        auxiliary = load_auxiliary(args.data_dir, stem)
        started = time.perf_counter()
        try:
            result = translator.translate(source, auxiliary)
            total += time.perf_counter() - started
            if not isinstance(result, Image.Image):
                raise TypeError(f"translate() returned {type(result).__name__}, expected a PIL image")
            result.convert("RGB").save(generated / f"{stem}.png")
        except Exception:  # noqa: BLE001 - report every failing scene, then fail
            errors += 1
            traceback.print_exc()
        print(f"[{index}/{len(scenes)}] {stem}", flush=True)

    (args.output / "generation_times.json").write_text(json.dumps(
        {"total_seconds": round(total, 3), "scenes": len(scenes), "hardware": platform.platform()}, indent=2) + "\n",
        encoding="utf-8")
    if errors:
        raise SystemExit(f"translate() failed for {errors} of {len(scenes)} scene(s).")
    print(f"Translated {len(scenes)} scene(s) in {total:.1f} s")


if __name__ == "__main__":
    main()
