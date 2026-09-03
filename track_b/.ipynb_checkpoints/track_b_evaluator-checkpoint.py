"""Evaluate a Track B in-car scene understanding system.

The evaluator reads labels and matching images from either a flat data
directory or the starter-kit layout with ``images/`` and ``labels/``
subdirectories. It queries Qwen2.5-VL-3B-Instruct and writes per-image
predictions and aggregate metrics as JSON.

Example:
	python track_b_evaluator.py --data-dir ../data
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_IGNORED_FEATURES = {
	"participant_driver", "car", "height_m", "weight_kg", "light_front",
	"light_back_left", "light_back_right", "env_strength",
	"exposure_compensation",
}


def normalize(value):
	if value is None:
		return None
	return str(value).strip().upper()


def comparable_features(label, ignored_features):
	return {
		key: value for key, value in label.items()
		if key not in ignored_features
		and value is not None
		and isinstance(value, (str, bool, int, float))
	}


def load_dataset(data_dir, image_kind):
	data_dir = Path(data_dir)
	labels_dir = data_dir / "labels" if (data_dir / "labels").is_dir() else data_dir
	images_dir = data_dir / "images" if (data_dir / "images").is_dir() else data_dir
	scenes = []
	for label_path in sorted(labels_dir.glob("*.json")):
		stem = label_path.stem
		if stem.endswith("_label"):
			stem = stem[: -len("_label")]
		if image_kind in ("real", "both"):
			image_path = images_dir / f"{stem}.jpg"
			if image_path.exists():
				scenes.append((stem, image_path, json.loads(label_path.read_text())))
		if image_kind in ("synthetic", "both"):
			image_path = images_dir / f"{stem}_sim.png"
			if image_path.exists():
				scenes.append((f"{stem}_sim", image_path, json.loads(label_path.read_text())))
	if not scenes:
		raise FileNotFoundError(
			f"No matching images and labels found in {data_dir}. Expected "
			"<stem>.jpg/<stem>.json or <stem>_sim.png/<stem>.json."
		)
	return scenes


def build_vocabulary(scenes, ignored_features):
	vocabulary = {}
	for _, _, label in scenes:
		for key, value in comparable_features(label, ignored_features).items():
			vocabulary.setdefault(key, set()).add(normalize(value))
	return {key: sorted(values) for key, values in sorted(vocabulary.items())}


def build_prompt(vocabulary):
	schema = json.dumps(vocabulary, ensure_ascii=True)
	return (
		"You are an expert security system for in-car scene understanding. "
		"Inspect the image and classify every requested scene feature. Pay "
		"attention to driver and passenger occupancy, seat-belt usage, driver "
		"behaviour, phones, child seats, luggage, and cabin objects. Choose "
		"exactly one allowed value for each feature. Use UNKNOWN only when the "
		"feature cannot be determined from the image. Return strict JSON only, "
		"with no Markdown, explanation, or additional keys. The JSON schema is: "
		f"{schema}"
	)


def parse_json_answer(answer):
	text = str(answer).strip()
	text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
	try:
		result = json.loads(text)
	except json.JSONDecodeError:
		match = re.search(r"\{.*\}", text, flags=re.DOTALL)
		if not match:
			return None
		try:
			result = json.loads(match.group(0))
		except json.JSONDecodeError:
			return None
	return result if isinstance(result, dict) else None


def query_model(model, processor, image, prompt, device):
	messages = [{
		"role": "user",
		"content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
	}]
	text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
	inputs = {key: value.to(device) if hasattr(value, "to") else value
			  for key, value in inputs.items()}
	with torch.inference_mode():
		generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
	generated = generated[:, inputs["input_ids"].shape[1]:]
	return processor.batch_decode(generated, skip_special_tokens=True)[0]


def feature_metrics(rows):
	matches = [match for row in rows for match in row["feature_matches"].values()]
	accuracy = sum(matches) / len(matches) if matches else None
	vectors = [row["expected"] for row in rows]
	distances = []
	for first, second in combinations(vectors, 2):
		keys = set(first) & set(second)
		if keys:
			distances.append(np.mean([
				normalize(first[key]) != normalize(second[key]) for key in keys
			]))
	unique_values = {
		key: len({normalize(vector[key]) for vector in vectors if key in vector})
		for key in sorted(set().union(*(vector.keys() for vector in vectors)))
	}
	return {
		"accuracy": float(accuracy) if accuracy is not None else None,
		"correct_features": sum(matches),
		"evaluated_features": len(matches),
		"feature_diversity": float(np.mean(distances)) if distances else 0.0,
		"unique_values": unique_values,
	}


def evaluate(args):
	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	scenes = load_dataset(args.data_dir, args.image_kind)
	ignored_features = DEFAULT_IGNORED_FEATURES | set(args.ignore_feature)
	vocabulary = build_vocabulary(scenes, ignored_features)
	prompt = build_prompt(vocabulary)

	processor = AutoProcessor.from_pretrained(args.model)
	model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
		args.model, torch_dtype="auto", device_map=device
	).eval()
	rows = []
	for index, (stem, image_path, label) in enumerate(scenes, 1):
		image = Image.open(image_path).convert("RGB")
		started = time.perf_counter()
		raw_answer = query_model(model, processor, image, prompt, device)
		prediction = parse_json_answer(raw_answer) or {}
		expected = comparable_features(label, ignored_features)
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
			"raw_answer": raw_answer,
			"inference_seconds": time.perf_counter() - started,
		})
		print(f"[{index}/{len(scenes)}] {stem}", flush=True)

	output = {
		"track": "B",
		"model": args.model,
		"device": device,
		"image_kind": args.image_kind,
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
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True))
	print(json.dumps(output["metrics"], indent=2), flush=True)
	print(f"Saved report to {output_path}", flush=True)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", default="data", help="Directory containing images and label JSON files.")
	parser.add_argument("--image-kind", choices=("real", "synthetic", "both"), default="both")
	parser.add_argument("--output", default="track_b_evaluation_output/report.json")
	parser.add_argument("--model", default=MODEL_NAME)
	parser.add_argument("--device", help="Torch device, for example cuda, mps, or cpu.")
	parser.add_argument("--ignore-feature", action="append", default=[],
						help="Additional label key to exclude from scoring.")
	return parser.parse_args()


if __name__ == "__main__":
	evaluate(parse_args())
