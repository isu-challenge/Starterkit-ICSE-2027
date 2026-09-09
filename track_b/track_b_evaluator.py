"""Evaluate a Track B in-car scene understanding system.

Evaluation workflow:

* 📂 Load matching scene images and label JSON files.
* 🧾 Build the evaluated feature vocabulary and strict JSON prompt.
* 🤖 Run Qwen2.5-VL on each RGB image.
* ✅ Compare each prediction with the applicable ground-truth features.
* 📊 Aggregate feature accuracy, scene exact-match accuracy, and latency.
* 💾 Save per-image predictions and aggregate metrics as JSON.

The evaluator accepts either a flat data directory or the starter-kit layout
with ``images/`` and ``labels/`` subdirectories. The participant system is
expected to receive only the RGB image; labels are used by the evaluator for
scoring and must not be passed to the model during inference.

Example:
	python track_b_evaluator.py --data-dir ../data
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_IGNORED_FEATURES = {
	"participant_driver", "car", "height_m", "weight_kg", "env_strength",
	"light_front", "light_back_left", "light_back_right",
	"exposure_compensation",
}
IGNORED_FEATURE_NAME_PARTS = ("env", "light")


def normalize(value):
	if value is None:
		return None
	return str(value).strip().upper()


def comparable_features(label, ignored_features):
	return {
		key: value for key, value in label.items()
		if key not in ignored_features
		and not any(part in key.lower() for part in IGNORED_FEATURE_NAME_PARTS)
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
		"You are a digital forensics expert in analysing interior scenes in cars "
		"for different scenes with more than 20 years experience. "
		"Your task is to answer the following questions about the scenes provided "
		"as images and return the result in strict JSON format. "
		"You can detect occluded objects, especially if they are in the back of "
		"the interior, or behind other objects or if they look similar to the car "
		"interior.\n\n"
		"Rules:\n"
		"- Return only the JSON.\n"
		"- Do not provide explanations.\n"
		"- For each question, choose exactly one answer from the given "
		"answer_options.\n"
		"- Do not create new answers, only choose from the provided options.\n"
		"- If a question references something not present, select the provided "
		"None.\n\n"
		f"Questions and answer_options: {schema}"
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
	if not isinstance(result, dict):
		return None
	return {
		key: value[0] if isinstance(value, list) and len(value) == 1 else value
		for key, value in result.items()
	}


def query_model(model, processor, image, prompt, device):
	messages = [{
		"role": "system",
		"content": [{"type": "text", "text": prompt}],
	}, {
		"role": "user",
		"content": [{"type": "image", "image": image}],
	}]
	text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
	inputs = {key: value.to(device) if hasattr(value, "to") else value
			  for key, value in inputs.items()}
	with torch.inference_mode():
		generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
	print("Generated tokens:", generated)
	generated = generated[:, inputs["input_ids"].shape[1]:]
	return processor.batch_decode(generated, skip_special_tokens=True)[0]


def feature_metrics(rows):
	matches = [match for row in rows for match in row["feature_matches"].values()]
	accuracy = sum(matches) / len(matches) if matches else None
	vectors = [row["expected"] for row in rows]
	unique_values = {
		key: len({normalize(vector[key]) for vector in vectors if key in vector})
		for key in sorted(set().union(*(vector.keys() for vector in vectors)))
	}
	return {
		"accuracy": float(accuracy) if accuracy is not None else None,
		"correct_features": sum(matches),
		"evaluated_features": len(matches),
		"unique_values": unique_values,
	}


def evaluate(args):
	"""Run the complete Track B evaluation pipeline and save its JSON report."""
	print("📂 Loading Track B scenes and labels...", flush=True)
	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	scenes = load_dataset(args.data_dir, args.image_kind)
	ignored_features = DEFAULT_IGNORED_FEATURES | set(args.ignore_feature)
	print(f"   Found {len(scenes)} scene-image pairs ({args.image_kind}).", flush=True)

	print("🧾 Building the evaluated feature vocabulary and prompt...", flush=True)
	vocabulary = build_vocabulary(scenes, ignored_features)
	prompt = build_prompt(vocabulary)

	print(f"🤖 Loading model {args.model} on {device}...", flush=True)
	processor = AutoProcessor.from_pretrained(args.model)
	model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
		args.model, torch_dtype="auto", device_map=device
	).eval()
	print("🔎 Running image-only predictions...", flush=True)
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

	print("📊 Aggregating feature, scene-level, and latency metrics...", flush=True)
	output = {
		"team_name": args.team_name,
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
	print(f"💾 Saving evaluation report to {output_path}...", flush=True)
	output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True))
	print(json.dumps(output["metrics"], indent=2), flush=True)
	print(f"Saved report to {output_path}", flush=True)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", default="data", help="Directory containing images and label JSON files.")
	parser.add_argument("--image-kind", choices=("real", "synthetic", "both"), default="both")
	parser.add_argument("--output", default="track_b_evaluation_output/report.json")
	parser.add_argument("--team-name", default="unnamed_team",
					help="Team name stored in the evaluation report.")
	parser.add_argument("--model", default=MODEL_NAME)
	parser.add_argument("--device", help="Torch device, for example cuda, mps, or cpu.")
	parser.add_argument("--ignore-feature", action="append", default=[],
						help="Additional label key to exclude from scoring.")
	return parser.parse_args()


if __name__ == "__main__":
	evaluate(parse_args())
