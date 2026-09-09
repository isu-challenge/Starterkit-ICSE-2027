from __future__ import annotations

import json
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import normalized_mutual_info_score
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from image_metrics_utils import (
    VGGPerceptualMetrics,
    evaluate_realism,
    evaluate_validity,
    files_by_stem,
    list_images,
    paired_files,
    save_report,
)


METRIC_DIRECTIONS = {
    "exact_match": "higher",
    "correlation_coefficient": "higher",
    "histogram_intersection": "higher",
    "lbp_histogram_similarity": "higher",
    "psnr": "higher",
    "ssim": "higher",
    "normalized_mutual_information": "higher",
    "fractal_dimension_similarity": "higher",
    "kl_divergence": "lower",
    "mse": "lower",
    "vgg_perceptual_distance": "lower",
    "glcm_texture_distance": "lower",
    "wasserstein_distance": "lower",
    "fid": "lower",
    "kid_mean": "lower",
    "inception_score_mean": "higher",
}

IGNORED_FEATURES = {
    "participant_driver", "car", "height_m", "weight_kg", "light_front",
    "light_back_left", "light_back_right", "env_strength", "exposure_compensation",
}
IGNORED_FEATURE_NAME_PARTS = (
    "color", "colour", "emotion", "gender", "env", "light"
)


def discover_contestants(root="submissions"):
    result = {}
    for folder in sorted(Path(root).iterdir()):
        paths = {name: folder / name for name in ("simulated", "generated", "reference", "labels")}
        if all(path.is_dir() for path in paths.values()) and list_images(paths["generated"]):
            result[folder.name] = paths
    return result


def _gray(image):
    return np.asarray(image.convert("L"), dtype=np.uint8).copy()


def _histogram(values):
    hist = np.histogram(values, bins=256, range=(0, 256))[0].astype(float)
    return hist / max(hist.sum(), 1)


def _fractal_dimension(gray):
    binary = gray < np.mean(gray)
    sizes = 2 ** np.arange(1, int(np.log2(min(binary.shape))))
    counts = []
    valid_sizes = []
    for size in sizes:
        height = binary.shape[0] // size * size
        width = binary.shape[1] // size * size
        if not height or not width:
            continue
        blocks = binary[:height, :width].reshape(height // size, size, width // size, size)
        counts.append(np.count_nonzero(blocks.any(axis=(1, 3))))
        valid_sizes.append(size)
    valid = [(size, count) for size, count in zip(valid_sizes, counts) if count > 0]
    if len(valid) < 2:
        return 0.0
    x, y = zip(*valid)
    return float(-np.polyfit(np.log(x), np.log(y), 1)[0])


def original_work_metrics(first, second, perceptual):
    second = second.resize(first.size, Image.Resampling.BILINEAR)
    rgb_a = np.asarray(first.convert("RGB"), dtype=np.uint8)
    rgb_b = np.asarray(second.convert("RGB"), dtype=np.uint8)
    gray_a, gray_b = _gray(first), _gray(second)
    hist_a, hist_b = _histogram(rgb_a), _histogram(rgb_b)
    lbp_a = local_binary_pattern(gray_a, 8, 1, method="uniform")
    lbp_b = local_binary_pattern(gray_b, 8, 1, method="uniform")
    lbp_hist_a = np.histogram(lbp_a, bins=10, range=(0, 10))[0].astype(float)
    lbp_hist_b = np.histogram(lbp_b, bins=10, range=(0, 10))[0].astype(float)
    lbp_hist_a /= max(lbp_hist_a.sum(), 1)
    lbp_hist_b /= max(lbp_hist_b.sum(), 1)
    glcm_a = graycomatrix(gray_a, (1, 2, 3), (0,), symmetric=True, normed=True)
    glcm_b = graycomatrix(gray_b, (1, 2, 3), (0,), symmetric=True, normed=True)
    glcm_properties = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
    texture = np.mean([
        np.mean(np.abs(graycoprops(glcm_a, prop) - graycoprops(glcm_b, prop)))
        for prop in glcm_properties
    ])
    flat_a, flat_b = gray_a.ravel(), gray_b.ravel()
    correlation = np.corrcoef(flat_a, flat_b)[0, 1]
    fractal_a, fractal_b = _fractal_dimension(gray_a), _fractal_dimension(gray_b)
    vgg = perceptual(first, second)["classifier_perceptual_loss"]
    epsilon = 1e-10
    return {
        "exact_match": float(np.array_equal(rgb_a, rgb_b)),
        "correlation_coefficient": float(correlation) if np.isfinite(correlation) else 0.0,
        "histogram_intersection": float(np.minimum(hist_a, hist_b).sum()),
        "lbp_histogram_similarity": float(np.minimum(lbp_hist_a, lbp_hist_b).sum()),
        "psnr": float(peak_signal_noise_ratio(rgb_a, rgb_b, data_range=255)),
        "ssim": float(structural_similarity(rgb_a, rgb_b, channel_axis=-1, data_range=255)),
        "normalized_mutual_information": float(normalized_mutual_info_score(flat_a, flat_b)),
        "fractal_dimension_similarity": float(1 / (1 + abs(fractal_a - fractal_b))),
        "kl_divergence": float(entropy(hist_a + epsilon, hist_b + epsilon)),
        "mse": float(np.mean((rgb_a.astype(float) - rgb_b.astype(float)) ** 2)),
        "vgg_perceptual_distance": float(vgg),
        "glcm_texture_distance": float(texture),
        "wasserstein_distance": float(wasserstein_distance(
            np.arange(256), np.arange(256), hist_a, hist_b
        )),
    }


def _mean_metrics(rows):
    if not rows:
        return {}
    result = {}
    for key in rows[0]["metrics"]:
        values = [row["metrics"][key] for row in rows if np.isfinite(row["metrics"][key])]
        result[key] = float(np.mean(values)) if values else None
    return result


def paired_public_metrics(first_dir, second_dir, first_name, second_name):
    pairs, pairing = paired_files(**{first_name: first_dir, second_name: second_dir})
    print(f"Found {len(pairs)} pairs for {first_name} vs {second_name} evaluation.", flush=True)
    print("  Loading VGG16...", flush=True)
    perceptual = VGGPerceptualMetrics()
    rows = []
    for index, pair in enumerate(pairs, 1):
        print(f"  {first_name} vs {second_name}: {index}/{len(pairs)} - {pair['stem']}", flush=True)
        first = Image.open(pair[first_name]).convert("RGB")
        second = Image.open(pair[second_name]).convert("RGB")
        rows.append({"image": pair["stem"], "metrics": original_work_metrics(first, second, perceptual)})
    return {"pairing": pairing, "summary": _mean_metrics(rows), "images": rows}


def _empty_paired_metrics(first_dir, second_dir, first_name, second_name):
    first = files_by_stem(first_dir, normalize_simulated=first_name == "simulated")
    second = files_by_stem(second_dir, normalize_simulated=second_name == "simulated")
    common = set(first) & set(second)
    print(
        f"  No paired {first_name}/{second_name} stems found. "
        f"{first_name} folder: {Path(first_dir)}; "
        f"{second_name} folder: {Path(second_dir)}",
        flush=True,
    )
    for name, files in ((first_name, first), (second_name, second)):
        print(f"  {name} stems ({len(files)}): {sorted(files) or 'none'}", flush=True)
        for stem, path in sorted(files.items()):
            print(f"    {stem}: {path.name}", flush=True)
    return {
        "pairing": {
            "paired": len(common),
            "counts": {first_name: len(first), second_name: len(second)},
            "unpaired": {
                first_name: sorted(set(first) - common),
                second_name: sorted(set(second) - common),
            },
        },
        "summary": {},
        "images": [],
        "note": (
            f"No matching filename stems were found for {first_name} and {second_name}; "
            "paired image metrics were skipped."
        ),
    }


def evaluate_public_realism(folders):
    generated_files = files_by_stem(folders["generated"])
    simulated_files = files_by_stem(folders["simulated"], normalize_simulated=True)
    reference_files = files_by_stem(folders["reference"])

    generated = (
        paired_public_metrics(folders["generated"], folders["reference"], "generated", "real")
        if set(generated_files) & set(reference_files)
        else _empty_paired_metrics(folders["generated"], folders["reference"], "generated", "real")
    )
    simulated = (
        paired_public_metrics(folders["simulated"], folders["reference"], "simulated", "real")
        if set(simulated_files) & set(reference_files)
        else _empty_paired_metrics(folders["simulated"], folders["reference"], "simulated", "real")
    )
    generated["distribution"] = evaluate_realism(folders["reference"], folders["generated"])
    simulated["distribution"] = evaluate_realism(folders["reference"], folders["simulated"])
    return {"generated_vs_real": generated, "simulated_vs_real": simulated,
            "metric_directions": METRIC_DIRECTIONS}


def evaluate_sam(folders, output_dir, segmenter):
    """Class-agnostic SAM validation for matched simulated/generated scenes.

    Real images are an unpaired reference distribution, so they are not used
    for per-image segmentation comparison.
    """
    sim = evaluate_validity(
        folders["simulated"], folders["generated"], segmenter,
        Path(output_dir) / "simulated_vs_generated", "simulated", "generated"
    )
    return {"simulated_vs_generated": sim}


def load_labels(folder):
    """Load labels/<stem>_label.json, keyed by <stem> to match image stems."""
    result = {}
    for path in sorted(Path(folder).glob("*.json")):
        stem = path.stem[: -len("_label")] if path.stem.endswith("_label") else path.stem
        result[stem] = json.loads(path.read_text())
    return result


def public_features(label):
    features = {}
    for key, value in label.items():
        if key not in IGNORED_FEATURES:
            if any(part in key.lower() for part in IGNORED_FEATURE_NAME_PARTS):
                continue
            if key == "baby" and value is None:
                features[key] = "UNKNOWN"
            elif value is not None and isinstance(value, (str, bool)):
                features[key] = value
    return features


def _normalize(value):
    if value is None:
        return None
    return str(value).strip().upper()


def moondream_feature_prompt(features):
    schema = {key: list(sorted(values)) for key, values in features.items()}
    return (
        "Inspect this in-car image. Return one strict JSON object with exactly one "
        "scalar string value for every key. For each key, choose one item from its "
        "allowed-values list, or use UNKNOWN when it is not visible. Do not return "
        "arrays and do not repeat the schema. "
        f"Allowed values: {json.dumps(schema)}"
    )


def parse_object(answer):
    text = str(answer).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def evaluate_failure_rate(folders, analyzer):
    labels = load_labels(folders["labels"])
    vocabulary = {}
    for label in labels.values():
        for key, value in public_features(label).items():
            vocabulary.setdefault(key, set()).add(_normalize(value))
    prompt = moondream_feature_prompt(vocabulary)
    images = files_by_stem(folders["generated"])
    references = files_by_stem(folders["reference"])
    stems = sorted(set(labels) & set(images) & set(references))

    # Debug output for empty stems
    if not stems:
        print(f"  WARNING: No matching stems found for failure detection")
        print(f"    Labels stems ({len(labels)}): {sorted(labels.keys())}", flush=True)
        print(f"    Generated stems ({len(images)}): {sorted(images.keys())}", flush=True)
        print(f"    Reference stems ({len(references)}): {sorted(references.keys())}", flush=True)
    rows = []
    analyzer_name = getattr(analyzer, '__class__', type(analyzer)).__name__
    for index, stem in enumerate(stems, 1):
        print(f"  {analyzer_name} generated-vs-real matching: {index}/{len(stems)} - {stem}", flush=True)
        started = time.perf_counter()
        real_image = Image.open(references[stem]).convert("RGB")
        generated_image = Image.open(images[stem]).convert("RGB")
        real_prediction, real_answers = analyzer.classify_features(real_image, vocabulary)
        identical_inputs = (
            real_image.size == generated_image.size
            and np.array_equal(np.asarray(real_image), np.asarray(generated_image))
        )
        # Identical pixels must have identical semantic measurements. Avoid a
        # second stochastic VLM call changing the score for the same image.
        if identical_inputs:
            generated_prediction = dict(real_prediction)
            generated_answers = dict(real_answers)
        else:
            generated_prediction, generated_answers = analyzer.classify_features(
                generated_image, vocabulary
            )
        keys = sorted(public_features(labels[stem]))
        matches = {
            key: _normalize(generated_prediction.get(key)) == _normalize(real_prediction.get(key))
            for key in keys
        }
        rows.append({"image": stem, "failure": not all(matches.values()),
                     "comparison": "generated_vs_real",
                     "identical_inputs": identical_inputs,
                     "expected": real_prediction, "prediction": generated_prediction,
                     "real_prediction": real_prediction,
                     "generated_prediction": generated_prediction,
                     "feature_matches": matches,
                     "raw_real_answers": real_answers,
                     "raw_generated_answers": generated_answers,
                     "inference_seconds": time.perf_counter() - started})
    failures = sum(row["failure"] for row in rows)
    return {"failure_rate": failures / len(rows) if rows else None, "failures": failures,
            "executed": len(rows), "prompt": prompt,
            "inference": {"model": "vikhyatk/moondream2",
                          "revision": analyzer.revision,
                          "seed": analyzer.seed,
                          "settings": analyzer.DETERMINISTIC_SETTINGS},
            "images": rows}


def feature_diversity(failure_report, labels):
    failing = [row["image"] for row in failure_report["images"] if row["failure"]]
    vectors = [public_features(labels[stem]) for stem in failing]
    keys = sorted(set().union(*(vector.keys() for vector in vectors))) if vectors else []
    distances = []
    for first, second in combinations(vectors, 2):
        shared = [key for key in keys if key in first and key in second]
        if shared:
            distances.append(np.mean([_normalize(first[key]) != _normalize(second[key]) for key in shared]))
    return {"failing_scenes": len(failing), "mean_pairwise_hamming":
            float(np.mean(distances)) if distances else 0.0,
            "unique_values": {key: len({_normalize(row[key]) for row in vectors if key in row}) for key in keys}}


def build_final_report(report, labels):
    """Assemble the final per-contestant report from the independent raw
    sections (sam, realism_raw, failure_raw). Every metric is a continuous
    distance or agreement rate — nothing here gates or excludes a scene, and
    no metric family depends on another having run."""
    failure = report["failure_raw"]
    return {
        "failure": failure,
        "realism": {
            "generated_vs_real": report["realism_raw"]["generated_vs_real"],
            "simulated_vs_real": report["realism_raw"]["simulated_vs_real"],
            "metric_directions": report["realism_raw"]["metric_directions"],
        },
        "feature_diversity": feature_diversity(failure, labels),
    }


def clip_visual_diversity(image_dir, stems, model_name="openai/clip-vit-base-patch32"):
    if len(stems) < 2:
        return {"mean_cosine_distance": 0.0, "images": len(stems)}
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    paths = files_by_stem(image_dir)
    images = [Image.open(paths[stem]).convert("RGB") for stem in stems]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarity = embeddings @ embeddings.T
    upper = torch.triu_indices(len(stems), len(stems), offset=1)
    return {"mean_cosine_distance": float((1 - similarity[upper[0], upper[1]]).mean()),
            "images": len(stems), "model": model_name}


def efficiency(folder, executed):
    path = Path(folder) / "generation_times.json"
    if not path.exists():
        return {"available": False, "note": "Provide generation_times.json from standardized execution"}
    data = json.loads(path.read_text())
    total = float(data["total_seconds"])
    return {"available": True, "total_seconds": total,
            "seconds_per_scene": total / executed if executed else None,
            "hardware": data.get("hardware")}


def pareto_front(rows, objectives):
    def dominates(first, second):
        values = []
        for key, direction in objectives.items():
            a, b = first[key], second[key]
            values.append((a >= b, a > b) if direction == "max" else (a <= b, a < b))
        return all(item[0] for item in values) and any(item[1] for item in values)
    remaining = list(rows)
    fronts = []
    while remaining:
        front = [row for row in remaining if not any(dominates(other, row) for other in remaining if other is not row)]
        fronts.append([row["contestant"] for row in front])
        remaining = [row for row in remaining if row not in front]
    return fronts


def tie_break_scores(rows, weights=None):
    weights = weights or {"failure": 0.3, "realism": 0.4, "diversity": 0.3}
    normalized = {row["contestant"]: {} for row in rows}
    for metric in weights:
        values = [row[metric] for row in rows]
        low, high = min(values), max(values)
        for row in rows:
            normalized[row["contestant"]][metric] = (
                (row[metric] - low) / (high - low) if high > low else 1.0
            )
    return {
        name: float(sum(weights[key] * values[key] for key in weights))
        for name, values in normalized.items()
    }
