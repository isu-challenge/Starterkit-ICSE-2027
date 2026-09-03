from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity
from skimage.segmentation import find_boundaries


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Segmenter(Protocol):
    class_names: None

    def predict(self, image: Image.Image) -> np.ndarray: ...


class SamAutomaticSegmenter:
    class_names = None

    def __init__(self, mask_generator: Any):
        self.mask_generator = mask_generator

    def predict(self, image):
        masks = self.mask_generator.generate(np.asarray(image.convert("RGB")))
        masks.sort(key=lambda item: item.get("area", 0), reverse=True)
        labels = np.zeros((image.height, image.width), dtype=np.int64)
        for index, item in enumerate(masks, 1):
            labels[np.asarray(item["segmentation"], dtype=bool)] = index
        return labels


class TransformersSamSegmenter:
    class_names = None

    def __init__(self, model_name="facebook/sam-vit-base", device=None, points_per_batch=64):
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline("mask-generation", model=model_name, device=device)
        self.points_per_batch = points_per_batch

    def predict(self, image):
        output = self.generator(image, points_per_batch=self.points_per_batch)
        masks = [np.asarray(mask, dtype=bool) for mask in output["masks"]]
        masks.sort(key=np.count_nonzero, reverse=True)
        labels = np.zeros((image.height, image.width), dtype=np.int64)
        for index, mask in enumerate(masks, 1):
            labels[mask] = index
        return labels


class MoondreamAnalyzer:
    DETERMINISTIC_SETTINGS = {
        "temperature": 0.0,
        "top_p": 0.0,
        "max_tokens": 768,
    }

    def __init__(self, revision="2025-06-21", device=None, seed=0):
        import torch
        from transformers import AutoModelForCausalLM

        self.revision = revision
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        # moondream2's own remote code patches adaptive_avg_pool2d to force its
        # output onto MPS whenever MPS is available on the machine (see its
        # vision.py, citing pytorch/pytorch#96056) — even when the model is
        # explicitly placed on CPU. That breaks a genuinely CPU-only,
        # reproducible run on any Apple-silicon machine. Hide MPS from the
        # module while it loads so a requested CPU run stays on CPU.
        hide_mps = device == "cpu" and torch.backends.mps.is_available()
        if hide_mps:
            real_is_available = torch.backends.mps.is_available
            torch.backends.mps.is_available = lambda: False
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision=revision,
                trust_remote_code=True,
            )
        finally:
            if hide_mps:
                torch.backends.mps.is_available = real_is_available
        self.model = self.model.to(device).eval()

    def query(self, image, prompt):
        """Greedy Moondream query with pinned decoding settings."""
        import torch

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        return self.model.query(
            image,
            prompt,
            reasoning=False,
            settings=dict(self.DETERMINISTIC_SETTINGS),
        )["answer"]

    def classify_features(self, image, feature_values):
        """Classify features with small greedy prompts and one image encoding."""
        encoded = self.model.encode_image(image)
        predictions = {}
        raw_answers = {}
        for key in sorted(feature_values):
            allowed = [str(value).strip().upper() for value in feature_values[key]]
            choices = allowed + ["UNKNOWN"]
            prompt = (
                f"Inspect the image for the feature {key}. "
                f"Answer with exactly one of: {', '.join(choices)}. "
                "Return only the chosen value, with no explanation."
            )
            answer = self.model.query(
                encoded,
                prompt,
                reasoning=False,
                settings={"temperature": 0.0, "top_p": 0.0, "max_tokens": 16},
            )["answer"]
            normalized = str(answer).strip().strip("`\"'. ").upper()
            prediction = normalized if normalized in choices else "UNKNOWN"
            predictions[key] = prediction
            raw_answers[key] = answer
        return predictions, raw_answers

    def analyze(self, image, prompt):
        query = self.query(image, prompt)
        return {
            "short_caption": self.model.caption(image, length="short")["caption"],
            "normal_caption": self.model.caption(image, length="normal")["caption"],
            "query": query,
            "query_json": parse_json_answer(query),
            "faces": self.model.detect(image, "face")["objects"],
            "people": self.model.point(image, "person")["points"],
        }


def evaluate_vlm_folders(folders, analyzer, prompt):
    domains = {}
    for domain, folder in folders.items():
        domains[domain] = {}
        paths = list_images(folder)
        for index, path in enumerate(paths, 1):
            print(f"  Moondream {domain}: {index}/{len(paths)} - {path.name}", flush=True)
            domains[domain][path.stem] = analyzer.analyze(Image.open(path).convert("RGB"), prompt)

    comparisons = {}
    for first, second in combinations(domains, 2):
        pair_stems = set(domains[first]) & set(domains[second])
        comparisons[f"{first}_vs_{second}"] = {
            stem: _vlm_comparison(domains[first][stem], domains[second][stem], first, second)
            for stem in sorted(pair_stems)
        }
    shared = set.intersection(*(set(items) for items in domains.values())) if domains else set()
    triplets = {
        stem: {domain: items[stem] for domain, items in domains.items()}
        for stem in sorted(shared)
    }
    return {
        "prompt": prompt,
        "domains": domains,
        "comparisons": comparisons,
        "simulated_generated_comparison": comparisons.get("simulated_vs_generated", {}),
        "matched_triplets": triplets,
    }


def parse_json_answer(answer):
    text = str(answer).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    score = value.get("score")
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return None
    return {"score": float(score), "explanation": str(value.get("explanation", ""))}


def _vlm_comparison(first_result, second_result, first_name="simulated", second_name="generated"):
    first_query = first_result.get("query_json") or {}
    second_query = second_result.get("query_json") or {}
    first_score, second_score = first_query.get("score"), second_query.get("score")
    return {
        f"{first_name}_score": first_score,
        f"{second_name}_score": second_score,
        "score_delta": (
            second_score - first_score
            if first_score is not None and second_score is not None else None
        ),
        f"{first_name}_faces": len(first_result.get("faces", [])),
        f"{second_name}_faces": len(second_result.get("faces", [])),
        f"{first_name}_people": len(first_result.get("people", [])),
        f"{second_name}_people": len(second_result.get("people", [])),
    }


def list_images(folder):
    folder = Path(folder)
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def files_by_stem(folder):
    result = {}
    for path in list_images(folder):
        if path.stem in result:
            raise ValueError(f"Duplicate stem '{path.stem}' in {folder}")
        result[path.stem] = path
    return result


def paired_files(**folders):
    mappings = {name: files_by_stem(folder) for name, folder in folders.items()}
    common = set.intersection(*(set(mapping) for mapping in mappings.values()))
    if not common:
        raise ValueError("No files with matching stems found")
    rows = [{"stem": stem, **{name: data[stem] for name, data in mappings.items()}}
            for stem in sorted(common)]
    diagnostics = {
        "paired": len(rows),
        "counts": {name: len(data) for name, data in mappings.items()},
        "unpaired": {name: sorted(set(data) - common) for name, data in mappings.items()},
    }
    return rows, diagnostics


def resize_labels(labels, size):
    if labels.shape == (size[1], size[0]):
        return labels
    image = Image.fromarray(labels.astype(np.int32), mode="I")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)).astype(np.int64)


def colorize_partitions(labels):
    ids = np.unique(labels)
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in ids:
        if label == 0:
            color = (20, 20, 20)
        else:
            color = ((53 * label) % 256, (97 * label) % 256, (193 * label) % 256)
        colors[labels == label] = color
    return Image.fromarray(colors)


def _image_fingerprint(image):
    rgb = image.convert("RGB")
    digest = hashlib.sha256(np.asarray(rgb, dtype=np.uint8).tobytes()).hexdigest()
    return {"sha256_rgb": digest, "size": list(rgb.size)}


def save_partition_visuals(image, labels, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colorized = colorize_partitions(labels)
    overlay = Image.blend(image.convert("RGB").resize(colorized.size), colorized, 0.45)
    colorized.save(output_dir / f"{stem}_segments.png")
    overlay.save(output_dir / f"{stem}_overlay.png")
    np.save(output_dir / f"{stem}_labels.npy", labels)
    (output_dir / f"{stem}_cache.json").write_text(
        json.dumps(_image_fingerprint(image)), encoding="utf-8"
    )


def load_partition_cache(output_dir, stem, image=None):
    output_dir = Path(output_dir)
    metadata_path = output_dir / f"{stem}_cache.json"
    if image is not None:
        if not metadata_path.exists():
            return None
        try:
            if json.loads(metadata_path.read_text(encoding="utf-8")) != _image_fingerprint(image):
                return None
        except (OSError, ValueError, TypeError):
            return None
    labels_path = output_dir / f"{stem}_labels.npy"
    if labels_path.exists():
        return np.load(labels_path)
    segments_path = output_dir / f"{stem}_segments.png"
    if not segments_path.exists():
        return None
    colors = np.asarray(Image.open(segments_path).convert("RGB"))
    _, labels = np.unique(colors.reshape(-1, 3), axis=0, return_inverse=True)
    return labels.reshape(colors.shape[:2]).astype(np.int64)


class VGGPerceptualMetrics:
    def __init__(self, device=None):
        import torch
        from torchvision.models import VGG16_Weights, vgg16

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = VGG16_Weights.IMAGENET1K_V1
        self.model = vgg16(weights=weights).features[:30].to(self.device).eval()
        self.transform = weights.transforms()

    def __call__(self, first, second):
        import torch.nn.functional as functional

        a = self.transform(first).unsqueeze(0).to(self.device)
        b = self.transform(second).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            feature_a, feature_b = self.model(a), self.model(b)
        return {
            "cosine_similarity": float(functional.cosine_similarity(
                feature_a.flatten(1), feature_b.flatten(1)
            ).item()),
            "classifier_perceptual_loss": float(functional.mse_loss(feature_a, feature_b).item()),
        }


def paired_fidelity(first: Image.Image, second: Image.Image, perceptual=None):
    second = second.resize(first.size, Image.Resampling.BILINEAR)
    a = np.asarray(first.convert("RGB"), dtype=np.uint8)
    b = np.asarray(second.convert("RGB"), dtype=np.uint8)
    mse = float(np.mean((a.astype(float) - b.astype(float)) ** 2))
    gray_a = np.asarray(first.convert("L"), dtype=np.uint8).copy()
    gray_b = np.asarray(second.convert("L"), dtype=np.uint8).copy()
    properties = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
    glcm_a = graycomatrix(gray_a, distances=(1, 2, 3), angles=(0,), symmetric=True, normed=True)
    glcm_b = graycomatrix(gray_b, distances=(1, 2, 3), angles=(0,), symmetric=True, normed=True)
    texture = np.mean([
        np.mean(np.abs(graycoprops(glcm_a, prop) - graycoprops(glcm_b, prop)))
        for prop in properties
    ])
    hist_a = np.histogram(a, bins=256, range=(0, 256))[0].astype(float)
    hist_b = np.histogram(b, bins=256, range=(0, 256))[0].astype(float)
    hist_a /= max(hist_a.sum(), 1)
    hist_b /= max(hist_b.sum(), 1)
    epsilon = 1e-10
    metrics = {
        "exact_match": float(np.array_equal(a, b)),
        "ssim": float(structural_similarity(a, b, channel_axis=-1, data_range=255)),
        "psnr": math.inf if mse == 0 else float(20 * np.log10(255 / np.sqrt(mse))),
        "mse": mse,
        "texture_similarity": float(texture),
        "wasserstein_distance": float(wasserstein_distance(
            np.arange(256), np.arange(256), hist_a, hist_b
        )),
        "kl_divergence": float(entropy(hist_a + epsilon, hist_b + epsilon)),
        "histogram_intersection": float(np.minimum(hist_a, hist_b).sum()),
    }
    if perceptual is not None:
        metrics.update(perceptual(first, second))
    return metrics


def partition_metrics(truth, prediction, ignore_value=255):
    valid = truth != ignore_value
    a, b = truth[valid].astype(int), prediction[valid].astype(int)
    if not a.size:
        return {
            "adjusted_rand": None,
            "normalized_mutual_info": None,
            "matched_iou": None,
            "symmetric_region_covering": None,
            "boundary_f1": None,
        }
    _, a_inverse = np.unique(a, return_inverse=True)
    _, b_inverse = np.unique(b, return_inverse=True)
    overlap = np.zeros((a_inverse.max() + 1, b_inverse.max() + 1), dtype=np.int64)
    np.add.at(overlap, (a_inverse, b_inverse), 1)
    a_area, b_area = overlap.sum(1)[:, None], overlap.sum(0)[None, :]
    iou = overlap / np.maximum(a_area + b_area - overlap, 1)
    rows, columns = linear_sum_assignment(-iou)
    truth_covering = np.average(iou.max(axis=1), weights=a_area[:, 0])
    prediction_covering = np.average(iou.max(axis=0), weights=b_area[0])
    return {
        "adjusted_rand": float(adjusted_rand_score(a, b)),
        "normalized_mutual_info": float(normalized_mutual_info_score(a, b)),
        "matched_iou": float(np.average(iou[rows, columns], weights=a_area[rows, 0])),
        "symmetric_region_covering": float((truth_covering + prediction_covering) / 2),
        "boundary_f1": boundary_f1(truth, prediction, valid),
    }


def boundary_f1(first, second, valid=None, tolerance=2):
    first_boundary = find_boundaries(first, mode="thick")
    second_boundary = find_boundaries(second, mode="thick")
    if valid is not None:
        first_boundary &= valid
        second_boundary &= valid
    if not first_boundary.any() and not second_boundary.any():
        return 1.0
    if not first_boundary.any() or not second_boundary.any():
        return 0.0
    first_near = binary_dilation(first_boundary, iterations=tolerance)
    second_near = binary_dilation(second_boundary, iterations=tolerance)
    precision = np.count_nonzero(second_boundary & first_near) / np.count_nonzero(second_boundary)
    recall = np.count_nonzero(first_boundary & second_near) / np.count_nonzero(first_boundary)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def evaluate_realism(real_dir, generated_dir, batch_size=16):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    real_paths, generated_paths = list_images(real_dir), list_images(generated_dir)
    if len(real_paths) < 2 or len(generated_paths) < 2:
        return {"inception_score_mean": None, "inception_score_std": None,
                "fid": None, "kid_mean": None, "kid_std": None,
                "real_images": len(real_paths), "generated_images": len(generated_paths)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    subset = min(50, len(real_paths), len(generated_paths))
    kid = KernelInceptionDistance(feature=2048, subsets=20, subset_size=subset,
                                  normalize=False).to(device)
    inception = InceptionScore(splits=min(10, len(generated_paths)), normalize=False).to(device)
    for is_real, paths in ((True, real_paths), (False, generated_paths)):
        domain = "real" if is_real else "candidate"
        total_batches = math.ceil(len(paths) / batch_size)
        for start in range(0, len(paths), batch_size):
            batch_number = start // batch_size + 1
            print(f"    Distribution features {domain}: batch {batch_number}/{total_batches}", flush=True)
            images = []
            for path in paths[start:start + batch_size]:
                image = Image.open(path).convert("RGB").resize((299, 299), Image.Resampling.BILINEAR)
                images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
            batch = torch.stack(images).to(device)
            fid.update(batch, real=is_real)
            kid.update(batch, real=is_real)
            if not is_real:
                inception.update(batch)
    kid_mean, kid_std = kid.compute()
    is_mean, is_std = inception.compute()
    return {"inception_score_mean": float(is_mean), "inception_score_std": float(is_std),
            "fid": float(fid.compute()), "kid_mean": float(kid_mean),
            "kid_std": float(kid_std), "real_images": len(real_paths),
            "generated_images": len(generated_paths)}


def evaluate_segmentation(first_dir, second_dir, segmenter,
                          segmentation_output_dir=None, first_name="simulated",
                          second_name="generated"):
    pairs, diagnostics = paired_files(**{first_name: first_dir, second_name: second_dir})
    rows = []
    for index, item in enumerate(pairs, 1):
        first = Image.open(item[first_name]).convert("RGB")
        second = Image.open(item[second_name]).convert("RGB")
        root = Path(segmentation_output_dir) if segmentation_output_dir is not None else None
        first_partition = (
            load_partition_cache(root / first_name, item["stem"], first) if root else None
        )
        second_partition = (
            load_partition_cache(root / second_name, item["stem"], second) if root else None
        )
        identical_inputs = (
            first.size == second.size
            and np.array_equal(np.asarray(first), np.asarray(second))
        )
        if identical_inputs:
            if first_partition is None and second_partition is not None:
                first_partition = second_partition.copy()
            elif second_partition is None and first_partition is not None:
                second_partition = first_partition.copy()
            elif first_partition is None and second_partition is None:
                if segmenter is None:
                    print("  Loading SAM because this pair is not cached...", flush=True)
                    segmenter = TransformersSamSegmenter()
                first_partition = segmenter.predict(first)
                second_partition = first_partition.copy()
            else:
                # Even deterministic models can have separately-created stale
                # caches. Pixel identity is stronger evidence than two model runs.
                second_partition = first_partition.copy()
        if first_partition is not None and second_partition is not None:
            print(
                f"  SAM {first_name} vs {second_name}: {index}/{len(pairs)} - "
                f"{item['stem']} (cached)", flush=True
            )
        else:
            print(
                f"  SAM {first_name} vs {second_name}: {index}/{len(pairs)} - "
                f"{item['stem']} (running SAM)", flush=True
            )
            if segmenter is None:
                print("  Loading SAM because this pair is not cached...", flush=True)
                segmenter = TransformersSamSegmenter()
            first_partition = segmenter.predict(first)
            second_partition = segmenter.predict(second)
        first_partition = resize_labels(first_partition, second.size)
        prediction = resize_labels(second_partition, second.size)
        if root is not None:
            save_partition_visuals(first, first_partition, root / first_name, item["stem"])
            save_partition_visuals(second, prediction, root / second_name, item["stem"])
        semantic = {"partition": partition_metrics(first_partition, prediction, -1)}
        matched_iou = semantic["partition"]["matched_iou"]
        semantic["semantic_segmentation_score"] = 1.0 - matched_iou
        rows.append({"image": item["stem"], "semantic": semantic})
    return {
        "comparison": f"{first_name}_vs_{second_name}",
        "pairing": diagnostics,
        "semantic_mode": "class_agnostic_sam_partitions",
        "segmentation_visuals": (
            str(segmentation_output_dir) if segmentation_output_dir is not None else None
        ),
        "summary": {
            "partition": _mean_dict([row["semantic"]["partition"] for row in rows]),
            "semantic_segmentation_score": _mean(
                row["semantic"]["semantic_segmentation_score"] for row in rows
            ),
        },
        "images": rows,
    }


def _paired_image_metrics(first_dir, second_dir, first_name, second_name, perceptual):
    pairs, diagnostics = paired_files(**{first_name: first_dir, second_name: second_dir})
    rows = []
    for index, item in enumerate(pairs, 1):
        print(
            f"  Image metrics {first_name} vs {second_name}: "
            f"{index}/{len(pairs)} - {item['stem']}",
            flush=True,
        )
        first = Image.open(item[first_name]).convert("RGB")
        second = Image.open(item[second_name]).convert("RGB")
        rows.append({"image": item["stem"], "metrics": paired_fidelity(first, second, perceptual)})
    return {
        "comparison": f"{first_name}_vs_{second_name}",
        "pairing": diagnostics,
        "summary": _mean_dict([row["metrics"] for row in rows]),
        "images": rows,
    }


def evaluate_image_metrics(simulated_dir, generated_dir, real_dir):
    print("  Loading VGG perceptual model...", flush=True)
    perceptual = VGGPerceptualMetrics()
    print("  Paired metrics: generated vs real", flush=True)
    generated_vs_real = _paired_image_metrics(
        generated_dir, real_dir, "generated", "real", perceptual
    )
    print("  Paired metrics: simulated vs real", flush=True)
    simulated_vs_real = _paired_image_metrics(
        simulated_dir, real_dir, "simulated", "real", perceptual
    )
    print("  Distribution metrics: generated vs real", flush=True)
    generated_vs_real["distribution"] = evaluate_realism(real_dir, generated_dir)
    print("  Distribution metrics: simulated vs real", flush=True)
    simulated_vs_real["distribution"] = evaluate_realism(real_dir, simulated_dir)
    return {
        "generated_vs_real": generated_vs_real,
        "simulated_vs_real": simulated_vs_real,
        "improvement": {
            "positive_means_generated_is_better": True,
            "paired": _metric_improvement(
                generated_vs_real["summary"], simulated_vs_real["summary"]
            ),
            "distribution": _metric_improvement(
                generated_vs_real["distribution"], simulated_vs_real["distribution"]
            ),
        },
    }


def _metric_improvement(generated, simulated):
    higher_is_better = {
        "ssim", "psnr", "cosine_similarity", "histogram_intersection",
        "inception_score_mean",
    }
    ignored = {"inception_score_std", "kid_std", "real_images", "generated_images"}
    result = {}
    for metric in generated.keys() & simulated.keys():
        if metric in ignored or generated[metric] is None or simulated[metric] is None:
            continue
        delta = float(generated[metric]) - float(simulated[metric])
        result[metric] = delta if metric in higher_is_better else -delta
    return result


def evaluate_dataset(simulated_dir, generated_dir, segmenter=None, real_dir=None,
                     segmentation_output_dir=None):
    segmentation = evaluate_segmentation(
        simulated_dir, generated_dir, segmenter, segmentation_output_dir
    )
    image_metrics = (
        evaluate_image_metrics(simulated_dir, generated_dir, real_dir) if real_dir else None
    )
    return {
        "dataset": segmentation["pairing"],
        "segmentation": segmentation,
        "image_metrics": image_metrics,
        "semantic_mode": segmentation["semantic_mode"],
        "segmentation_visuals": segmentation["segmentation_visuals"],
        "metric_direction": {
            "exact_match": "higher_is_better",
            "inception_score": "higher_is_better",
            "fid": "lower_is_better",
            "kid": "lower_is_better",
            "ssim": "higher_is_better",
            "psnr": "higher_is_better",
            "mse": "lower_is_better",
            "cosine_similarity": "higher_is_better",
            "texture_similarity": "lower_is_better",
            "wasserstein_distance": "lower_is_better",
            "kl_divergence": "lower_is_better",
            "histogram_intersection": "higher_is_better",
            "classifier_perceptual_loss": "lower_is_better",
            "semantic_segmentation_score": "lower_is_better",
            "symmetric_region_covering": "higher_is_better",
            "boundary_f1": "higher_is_better",
        },
        "summary": {
            "segmentation": segmentation["summary"],
            "image_metrics": {
                name: {"paired": result["summary"], "distribution": result["distribution"]}
                for name, result in (image_metrics or {}).items()
                if name != "improvement"
            },
            "improvement": (image_metrics or {}).get("improvement"),
        },
    }


def _mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _mean_dict(items):
    return {key: _mean(item[key] for item in items) for key in items[0]} if items else {}


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description="Evaluate sim-to-real image translations")
    parser.add_argument("--simulated", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    parser.add_argument("--segments", default="sam_segments")
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()
    report = evaluate_dataset(
        args.simulated,
        args.generated,
        segmenter=TransformersSamSegmenter(args.sam_model),
        real_dir=args.reference,
        segmentation_output_dir=args.segments,
    )
    save_report(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
