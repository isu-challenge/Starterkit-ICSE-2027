# ISU-Challenge Evaluation CLI Guide

Complete reference for Track A and Track B evaluation scripts with structured Python classes.

## Table of Contents

1. [Quick Navigation](#quick-navigation)
2. [Architecture Overview](#architecture-overview)
3. [Track A Evaluation](#track-a-evaluation)
4. [Track B Evaluation](#track-b-evaluation)
5. [Implementing Custom Systems](#implementing-custom-systems)
6. [Data Formats](#data-formats)
7. [Report Formats](#report-formats)
8. [Complete CLI Reference](#complete-cli-reference)
9. [Troubleshooting](#troubleshooting)

---

## Quick Navigation

### By Task

| Task | Location | Time |
|------|----------|------|
| **Get started immediately** | [Quick Start](#track-a-evaluation--usage) | 5 min |
| **Understand architecture** | [Architecture Overview](#architecture-overview) | 15 min |
| **Evaluate submissions** | [Track A](#track-a-evaluation) | 5 min |
| **Evaluate ISU systems** | [Track B](#track-b-evaluation) | 5 min |
| **Implement custom system** | [Implementing Custom Systems](#implementing-custom-systems) | 30 min |
| **Debug issues** | [Troubleshooting](#troubleshooting) | varies |

### By Role

**Track A Participant (Test Generator):**
- Read [Track A Translator Implementation](#implementing-track-a-translators)
- See `track_a/example_custom_translator.py`
- Generate and submit scenes

**Track B Participant (ISU System):**
- Read [Track B System Implementation](#implementing-track-b-isu-systems)
- See `track_b/example_custom_system.py`
- Implement and evaluate your system

**Developer/Researcher:**
- Read [Architecture Overview](#architecture-overview)
- Study [Class Hierarchies](#class-hierarchies)
- Review example implementations

---

## Architecture Overview

### Design Principles

The evaluation system is built on these principles:

1. **Separation of Concerns** — Abstract base classes define contracts
2. **CLI-First** — All evaluations accessible via command-line flags
3. **Extensibility** — Participants extend through class subclassing
4. **Standardization** — JSON output for reproducibility and comparison
5. **Backward Compatibility** — Supports multiple data directory structures

### Class Hierarchies

**Track A:**

```
DataLoader (ABC)
└── TrackADataLoader
    ├── load(submission_path) → dict[str, Path]
    └── validate(data) → bool

Translator (ABC)
└── [Participants implement]
    └── translate(image, auxiliary_data) → Image

EvaluatorTrackA
├── evaluate_segmentation() → dict      [SAM distance]
├── evaluate_realism() → dict           [Image metrics]
├── evaluate_failures() → dict          [VLM failure rate]
├── evaluate_efficiency() → dict        [Generation time]
└── build_report() → dict               [Assemble final]

ReportWriter
├── write(report, path) → None
└── write_summary(reports, dir) → None

SubmissionEvaluator
├── evaluate_all() → dict[str, dict]
└── evaluate_submission(name, folders) → None
```

**Track B:**

```
DataLoader (ABC)
└── TrackBDataLoader
    └── load(data_dir, image_kind) → list[tuple]

ISUSystem (ABC)
├── predict(image: Image) → dict[str, str]
├── name() → str
└── set_prompt(prompt) → None

├── QwenISUSystem
│   └── predict() using Qwen2.5-VL
└── MoondreamISUSystem
    └── predict() using Moondream (stub)

EvaluatorTrackB
├── evaluate() → dict
├── _run_predictions(scenes) → list
└── _build_report(rows, vocab, prompt) → dict

ReportWriter
├── write(report, path) → None
└── write_readable_summary(report) → None
```

### Evaluation Pipelines

**Track A: `SubmissionEvaluator.evaluate_all()`**

```
1. Discover contestants
2. For each submission:
   ├─ Validate structure
   ├─ Run SAM segmentation distance → report["sam"]
   ├─ Run public realism metrics → report["realism_raw"]
   ├─ Run failure detection → report["failure_raw"]
   ├─ Build final report with diversity → report["final"]
   └─ Calculate efficiency → report["efficiency"]
3. Write summary.json
```

**Track B: `EvaluatorTrackB.evaluate()`**

```
1. Load dataset (images + labels)
2. Build vocabulary and prompt
3. For each scene:
   ├─ Load image
   ├─ Call system.predict(image)
   ├─ Match features with ground truth
   └─ Record latency
4. Aggregate metrics (accuracy, exact-match, latency)
5. Save JSON report
```

---

## Track A Evaluation

### Overview

Track A evaluates test generators that create realistic, diverse in-car scenes while preserving semantic structure. The evaluator measures:

- **Segmentation distance** — Structural preservation via SAM
- **Realism metrics** — Visual similarity (SSIM, FID, KID, etc.)
- **Failure detection** — VLM-based semantic consistency
- **Feature diversity** — Diversity of detected failures
- **Efficiency** — Generation time per scene

### Classes

#### DataLoader (ABC)
Abstract base class for loading submission data.

```python
class DataLoader(ABC):
    @abstractmethod
    def load(self, submission_path: Path) -> dict[str, Path]:
        """Load submission data and return paths to folders."""
        pass

    @abstractmethod
    def validate(self, data: dict[str, Path]) -> bool:
        """Validate that required folders exist."""
        pass
```

#### TrackADataLoader
Default implementation for standard submission structure.

```python
class TrackADataLoader(DataLoader):
    def load(self, submission_path: Path) -> dict[str, Path]:
        return {
            "simulated": submission_path / "simulated",
            "generated": submission_path / "generated",
            "reference": submission_path / "reference",
            "labels": submission_path / "labels",
        }

    def validate(self, data: dict[str, Path]) -> bool:
        return all(path.is_dir() for path in data.values())
```

#### Translator (ABC)
Abstract base class for scene transformation implementations.

```python
class Translator(ABC):
    @abstractmethod
    def translate(self, image: Image.Image, 
                  auxiliary_data: Optional[dict] = None) -> Image.Image:
        """Transform an image while preserving semantic structure."""
        pass
```

**Constraint:** Transformations must NOT modify:
- Object positions or boundaries
- Scene semantic structure
- Ground-truth label validity
- Spatial relationships between objects

#### EvaluatorTrackA
Static methods for all evaluation stages.

```python
class EvaluatorTrackA:
    @staticmethod
    def evaluate_segmentation(folders: dict[str, Path], 
                              output_dir: Path) -> dict[str, Any]:
        """Run SAM-based segmentation distance evaluation."""
        # Returns partition agreement metrics (boundary F1, IoU, etc.)

    @staticmethod
    def evaluate_realism(folders: dict[str, Path]) -> dict[str, Any]:
        """Run public image-quality metrics."""
        # Returns SSIM, FID, KID, PSNR, etc.

    @staticmethod
    def evaluate_failures(folders: dict[str, Path],
                          failure_model: str = "moondream",
                          device: str = "cpu") -> dict[str, Any]:
        """Run VLM-based failure detection."""
        # Returns failure rate, per-image results

    @staticmethod
    def evaluate_efficiency(submission_path: Path,
                            executed_scenes: int) -> dict[str, Any]:
        """Evaluate generation time."""
        # Returns seconds per scene

    @staticmethod
    def build_report(evaluation_results: dict,
                     labels: dict) -> dict[str, Any]:
        """Assemble final report with diversity metrics."""
```

#### ReportWriter & SubmissionEvaluator
See [classes section](#class-hierarchies) for details.

### Usage

#### Basic evaluation:
```bash
cd track_a
python evaluator_cli.py evaluate \
    --submissions-dir ../submissions \
    --output-dir results
```

#### With Qwen2.5-VL (GPU):
```bash
python evaluator_cli.py evaluate \
    --submissions-dir submissions \
    --output-dir results \
    --failure-model qwen \
    --device cuda
```

#### With Moondream (CPU):
```bash
python evaluator_cli.py evaluate \
    --failure-model moondream \
    --device cpu
```

### Output

**Per-submission:** `{contestant_name}.json`
```json
{
  "contestant": "team_name",
  "sam": { "simulated_vs_generated": {...} },
  "realism_raw": { "generated_vs_real": {...}, "simulated_vs_real": {...} },
  "failure_raw": { "failure_rate": 0.15, "failures": 2, "executed": 13 },
  "final": { "feature_diversity": {...}, "clip_visual_diversity": {...} },
  "efficiency": { "seconds_per_scene": 2.5 }
}
```

**Cross-submission:** `summary.json`
```json
{
  "team_name": {
    "failure_rate": 0.15,
    "ssim_generated": 0.72,
    "feature_diversity": 0.35,
    "seconds_per_scene": 2.5
  }
}
```

---

## Track B Evaluation

### Overview

Track B evaluates in-car scene understanding (ISU) systems that predict categorical features from images. The evaluator measures:

- **Feature accuracy** — Correct predictions per feature
- **Scene exact-match** — All features correct per scene
- **Latency** — Inference time per scene

### Classes

#### DataLoader (ABC)
Abstract base for loading image-label pairs.

```python
class DataLoader(ABC):
    @abstractmethod
    def load(self, data_dir: Path, 
             image_kind: str) -> list[tuple[str, Path, dict]]:
        """Load scenes with matching images and labels."""
        pass
```

#### TrackBDataLoader
Loads images and labels with support for real/ folder structure.

```python
class TrackBDataLoader(DataLoader):
    def load(self, data_dir: Path, image_kind: str):
        """Load dataset supporting:
        - New: data/real/, data/images/, data/labels/
        - Old: data/images/ (all), data/labels/
        """
```

**Supports:** `image_kind` of `"real"`, `"synthetic"`, or `"both"`

#### ISUSystem (ABC)
Abstract interface participants must implement.

```python
class ISUSystem(ABC):
    @abstractmethod
    def predict(self, image: Image.Image) -> dict[str, str]:
        """Predict features from RGB image.
        
        Returns:
            {"driver_phone": "NO", "baby_seat": "YES", ...}
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return system identifier."""
        pass

    def set_prompt(self, prompt: str) -> None:
        """Called by evaluator with feature vocabulary."""
        pass
```

#### QwenISUSystem
Reference implementation using Qwen2.5-VL-3B-Instruct.

```python
class QwenISUSystem(ISUSystem):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", 
                 device: str = "cuda"):
        # Loads model from Hugging Face

    def predict(self, image: Image.Image) -> dict[str, str]:
        # Queries model with structured prompt
```

#### MoondreamISUSystem
Lightweight CPU-friendly option (stub in current version).

```python
class MoondreamISUSystem(ISUSystem):
    def __init__(self, revision: str = "2025-06-21", device: str = "cpu"):
        pass
```

#### EvaluatorTrackB
Complete evaluation pipeline.

```python
class EvaluatorTrackB:
    def __init__(self, data_dir: Path, isu_system: ISUSystem,
                 image_kind: str = "both",
                 ignored_features: Optional[set[str]] = None,
                 output_path: Path = None):
        pass

    def evaluate(self) -> dict[str, Any]:
        """Run complete evaluation and save report."""
```

### Usage

#### Basic evaluation (Qwen on GPU):
```bash
cd track_b
python evaluator_cli.py evaluate --data-dir ../data
```

#### Custom output:
```bash
python evaluator_cli.py evaluate \
    --data-dir data \
    --output my_report.json
```

#### Real images only:
```bash
python evaluator_cli.py evaluate \
    --data-dir data \
    --image-kind real \
    --output real_only.json
```

#### Synthetic only:
```bash
python evaluator_cli.py evaluate \
    --data-dir data \
    --image-kind synthetic
```

#### Moondream on CPU:
```bash
python evaluator_cli.py evaluate \
    --data-dir data \
    --model moondream \
    --device cpu
```

#### Ignore features:
```bash
python evaluator_cli.py evaluate \
    --data-dir data \
    --ignore-feature light_front \
    --ignore-feature env_strength
```

### Output

**Report:** `report.json`
```json
{
  "track": "B",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "device": "cuda",
  "image_kind": "both",
  "prompt": "...",
  "features": {
    "driver_phone": ["NO", "YES"],
    "baby_seat": ["NO", "YES"]
  },
  "metrics": {
    "accuracy": 0.92,
    "scene_exact_match": 0.85,
    "correct_features": 184,
    "evaluated_features": 200,
    "latency_seconds_total": 45.2,
    "latency_seconds_mean": 0.226,
    "scenes": 200
  },
  "predictions": [
    {
      "image": "sample_0001_sim",
      "expected": {"driver_phone": "NO"},
      "prediction": {"driver_phone": "NO"},
      "feature_matches": {"driver_phone": true},
      "all_features_match": true,
      "inference_seconds": 0.22
    }
  ]
}
```

---

## Implementing Custom Systems

### Implementing Track A Translators

Create a `Translator` subclass for test generation:

```python
from pathlib import Path
from evaluator_cli import Translator
from PIL import Image

class MyTranslator(Translator):
    """Custom scene transformation."""

    def __init__(self, config=None):
        self.config = config or {}

    def translate(self, image: Image.Image, 
                  auxiliary_data=None) -> Image.Image:
        """Transform image while preserving structure.
        
        Args:
            image: RGB image of source scene
            auxiliary_data: dict with segmentation, depth, etc.
                           Use for validation, NOT for transformation
        
        Returns:
            Transformed image with same scene structure
        """
        # Example: modify appearance
        transformed = self._adjust_lighting(image)
        transformed = self._adjust_colors(transformed)
        
        # CRITICAL: Validate structure preserved
        if auxiliary_data:
            if not self._validate_structure(image, transformed, auxiliary_data):
                raise RuntimeError("Structure validation failed")
        
        return transformed

    def _adjust_lighting(self, image: Image.Image) -> Image.Image:
        # Example: brightness/contrast modifications
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)

    def _adjust_colors(self, image: Image.Image) -> Image.Image:
        # Example: hue/saturation modifications
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(0.9)

    def _validate_structure(self, original, transformed, auxiliary):
        # Validate using segmentation
        # Must NOT move/add/remove objects
        return True
```

**See:** `track_a/example_custom_translator.py` for complete example with:
- Lighting, color, texture, exposure adjustments
- Diffusion-based transformations
- Bounding box validation utilities

### Implementing Track B ISU Systems

Create an `ISUSystem` subclass for scene understanding:

```python
from evaluator_cli import ISUSystem, EvaluatorTrackB
from PIL import Image

class MyISUSystem(ISUSystem):
    """Custom in-car scene understanding system."""

    def __init__(self):
        self.model = self._load_model()
        self.prompt = None

    def _load_model(self):
        # Load your model here
        # Could be: fine-tuned VLM, ensemble, CLIP-based, etc.
        pass

    def set_prompt(self, prompt: str) -> None:
        """Called by evaluator with feature vocabulary."""
        self.prompt = prompt

    def predict(self, image: Image.Image) -> dict[str, str]:
        """Predict scene features from RGB image.
        
        Args:
            image: PIL Image in RGB mode
        
        Returns:
            Dictionary mapping feature names to predicted values:
            {"driver_phone": "NO", "baby_seat": "YES", ...}
        """
        # Your inference logic
        features = self.model.predict(image)
        
        # Ensure output matches vocabulary
        predictions = {}
        for key, value in features.items():
            # Normalize to uppercase strings
            predictions[key] = str(value).upper()
        
        return predictions

    def name(self) -> str:
        """Return system identifier for reports."""
        return "MyISUSystem-v1.0"

# Evaluate your system
if __name__ == "__main__":
    system = MyISUSystem()
    evaluator = EvaluatorTrackB(
        data_dir="data",
        isu_system=system,
        output_path="my_results.json"
    )
    report = evaluator.evaluate()
    print(f"Accuracy: {report['metrics']['accuracy']:.1%}")
```

**See:** `track_b/example_custom_system.py` for complete examples with:
- CLIP-based baseline
- Ensemble systems
- Integration patterns

### Building Ensemble Systems

Combine multiple models for improved robustness:

```python
from evaluator_cli import ISUSystem

class EnsembleISUSystem(ISUSystem):
    def __init__(self, models: list[ISUSystem]):
        self.models = models

    def set_prompt(self, prompt: str) -> None:
        for model in self.models:
            model.set_prompt(prompt)

    def predict(self, image) -> dict[str, str]:
        # Get predictions from all models
        all_predictions = [model.predict(image) for model in self.models]
        
        # Ensemble by voting
        ensemble = {}
        for feature in all_predictions[0].keys():
            values = [p[feature] for p in all_predictions]
            ensemble[feature] = max(set(values), key=values.count)
        
        return ensemble

    def name(self) -> str:
        names = ", ".join(m.name() for m in self.models)
        return f"Ensemble[{names}]"
```

---

## Data Formats

### Track A Submission Structure

```
submissions/
├── your_team_name/
│   ├── simulated/           # Original synthetic images (*_sim.png)
│   │   ├── sample_0001_sim.png
│   │   └── ...
│   ├── generated/           # Your transformed images
│   │   ├── sample_0001.png
│   │   └── ...
│   ├── reference/           # Real reference images (for evaluation)
│   │   ├── sample_0001.jpg
│   │   └── ...
│   ├── labels/              # Ground-truth JSON files
│   │   ├── sample_0001.json
│   │   └── ...
│   └── generation_times.json # {"total_seconds": 123.45, "hardware": "..."}
```

**generation_times.json format:**
```json
{
  "total_seconds": 123.45,
  "hardware": "g6e.2xlarge EC2 instance with NVIDIA L40S GPU"
}
```

### Track B Data Structure

**New format (recommended - with real/ folder):**
```
data/
├── real/                    # Real in-car images
│   ├── sample_0001.jpg
│   ├── sample_0002.jpg
│   └── ...
├── images/                  # Synthetic images only
│   ├── sample_0001_sim.png
│   ├── sample_0002_sim.png
│   └── ...
└── labels/                  # Ground-truth labels
    ├── sample_0001.json
    ├── sample_0002.json
    └── ...
```

**Legacy format (still supported):**
```
data/
├── images/                  # Both real (*.jpg) and synthetic (*_sim.png)
│   ├── sample_0001.jpg
│   ├── sample_0001_sim.png
│   └── ...
└── labels/                  # Ground-truth labels
    ├── sample_0001.json
    └── ...
```

**Label JSON format:**
```json
{
  "driver_gender": "MALE",
  "driver_tshirt_color": "WHITE",
  "driver_emotion": "HAPPY",
  "driver_phone": "NO",
  "driver_safety_belt": "YES",
  "passenger_codriver": "NO",
  "codriver_safety_belt": "NO",
  "passenger_rear_seat_left": "YES",
  "passenger_rear_left_tshirt_color": "BLACK",
  "passenger_rear_left_emotion": "HAPPY",
  "passenger_rear_left_safety_belt": "YES",
  "passenger_rear_seat_right": "YES",
  "passenger_rear_right_tshirt_color": "WHITE",
  "passenger_rear_right_emotion": "SERIOUS",
  "passenger_rear_right_safety_belt": "YES",
  "suitcase": "NO",
  "phone_codriver_seat": "YES",
  "phone_codriver_seat_color": "BLACK",
  "colabottle_codriver_seat": "YES",
  "colacan_codriver_seat": "YES",
  "baby_seat": "NO"
}
```

---

## Report Formats

### Track A Final Report

```json
{
  "contestant": "team_name",
  
  "sam": {
    "simulated_vs_generated": {
      "summary": {
        "partition": {
          "boundary_f1": 0.95,
          "matched_iou": 0.88
        }
      },
      "images": [...]
    }
  },
  
  "realism_raw": {
    "generated_vs_real": {
      "summary": {
        "ssim": 0.72,
        "fid": 42.1,
        "kid_mean": 0.35
      },
      "images": [...]
    },
    "simulated_vs_real": {...},
    "metric_directions": {"ssim": "higher", "fid": "lower", ...}
  },
  
  "failure_raw": {
    "failure_rate": 0.15,
    "failures": 2,
    "executed": 13,
    "images": [...]
  },
  
  "final": {
    "failure": {...},
    "realism": {...},
    "feature_diversity": {
      "mean_pairwise_hamming": 0.35,
      "failing_scenes": 2
    },
    "clip_visual_diversity": {
      "mean_cosine_distance": 0.42
    }
  },
  
  "efficiency": {
    "available": true,
    "total_seconds": 1234.5,
    "seconds_per_scene": 94.96,
    "hardware": "g6e.2xlarge EC2 with NVIDIA L40S"
  }
}
```

### Track B Evaluation Report

```json
{
  "track": "B",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "device": "cuda:0",
  "image_kind": "both",
  
  "prompt": "You are an expert...",
  
  "features": {
    "driver_phone": ["NO", "YES"],
    "baby_seat": ["NO", "YES"],
    "driver_safety_belt": ["NO", "YES"]
  },
  
  "metrics": {
    "accuracy": 0.92,
    "correct_features": 184,
    "evaluated_features": 200,
    "unique_values": {
      "driver_phone": 2,
      "baby_seat": 2
    },
    "scene_exact_match": 0.85,
    "latency_seconds_total": 45.2,
    "latency_seconds_mean": 0.226,
    "scenes": 200
  },
  
  "predictions": [
    {
      "image": "sample_0001_sim",
      "image_path": "/path/to/image.png",
      "expected": {
        "driver_phone": "NO",
        "baby_seat": "YES"
      },
      "prediction": {
        "driver_phone": "NO",
        "baby_seat": "YES"
      },
      "feature_matches": {
        "driver_phone": true,
        "baby_seat": true
      },
      "all_features_match": true,
      "inference_seconds": 0.22
    }
  ]
}
```

---

## Complete CLI Reference

### Track A: `python evaluator_cli.py evaluate`

```
Options:
  --submissions-dir DIR      Root directory containing contestant folders
                             (default: submissions)
                             
  --output-dir DIR          Directory for evaluation output
                             (default: track_a_evaluation_output)
                             
  --failure-model {moondream,qwen}
                             VLM for failure detection
                             - moondream: lightweight, CPU-friendly
                             - qwen: stronger, GPU recommended
                             (default: moondream)
                             
  --device {cpu,cuda,mps}    Torch device for model inference
                             (default: cpu)
```

**Examples:**
```bash
# Basic evaluation
python evaluator_cli.py evaluate

# Custom locations
python evaluator_cli.py evaluate \
    --submissions-dir /path/to/subs \
    --output-dir /path/to/results

# Qwen on GPU
python evaluator_cli.py evaluate \
    --failure-model qwen \
    --device cuda

# Moondream on CPU
python evaluator_cli.py evaluate \
    --failure-model moondream \
    --device cpu
```

### Track B: `python evaluator_cli.py evaluate`

```
Options:
  --data-dir DIR            Directory with images/ and labels/ subdirectories
                             Supports both new (with real/) and legacy formats
                             (default: data)
                             
  --image-kind {real,synthetic,both}
                             Types of images to evaluate
                             - real: only from data/real/ or data/images/ *.jpg
                             - synthetic: only from data/images/ *_sim.png
                             - both: all images
                             (default: both)
                             
  --model {qwen,moondream}  ISU model to use for evaluation
                             (default: qwen)
                             
  --device {cuda,cpu,mps}    Torch device for inference
                             (default: cuda)
                             
  --output FILE             Path for evaluation report
                             (default: track_b_evaluation_output/report.json)
                             
  --ignore-feature FEAT      Feature keys to exclude from scoring
                             Repeatable: --ignore-feature X --ignore-feature Y
```

**Examples:**
```bash
# Basic evaluation (both real and synthetic)
python evaluator_cli.py evaluate --data-dir data

# Real images only
python evaluator_cli.py evaluate \
    --data-dir data \
    --image-kind real

# Custom output
python evaluator_cli.py evaluate \
    --data-dir data \
    --output my_report.json

# Ignore features
python evaluator_cli.py evaluate \
    --data-dir data \
    --ignore-feature light_front \
    --ignore-feature env_strength

# CPU evaluation
python evaluator_cli.py evaluate \
    --data-dir data \
    --model moondream \
    --device cpu
```

---

## Troubleshooting

### Common Issues

#### "Module not found" / "ImportError"

**Problem:** Missing dependencies or running from wrong directory

**Solution:**
```bash
# Install requirements
cd track_a  # or track_b
pip install -r requirements.txt

# Run from correct directory
python evaluator_cli.py evaluate
```

#### "Cannot find images/labels"

**Problem:** Incorrect data directory structure

**Solution - Track B:**
```bash
# Check directory structure
ls data/images/
ls data/labels/
ls data/real/  # for new format

# Ensure files exist
ls data/images/*_sim.png    # Synthetic: should find files
ls data/real/*.jpg          # Real: should find files
ls data/labels/*.json       # Labels: should find files
```

**Solution - Track A:**
```bash
# Check submission structure
ls submissions/your_team/simulated/
ls submissions/your_team/generated/
ls submissions/your_team/reference/
ls submissions/your_team/labels/
```

#### GPU out of memory (OOM)

**Problem:** Model too large for GPU

**Solution:**
```bash
# Use CPU instead
python evaluator_cli.py evaluate --device cpu

# Or use lighter model
python evaluator_cli.py evaluate --model moondream --device cpu

# Or evaluate subset first
python evaluator_cli.py evaluate --image-kind synthetic  # just one kind
```

#### Model download fails

**Problem:** Internet issue or Hugging Face authentication

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Authenticate with HF
huggingface-cli login

# Try again
python evaluator_cli.py evaluate
```

#### CUDA not detected

**Problem:** PyTorch CUDA support not installed

**Solution:**
```bash
# Use CPU
python evaluator_cli.py evaluate --device cpu

# Or reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### File permission denied

**Problem:** Output directory not writable

**Solution:**
```bash
# Check permissions
ls -ld results/

# Create directory with proper permissions
mkdir -p results/
chmod 755 results/

# Or use different output directory
python evaluator_cli.py evaluate --output-dir /tmp/results
```

### Getting Help

1. Check this guide's [Troubleshooting section](#troubleshooting)
2. Review [Quick Start](QUICK_START.md) examples
3. Check example implementations in `track_a/example_*.py` or `track_b/example_*.py`
4. Visit [ISU-Challenge repository](https://isu-challenge.github.io)

---

## Learning Paths

### Path 1: Run Evaluations (5-10 min)
1. Read this guide's [Quick Start](#quick-navigation)
2. Run basic evaluation
3. Customize with CLI flags
4. Check generated reports

### Path 2: Implement Custom System (30-60 min)
1. Read [Implementing Custom Systems](#implementing-custom-systems)
2. Review relevant example file
3. Copy example to start your implementation
4. Integrate with evaluator
5. Test and iterate

### Path 3: Understand Architecture (20-30 min)
1. Read [Architecture Overview](#architecture-overview)
2. Study class hierarchies
3. Review example implementations
4. Explore source code

### Path 4: Debug/Extend (varies)
1. Find issue in [Troubleshooting](#troubleshooting)
2. Review relevant class implementation
3. Check data formats in [Data Formats](#data-formats)
4. Examine report structure in [Report Formats](#report-formats)

---

## File Organization

```
Starterkit-ICSE-2027/
├── QUICK_START.md                    ← 5-minute guide
├── EVALUATION_GUIDE.md               ← This file (complete reference)
├── track_a/
│   ├── evaluator_cli.py              ← Track A CLI
│   ├── example_custom_translator.py  ← Implementation example
│   ├── track_a_metrics.py            ← Core functions
│   └── requirements.txt
└── track_b/
    ├── evaluator_cli.py              ← Track B CLI
    ├── example_custom_system.py      ← Implementation example
    ├── track_b_evaluator.py          ← Core functions
    └── requirements.txt
```
---

For quick onboarding: See [QUICK_START.md](QUICK_START.md)

---

*Last updated: 2026-09-06*  
*For questions, see [ISU-Challenge](https://isu-challenge.github.io)*
