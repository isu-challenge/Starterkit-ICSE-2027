# Quick Start: Using the Evaluation CLI

This guide helps you get started with Track A and Track B evaluations using the new CLI scripts.

## Installation

Install dependencies for each track:

```bash
# Track A
cd track_a
pip install -r requirements.txt

# Track B
cd track_b
pip install -r requirements.txt
```

## Track A: Test Generation

### Evaluate Your Submissions

Generate test scenes and evaluate them:

```bash
cd track_a

# Evaluate all submissions in the submissions/ directory
python evaluator_cli.py evaluate \
    --submissions-dir ../submissions \
    --output-dir results

# Use Qwen for better failure detection (GPU required)
python evaluator_cli.py evaluate \
    --submissions-dir ../submissions \
    --output-dir results \
    --failure-model qwen \
    --device cuda
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `DataLoader` | Abstract base; implement to load custom data |
| `TrackADataLoader` | Default loader for submission structure |
| `Translator` | Abstract base; participants implement custom translators |
| `EvaluatorTrackA` | Static methods for all evaluation stages |
| `ReportWriter` | Save JSON reports |
| `SubmissionEvaluator` | Orchestrates full pipeline |

### Implement a Custom Translator

See `example_custom_translator.py` for complete examples.

```python
from evaluator_cli import Translator
from PIL import Image

class MyTranslator(Translator):
    def translate(self, image: Image.Image, auxiliary_data=None) -> Image.Image:
        # Your transformation logic here
        # IMPORTANT: Preserve semantic structure!
        return transformed_image
```

### Submission Structure

```
submissions/
└── your_team_name/
    ├── simulated/           # Original scenes (*_sim.png)
    ├── generated/           # Your transformed scenes + generation_times.json
    ├── reference/           # Real reference images
    └── labels/              # Ground-truth JSON files
```

### Generated Reports

- `results/your_team_name.json` — Complete evaluation
- `results/summary.json` — All teams comparison

**Key metrics:**
- Failure rate (lower is better for diversity)
- SSIM to real (higher is better)
- Feature diversity (higher is better)
- Generation time (lower is better)

---

## Track B: Perception Robustness

### Evaluate Your System

Implement your ISU system and evaluate:

```bash
cd track_b

# Evaluate with Qwen2.5-VL (GPU recommended)
python evaluator_cli.py evaluate \
    --data-dir ../data \
    --output report.json

# Lightweight CPU evaluation with Moondream
python evaluator_cli.py evaluate \
    --data-dir ../data \
    --model moondream \
    --device cpu \
    --output report_moondream.json

# Evaluate only real images
python evaluator_cli.py evaluate \
    --data-dir ../data \
    --image-kind real \
    --output real_only.json
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `DataLoader` | Abstract base; implement to load custom data |
| `TrackBDataLoader` | Default loader for images+labels |
| `ISUSystem` | Abstract base; participants implement this |
| `QwenISUSystem` | Reference implementation using Qwen |
| `MoondreamISUSystem` | Lightweight reference (stub) |
| `EvaluatorTrackB` | Runs complete evaluation pipeline |
| `ReportWriter` | Save JSON reports |

### Implement a Custom System

See `example_custom_system.py` for complete examples.

```python
from evaluator_cli import ISUSystem
from PIL import Image

class MyISUSystem(ISUSystem):
    def predict(self, image: Image.Image) -> dict[str, str]:
        # Return predictions like:
        # {"driver_phone": "NO", "baby_seat": "YES", ...}
        return predictions

    def name(self) -> str:
        return "MySystem-v1"

# Evaluate your system
from evaluator_cli import EvaluatorTrackB

system = MyISUSystem()
evaluator = EvaluatorTrackB(
    data_dir="data",
    isu_system=system,
    output_path="my_results.json"
)
report = evaluator.evaluate()
```

### Data Format

**Recommended (separate real/ folder):**
```
data/
├── real/                    # Real images
│   ├── sample_0001.jpg
│   └── ...
├── images/                  # Synthetic images (*_sim.png)
│   ├── sample_0001_sim.png
│   └── ...
└── labels/                  # Ground-truth JSON
    ├── sample_0001.json
    └── ...
```

**Also supported (legacy):**
```
data/
├── images/                  # Both real (.jpg) and synthetic (*_sim.png)
│   ├── sample_0001.jpg
│   ├── sample_0001_sim.png
│   └── ...
└── labels/
    ├── sample_0001.json
    └── ...
```

### Generated Reports

- `report.json` — Complete evaluation with predictions
- Console output shows metrics summary

**Key metrics:**
- Accuracy (percentage of correct predictions)
- Scene exact-match (all features correct)
- Latency (seconds per scene)
- Per-feature accuracy breakdown

---

## Common Tasks

### Test with a Small Dataset

Track B only:
```bash
python evaluator_cli.py evaluate --data-dir data --output quick_test.json
```

### Compare Multiple Systems

```python
from evaluator_cli import EvaluatorTrackB, QwenISUSystem
from pathlib import Path

systems = [QwenISUSystem()]  # Add your systems here

results = {}
for system in systems:
    evaluator = EvaluatorTrackB(
        data_dir="data",
        isu_system=system,
        output_path=f"results/{system.name()}.json"
    )
    results[system.name()] = evaluator.evaluate()

# Compare results
for name, report in results.items():
    print(f"{name}: {report['metrics']['accuracy']:.1%} accuracy")
```

### Ignore Specific Features

Track B only (useful for debugging):
```bash
python evaluator_cli.py evaluate \
    --data-dir data \
    --ignore-feature light_front \
    --ignore-feature env_strength \
    --output no_lighting.json
```

---

## Troubleshooting

### "Model not found" error

Models download automatically from Hugging Face on first run. Ensure:
```bash
pip install transformers torch
huggingface-cli login  # If needed for gated models
```

### GPU memory error (OOM)

```bash
# Use CPU instead
python evaluator_cli.py evaluate --device cpu

# Or use lighter model
python evaluator_cli.py evaluate --model moondream --device cpu
```

### "No matching images and labels found"

Check your data directory structure:
```bash
# Track B expects:
ls data/images/sample_*.{jpg,png}
ls data/labels/sample_*.json
```

### Segmentation fault on SAM

For Track A, SAM uses GPU memory. Try:
```bash
# Specify GPU device
python evaluator_cli.py evaluate --device cuda:0
```

---

## Next Steps

1. **Track A:** Read `example_custom_translator.py` for test generation patterns
2. **Track B:** Read `example_custom_system.py` for ISU system patterns
3. **Full Details:** See `EVALUATION_CLI.md` for complete reference
4. **Notebooks:** Original Jupyter notebooks in `track_a/` and `track_b/` for exploration

---

## Questions?

- See `EVALUATION_CLI.md` for detailed API reference
- Check example files for implementation patterns
- Review original Jupyter notebooks for evaluation details
- Contact: [ISU-Challenge](https://isu-challenge.github.io)
