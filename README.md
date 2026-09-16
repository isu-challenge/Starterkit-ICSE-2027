
<h1 align="center">ISU-Challenge: Benchmarking Vision-Language Models for In-Car Scene Understanding</h1>

<p align="center">
	<a href="https://isu-challenge.github.io"><img src="https://img.shields.io/badge/ISU--Challenge-Competition-42c2ff" alt="ISU-Challenge Competition Website"></a>
	<a href="https://arxiv.org/abs/2607.02300"><img src="https://img.shields.io/badge/ISU--Test-Paper-b31b1b" alt="ISU-Test Paper"></a>
	<a href="https://huggingface.co/datasets/ISU-Test/isu-challenge-dataset"><img src="https://img.shields.io/badge/Hugging%20Face-dataset-yellow" alt="Hugging Face dataset"></a>
</p>

The **ISU-Challenge** is the first competition focused on testing and improving vision-language model (VLM) systems for in-car scene understanding (ISU), held at <a href="https://conf.researchr.org/home/icse-2027"> ICSE 2027</a>.

In-cabin monitoring systems are increasingly important for detecting safety-relevant events such as driver distraction, while also supporting comfort and personalization features. VLMs offer a flexible way to interpret camera images from a vehicle interior, but they can still produce incorrect, incomplete, or overconfident descriptions. These failures are particularly important when a system is expected to answer questions about seat occupancy, seat-belt status, passengers, objects, or driver behavior.

Real-world in-cabin edge cases are expensive and difficult to collect at scale. The ISU-Challenge addresses this problem with a dual-track competition combining realistic scene generation, automated evaluation, and software-testing techniques.

## 🧰 Starter Kit

This repository provides code, data, and documentation for both competition tracks. The starter kit dataset is available on [Hugging Face](https://huggingface.co/datasets/ISU-Test/isu-challenge-dataset).

### 📚 Documentation

**Getting started with evaluation scripts:**

- **[QUICK_START.md](QUICK_START.md)** — 5-minute guide to run evaluations via CLI (recommended first read)
- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** — Complete reference with architecture, API, and troubleshooting
- **[Track A README](track_a/README.md)** — Participant guide for in-car scene translation and evaluation
- **[Track B README](track_b/README.md)** — Participant guide for in-car scene understanding and evaluation

**Dataset:**
- [ISU-Challenge Dataset on Hugging Face](https://huggingface.co/datasets/ISU-Test/isu-challenge-dataset) — Synthetic and real in-car images with labels

Do get access the to the real data, please send us an email first and register for the competition.

For evaluation details: See [Track A](#-track-a-test-generation) and [Track B](#-track-b-perception-robustness) sections below.

### Example Data Channels

The starter kit includes synthetic scenes with auxiliary channels for Track A generated with [ISU-Test](https://github.com/ast-fortiss-tum/ISU-Test) [1], as well as real
images for Track B. The examples below use the same synthetic scene where possible so that the
relationship between the RGB image, segmentation map, and Canny image is easy to see.

| Synthetic RGB image | Instance segmentation map | Canny edge map | Real in-car image |
| --- | --- | --- | --- |
| ![Synthetic RGB in-car scene](readme_assets/sample_0001_sim.png) | ![Instance segmentation map](readme_assets/sample_0001_instance_seg.png) | ![Canny edge map](readme_assets/sample_0001_canny.png) | ![Real in-car scene](readme_assets/20251207_110606.jpg) |
| Source scene used by Track A. | False-color mask; see the [instance-segmentation legend](readme_assets/instance_seg_legend.json). | Edge information extracted from the synthetic scene. | Example real image provided for Track B. |

## 🏁 Competition Tracks

### 🧪 Track A: Test Generation
Track A is for participants who develop automated test generators. The goal is to generate realistic, diverse, and challenging in-car scenes that expose failures in VLM-based ISU systems while preserving the semantic and geometric ground truth of the source scene.

Participants use synthetic base scenes together with ground-truth annotations and auxiliary channels, such as depth maps and semantic segmentation maps. Test generators may diversify the visual appearance of the scenes, including:

- textures and materials;
- illumination and environmental conditions;
- passenger appearance and clothing
- other appearance-level variations that do not modify the underlying scene geometry, semantic structure and interior look and feel.

For the generated scenes, the ground-truth annotations of the source scene must remain valid. In particular, appearance transformations must not add, remove, move, or reshape semantically annotated objects, alter their spatial relationships, or otherwise modify the scene structure represented by the ground-truth annotations. For example, a participant may change a person's clothing or appearance, but may not change the person's location, body geometry, or the presence of safety-critical objects such as seat belts.

Transformations that modify the semantic or geometric structure of the scene are not permitted in Track A unless the corresponding ground-truth annotations are generated consistently and submitted with the transformed scene.

**Inputs**: Synthetic base scenes with ground-truth labels and auxiliary channels.

**Outputs**: A generated in-car scene dataset, together with the code required to generate the scenes and, where applicable, the corresponding ground-truth annotations.

**Track A evaluation:**

Download data from [Hugging Face](https://huggingface.co/datasets/ISU-Test/isu-challenge-dataset), then evaluate submissions via CLI:
```bash
cd track_a
python evaluator_cli.py evaluate --submissions-dir submissions --output-dir results
```

See [`track_a/evaluator_cli.py`](track_a/evaluator_cli.py) for CLI interface. Implementation example in [`track_a/example_custom_translator.py`](track_a/example_custom_translator.py).

**Additional resources** in [`track_a/`](track_a/): [Track A README](track_a/README.md), [participant starter notebook](track_a/participant_starter.ipynb), [challenge evaluator notebook](track_a/track_a_challenge_evaluator.ipynb), [metrics implementation](track_a/track_a_metrics.py), and [image metrics utilities](track_a/image_metrics_utils.py).

### 👁️ Track B: Perception Robustness

Track B is for participants who build robust ISU systems. The goal is to answer safety-relevant questions about synthetic and real in-car scenes, including questions about occupants, seat belts, objects, and driver or passenger behavior.

Systems should process visual in-car scenes and return predictions in the prescribed JSON format. The Track B evaluator uses `Qwen/Qwen2.5-VL-3B-Instruct` as its reference model. A starter collection of real in-car images is provided for training and validation, and the final evaluation uses a withheld test set containing both synthetic and real scenes.

**Inputs:** Varied synthetic and real in-car scenes together with the competition questions.

**Outputs:** A model, model parameters, reproducible inference code, and JSON-formatted visual question-answering predictions.

**Track B evaluation:**

Download data from [Hugging Face](https://huggingface.co/datasets/ISU-Test/isu-challenge-dataset), then evaluate an ISU system via CLI:
```bash
cd track_b
python evaluator_cli.py evaluate --data-dir data --output report.json
```

Implement a custom ISU system by extending `ISUSystem`:
```python
from evaluator_cli import ISUSystem, EvaluatorTrackB

class MyISUSystem(ISUSystem):
    def predict(self, image):
        return {"driver_phone": "NO", ...}
    def name(self):
        return "MySystem"

evaluator = EvaluatorTrackB(data_dir="data", isu_system=MyISUSystem())
report = evaluator.evaluate()
```

See [`track_b/evaluator_cli.py`](track_b/evaluator_cli.py) for CLI interface. Implementation example in [`track_b/example_custom_system.py`](track_b/example_custom_system.py).

**Additional resources** in [`track_b/`](track_b/): [Track B README](track_b/README.md), [evaluator notebook](track_b/track_b_evaluator.ipynb), and [Python evaluator](track_b/track_b_evaluator.py).

### 🧩 Features

The benchmark describes each scene through a set of observable features. These labels cover driver behaviour, passenger occupancy, seat-belt usage, and the presence and location of objects in the cabin. Together, they define the questions that Track B systems must address and the semantic information that Track A generators must preserve when changing a scene's visual appearance.

The feature representation is intentionally structured so that every prediction can be compared directly with the ground truth. A scene may contain several simultaneous conditions, such as an occupied rear seat, an unfastened belt, and a suitcase in the rear. This supports both fine-grained failure analysis and diversity measurements across generated scenes.

The full schema is listed below; however, Track A and Track B predictions and evaluations use only the feature subsets specified in their respective notebooks: the [Track A participant starter notebook](track_a/participant_starter.ipynb) and the [Track B evaluator notebook](track_b/track_b_evaluator.ipynb).

The feature schema is summarized below. Values are `null` when the corresponding object or passenger is not present.

| Category | Feature | Possible values | Description |
| --- | --- | --- | --- |
| Environment | `env` | `URBAN`, `HIGHWAY`, `NATURE`, `COUNTRYSIDE` | Scene environment. |
| Driver | `driver_gender` | `MALE`, `FEMALE` | Driver gender. |
| Driver | `driver_tshirt_color` | `BLACK`, `WHITE` | Driver shirt color. |
| Driver | `driver_emotion` | `HAPPY`, `SERIOUS` | Driver facial expression. |
| Driver | `driver_phone` | `NO`, `YES` | Whether the driver is using a phone. |
| Driver | `driver_safety_belt` | `NO`, `YES` | Driver safety-belt status. |
| Passenger occupancy | `passenger_codriver` | `NO`, `YES` | Whether the front-right passenger seat is occupied. |
| Codriver | `passenger_codriver_tshirt_color` | `BLACK`, `WHITE` | Codriver shirt color. |
| Codriver | `passenger_codriver_emotion` | `HAPPY`, `SERIOUS` | Codriver facial expression. |
| Codriver | `passenger_codriver_head_angle` | `-45`, `-30`, `-15`, `0`, `15`, `30`, `45` | Codriver head angle in degrees. |
| Codriver | `codriver_safety_belt` | `NO`, `YES` | Codriver safety-belt status. |
| Passenger occupancy | `passenger_back_seat_left` | `NO`, `YES` | Whether the rear-left passenger seat is occupied. |
| Rear-left passenger | `passenger_rear_left_tshirt_color` | `BLACK`, `WHITE` | Rear-left passenger shirt color. |
| Rear-left passenger | `passenger_rear_left_emotion` | `HAPPY`, `SERIOUS` | Rear-left passenger facial expression. |
| Rear-left passenger | `passenger_rear_left_head_angle` | `-45`, `-30`, `-15`, `0`, `15`, `30`, `45` | Rear-left passenger head angle in degrees. |
| Rear-left passenger | `passenger_rear_left_safety_belt` | `NO`, `YES` | Rear-left passenger safety-belt status. |
| Passenger occupancy | `passenger_back_seat_right` | `NO`, `YES` | Whether the rear-right passenger seat is occupied. |
| Rear-right passenger | `passenger_rear_right_tshirt_color` | `BLACK`, `WHITE` | Rear-right passenger shirt color. |
| Rear-right passenger | `passenger_rear_right_emotion` | `HAPPY`, `SERIOUS` | Rear-right passenger facial expression. |
| Rear-right passenger | `passenger_rear_right_head_angle` | `-45`, `-30`, `-15`, `0`, `15`, `30`, `45` | Rear-right passenger head angle in degrees. |
| Rear-right passenger | `passenger_rear_right_safety_belt` | `NO`, `YES` | Rear-right passenger safety-belt status. |
| Objects | `phone_codriver_seat` | `NO`, `YES` | Whether a phone is present on the codriver seat. |
| Objects | `phone_codriver_seat_color` | `BLACK`, `WHITE` | Phone color. |
| Objects | `colabottle_codriver_seat` | `NO`, `YES` | Whether a cola bottle is present on the codriver seat. |
| Objects | `colacan_codriver_seat` | `NO`, `YES` | Whether a cola can is present on the codriver seat. |
| Objects | `suitcase` | `NO`, `YES` | Whether a suitcase is present. |
| Objects | `suitcase_color` | `ANTRACITE`, `YELLOW` | Suitcase color. |
| Objects | `suitcase_location` | `CO_DRIVER_SEAT`, `REAR_SEAT` | Suitcase location. |
| Objects | `suitcase_pose` | `UPWARDS`, `FLAT` | Suitcase orientation. |
| Child safety | `baby_seat` | `NO`, `YES` | Whether a baby seat is present. |
| Child safety | `baby_seat_orientation` | `FRONT_FACING`, `REAR_FACING` | Baby-seat orientation. |
| Child safety | `baby` | `NO`, `YES` | Whether a baby is present. |
| Driver attributes | `driver_height_m` | `1.6` to `2.0` | Driver height in meters. |
| Driver attributes | `driver_weight_kg` | `50` to `100` | Driver weight in kilograms. |
| Lighting | `light_front` | `0.0` to `0.5` | Front lighting level. |
| Lighting | `light_back_left` | `0.0` to `0.5` | Rear-left lighting level. |
| Lighting | `light_back_right` | `0.0` to `0.5` | Rear-right lighting level. |
| Environment | `env_strength` | `0.5` to `1.0` | Environment intensity. |
| Camera | `exposure_compensation` | `0.0` to `2.0` | Camera exposure compensation. |

#### Example Feature Labels

The following synthetic scene demonstrates how the feature labels apply to an image. The table shows only the features predicted for the relevant track; the complete track-specific prediction subsets are defined in the corresponding notebooks. It shows a male driver wearing a white shirt and safety belt, two rear passengers, and several objects on the front-right seat.

<img src="readme_assets/sample_0001_sim.png" alt="Example scene with feature labels" width="600">

| Feature | Label | Feature | Label |
| --- | --- | --- | --- |
| `driver_safety_belt` | `YES` | `driver_phone` | `NO` |
| `passenger_front_seat_right` | `NO` | `passenger_codriver_tshirt_color` | `null` |
| `passenger_codriver_emotion` | `null` | `codriver_safety_belt` | `NO` |
| `passenger_back_seat_left` | `YES` | `passenger_rear_left_tshirt_color` | `BLACK` |
| `passenger_rear_left_emotion` | `HAPPY` | `rear_left_safety_belt` | `YES` |
| `passenger_back_seat_right` | `YES` | `passenger_rear_right_tshirt_color` | `WHITE` |
| `passenger_rear_right_emotion` | `SERIOUS` | `rear_right_safety_belt` | `YES` |
| `suitcase` | `NO` | `suitcase_color` | `null` |
| `suitcase_location` | `null` | `suitcase_pose` | `null` |
| `phone_codriver_seat` | `YES` | `phone_codriver_seat_color` | `BLACK` |
| `colabottle_codriver_seat` | `YES` | `colacan_codriver_seat` | `YES` |
| `baby_seat` | `NO` | `baby_seat_orientation` | `null` |
| `baby` | `NO` |  |  |

## 📊 Evaluation

Both tracks use executable, automated evaluation pipelines.

### 🧪 Track A Metrics

- **Failure rate:** The number of scenes that expose a failure divided by the number of executed scenes, based on feature matching. The reference system under test is `Qwen/Qwen2.5-VL-3B-Instruct`. A separate industrial system will be used for the final evaluation.
- **Realism:** Apply default sim2real gap assessment techqniques [2].
- **Diversity:** Feature diversity of failing scenes, following ISU-Test. Visual diversity of failing scenes is measured with CLIP; this captures changes such as clothing colour and style.
- **Efficiency:** Generation time. Is to be measured on the standardized `g6e.2xlarge` EC2 instance in final evaluation.

Track A results are ranked using Pareto non-dominance sorting across the competition objectives. In case of a tie, a weighted linear combination is used.

### 👁️ Track B Metrics

- **Accuracy:** Performance of visual question answering predictions; scene & feature level based.
- **Latency:** Time required to process a scene and produce its answer, measured on the predefined EC2 instance with an NVIDIA L40S GPU.
- **Extended evaluation:** Submitted systems are evaluated on an extended real and synthetic dataset.

Track B participants train and evaluate with synthetic image data and a set of 50 real images. Participants are also invited to collect additional real data. The final evaluation uses withheld real and synthetic scenes.

Participants can use [ISU-Test](https://github.com/ast-fortiss-tum/ISU-Test) to generate additional controllable synthetic in-car scenes for Track B development and training. Alternatively, they can use the synthetic and real images provided in the starter kit. Generated or provided scenes should use the competition feature definitions and labels so that predictions can be evaluated with the Track B pipeline.

### 🔍 Evaluation Models and Data

The baseline for Track A is `Qwen/Qwen2.5-VL-3B-Instruct`, and the final evaluation includes an industrial system to assess generalizability.

The starter kit provides 1,000 synthetically generated images and 50 real in-car images. These data are released under the MIT license. Participants can also collect real data for the features defined by the competition.

The exact output schemas, executable evaluation commands, and metric implementations are available in the starter kit.

## 🤝 Participation

The competition is open to participants from software engineering, computer vision, machine learning, and related communities. Teams may participate in one or both tracks.

To register, send the team or participant name, affiliation, and address to [lev.sorokin@tum.de](mailto:lev.sorokin@tum.de) with the subject **ISU-Challenge Registration Track A/B**. Registration details and deadlines will be announced with the competition launch. Provide ideally a Team name in case you participate as a group.

The competition is organized in connection with A* conference ICSE 2027. The top-ranked teams are invited to submit solution papers to the ICSE 2027 Competition Track proceedings. Novel approaches will be invited based on their methodological contribution, even if they do not achieve the highest leaderboard position.

## 📦 Submission Format

Submissions are made through the competition platform are to be evaluated on a withheld test set.

- **Track A:** Generated image archive and reproducible generation code.
- **Track B:** Model files, parameters, inference or training code, and predictions in the required JSON format.

Submissions must pass automated integrity and reproducibility checks. Track A outputs must preserve the required semantic and spatial information. Track B code submissions may be required for verification. All submission will be in addition manually reviewed.

## 🗓️ Timeline

The timeline is as follows:

| Date | Event |
| --- | --- |
| September 2026 | Competition launch and starter-kit release |
| 27 November 2026 | Participant submissions (code & report) |
| 4 December 2026 | Solution papers due |
| 20 January 2027 | Camera-ready deadline |
| 25 April - 1 May 2027 | ICSE 2027 in-person sessions |

## 📝 Paper Selection

We will invite solution papers from the competition participants and accept at most five papers for the ICSE 2027 Competition Track proceedings. Submitted papers will be reviewed using soundness, writing quality and replicability. Approaches outside the top three teams in the track-metric ranking may still be invited for publication if they receive a sufficiently high paper-review score.

The best accompanying papers will be published in the ICSE proceedings, and their authors will be invited to present their results at the conference in Dublin.


# ❓ FAQs

## 🧪 Track A

<details>
<summary>What is a valid test input?</summary>

A valid Track A input is a synthetic in-car scene from the starter kit together with its available auxiliary channels, such as depth maps, semantic segmentation maps, and edge information. Participants may use these inputs to generate visually transformed scenes.

</details>

<details>
<summary>What is the system output?</summary>

The output is a generated in-car scene dataset together with the reproducible code used to generate it. Each generated scene must preserve the semantic content and spatial geometry of its source scene such that the provided ground-truth labels remain valid after transformation.

</details>

<details>
<summary>What does it mean to preserve the ground-truth labels?</summary>

Participants may modify the visual appearance of a scene, but must not modify the underlying semantic or geometric structure represented by the ground truth. In particular, transformations must not add, remove, move, or reshape semantically annotated objects or people, or change their spatial relationships.

For example, participants may change a person's clothing or appearance, or modify the texture and material of an object, provided that its location and geometry remain unchanged.

</details>

<details>
<summary>Am I allowed to generate new labels?</summary>

Participants do not need to generate new labels for transformed scenes. Track A is designed around ground-truth-preserving transformations; therefore, the existing labels must remain valid after transformation.

</details>

<details>
<summary>What happens if a transformation changes the segmentation?</summary>

If a transformation changes the semantic or geometric structure of the scene such that the provided segmentation or other required ground-truth labels are no longer valid, the generated scene does not satisfy the Track A validity requirements.

</details>

<details>
<summary>Can I move, add, or remove objects or people?</summary>

No. Moving, adding, removing, or reshaping semantically annotated objects or people is not permitted in Track A because these operations can invalidate the provided ground-truth labels.

</details>

<details>
<summary>How will realism be assessed?</summary>

Realism is assessed using the published metrics from the challenge and additional image metrics. Generated scenes must also pass semantic-validity checks, including preservation of the relevant objects, people, and spatial information.

</details>

<details>
<summary>Can I use LLMs for image generation?</summary>

Yes. Participants may use LLMs, VLMs, diffusion models, image-to-image translation systems, or other generative techniques, provided that the resulting scenes are realistic, reproducible, and semantically valid. Any required model or API dependencies must be declared in the submission, and applicable token or compute limits must be respected.

</details>

## 👁️ Track B

<details>
<summary>What is an interior scene understanding system?</summary>

An interior scene understanding system analyzes images from inside a vehicle and provides information, such as occupants, seat-belt status, driver behaviour, and objects in the cabin. In Track B, the system returns these predictions using the prescribed feature names and JSON format.

</details>

<details>
<summary>Where can I get real data for training?</summary>

The starter kit provides 50 real in-car images together with the relevant labels. The data are provided for training and validation and are released under the MIT license. Participants may also collect additional real data using the instructions and recording script provided with the challenge. Additional synthetic training data can be generated with [ISU-Test](https://github.com/ast-fortiss-tum/ISU-Test) [1].

</details>

<details>
<summary>What does the interior scene understanding model need to predict?</summary>

The model must predict the observable scene features represented in the label JSON files. These include driver phone use, passenger occupancy, safety-belt status, suitcase presence and location, beverage objects, baby-seat configuration, and related cabin features. The required feature names and allowed values are defined by the provided labels and evaluation prompt.

</details>

<details>
<summary>What format should predictions use?</summary>

Predictions must be returned as strict JSON objects using the prescribed feature names. Each evaluated feature should have one value, selected from the values allowed for that feature.

</details>

<details>
<summary>How is Track B accuracy calculated?</summary>

The evaluator compares each predicted feature with the corresponding ground-truth value in the label JSON. Accuracy is reported at feature level, and scene-level exact match is also reported when all evaluated features for a scene are correct. Missing predictions for labeled features are counted as incorrect.

</details>

<details>
<summary>Which images are used for evaluation?</summary>

Participants use the synthetic and real images supplied in the starter kit for development. The final evaluation uses a withheld extended dataset containing both synthetic and real in-car scenes, so systems should not rely on a single visual domain.

</details>

<details>
<summary>Which model should I use?</summary>

The starter kit includes a baseline using the open-weight `Qwen/Qwen2.5-VL-3B-Instruct` model. The final evaluation is performed on the submitted system, not by requiring every participant to use the baseline model.

</details>

<details>
<summary>How is latency measured?</summary>

Latency is the time required to process an image and produce its Track B prediction. For comparable results, final measurements are performed on the predefined `g6e.2xlarge` EC2 instance with an NVIDIA L40S GPU.

</details>

<details>
<summary>Can I use external data or models?</summary>

Yes, but participants must document external datasets, pretrained models, and dependencies, and provide reproducible inference or training code. Submitted artefacts must comply with the applicable licenses.




</details>

## 📚 Background

The ISU-Challenge builds on [ISU-Test](https://github.com/ast-fortiss-tum/ISU-Test), which can generate controllable synthetic interior scenes and evaluate in-car scene understanding systems. The underlying search-based testing approach is described in [Search-based Testing of Vision Language Models for In-Car Scene Understanding](https://arxiv.org/abs/2607.02300), which demonstrates systematic exploration of in-car scenarios to expose failures in both industrial and open-source VLM-based ISU systems. The competition extends this direction with more varied environments, additional passengers, real in-car data, and a focus on failure-oriented testing of VLMs.

## 📖 Reference

[1] L. Sorokin, C. Yang, K. E. Friedl, and A. Stocco, "Search-based Testing of Vision Language Models for In-Car Scene Understanding," arXiv:2607.02300, 2026. [Online]. Available: [https://arxiv.org/abs/2607.02300](https://arxiv.org/abs/2607.02300). Accepted at the Industry Track of the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026).


[2] S. C. Lambertenghi and A. Stocco, "Assessing Quality Metrics for Neural Reality Gap Input Mitigation in Autonomous Driving Testing," in *Proceedings of the 17th IEEE International Conference on Software Testing, Verification and Validation (ICST)*, 2024. doi: [10.1109/ICST60714.2024.00016](https://doi.org/10.1109/ICST60714.2024.00016).

## ⚖️ License

This repository is licensed under the [MIT License](LICENSE). 

## ✉️ Contact

For questions about the competition, registration, or collaboration, please contact [Lev Sorokin](mailto:lev.sorokin@tum.de) or [Rifaath Ameen](mailto:rifaath.ameen@fau.de).
