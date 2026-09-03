
# ISU-Challenge: Benchmarking Vision-Language Models for In-Car Scene Understanding

The **ISU-Challenge** is the 1st ICSE 2027 competition on testing and improving vision-language-model (VLM) systems for in-car scene understanding (ISU).

In-cabin monitoring systems are increasingly important for detecting safety-relevant events such as driver distraction, while also supporting comfort and personalization features. VLMs offer a flexible way to interpret camera images from a vehicle interior, but they can still produce incorrect, incomplete, or overconfident descriptions. These failures are particularly important when a system is expected to answer questions about seat occupancy, seat-belt status, passengers, objects, or driver behavior.

Real-world in-cabin edge cases are expensive and difficult to collect at scale. The ISU-Challenge addresses this problem with a dual-track competition combining realistic scene generation, automated evaluation, and software-testing techniques.

## 🏁 Competition Tracks

### 🧪 Track A: Test Generation

Track A is for participants who build automated test generators. The goal is to create realistic, diverse, and challenging in-car scenes that expose failures in VLM-based ISU systems while preserving the semantic content and spatial geometry of the source scene.

Participants will work with provided synthetic scenes and auxiliary channels such as depth maps, semantic segmentation maps. These inputs can be used to diversify visual appearance through changes such as:

- textures and materials;
- environmental conditions;
- realistic passenger and object variations,

as long as the generated scenes remain semantically valid. For example, a transformation must not move a person, remove a seat belt, or change the position of an object in a way that invalidates the ground-truth labels. However, the humans appearance or the clothing can change.

**Inputs:** Synthetic base scenes with ground-truth labels and auxiliary channels.

**Outputs:** A realistic in-car scene dataset and the code required to generate it.

### 👁️ Track B: Perception Robustness

Track B is for participants who build robust ISU systems. The goal is to answer safety-relevant questions about synthetic and real in-car scenes, including questions about occupants, seat belts, objects, and driver or passenger behavior.

Systems should process visual in-car scenes and return predictions in the prescribed JSON format. A starter collection of real in-car images will be provided for training and validation, and the final evaluation will use a withheld test set containing both synthetic and real scenes.

**Inputs:** Varied synthetic and real in-car scenes together with the competition questions.

**Outputs:** A model, model parameters, reproducible inference code, and JSON-formatted visual question-answering predictions.

### 🧩 Features

The benchmark describes each scene through a set of observable, safety-relevant features. These labels cover driver behaviour, passenger occupancy, seat-belt usage, and the presence and location of objects in the cabin. Together, they define the questions that Track B systems must answer and the semantic information that Track A generators must preserve when changing a scene's visual appearance.

The feature representation is intentionally structured so that every prediction can be compared directly with the ground truth. A scene may contain several simultaneous conditions, such as an occupied rear seat, an unfastened belt, and a suitcase in the rear. This supports both fine-grained failure analysis and diversity measurements across generated scenes.

An example feature record is shown below:

{
  "driver_phone": "NO",
  "phone_codriver_seat": "NO",

  "passenger_front_seat_right": "NO",
  "passenger_back_seat_left": "NO",
  "passenger_back_seat_right": "YES",

  "front_left_safety_belt": "YES",
  "front_right_safety_belt": "NO",
  "rear_left_safety_belt": "NO",
  "rear_right_safety_belt": "NO",

  "suitcase": "YES",
  "suitcase_location": "REAR_SEAT",
  "colabottle_codriver_seat": "NO",
  "colacan_codriver_seat": "NO",

  "baby_seat": "YES",
  "baby_seat_orientation": "FRONT_FACING",
  "baby": "NO"
}

## 🧰 Starter Kit

This repository provides the starter kit for the ISU-Challenge. It is intended to help participants understand the data format, run the evaluation pipeline, and develop a first baseline for either track.

The starter kit includes:

- synthetic in-car scenes and their annotations;
- auxiliary channels such as depth, segmentation, and edge information;
- example real in-car data for the perception track;
- evaluation scripts for test generators and perception systems;
- Track B dependencies in `track_b/requirements.txt`;

- baseline Track A generators using image augmentation; and
- baseline Track B inference scripts using open-weight VLMs.

The starter kit is available on Kaggle and on the competition website.

## 🔄 Example ISU Workflow

The competition follows the workflow below:

1. A scene is defined using controllable semantic features and ground-truth labels.
2. Track A participants generate a visually diverse version of the scene while preserving its labels and geometry.
3. The scene is submitted to a VLM-based ISU system together with a visual question.
4. The system response is compared with the ground truth.
5. Evaluation aggregates: semantic validity, visual realism, failure-inducing capability, and computational cost according to the selected track (data generation vs. inference).

## 📊 Evaluation

Both tracks use executable, automated evaluation pipelines.

### 🧪 Track A Metrics

- **Failure rate:** The number of scenes that expose a failure divided by the number of executed scenes, based on feature matching. The public system under test is Qwen-3B-Instruct. A separate industrial system will be used for the final evaluation.
- **Realism:** Will be disclosed after evaluation.
- **Diversity:** Feature diversity of failing scenes, following ISU-Test. Visual diversity of failing scenes is measured with CLIP and is private; this captures changes such as clothing colour and style.
- **Efficiency (public):** Generation time. Will be measured on the standardized `g6e.2xlarge` EC2 instance in final evaluation.

Track A results are ranked using Pareto non-dominance sorting across the competition objectives. In case of a tie, a weighted linear combination is used.

### 👁️ Track B Metrics

- **Accuracy (public):** Performance of visual question answering predictions.
- **Feature diversity (public):** Diversity of the features represented in the evaluated scenes.
- **Latency (public):** Time required to process a scene and produce its answer, measured on the predefined `g6e.2xlarge` EC2 instance with an NVIDIA L40S GPU.
- **Extended evaluation (private):** Submitted systems are evaluated on an extended real and synthetic dataset.

Track B participants train and evaluate with synthetic image data and a set of 60 real images collected by the organizers. Participants are also invited to collect additional real data. The final evaluation uses withheld real and synthetic scenes.

Participants can use [ISU-Test](https://github.com/ast-fortiss-tum/ISU-Test) to generate additional controllable synthetic in-car scenes for Track B development and training. Alternatively, they can use the synthetic and real images provided in the starter kit. Generated or provided scenes should use the competition feature definitions and labels so that predictions can be evaluated with the Track B pipeline.

### 🔍 Evaluation Models and Data

The public baseline for Track A is MoonDream:2B, the final evaluation includes an industrial system to assess generalizability.

The organizers provide 1,000 synthetically generated images and 60 real in-car images collected by the organizers. These data are released under the MIT license. Participants may also collect real data for the features defined by the competition.

The exact output schemas, executable evaluation commands, and metric implementations will be included in the starter kit.

## 🤝 Participation

The competition is open to participants from software engineering, computer vision, machine learning, and related communities. Teams may participate in one or both tracks.

To register, send the team or participant name, affiliation, and address to [lev.sorokin@tum.de](mailto:lev.sorokin@tum.de) with the subject **ISU-Challenge Registration**. Registration details and deadlines will be announced with the competition launch. And ideally with a Team name in case you participate as a group.

The competition is organized in connection with A* conference ICSE 2027. The top-ranked teams may be invited to submit solution papers to the ICSE 2027 Competition Track proceedings. Novel approaches may also be invited based on their methodological contribution, even if they do not achieve the highest leaderboard position.

## 📦 Submission Format

Submissions will be made through the competition platform and will be evaluated on a private test set.

- **Track A:** Generated image archive and reproducible generation code.
- **Track B:** Model files, parameters, inference or training code, and predictions in the required JSON format.

Submissions must pass automated integrity and reproducibility checks. Track A outputs must preserve the required semantic and spatial information. Track B code submissions may be required for verification. All submission will be in addition manually reviewed.

## 🗓️ Timeline

The expected timeline is synchronized with ICSE 2027:

| Date | Event |
| --- | --- |
| September 2026 | Competition launch and starter-kit release |
| Early November 2026 | Final competition submission deadline and winner verification |
| 4 December 2026 | Solution papers and organizer reports due |
| 16 December 2026 | Reviewer response for solution papers and organizer reports |
| 6 January 2027 | Revisions due for solution papers |
| 13 January 2027 | Final notification for solution papers |
| 20 January 2027 | Camera-ready deadline |
| 25 April - 1 May 2027 | ICSE 2027 in-person sessions |

## 🏆 Prizes

Prizes are provide for each tracks winner include token packages after the competition. In addition the sponsorship is supporting participant token costs for VLM and LLM use during the competition. 

## 📝 Paper Selection

We will invite solution papers from the competition participants and accept at most five papers for the ICSE 2027 Competition Track proceedings. Submitted papers will be reviewed using soundness, writing quality and replicability. Approaches outside the top three teams in the track-metric ranking may still be invited for publication if they receive a sufficiently high paper-review score.

# ❓ FAQs

## 🧪 Track A

### What is a valid test input?

A valid Track A input is a synthetic in-car scene from the starter kit together with its available auxiliary channels, such as depth, semantic segmentation, and edge information. Participants may use these inputs to generate a visually transformed scene.

### What is the system output?

The output is a generated in-car scene dataset and the reproducible code used to generate it. Each generated scene must remain aligned with the required source scene and its ground-truth labels.

### Am I allowed to generate new labels?

Participants do not need to generate new labels for transformed scenes. The existing labels must remain valid after transformation.

### How will realism be assessed?

Realism is assessed using the published metrics from the challenge and additional private image metrics. The private metrics are used to reduce overfitting to a fixed metric set. Generated scenes must also pass semantic-validity checks, including preservation of the relevant objects, people, and spatial information.

### Can I use LLMs for image generation?

Yes. Participants may use LLMs, VLMs, diffusion models, image-to-image translation systems, or other generative techniques, provided that the resulting scenes are realistic, reproducible, and semantically valid. Any required model or API dependencies must be declared in the submission, and  token or compute limits must be considered.


## 👁️ Track B

### Where can I get real data for training?

The starter kit provides 60 real in-car images collected by the organizers, together with the relevant labels. The data are provided for training and validation and are released under the MIT license. Participants may also collect additional real data using the instructions and recording script provided with the challenge.

### What does the model need to predict?

The model must predict the observable scene features represented in the label JSON files. These include driver phone use, passenger occupancy, safety-belt status, suitcase presence and location, beverage objects, baby-seat configuration, and related cabin features. The required feature names and allowed values are defined by the provided labels and evaluation prompt.

### What format should predictions use?

Predictions must be returned as strict JSON objects using the prescribed feature names. Each evaluated feature should have one value, selected from the values allowed for that feature.

### How is Track B accuracy calculated?

The evaluator compares each predicted feature with the corresponding ground-truth value in the label JSON. Accuracy is reported at feature level, and scene-level exact match is also reported when all evaluated features for a scene are correct. Missing predictions for labeled features are counted as incorrect.

### Which images are used for evaluation?

Participants use the synthetic and real images supplied in the starter kit for development. The final evaluation uses a withheld extended dataset containing both synthetic and real in-car scenes, so systems should not rely on a single visual domain.

### Which model should I use?

The starter kit includes a baseline using the open-weight `Qwen/Qwen2.5-VL-3B-Instruct` model. The final evaluation is performed on the submitted system, not by requiring every participant to use the baseline model.

### How is latency measured?

Latency is the time required to process an image and produce its Track B prediction. For comparable results, final measurements are performed on the predefined `g6e.2xlarge` EC2 instance with an NVIDIA L40S GPU.

### Can I use external data or models?

Yes, but participants must document external datasets, pretrained models, and dependencies, and provide reproducible inference or training code. Submitted artefacts must comply with the applicable licenses.




## 👥 Organizers

- **Lev Sorokin**, BMW Group and Technical University of Munich
- **Rifaath Ameen**, BMW Group and FAU Erlangen-Nuremberg
- **Stefano Carlo Lambertenghi**, Technical University of Munich and fortiss GmbH
- **Chen Yang**, Technical University of Munich.

## 📚 Background

The ISU-Challenge builds on [ISU-Test](https://github.com/ast-fortiss-tum/ISU-Test), which can generate controllable synthetic interior scenes and evaluate in-car scene understanding systems. The underlying search-based testing approach is described in [Search-based Testing of Vision Language Models for In-Car Scene Understanding](https://arxiv.org/abs/2607.02300), which demonstrates systematic exploration of in-car scenarios to expose failures in both industrial and open-source VLM-based ISU systems. The competition extends this direction with more varied environments, additional passengers, real in-car data, and a focus on failure-oriented testing of VLMs.

## 📖 Reference

[1] S. C. Lambertenghi and A. Stocco, "Assessing Quality Metrics for Neural Reality Gap Input Mitigation in Autonomous Driving Testing," in *Proceedings of the 17th IEEE International Conference on Software Testing, Verification and Validation (ICST)*, 2024. doi: [10.1109/ICST60714.2024.00016](https://doi.org/10.1109/ICST60714.2024.00016).

[2] L. Sorokin, C. Yang, K. E. Friedl, and A. Stocco, "Search-based Testing of Vision Language Models for In-Car Scene Understanding," arXiv:2607.02300, 2026. [Online]. Available: [https://arxiv.org/abs/2607.02300](https://arxiv.org/abs/2607.02300). Accepted at the Industry Track of the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026).

## ⚖️ License

See the repository files and the competition guidelines for the applicable licenses and data-use conditions.

## ✉️ Contact

For questions about the competition, registration, or collaboration, please contact [Lev Sorokin](mailto:lev.sorokin@tum.de) or [Rifaath Ameen](mailto:rifaath.ameen@fau.de).
