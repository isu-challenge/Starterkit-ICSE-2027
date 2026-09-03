# TRACK A SUBMISSION GUIDE

Submit one translated image for every provided simulated image. Your model may change appearance, lighting, color, and texture, but it must preserve the objects and geometry of the original scene.

## Submission layout

Each team receives one folder under `submissions`:

```text
submissions/
└── your_team_name/
    ├── simulated/
    │   ├── image_001.png
    │   └── image_002.png
    ├── generated/
    │   ├── image_001.png
    │   └── image_002.png
    └── reference/
        ├── image_001.png
        └── image_002.png
```

`simulated` contains the source images, `generated` is where you place your translated outputs, and `reference` contains the real target-domain images supplied for evaluation. Do not modify `simulated` or `reference`.

## File names

Every submitted image must have the same filename stem as its corresponding simulated image:

```text
submissions/your_team_name/simulated/scene_0042.png
submissions/your_team_name/generated/scene_0042.png
```

The extensions may differ, but using PNG is recommended. Supported formats are PNG, JPEG, BMP, and TIFF. Filename stems must be unique.

Files in `simulated` and `generated` must pair by filename stem. Reference images are evaluated as a distribution and do not need to share those stems. Do not rename or omit provided simulated scenes.

Moondream outputs are recorded for every image. Use matching filename stems in all three folders to receive a direct simulated/generated/reference comparison. The report includes captions, parsed distraction scores, score differences, face counts, and person counts.

## Image requirements

- Submit RGB images.
- Keep the original width, height, framing, and camera viewpoint.
- Do not crop, rotate, mirror, or change the scene geometry.
- Do not remove, add, or move scene objects.
- Do not include semantic masks, model files, or metadata in the team folder.

Images with a different resolution are resized for evaluation, but changing resolution may reduce fidelity and semantic scores.

## Evaluation

Submissions are evaluated on three parts:

- Realism: FID and KID against the supplied real-image reference set (can be less then number of generated images)
- Semantic preservation: class-independent SAM agreement for matched simulated vs. generated images. Boundary F1 and symmetric region covering ignore class IDs and tolerate region splits and merges; matched IoU, ARI, and NMI remain available as diagnostics. Real/reference images are unpaired and are used for distribution-level realism metrics instead.
- Image quality: generated vs. real/reference and simulated vs. real/reference are scored separately. The report also shows the generated improvement over the simulated baseline.
- Content fidelity: SSIM, PSNR, and MSE against the simulated source image.

The full ICST 2024 metric set is reported: Inception Score (IS), FID, KID, SSIM, PSNR, MSE, cosine similarity, texture similarity, Wasserstein distance, KL divergence, histogram intersection, classifier perceptual loss, and a class-agnostic SAM segmentation score. Moondream runs on simulated, generated, and real/reference images. No predefined class dictionary is required.

Lower values are better for FID, KID, and MSE. Higher values are better for the segmentation scores, SSIM, and PSNR.

The organizer runs the evaluator. Contestants only need to provide their translated images in the required folder structure and the code for the translation. 

The translation and evaluation will be performed on a hidden dataset, in addition.
