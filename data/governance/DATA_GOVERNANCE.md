# Data Governance

This project works with face images and age-related labels. That makes the data layer sensitive even when the images come from public research datasets.

## Source

The image source is FFHQ-derived data. The global branch is designed around an FFHQ subset stored outside the repository. The local branch includes a small fixture subset for development and reproducibility checks.

The project also includes derived metadata:

- apparent age predictions;
- gender labels/predictions;
- skin tone labels;
- hair/glasses labels;
- enriched prompts;
- local facial-region aging scores and bounding boxes.

These labels are derived or annotated signals. They should not be treated as ground truth demographic identity.

## Intended Use

The intended use is research and engineering experimentation around controllable face aging with diffusion models.

The data should be used for:

- dataloader development;
- image quality validation;
- local/global branch training experiments;
- reproducibility and audit checks;
- controlled model evaluation.

## Out of Scope

The data should not be used for:

- identity verification;
- demographic classification products;
- decision-making about real people;
- surveillance or biometric matching;
- claims about biological age or health.

## Sensitive Attributes

Some CSV fields refer to gender, skin tone, hair, glasses, and apparent age. These fields are noisy model-derived attributes and are included only to support prompt construction and training diagnostics.

When reporting results, avoid overclaiming demographic validity. Treat these fields as weak conditioning metadata.

## Versioning Expectations

Every dataset snapshot should define:

- image source files;
- annotation source files;
- extraction directory;
- generated manifest;
- quality thresholds;
- creation time;
- code version when available.

## Retention

Large raw datasets and generated training outputs should generally remain outside version control. Small fixtures and manifests may be versioned when they are useful for reproducibility.

