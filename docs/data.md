# Data Pipeline

The project has two dataset branches: local and global. The branches are intentionally separate because they represent different supervision signals and different image scales.

## Local Branch Dataset

The local branch trains on facial region crops. Each training sample corresponds to one annotated zone from one face image.

The local assets are stored in the repository:

```text
data/data_subset/subset_50.zip
data/results_labeling/results_translated_json.zip
```

The code extracts these assets into local working directories and matches annotations to images by filename stem. For example:

```text
00632_annotations.json -> 00632.png
```

The main local dataset builder is:

```text
data/local_path_dataset.py
```

The high-level dataloader entrypoint is:

```text
data/create_data.py::build_local_dataloaders
```

## Local Sample Contract

The local dataloader returns batches shaped for diffusion training:

```python
{
    "pixel_values": Tensor[B, 3, 256, 256],  # crop in [-1, 1]
    "score": Tensor[B],                      # normalized local aging score in [0, 1]
    "score_raw": Tensor[B],                  # jittered score in 0-100 scale
    "score_original": Tensor[B],             # original annotation score
    "prompt": List[str],                     # full local aging prompt
    "zone_prompt": List[str],                # zone-only prompt
    "region_key": List[str],
    "region_name": List[str],
    "region_alias": List[str],
    "bbox_crop": Tensor[B, 4],
    "image_id": List[str],
}
```

The default local resolution is `256x256`, defined by `LOCAL_RESOLUTION`.

## Local Regions

The local annotations cover facial zones such as:

- forehead;
- glabella;
- crow's feet;
- under-eye wrinkles;
- nasolabial folds;
- marionette lines;
- nasal bridge;
- upper lip.

The sampler uses score-aware and region-aware weighting so high-aging examples and underrepresented anatomical signals can appear more often during training.

## Global Branch Dataset

The global branch trains on full-face images. Its purpose is to learn full-face aging semantics.

The global data paths are intentionally Drive/Colab-oriented by default, because the full global dataset is not stored in the repository. The main global dataset builder is:

```text
data/global_path_datasets.py
```

The high-level dataloader entrypoint is:

```text
data/create_data.py::build_global_dataloaders
```

The global dataset uses an attribute CSV with filenames and age predictions. It builds prompts that describe the apparent age and demographic/image attributes.

## Global Batch Contract

The global dataloader returns batches for full-face diffusion training:

```python
{
    "pixel_values": Tensor[B, 3, 512, 512],  # full face in [-1, 1]
    "age": Tensor[B],
    "age_norm": Tensor[B],
    "prompt": List[str],
    "filename": List[str],
    ...
}
```

The default global resolution is `512x512`, defined by `GLOBAL_RESOLUTION`.

## Auditing Loaders

Loader audits are implemented in:

```text
data/analyze_loaders.py
data/local_utils.py
data/global_utils.py
```

The audit checks:

- dataset sizes;
- unique image counts;
- score distribution;
- score bins;
- score distribution by region;
- sampled loader distribution;
- batch keys and tensor shapes.

Run the local audit with:

```bash
python -m scripts.data_cli --config configs/data/local_data.yaml --branch local
```

Run the global audit in an environment where the Drive paths exist:

```bash
python -m scripts.data_cli --config configs/data/global_data.yaml --branch global
```

## DataOps Quality Manifests

The DataOps layer adds dataset version definitions, schemas, governance notes, and reproducible quality manifests under `data/`.

The local quality audit is implemented in:

```text
data/preprocessing/quality_audit.py
```

It checks:

- blur;
- overexposure;
- underexposure;
- excessive noise;
- low resolution;
- strong compression proxy;
- incorrect crops;
- annotation/image mismatches.

Run:

```bash
python -m data.preprocessing.quality_audit \
  --dataset-version data/configs/dataset_versions.yaml \
  --version local_subset_v1 \
  --output-dir data/manifests
```

The generated manifest records hashes, dimensions, quality metrics, annotation counts, invalid crop counts, region coverage, and flags.

## Important Data Assumptions

The local branch can be run from the repository fixtures. The global branch assumes the external dataset is available at the configured Drive/Colab paths unless the user overrides those paths in code or config.

The tests do not require downloading the global dataset. They use smoke fixtures and shape-level contracts.
