# DataOps Overview

This folder contains the data layer for the dual-scale face aging pipeline. It is organized around reproducibility, governance, and auditability rather than only dataloader code.

## Data Sources

The project uses FFHQ-derived face images and model/annotation-derived metadata:

- local branch: a small repository fixture subset under `data/data_subset`;
- local annotations: translated region-level JSON files under `data/results_labeling`;
- global branch: FFHQ subset ZIP files expected from external storage, usually Drive/Colab;
- global attributes: CSV files under `data/ffhq_predictions`.

The repository does not claim ownership over FFHQ. Users are responsible for complying with the original dataset terms, any institutional review requirements, and any privacy constraints that apply to face data.

## DataOps Structure

```text
data/
|-- configs/                 Dataset version definitions and audit thresholds
|-- data_subset/             Local fixture image ZIP
|-- ffhq_predictions/        Global attribute CSVs
|-- governance/              Dataset governance and intended-use notes
|-- manifests/               Generated reproducibility manifests
|-- preprocessing/           Image quality and preprocessing audit pipeline
|-- results_labeling/        Local region-level annotation ZIP/JSON files
|-- schemas/                 JSON schemas for manifests and annotations
`-- snapshots/               Dataset snapshot summaries
```

## Reproducibility Contract

A dataset version should define:

- source files and expected paths;
- extraction directory;
- annotation files;
- manifest output paths;
- quality thresholds;
- known limitations.

Generated manifests should record:

- file path;
- SHA-256 hash;
- dimensions;
- mode/format;
- file size;
- quality metrics;
- annotation linkage;
- preprocessing flags.

## Local Quality Audit

Run the local subset audit:

```bash
python -m data.preprocessing.quality_audit \
  --dataset-version data/configs/dataset_versions.yaml \
  --version local_subset_v1 \
  --output-dir data/manifests
```

This creates:

```text
data/manifests/local_subset_v1_quality_manifest.csv
data/manifests/local_subset_v1_quality_manifest.json
data/manifests/local_subset_v1_snapshot.json
```

## Governance

See [governance/DATA_GOVERNANCE.md](governance/DATA_GOVERNANCE.md).

