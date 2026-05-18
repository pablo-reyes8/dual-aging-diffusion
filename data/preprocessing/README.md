# Preprocessing and Quality Audit

This module contains dataset quality checks used before training.

The current audit covers:

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

The audit is intentionally deterministic. It does not modify images. It only reads source assets and writes manifests.

