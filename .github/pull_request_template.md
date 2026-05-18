## Summary

Describe the change and why it is needed.

## Type of Change

- [ ] Bug fix
- [ ] Feature
- [ ] Data/DataOps
- [ ] Training/inference behavior
- [ ] Documentation
- [ ] Tests/CI/dev tooling

## Scope

- Affected modules:
- Configs changed:
- Data paths or dataset versions changed:

## Validation

- [ ] `pytest -q`
- [ ] Focused tests:
- [ ] CLI dry-run:
- [ ] Local data audit, if data behavior changed:

## ML/Data Safety Checklist

- [ ] No private face images, checkpoints, cache directories, or generated outputs were committed.
- [ ] No new import-time downloads, extraction, or model loading were added.
- [ ] Default tests remain CPU-safe and do not download diffusion models.
- [ ] If a dataset path changed, the relevant config/docs were updated.
- [ ] If a dependency can reinstall PyTorch/CUDA packages, installation notes were updated.

## Notes for Reviewers

Mention risks, tradeoffs, or follow-up work.

