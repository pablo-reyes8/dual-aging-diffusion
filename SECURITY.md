# Security Policy

## Supported Versions

This repository is pre-release research code. Security fixes target the active default branch.

## Reporting a Vulnerability

Do not open public issues for security-sensitive reports.

Report privately to the maintainer and include:

- affected files, commands, notebooks, or workflows;
- reproduction steps;
- expected impact;
- whether the issue involves data exposure, model artifact exposure, dependency risk, or code execution;
- suggested mitigation, if known.

## Sensitive Data

This project may process face images and demographic-like metadata. Treat all non-public face data as sensitive.

Do not commit:

- private face images or videos;
- private dataset ZIPs;
- generated personal images;
- checkpoints trained on non-public data;
- model caches that contain private artifacts;
- access tokens, API keys, Drive links with credentials, or cloud storage secrets.

## Dependency Safety

Install PyTorch and torchvision from the explicit wheel index appropriate for the environment. Some packages can attempt to reinstall PyTorch through transitive dependencies.

For identity loss, install FaceNet without dependencies:

```bash
pip install --no-deps facenet-pytorch
```

This avoids accidentally replacing the selected CPU/CUDA PyTorch build.

## Model Safety

This repository is for controlled research experiments. It should not be used for identity verification, surveillance, demographic decision-making, or claims about biological age or health.

## CI and Secrets

CI should not require private dataset access, model-provider secrets, or GPU-only resources. If future workflows need secrets, they must avoid printing paths, tokens, signed URLs, or personally identifying data in logs.
