# Losses

The project uses different objectives for the local and global branches because they solve different aging problems.

## Local Branch Loss

The local branch loss is implemented in:

```text
src/loss/local_loss.py
```

The main class is:

```python
LDLALocalAgingLoss
```

It follows an LDLA-style objective for local facial aging:

```text
L_local =
    lambda_full  * L_full
  + lambda_zone  * L_zone
  + lambda_score * L_score
  + lambda_cycle * L_cycle
```

## Local Loss Components

`L_full` trains the crop using the full local prompt. This is the default diffusion reconstruction/denoising objective conditioned on a prompt that includes the region and aging score.

`L_zone` trains the model with a zone-only prompt. This helps the model understand anatomy and local region context even when score conditioning is reduced.

`L_score` decodes a one-step estimate and sends it through a frozen ScoreNet. It encourages generated local crops to match the target local aging score.

`L_cycle` is reserved for cycle-style consistency. It is expensive and is disabled by default.

Default local weights:

```yaml
lambda_full: 1.0
lambda_zone: 0.25
lambda_score: 0.05
lambda_cycle: 0.0
```

## ScoreNet

ScoreNet is the auxiliary local aging score predictor. It is loaded through:

```text
src/score_net/load_scorenet.py
```

ScoreNet receives decoded RGB crops, not latents. It may be frozen, but its forward pass must remain differentiable with respect to the generated image when used inside `L_score`.

Default ScoreNet config:

```yaml
checkpoint_path:
base_channels: 32
dropout: 0.15
strict: true
freeze: true
```

If `lambda_score > 0`, a ScoreNet checkpoint must be provided.

## Global Branch Loss

The global branch loss is implemented in:

```text
src/loss/global_loss.py
```

The main class is:

```python
GlobalAgingLoss
```

It combines diffusion learning with semantic constraints:

```text
L_global =
    lambda_diff      * L_diff
  + lambda_id        * L_id
  + lambda_age       * L_age
  + lambda_delta_age * L_delta_age
  + lambda_perc      * L_perc
```

## Global Loss Modes

The global loss is memory-aware. It supports three conceptual modes:

`diff` uses only the diffusion noise-prediction objective. It requires one UNet forward and no VAE decode.

`semantic` uses semantic losses on a decoded one-step image estimate. It uses age, identity, delta-age, and optional perceptual objectives.

`all` combines diffusion and semantic objectives. This is expensive because it requires multiple large graph components.

The training wrapper samples global modes according to:

```yaml
global_p_diff: 0.55
global_p_semantic: 0.45
global_enable_diff: true
global_enable_semantic: true
global_semantic_components: [age, delta_age, id]
```

## Global Auxiliary Bundle

The semantic global losses use:

```text
src/loss/global_aux_bundle.py
```

This bundle can load frozen auxiliary models for:

- apparent age prediction;
- identity similarity;
- optional perceptual distance.

Default global weights:

```yaml
lambda_diff: 1.0
lambda_id: 0.5
lambda_age: 0.25
lambda_delta_age: 0.25
lambda_perc: 0.0
```

## Prompt Dropout and Neutral Prompts

Both branches use CFG-style prompt dropout. Instead of always training with explicit age/score prompts, the wrapper sometimes replaces the conditioning with a neutral prompt.

This encourages the adapters to distinguish identity and facial content from age-specific information.

Relevant hyperparameters:

```yaml
local_p_neutral: 0.10
local_p_double_full: 0.15
global_p_neutral: 0.10
global_p_double_diff: 0.05
```

## Memory Principle

The losses are intentionally sampled rather than all activated on every batch. LoRA and DoRA reduce optimizer memory, but UNet activations still dominate training memory. For that reason, the wrapper alternates loss modes and can offload inactive branches.

