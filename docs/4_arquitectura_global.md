# 4. Arquitectura global

La rama global procesa la cara completa. Debe cambiar la edad aparente manteniendo identidad, composición y rasgos generales. Es también la rama que recibe la supervisión longitudinal opcional.

## Componentes

1. El backbone recomendado es `Realistic_Vision_V6.0_B1_noVAE` con `sd-vae-ft-mse` explícito.
2. El UNet base permanece congelado y se entrenan adaptadores **LoRA** en atención (`to_q`, `to_k`, `to_v`, `to_out.0`).
3. La condición textual incluye la edad objetivo.
4. [`GlobalAgingLoss`](../src/loss/global_loss.py) mezcla difusión e información semántica de edad e identidad.
5. [`GlobalLossAuxBundle`](../src/loss/global_aux_bundle.py) encapsula los estimadores auxiliares congelados.

La creación y congelamiento de bundles se implementa en [`src/diffusion_pipeline/load_diffusion_models.py`](../src/diffusion_pipeline/load_diffusion_models.py); LoRA está en [`src/diffusion_pipeline/LoRa.py`](../src/diffusion_pipeline/LoRa.py).

## Decisiones conservadoras

- LoRA `rank: 8`, `alpha: 8`, `dropout: 0.05`.
- Learning rate global `5e-5`, menor que el local.
- Dos épocas globales frente a cinco locales.
- La pérdida semántica se calcula en timesteps bajos (`5–120`) para no interpretar como cara una muestra excesivamente ruidosa.
- `delta_age_target_mode: chronological_gap` evita contar dos veces la edad de origen: el objetivo es `target_age - source_age`.

La rama longitudinal añade un paso supervisado separado cada cuatro batches regulares. No reemplaza el flujo global original ni se ejecuta cuando está desactivada.

