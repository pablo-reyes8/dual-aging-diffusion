# 5. Losses

El entrenamiento no calcula todos los términos costosos en cada forward. [`src/training/train_aging_model.py`](../src/training/train_aging_model.py) muestrea qué objetivo usar según las probabilidades del YAML.

## Rama local

Implementación: [`src/loss/local_loss.py`](../src/loss/local_loss.py).

`L_local = λ_full L_diff + λ_zone L_zone + λ_score L_score + λ_cycle L_cycle`

- `L_diff`: predicción de ruido del recorte completo.
- `L_zone`: enfatiza la zona anatómica solicitada.
- `L_score`: favorece detalle útil según ScoreNet.
- `L_cycle`: reconstrucción adicional costosa; recomendado `0.0`.

## Rama global

Implementación: [`src/loss/global_loss.py`](../src/loss/global_loss.py).

`L_global = λ_diff L_diff + λ_id L_id + λ_age L_age + λ_delta L_delta + λ_perc L_perc`

- `L_diff`: objetivo estándar de difusión.
- `L_id`: mantiene la identidad entre fuente y predicción.
- `L_age`: acerca la edad estimada a la edad objetivo.
- `L_delta`: acerca el cambio estimado a `target_age - source_age`.
- `L_perc`: LPIPS opcional; recomendado apagado para no aumentar costo sin validación previa.

Los pesos recomendados son `1.0 / 0.35 / 0.10 / 0.15 / 0.0`. Se usa Min-SNR para reducir el dominio de timesteps poco informativos.

## Supervisión longitudinal opcional

Implementación: [`src/loss/paired_supervision_loss.py`](../src/loss/paired_supervision_loss.py).

`L_pair = λ_target L_diff(target | target_age) + λ_source L_diff(source | source_age) + λ_latent L_delta_latent`

El término principal entrena la rama global con una foto real posterior de la misma persona y edad conocida; el endpoint source de esa identidad actúa como regularización. No se afirma alineación espacial ni se concatena la foto source al UNet. `λ_latent = 0` es deliberado porque las fotos no están registradas.

`weight` escala todo `L_pair`; `every_n_steps` controla su frecuencia. FG-NET comienza en `0.25` cada 4 pasos y AgeDB en `0.20` cada 4 pasos.
