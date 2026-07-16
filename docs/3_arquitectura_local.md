# 3. Arquitectura local

La rama local aprende cambios finos en zonas faciales: frente, ojos, mejillas, labios y otras regiones anotadas. Su objetivo es aportar textura localizada sin decidir por sí sola la transformación completa de la cara.

## Componentes

1. Un backbone de Stable Diffusion recibe el recorte y una condición textual de envejecimiento.
2. El UNet base permanece congelado y se entrenan adaptadores **DoRA**. La configuración recomendada usa rango 16.
3. [`LDLALocalAgingLoss`](../src/loss/local_loss.py) combina denoising completo, términos por zona y la señal de ScoreNet.
4. [`ScoreNet`](../src/score_net/arquitecture.py) estima calidad/intensidad de los detalles. Su checkpoint se carga en [`src/score_net/load_scorenet.py`](../src/score_net/load_scorenet.py) y está congelado durante este entrenamiento.
5. En inferencia, los recortes modificados vuelven a su posición mediante máscaras suaves y ajuste de color.

La creación del bundle y la inyección de DoRA ocurren en [`src/diffusion_pipeline/load_diffusion_models.py`](../src/diffusion_pipeline/load_diffusion_models.py) y [`src/diffusion_pipeline/DoRa.py`](../src/diffusion_pipeline/DoRa.py).

## Valores recomendados

- DoRA `rank: 16`, `alpha: 16`, `dropout: 0.05`.
- Learning rate local `7e-5`.
- Probabilidades: full `0.50`, score `0.35`, zone `0.15`.
- Pesos: full `1.0`, zone `0.25`, score `0.05`, cycle `0.0`.

Estos valores están centralizados en [`configs/training/default_train.yaml`](../configs/training/default_train.yaml). El notebook no redefine otros valores ocultos.

