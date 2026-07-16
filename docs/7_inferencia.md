# 7. Inferencia

La inferencia combina transformación estructural y detalle:

1. La rama global envejece el rostro completo hacia la edad objetivo.
2. La rama local procesa recortes anatómicos para producir detalle fino.
3. Los recortes se reinsertan con máscaras suaves.
4. Se ajusta color y se mezcla un residual para reducir bordes visibles.

La fachada es [`src/inference/inference_wrapper.py`](../src/inference/inference_wrapper.py). La composición principal está en [`src/inference/global_local_fusion.py`](../src/inference/global_local_fusion.py), y las operaciones deterministas en [`src/inference/deterministic_fusion_ops.py`](../src/inference/deterministic_fusion_ops.py).

La configuración reproducible se encuentra en [`configs/inference/default_inference.yaml`](../configs/inference/default_inference.yaml), y el comando de alto nivel en [`scripts/inference_cli.py`](../scripts/inference_cli.py).

La supervisión FG-NET/AgeDB no cambia este contrato de inferencia: solo mejora los pesos aprendidos por la rama global durante entrenamiento. Por tanto, un checkpoint nuevo continúa entrando al mismo pipeline global-local.

