# 6. Entrenamiento con YAML

El high level está en [`scripts/train_cli.py`](../scripts/train_cli.py). Es la misma ruta que utiliza [`notebooks/train_model_yaml.ipynb`](../notebooks/train_model_yaml.ipynb), por lo que terminal y Jupyter no mantienen pipelines distintos.

## Configuraciones listas

| YAML | Uso |
|---|---|
| [`default_train.yaml`](../configs/training/default_train.yaml) | flujo original, sin datos longitudinales |
| [`paired_fgnet_train.yaml`](../configs/training/paired_fgnet_train.yaml) | primera prueba supervisada; descarga/reutiliza FG-NET |
| [`paired_agedb_train.yaml`](../configs/training/paired_agedb_train.yaml) | prueba grande; descarga/reutiliza AgeDB |

Los dos experimentos usan `_base_: default_train.yaml`: solo declaran las diferencias. El merge recursivo está en [`scripts/common.py`](../scripts/common.py).

## Flujo recomendado en Jupyter

```python
from scripts.train_cli import load_training_config, run_training

config = load_training_config("configs/training/paired_fgnet_train.yaml")
result = run_training(config)
```

Para una prueba corta se pueden modificar valores en memoria antes de llamar `run_training`, sin editar el YAML:

```python
config["training"]["local_max_batches"] = 2
config["training"]["global_max_batches"] = 2
config["training"]["local_num_epochs"] = 1
config["training"]["global_num_epochs"] = 1
```

## Terminal

Primero valide sin modelos, datos ni descarga:

```bash
python -m scripts.train_cli --config configs/training/paired_fgnet_train.yaml --dry-run
```

Luego entrene:

```bash
python -m scripts.train_cli --config configs/training/paired_fgnet_train.yaml
```

## Qué modificar

- `run`: nombre y carpeta de checkpoints.
- `models` y `adapters`: backbone, LoRA/DoRA y learning rates.
- `losses`: pesos internos de las ramas.
- `training`: épocas, probabilidades, acumulación y límites de batches.
- `paired_supervision`: encendido y YAML longitudinal elegido.
- `sampling`: muestras visuales periódicas.

Los parámetros propios del dataset longitudinal —caché, pares, split, frecuencia y peso— deben cambiarse en `configs/data/paired_*.yaml`. Así una corrida queda auditable en un solo par de archivos.

