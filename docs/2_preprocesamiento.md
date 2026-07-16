# 2. Preprocesamiento

## Datos local y global

[`data/create_data.py`](../data/create_data.py) es la fachada común. Construye los dataloaders y conserva la separación entre ramas:

- **Local:** lee las anotaciones anatómicas, recorta la zona facial requerida, normaliza el recorte y entrega la etiqueta de región. `skip_regions` permite excluir zonas exactas desde YAML.
- **Global:** usa el rostro completo, lo transforma a la resolución esperada por difusión y conserva edad/condición necesarias para el prompt y las pérdidas semánticas.
- **Fused local:** es opcional; agrupa varios recortes de una imagen para evaluar coherencia de la reconstrucción. Está apagado por defecto.

Los adaptadores concretos de Dataset están en [`data/local_path_dataset.py`](../data/local_path_dataset.py), [`data/local_fused_dataset.py`](../data/local_fused_dataset.py) y [`data/global_path_datasets.py`](../data/global_path_datasets.py).

## Datos longitudinales

[`data/paired_aging_dataset.py`](../data/paired_aging_dataset.py) aplica este flujo:

1. Descubre imágenes recursivamente.
2. Extrae identidad y edad del nombre oficial de FG-NET o de la estructura class-wise de AgeDB.
3. Descarta archivos ilegibles y thumbnails menores que `min_image_side`.
4. Separa identidades de train/validation.
5. Crea pares hacia adelante dentro de `min_age_gap` y `max_age_gap`.
6. Limita pares por persona con `max_pairs_per_identity` para evitar que unas pocas identidades dominen.
7. Redimensiona y normaliza ambas imágenes a `[-1, 1]`.

Las fotos longitudinales no están alineadas píxel a píxel: cambian pose, luz, fondo y cámara. Por esa razón `lambda_latent_delta` se mantiene en `0.0`; se usa supervisión de difusión condicionada, no una pérdida L1 entre las dos fotos.

