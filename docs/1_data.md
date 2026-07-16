# 1. Datos

El entrenamiento puede usar tres fuentes. Las dos originales siguen funcionando sin cambios; la tercera es opcional.

| Fuente | Unidad | Función principal | Implementación |
|---|---|---|---|
| Local | recortes de zonas faciales | aprender arrugas y textura por región | [`data/local_path_dataset.py`](../data/local_path_dataset.py) |
| Global | rostro completo con edad | aprender una transformación global condicionada por edad | [`data/global_path_datasets.py`](../data/global_path_datasets.py) |
| Longitudinal | dos fotos reales de la misma identidad | anclar el cambio joven → mayor con supervisión real | [`data/paired_aging_dataset.py`](../data/paired_aging_dataset.py) |

Los dataloaders originales se construyen en [`data/create_data.py`](../data/create_data.py). Sus rutas y subconjuntos versionados se describen en [`data/configs/dataset_versions.yaml`](../data/configs/dataset_versions.yaml).

## FG-NET y AgeDB

La fuente longitudinal admite:

- **FG-NET:** pequeño y sencillo para comprobar la integración.
- **AgeDB:** más grande y heterogéneo; conviene probarlo después de FG-NET.

Para cada identidad se crean solamente pares con `source_age < target_age`. El split de entrenamiento/validación se hace por identidad: una persona nunca queda en ambos grupos.

La configuración vive en:

- [`configs/data/paired_fgnet.yaml`](../configs/data/paired_fgnet.yaml)
- [`configs/data/paired_agedb.yaml`](../configs/data/paired_agedb.yaml)

Cuando `paired_supervision.enabled: true`, el high level hace lo siguiente:

1. Usa `root` si allí ya hay imágenes válidas.
2. Si no, revisa `cache_dir/dataset`.
3. Descarga el ZIP público de Kaggle solo si no existe.
4. Verifica el archivo, extrae de forma segura y escribe `.paired_dataset_complete.json`.
5. En las siguientes corridas reutiliza la extracción.

La caché recomendada es `data/external/paired_aging/`. `download_if_missing: false` permite prohibir cualquier descarga automática.

Las imágenes de [`data/paired_samples`](../data/paired_samples) son solo una muestra visual; el entrenamiento usa el dataset completo de la caché.

