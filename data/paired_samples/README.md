# Muestras longitudinales

Estas cuatro imágenes permiten inspeccionar la estructura y dimensiones antes de descargar los datasets completos. No son un dataset de entrenamiento.

| Dataset | Identidad | Edades | Archivos |
|---|---|---:|---|
| FG-NET | `048` | 18 → 54 | `fgnet/048A18.JPG`, `fgnet/048A54.JPG` |
| AgeDB | `Abe Vigoda` | 49 → 93 | `agedb/Abe_Vigoda/2325_AbeVigoda_49_m.jpg`, `agedb/Abe_Vigoda/2358_AbeVigoda_93_m.jpg` |

Las muestras conservan los nombres de los mirrors de Kaggle configurados en [`data/paired_aging_dataset.py`](../paired_aging_dataset.py). Revise y respete las condiciones de uso de FG-NET, AgeDB y del mirror correspondiente antes de redistribuirlas.

Para entrenar no copie archivos aquí. Active uno de los YAML `configs/training/paired_*_train.yaml`; el high level descargará y reutilizará el dataset completo bajo `data/external/paired_aging/`.

