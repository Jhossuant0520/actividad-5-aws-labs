# AWS Academy — Laboratorio 3.4: Training a Machine Learning Model

## 1. Información general

- **Estudiante:** Jhossuant Cabezas
- **Correo institucional:** jcabezas45@unisalle.edu.co
- **Actividad:** Actividad 5
- **Laboratorio:** 3.4 — Training a Machine Learning Model
- **Plataforma:** AWS Academy / Amazon SageMaker
- **Lenguaje:** Python
- **Bibliotecas:** Pandas, SciPy, scikit-learn, boto3 y SageMaker SDK
- **Algoritmo:** XGBoost

## 2. Objetivo

Preparar un conjunto de datos biomédicos, dividirlo en conjuntos de entrenamiento, validación y prueba, cargar los archivos a Amazon S3 y entrenar un modelo de clasificación con XGBoost mediante Amazon SageMaker.

Este laboratorio se centra en el entrenamiento. Las pruebas y la evaluación del modelo se realizan en laboratorios posteriores.

## 3. Escenario y conjunto de datos

El escenario plantea utilizar aprendizaje automático para apoyar la detección de anomalías en pacientes ortopédicos. El conjunto de datos contiene seis características biomecánicas y una etiqueta objetivo.

Las seis características son:

1. Incidencia pélvica (`pelvic_incidence`).
2. Inclinación pélvica (`pelvic_tilt`).
3. Ángulo de lordosis lumbar (`lumbar_lordosis_angle`).
4. Inclinación del sacro (`sacral_slope`).
5. Radio pélvico (`pelvic_radius`).
6. Grado de espondilolistesis (`degree_spondylolisthesis`).

La clasificación binaria utiliza `Normal`, codificada como `0`, y `Abnormal`, codificada como `1`. El conjunto contiene **310 registros y 7 columnas**: seis características y la columna objetivo `class`.

Fuente: [UCI Machine Learning Repository — Vertebral Column](http://archive.ics.uci.edu/ml/datasets/Vertebral+Column).

## 4. Desarrollo paso a paso

### Paso 1. Instalar e importar las bibliotecas

Se instala SciPy y se importan las bibliotecas necesarias para descargar y leer los datos, manipularlos y acceder a los servicios de AWS.

```python
!pip install scipy

import warnings
import requests
import zipfile
import io
import pandas as pd
from scipy.io import arff
import boto3

warnings.simplefilter("ignore")
```

**¿Por qué?** Estas bibliotecas permiten procesar el conjunto de datos y trabajar con Amazon S3 y SageMaker.

### Paso 2. Descargar y cargar el conjunto de datos

Se descarga el archivo ZIP del repositorio UCI, se extrae y se carga `column_2C_weka.arff` en un DataFrame.

```python
f_zip = "http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"

r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

data = arff.loadarff("column_2C_weka.arff")
df = pd.DataFrame(data[0])

class_mapper = {b"Abnormal": 1, b"Normal": 0}
df["class"] = df["class"].replace(class_mapper)
```

**¿Por qué?** El modelo necesita datos estructurados y una etiqueta objetivo numérica para aprender a distinguir las clases.

### Paso 3. Explorar los datos

Se comprueban las dimensiones y los nombres de las columnas:

```python
df.shape
df.columns
```

El notebook registra el resultado `(310, 7)`. Esto confirma que hay 310 registros y siete columnas: seis características y la variable objetivo `class`.

**¿Por qué?** La exploración permite verificar la estructura del conjunto de datos antes de prepararlo.

### Paso 4. Mover la columna objetivo a la primera posición

XGBoost espera que, en el formato CSV utilizado en este laboratorio, la etiqueta objetivo esté en la primera columna.

```python
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

df.columns
```

Se verifica que `class` sea ahora la primera columna.

**¿Por qué?** Así el algoritmo puede identificar correctamente la etiqueta objetivo en los archivos de entrenamiento.

### Paso 5. Dividir los datos

Se utiliza `train_test_split` de scikit-learn. El parámetro `stratify` conserva una distribución representativa de las clases y `random_state=42` hace reproducible la división.

Primero se reserva el 20 % de los datos para prueba y validación:

```python
from sklearn.model_selection import train_test_split

train, test_and_validate = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["class"]
)
```

Después se divide ese conjunto restante en dos partes iguales:

```python
test, validate = train_test_split(
    test_and_validate,
    test_size=0.5,
    random_state=42,
    stratify=test_and_validate["class"]
)
```

Se comprueban las dimensiones y la distribución de clases:

```python
print(train.shape)
print(test.shape)
print(validate.shape)

print(train["class"].value_counts())
print(test["class"].value_counts())
print(validate["class"].value_counts())
```

Resultados registrados en el notebook:

| Conjunto | Registros | Columnas |
|---|---:|---:|
| Entrenamiento | 248 | 7 |
| Prueba | 31 | 7 |
| Validación | 31 | 7 |

El conjunto de entrenamiento contiene 168 registros de clase `1` y 80 de clase `0`. Los conjuntos de prueba y validación contienen 21 registros de clase `1` y 10 de clase `0` cada uno.

**¿Por qué?** El conjunto de entrenamiento se usa para ajustar el modelo, el de validación para validarlo durante el entrenamiento y el de prueba queda reservado para una evaluación posterior.

### Paso 6. Preparar y cargar los CSV a Amazon S3

Se definen el bucket, el prefijo y los nombres de los archivos CSV.

```python
bucket = "c221302a5580007l17609262t1w339237998423-labbucket-krriqpxvpubh"
prefix = "lab3"

train_file = "vertebral_train.csv"
test_file = "vertebral_test.csv"
validate_file = "vertebral_validate.csv"
```

Se crea una función para guardar cada DataFrame como CSV en S3:

```python
import os

s3_resource = boto3.Session().resource("s3")

def upload_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False)

    s3_resource.Bucket(bucket).Object(
        os.path.join(prefix, folder, filename)
    ).put(Body=csv_buffer.getvalue())
```

Se cargan los tres conjuntos:

```python
upload_s3_csv(train_file, "train", train)
upload_s3_csv(test_file, "test", test)
upload_s3_csv(validate_file, "validate", validate)
```

`header=False` evita incluir encabezados y `index=False` evita escribir el índice de Pandas.

**¿Por qué?** SageMaker necesita acceder a los archivos de datos almacenados en Amazon S3 para ejecutar el entrenamiento.

### Paso 7. Obtener el contenedor de XGBoost

Se obtiene la imagen de XGBoost correspondiente a la región de AWS de la sesión:

```python
import boto3
from sagemaker.image_uris import retrieve

container = retrieve(
    "xgboost",
    boto3.Session().region_name,
    "1.0-1"
)
```

**¿Por qué?** SageMaker utiliza este contenedor para ejecutar el algoritmo durante el entrenamiento.

### Paso 8. Configurar los hiperparámetros y el estimador

Se definen los parámetros iniciales del modelo:

```python
hyperparams = {
    "num_round": "42",
    "eval_metric": "auc",
    "objective": "binary:logistic"
}
```

`num_round` establece 42 rondas; `eval_metric` indica que se utiliza AUC como métrica y `objective` configura una clasificación binaria.

Se configura el estimador de SageMaker:

```python
import sagemaker

s3_output_location = "s3://{}/{}/output/".format(bucket, prefix)

xgb_model = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path=s3_output_location,
    hyperparameters=hyperparams,
    sagemaker_session=sagemaker.Session()
)
```

**¿Por qué?** El estimador reúne la configuración de infraestructura, el contenedor, los hiperparámetros y la ubicación de salida del modelo.

### Paso 9. Configurar los canales de datos

Se especifican las ubicaciones de los datos de entrenamiento y validación en S3:

```python
train_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train/".format(bucket, prefix, train_file),
    content_type="text/csv"
)

validate_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validate/".format(bucket, prefix, validate_file),
    content_type="text/csv"
)

data_channels = {
    "train": train_channel,
    "validation": validate_channel
}
```

**¿Por qué?** Los canales indican a SageMaker qué datos debe utilizar durante el entrenamiento. En este laboratorio se envían los conjuntos de entrenamiento y validación; el conjunto de prueba queda reservado para después.

### Paso 10. Entrenar el modelo

Se ejecuta el entrenamiento:

```python
xgb_model.fit(
    inputs=data_channels,
    logs=False
)
```

El proceso puede tardar varios minutos mientras SageMaker prepara la instancia, descarga los datos y el contenedor, entrena el modelo y guarda el artefacto resultante.

**Resultado registrado:** el notebook muestra el mensaje `Training job completed`.

**¿Por qué?** La llamada a `fit()` inicia el trabajo de entrenamiento de XGBoost con los datos y parámetros configurados.

## 5. Resultados obtenidos

De acuerdo con las salidas del notebook:

- Se cargó un conjunto de 310 registros y 7 columnas.
- Se dividieron los datos en 248 registros de entrenamiento, 31 de prueba y 31 de validación.
- Se prepararon los tres conjuntos para su almacenamiento en Amazon S3.
- Se configuró XGBoost con el objetivo `binary:logistic`, 42 rondas y la métrica AUC.
- El trabajo de entrenamiento finalizó correctamente con el mensaje `Training job completed`.

**Alcance:** las capturas revisadas muestran el entrenamiento, pero no muestran la creación de un endpoint ni una predicción con `xgb_predictor`. Estas actividades deben documentarse cuando se realicen en el laboratorio correspondiente.

## 6. Evidencias del laboratorio

Guarda las capturas de pantalla en la carpeta `pantallazos/` del repositorio. Se recomienda que sean capturas de pantalla completa y permitan leer el código ejecutado y sus resultados.

Evidencias recomendadas:

1. Importación de bibliotecas y carga de los datos.
2. Exploración de dimensiones y columnas.
3. Reorganización de la columna `class`.
4. División de datos y comprobación de dimensiones.
5. Distribución de las clases.
6. Carga de archivos CSV a Amazon S3.
7. Configuración del contenedor, hiperparámetros y estimador.
8. Configuración de los canales de datos.
9. Resultado del entrenamiento con `Training job completed`.

Comprueba que las capturas estén realmente guardadas en la carpeta antes de afirmar que fueron entregadas.

## 7. Conclusiones

En este laboratorio se preparó un conjunto de datos biomédicos para entrenar un modelo de clasificación binaria. Se exploraron las características, se convirtió la variable objetivo a valores numéricos, se reorganizaron las columnas y se dividieron los registros en conjuntos de entrenamiento, validación y prueba.

Después, los datos se prepararon en formato CSV y se cargaron a Amazon S3. Finalmente, se configuró y ejecutó el entrenamiento de XGBoost en Amazon SageMaker. El notebook registra que el trabajo terminó correctamente.

El entrenamiento es solo una etapa del ciclo de vida de un modelo. Para conocer su rendimiento predictivo se requieren las pruebas y evaluaciones correspondientes en laboratorios posteriores.

## 8. Referencias

- Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine, School of Information and Computer Sciences. http://archive.ics.uci.edu/ml
- UCI Machine Learning Repository. (s. f.). *Vertebral Column Data Set*. http://archive.ics.uci.edu/ml/datasets/Vertebral+Column
- scikit-learn. (s. f.). *train_test_split*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Amazon Web Services. (s. f.). *Amazon SageMaker Documentation*. https://docs.aws.amazon.com/sagemaker/