# AWS Academy — Laboratorio 3.5: Implementación de un modelo

## 1. Información general

- **Estudiante:** Jhossuant Cabezas
- **Correo institucional:** jcabezas45@unisalle.edu.co
- **Laboratorio:** 3.5 — Amazon SageMaker: Implementación de un modelo
- **Repositorio:** https://github.com/Jhossuant0520/actividad-5-aws-labs
- **Rama:** `dev_jcabezas45`
- **Notebook:** `Notebook_Jhossuant_Lab3.5.ipynb`
- **Laboratorio previo:** 3.4 — Formación de un modelo

## 2. Objetivo

Implementar en Amazon SageMaker el modelo de clasificación XGBoost entrenado en el laboratorio 3.4, realizar inferencias con datos de prueba, probar la entrada indicada por el profesor y conocer el proceso de transformación por lotes.

## 3. Procedimiento realizado

### 3.1 Preparación del entorno y datos

Se ejecutó la configuración del notebook del laboratorio 3.5 para cargar las bibliotecas, preparar el conjunto de datos de columna vertebral y recuperar el modelo entrenado (`xgb_model`) desde el flujo de trabajo de SageMaker. El conjunto contiene seis características biomecánicas y una etiqueta de clase.

### 3.2 Despliegue del modelo

Se desplegó el modelo mediante el SDK de SageMaker y se creó el predictor `xgb_predictor`, utilizando una instancia `ml.m4.xlarge` y un serializador CSV, tal como indica el notebook de la actividad. La salida del notebook muestra la creación del modelo, la configuración del endpoint y el endpoint de inferencia.

Código utilizado:

```python
xgb_predictor = xgb_model.deploy(
    initial_instance_count=1,
    serializer=sagemaker.serializers.CSVSerializer(),
    instance_type='ml.m4.xlarge'
)
```

### 3.3 Validación solicitada por el profesor

Se envió al endpoint la siguiente entrada de seis características, sin incluir la etiqueta objetivo:

```python
entrada = "63.0,12.5,45.0,50.5,125.0,3.0"
resultado = xgb_predictor.predict(entrada)
print("Entrada enviada:", entrada)
print("Resultado de la predicción:", resultado)
```

**Resultado observado en la ejecución:** `0.01694011501967907`.

El valor es una puntuación de probabilidad generada por el modelo, no una etiqueta binaria directamente. Con el umbral habitual de 0.5, esta puntuación se asignaría a la clase `0`. En el mapeo definido durante el laboratorio 3.4, la clase `0` corresponde a `Normal` y la clase `1` a `Abnormal`. Por tanto, la lectura del modelo para esta entrada es **Normal (clase 0)**. Es una salida de un modelo académico, no un diagnóstico médico.

### 3.4 Predicciones sobre el conjunto de prueba

El notebook muestra que el conjunto de prueba contiene 31 filas y siete columnas, incluida la etiqueta. Para inferencia se seleccionan únicamente las seis características, se convierten a CSV y se envían al predictor. El resultado de una predicción individual también se muestra como puntuación de probabilidad.

### 3.5 Eliminación del endpoint

Después de registrar la predicción, se ejecutó la instrucción del notebook para eliminar el endpoint y su configuración:

```python
xgb_predictor.delete_endpoint(delete_endpoint_config=True)
```

La salida guardada en el notebook muestra los mensajes de eliminación del endpoint y de su configuración. Como SageMaker cobra recursos de inferencia mientras permanecen activos, es importante verificar en la consola que el endpoint ya no esté activo.

## 4. Transformación por lotes: observaciones

El notebook continúa con una transformación por lotes usando SageMaker y S3. La ejecución registrada muestra la creación del modelo de transformación y del trabajo por lotes. Sin embargo, en la captura general aparecen errores en la lectura del archivo de salida con `pd.read_csv()` y, posteriormente, una referencia a `target_predicted` cuando la variable no quedó definida. Por transparencia, esta sección **no se reporta como completamente validada** y requiere corregir la lectura del CSV de salida antes de afirmar que la conversión binaria por lotes finalizó correctamente.

Estos errores no invalidan la evidencia independiente de despliegue y predicción personalizada mostrada en el notebook, pero sí deben tenerse en cuenta al describir el estado general del ejercicio.

## 5. Evidencias

- `pantallazos/Evidencia_General_Laboratorio_3.5.pdf`: captura multipágina del notebook de SageMaker.
- En las páginas 3 y 4 se observa el despliegue del modelo y la predicción de la entrada solicitada.
- En la página 4 se muestra la eliminación del endpoint.
- En las páginas 4 y 5 se aprecian los pasos de transformación por lotes y los errores pendientes de corrección.

## 6. Conclusiones

El laboratorio permitió continuar el flujo de trabajo del laboratorio 3.4 al desplegar el modelo XGBoost entrenado y consumirlo mediante un endpoint de SageMaker. La entrada solicitada por el profesor devolvió una puntuación de `0.01694011501967907`, que se interpreta como clase `0` (`Normal`) con el mapeo de clases empleado y un umbral de 0.5.

También se identificó la importancia de liberar los recursos de inferencia después de utilizarlos y de verificar por separado cada etapa del flujo. La sección de transformación por lotes quedó con errores en el procesamiento de la salida, por lo que no se considera completamente validada en esta entrega documental.

## 7. Referencias

- Amazon Web Services. (s. f.). *Amazon SageMaker Developer Guide*. https://docs.aws.amazon.com/sagemaker/
- Dua, D. y Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine, School of Information and Computer Science. https://archive.ics.uci.edu/
- UCI Machine Learning Repository. (s. f.). *Vertebral Column Data Set*. https://archive.ics.uci.edu/

## 8. Control de versiones

La entrega debe incorporarse a la rama `dev_jcabezas45`. Antes de subir los archivos, verificar el estado del repositorio y revisar el historial de commits. La guía de la actividad solicita un mínimo de tres commits por laboratorio; los commits deben representar cambios reales y realizarse desde el workspace de AWS asignado al estudiante.
