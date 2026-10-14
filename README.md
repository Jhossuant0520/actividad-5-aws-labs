# AWS Academy — Evidencias de laboratorios de Machine Learning

Repositorio académico de los laboratorios desarrollados en Amazon SageMaker. Cada laboratorio cuenta con su notebook, explicación del procedimiento y evidencias disponibles.

## Laboratorios documentados

| Laboratorio | Tema | Archivos principales |
|---|---|---|
| **3.3** | Encoding Categorical Variables | [`README.md`](README.md) — documentación del laboratorio 3.3; [`Nootebook_JhossuantCabezas.ipynb`](Nootebook_JhossuantCabezas.ipynb) |
| **3.4** | Training a Machine Learning Model | [`README_Lab3.4.md`](README_Lab3.4.md); [`NotebookJhossuantLab3.4.ipynb`](NotebookJhossuantLab3.4.ipynb) |
| **3.5** | Implementación y uso del modelo XGBoost | [`README_Lab3.5.md`](README_Lab3.5.md); [`Notebook_Jhossuant_Lab3.5.ipynb`](Notebook_Jhossuant_Lab3.5.ipynb) |

> **Nota sobre la estructura:** el `README.md` raíz conserva la documentación del laboratorio 3.3 para no perder el contenido ya preparado. La documentación de los laboratorios 3.4 y 3.5 se encuentra en archivos independientes.

## Evidencias

Las capturas y exportaciones completas de los notebooks se guardan en [`pantallazos/`](pantallazos/). Para el laboratorio 3.5 se incluye `Evidencia_General_Laboratorio_3.5.pdf`, una captura multipágina del notebook con el despliegue, la prueba de predicción y los pasos de transformación por lotes.

## Repositorio y rama de trabajo

- **Estudiante:** Jhossuant Cabezas
- **Rama requerida:** `dev_jcabezas45`
- **Repositorio:** https://github.com/Jhossuant0520/actividad-5-aws-labs

---

# AWS Lab 3.3 — Encoding Categorical Variables

## 1. Datos del estudiante

- **Nombre completo:** [Escribe tu nombre completo]
- **Correo institucional:** [Escribe tu correo institucional]
- **Asignatura:** [Nombre de la asignatura]
- **Actividad:** [Número y nombre de la actividad, según indique el profesor]
- **Laboratorio:** 3.3 — Encoding Categorical Variables
- **Fecha de realización:** [AAAA-MM-DD]
- **Repositorio:** [URL del repositorio de GitHub]
- **Rama de trabajo:** `dev_[prefijo_de_tu_correo]`

## 2. Objetivo del laboratorio

Practicar la conversión de variables categóricas a representaciones numéricas para que puedan utilizarse en flujos de trabajo de análisis de datos y *machine learning*. El laboratorio aborda dos estrategias:

1. **Codificación ordinal:** reemplazar categorías que representan una cantidad u orden interpretable por valores numéricos apropiados.
2. **Codificación no ordinal:** representar categorías sin orden inherente mediante variables indicadoras (*one-hot encoding* o variables ficticias).

## 3. Conjunto de datos

Se trabaja con el conjunto de datos **Automobile** del repositorio de aprendizaje automático de la Universidad de California, Irvine (UCI).

- **Fuente:** [UCI Machine Learning Repository — Automobile](https://archive.ics.uci.edu/dataset/10/automobile)
- **Descripción general:** características técnicas de automóviles, clasificación de riesgo de seguro y pérdidas normalizadas.
- **Archivo utilizado en el entorno:** [Escribe el nombre exacto del archivo de datos que utilizaste en AWS].
- **Preparación:** el archivo de origen no necesariamente contiene encabezados; por eso se asignan nombres a las columnas al leerlo, de acuerdo con las instrucciones del cuaderno del laboratorio.

En el cuaderno de referencia se cargan **205 filas y 25 columnas** antes de seleccionar las variables del ejercicio. Para la práctica se conservan inicialmente estas cuatro columnas:

| Variable | Tipo de dato antes de codificar | Tratamiento |
|---|---|---|
| `num-of-doors` | Categórica textual | Mapeo a la cantidad de puertas |
| `num-of-cylinders` | Categórica textual | Mapeo a la cantidad de cilindros |
| `drive-wheels` | Categórica nominal | Codificación one-hot |
| `aspiration` | Categórica nominal | Variable indicadora con `drop_first=True` |

## 4. Procedimiento realizado

> Completa esta sección con base en el cuaderno ejecutado en AWS. Guarda el notebook con sus salidas visibles cuando corresponda y no agregues resultados que no hayas verificado.

### 4.1 Importación y exploración

Se importa `pandas`, se carga el conjunto de datos en un `DataFrame` y se revisa su estructura mediante operaciones como:

- `df_car.shape`: dimensiones del conjunto de datos.
- `df_car.head()`: primeras filas para conocer la estructura y los valores.
- `df_car.info()`: tipos de datos y cantidad de valores no nulos.
- `df_car.columns`: nombres de las columnas disponibles.
- `value_counts()`: frecuencia de las categorías seleccionadas.

**Observaciones obtenidas en mi ejecución:**

[Describe brevemente las dimensiones, los tipos de variables y los valores faltantes que observaste.]

### 4.2 Codificación ordinal

Las columnas `num-of-doors` y `num-of-cylinders` expresan cantidades. En el cuaderno se definen diccionarios de correspondencia y se utiliza `replace()` para crear columnas numéricas nuevas.

Mapeo de puertas empleado:

| Categoría original | Valor numérico |
|---|---:|
| `two` | 2 |
| `four` | 4 |

Mapeo de cilindros empleado:

| Categoría original | Valor numérico |
|---|---:|
| `two` | 2 |
| `three` | 3 |
| `four` | 4 |
| `five` | 5 |
| `six` | 6 |
| `eight` | 8 |
| `twelve` | 12 |

Las columnas resultantes en el ejemplo del laboratorio son `doors` y `cylinders`. Se conserva la columna original para poder comparar los valores textuales con su representación numérica.

**Resultado verificado en mi ejecución:**

[Indica qué comprobaste después de aplicar los mapeos y adjunta la evidencia correspondiente.]

### 4.3 Codificación no ordinal

Las variables `drive-wheels` y `aspiration` no representan una escala numérica ordenada. Asignarles números arbitrarios podría sugerir relaciones que no existen entre sus categorías.

- Para `drive-wheels`, se utiliza `pd.get_dummies()` para crear una columna indicadora por categoría: `drive-wheels_4wd`, `drive-wheels_fwd` y `drive-wheels_rwd`.
- Para `aspiration`, se utiliza `pd.get_dummies(..., drop_first=True)`. En el ejemplo del laboratorio se conserva la columna `aspiration_turbo`; la categoría estándar (`std`) queda representada como la categoría de referencia.
- Según la versión y los parámetros de pandas, las columnas indicadoras pueden mostrarse como `True`/`False` en vez de `1`/`0`. Ambos formatos expresan indicadores binarios.

**Resultado verificado en mi ejecución:**

[Describe las nuevas columnas que aparecen y comprueba una o más filas del resultado.]

### 4.4 Tarea de desafío (si fue solicitada)

La guía propone regresar al conjunto de datos e incorporar otras variables. Esta parte debe marcarse como realizada solo si la desarrollaste.

- **Variables adicionales seleccionadas:** [Escribe las variables].
- **Estrategia aplicada a cada una:** [Explica el mapeo ordinal o la codificación one-hot y por qué corresponde].
- **Resultado:** [Describe lo que comprobaste].

## 5. Evidencias de ejecución

Crea la carpeta `pantallazos/` en el repositorio y agrega capturas tomadas en el entorno AWS. Las capturas deben mostrar la pantalla completa, como lo solicita la rúbrica. Evita recortar la imagen de modo que se pierdan el contexto del notebook o las salidas relevantes.

Nombres sugeridos para los archivos; utiliza solo los que realmente hayas capturado:

| Archivo sugerido | Qué debe mostrar |
|---|---|
| `pantallazos/01_exploracion_dataset.png` | Notebook en AWS con carga del conjunto de datos y exploración (`shape`, `head` o `info`). |
| `pantallazos/02_frecuencias_categorias.png` | Frecuencias de puertas, cilindros, tipo de tracción o aspiración. |
| `pantallazos/03_codificacion_ordinal.png` | Código ejecutado y columnas `doors` y `cylinders`. |
| `pantallazos/04_codificacion_one_hot.png` | Código y resultado de la codificación de `drive-wheels` y `aspiration`. |
| `pantallazos/05_rama_y_commits.png` | Terminal de AWS mostrando la rama de trabajo y el historial de commits. |

### Evidencias incorporadas

- [ ] Nombre completo visible en una celda Markdown del notebook.
- [ ] Exploración inicial del conjunto de datos.
- [ ] Codificación ordinal y resultado.
- [ ] Codificación one-hot y resultado.
- [ ] Rama de trabajo e historial de commits.

## 6. Resultados y análisis

[Redacta aquí tus resultados después de ejecutar el notebook. Puedes explicar cómo cambió la estructura del `DataFrame`, qué columnas se añadieron, cómo se representó cada categoría y si encontraste valores faltantes en las columnas originales. No incluyas afirmaciones que no hayas comprobado.]

## 7. Conclusiones personales

[Escribe entre uno y tres párrafos con tus propias palabras. Explica la diferencia entre codificar categorías que representan cantidades y codificar categorías nominales; menciona qué aprendiste de `replace()` y `get_dummies()` y en qué parte tuviste dificultades, si las hubo.]

## 8. Entorno utilizado

Completa esta información con el entorno real de AWS:

- **Servicio / entorno:** AWS Academy / [nombre exacto del recurso o entorno]
- **Notebook:** [nombre exacto del archivo `.ipynb`]
- **Python:** [versión, si la verificaste]
- **pandas:** [versión, si la verificaste]

## 9. Control de versiones

La rúbrica compartida solicita una rama de trabajo con el patrón `dev_` seguido del identificador derivado del correo institucional, y un mínimo de tres commits por laboratorio realizados desde el recurso aprovisionado en AWS. Confirma con las instrucciones del profesor el nombre exacto del repositorio y de la rama.

Los mensajes de commit deben describir cambios reales. Ejemplos de mensajes posibles, solo cuando correspondan a los cambios hechos:

1. `docs: agrega identificación y documentación inicial del laboratorio`
2. `docs: incorpora evidencias de exploración y codificación ordinal`
3. `docs: documenta codificación one-hot y conclusiones`

No crees commits vacíos ni registres capturas o resultados que no hayas generado. Consulta el historial con `git log --oneline` y verifica la rama con `git branch --show-current`.

## 10. Referencias

- Dua, D. y Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine, School of Information and Computer Science. [https://archive.ics.uci.edu/](https://archive.ics.uci.edu/)
- UCI Machine Learning Repository. (s. f.). *Automobile*. [https://archive.ics.uci.edu/dataset/10/automobile](https://archive.ics.uci.edu/dataset/10/automobile)
- pandas. (s. f.). *pandas.DataFrame.replace*. [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html)
- pandas. (s. f.). *pandas.get_dummies*. [https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

---

**Nota:** esta documentación corresponde al material del **Lab 3.3** compartido en el cuaderno. La rúbrica que se mostró menciona los laboratorios **3.4 y 3.5**; confirma si las mismas exigencias de repositorio, rama, commits y pantallazos también aplican al Lab 3.3.
