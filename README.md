 AWS Academy — Lab 3.3: Encoding Categorical Variabl

## 1. Información general

**Estudiante:** Jhossuant Cabezas  
**Correo institucional:** jcabezas45@unisalle.edu.co  
**Actividad:** Actividad 5  
**Laboratorio:** 3.3 — Encoding Categorical Variables  
**Plataforma:** AWS Academy / Amazon SageMaker  
**Lenguaje:** Python  
**Biblioteca principal:** Pandas  

---

## 2. Objetivo del laboratorio

El objetivo de este laboratorio es aprender a transformar variables
categóricas en representaciones numéricas que puedan ser utilizadas
por algoritmos de aprendizaje automático (*Machine Learning*).

Durante el desarrollo se trabajan dos estrategias de codificación:

- Codificación ordinal: convierte categorías que representan cantidades
  ordenables en sus equivalentes numéricos.
- Codificación de variables no ordinales: representa las categorías
  mediante variables binarias usando el método `get_dummies()` de Pandas.

Para desarrollar el ejercicio se utiliza el conjunto de datos Automobile
del repositorio de Machine Learning de la Universidad de California en
Irvine (UCI).

## 3. Herramientas y conjunto de datos

### Herramientas utilizadas

- AWS Academy y Amazon SageMaker para ejecutar el notebook.
- Python como lenguaje de programación.
- Pandas para cargar, explorar y transformar los datos.
- Jupyter Notebook para ejecutar el código y visualizar los resultados.
- Git y GitHub para versionar y documentar el trabajo.

### Dataset

El archivo utilizado es `imports-85.csv`, correspondiente al conjunto
de datos de automóviles.

El archivo contiene información sobre características de los vehículos,
como el tipo de aspiración, número de puertas, tracción y cilindros,
además de otras características técnicas.

En la versión utilizada por el cuaderno del laboratorio se obtienen
205 registros y 25 columnas.

Fuente del conjunto de datos:
https://archive.ics.uci.edu/dataset/10/automobile

## 4. Desarrollo paso a paso

### Paso 1. Importación y exploración de los datos

Se importa Pandas y se configuran opciones de visualización para facilitar
la inspección de los resultados.

```python
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


Como el archivo de datos no contiene encabezados, se define una lista
con los nombres de las columnas y posteriormente se carga el archivo
en un DataFrame.
col_names = [    'symboling', 'normalized-losses', 'fuel-type', 'aspiration',    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',    'wheel-base', 'length', 'width', 'height', 'curb-weight',    'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',    'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',    'city-mpg', 'highway-mpg', 'price']df_car = pd.read_csv(    "imports-85.csv",    sep=",",    names=col_names,    na_values="?",    header=None)



El parámetro na_values="?" permite identificar los signos de
interrogación como valores faltantes.
Para comprobar las dimensiones del conjunto de datos se utiliza
el atributo shape.
    
df_car.shape

Resultado esperado:
(205, 25)


A continuación, se visualizan las primeras cinco filas y los tipos
de datos mediante head() e info().
    
df_car.head(5)df_car.info()


Esta exploración permite identificar columnas numéricas, variables
categóricas y posibles valores faltantes.
    
## Paso 2. Selección de las variables categóricas
    
Para concentrarse en los objetivos del laboratorio, se seleccionan
cuatro columnas:
df_car = df_car[    ['aspiration', 'num-of-doors',     'drive-wheels', 'num-of-cylinders']].copy()df_car.head()



Las variables seleccionadas son:

Variable	        Descripción
aspiration	        Tipo de aspiración del motor.
num-of-doors	    Número de puertas del vehículo.
drive-wheels	    Tipo de tracción del vehículo.
num-of-cylinders	Número de cilindros del motor.
    
Se seleccionan estas columnas porque permiten comparar la codificación
de categorías que representan cantidades con la codificación de
categorías que no tienen un orden numérico inherente.
    
## Paso 3. Codificación de variables ordinales
    
Una variable ordinal contiene categorías que pueden representarse
mediante valores ordenados. En este laboratorio, el número de puertas
y el número de cilindros se convierten a valores numéricos equivalentes.
    
##3.1. Codificación del número de puertas
Primero se identifican las categorías y su frecuencia:

df_car['num-of-doors'].value_counts()

En el conjunto de datos del laboratorio se observan:
- four: 114 registros.
- two: 89 registros.
    
Después se crea un diccionario que relaciona cada categoría con
su número correspondiente.
    
door_mapper = {    "two": 2,    "four": 4}

Se utiliza replace() para crear una columna numérica nueva:

df_car['doors'] = df_car['num-of-doors'].replace(door_mapper)

La nueva columna doors representa el número de puertas mediante
los valores 2 y 4. La columna original se conserva para comparar
la transformación.
    
## 3.2. Codificación del número de cilindros
Se revisan las categorías existentes:
    
df_car['num-of-cylinders'].value_counts()


Las frecuencias del cuaderno son:

Categoría	Cantidad
four	159
six	24
five	11
eight	5
two	4
twelve	1
three	1

A continuación, se crea el mapeador:

cylinder_mapper = {    "two": 2,    "three": 3,    "four": 4,    "five": 5,    "six": 6,    "eight": 8,    "twelve": 12}

Se aplica la transformación:
    
df_car['cylinders'] = df_car[    'num-of-cylinders'].replace(cylinder_mapper)df_car.head()



La columna cylinders contiene la representación numérica del número
de cilindros del vehículo.
    
Resultado de esta etapa: el DataFrame conserva las variables
originales y añade las nuevas columnas numéricas doors y cylinders.
    
##Paso 4. Codificación de variables no ordinales
Las variables nominales contienen categorías que no representan un
orden numérico. En este caso se trabajan drive-wheels y aspiration.
    
Asignar números consecutivos a las categorías podría introducir una
relación de orden que no existe en los datos. Por este motivo se utiliza
la codificación one-hot mediante pd.get_dummies().
    
## 4.1. Codificación del tipo de tracción

Se consultan las frecuencias:
    
df_car['drive-wheels'].value_counts()

Resultados del cuaderno:
- fwd: 120 registros.
- rwd: 76 registros.
- 4wd: 9 registros.
Se utiliza get_dummies() para generar una columna binaria por categoría.
df_car = pd.get_dummies(    df_car,    columns=['drive-wheels'])df_car.head()

Se generan las siguientes columnas:
- drive-wheels_4wd
- drive-wheels_fwd
- drive-wheels_rwd

Cada columna indica si el vehículo pertenece a la categoría
correspondiente. En cada fila, la columna asociada con la categoría
presente tiene valor verdadero y las demás tienen valor falso.
    
Esta representación permite codificar la tracción sin asignar un
orden artificial a las categorías.
    
## 4.2. Codificación del tipo de aspiración

Se comprueban las frecuencias de la variable:
df_car['aspiration'].value_counts()

En el conjunto de datos se observan:
- std: 168 registros.
- turbo: 37 registros.
    
Como existen únicamente dos categorías, se utiliza drop_first=True
para conservar una sola columna indicadora:

df_car = pd.get_dummies(    df_car,    columns=['aspiration'],    drop_first=True)df_car.head()

Se genera la columna aspiration_turbo.
Esta columna permite distinguir entre las categorías: un valor verdadero
indica que el vehículo utiliza turbo; un valor falso representa la
categoría de referencia, std.
    
## 5. Resultados obtenidos

Al terminar los pasos del laboratorio se han aplicado dos estrategias
de codificación:
    
1. La transformación mediante diccionarios y replace() para convertir
   el número de puertas y cilindros a valores numéricos.
2. La transformación mediante get_dummies() para representar las
   categorías de tracción y aspiración con variables indicadoras.
       
Las variables originales y las variables nuevas se pueden inspeccionar
mediante:
df_car.head()df_car.info()



La transformación genera las siguientes columnas relevantes:
- doors
- cylinders
- drive-wheels_4wd
- drive-wheels_fwd
- drive-wheels_rwd
- aspiration_turbo
Las columnas originales num-of-doors y num-of-cylinders se conservan
en el DataFrame mostrado por el laboratorio.
    
Los valores finales deben verificarse mediante la ejecución del notebook
en el entorno de AWS.
    
##6. Evidencias del laboratorio
En esta sección se almacenan las capturas de pantalla tomadas durante
la ejecución del laboratorio en AWS SageMaker.
    
Las imágenes deben permitir observar la pantalla completa y mostrar
el código ejecutado junto con sus resultados.
    
Evidencias recomendadas:
Archivo	Pantallazo
Imagen completa de todo el procedimeinto del laboratorio

## 7. Conclusiones
El laboratorio permitió comprender la importancia de convertir las
variables categóricas en representaciones que puedan ser procesadas
por algoritmos de aprendizaje automático.
    
La codificación mediante diccionarios resulta adecuada cuando las
categorías representan cantidades, como el número de puertas y cilindros.
    
Por su parte, la codificación one-hot permite representar categorías
nominales sin introducir un orden numérico artificial.
    
También se practicó la exploración de datos con Pandas, la revisión
de frecuencias mediante value_counts(), la sustitución de categorías
con replace() y la creación de variables indicadoras con
get_dummies().
    
Estos procedimientos constituyen una etapa importante del
preprocesamiento de datos para futuros proyectos de Machine Learning.
    
8. Referencias
Dua, D., & Graff, C. (2019). UCI Machine Learning Repository.
University of California, Irvine, School of Information and Computer
Sciences. https://archive.ics.uci.edu/dataset/10/automobile
pandas. (s. f.). pandas.DataFrame.replace.
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
pandas. (s. f.). pandas.get_dummies.
https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html# actividad-5-aws-labs
