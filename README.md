# Prediccion de egreso por mejoria

Proyecto de ciencia de datos para predecir si un paciente egresara por mejoria. El entregable automatiza la lectura de nuevos archivos Excel, el uso de catalogos, la preparacion de datos y la clasificacion mediante un Random Forest regularizado.

Consulta la [ficha tecnica del modelo](MODEL_CARD.md) para conocer su objetivo, variables, metricas, limitaciones, riesgos y recomendaciones de uso.

## Estructura del entregable

Los tre archivos ejecutables principales estan en la carpeta base del proyecto:

### `3) Ejecutable_en_py.py`

Es la version ejecutable de **Importacion y analisis.ipynb** en donde se realizacon todas las pruebas iniciales de los modelos incluyendo el tratamiento de la informacion a detalle, la eleccion de las variables, el modelado, la eleccion del modelo, la calibracion de los hiperparametos, metricas y salidas.

- Se debo tomar en cuenta que este el origen de todos los codigos, los siguientes son varaciones de este mismo

La recomendacion si se quiere ver los analisis es **Ver el IPYNB o la presentacion ejecutiva disponible en la carpeta**

```bash
python "3) Ejecutable_en_py.py"
```


### `4) entrenar_modelo.py`

Entrena nuevamente el modelo a partir de un archivo Excel. Realiza el flujo completo:

- Lee la hoja `Base de datos` y los catalogos.
- Usa `Estatus` para identificar automaticamente el codigo de egreso por mejoria.
- Usa `Enfermedades` y `Ciudades` para agregar descripciones de los identificadores.
- Genera reportes de analisis exploratorio.
- Limpia los datos mediante imputacion y preparacion dentro de un pipeline.
- Construye la variable objetivo `EGRESO_MEJORIA`.
- Entrena y valida el Random Forest.
- Guarda el modelo, los datos preparados, las metricas y los reportes en `entregable/Salida`.

Ejemplo:

```bash
python "4) entrenar_modelo.py" --input Notebooks/Inputs/Hospitales.xlsx --output entregable/Salida
```

### `5) api_fastapi.py`

Expone el modelo como un servicio web para clasificar un paciente individual.

Endpoints disponibles:

- `GET /health`: verifica que el servicio y el modelo esten disponibles.
- `POST /predict`: devuelve prediccion, probabilidad y clasificacion.
- `GET /docs`: documentacion interactiva de FastAPI.

Ejemplo de inicio mediante Docker, recomendado para conservar el nombre numerado del documento:

```bash
docker compose -f entregable/Docker/docker-compose.yml up --build
```

La API espera un JSON con las variables del modelo:

```json
{
  "DIAS_ESTANCIA": 2,
  "EDAD": 35,
  "GENERO": 1,
  "NIVEL_DE_GRAVEDAD": 3,
  "IMC": 24.5,
  "ES_INDIGENA": 2,
  "ENTIDAD": 15,
  "MES": 8
}
```

## Modulo de apoyo

`funciones_1.py` contiene las funciones compartidas por los dos scripts: lectura de libros Excel, union con catalogos, construccion del objetivo, ingenieria de variables y preparacion de las columnas del modelo.

## Importacion y analisis.ipynb



Notebook reproducible ubicado en la raiz, con lectura, catalogos, EDA, limpieza, ingenieria de variables, comparacion de modelos, calibracion y diagnostico de sesgo-varianza.
- `Notebooks/Outputs/comparacion_modelos.csv`: comparacion de escenarios y clasificadores.
- `Notebooks/Outputs/comparacion_modelos_calibrados.csv`: resultados de la calibracion de hiperparametros.
- `Notebooks/Outputs/reporte_sesgo_varianza.csv`: diferencia entre ROC-AUC de entrenamiento y validacion.
- `Notebooks/Outputs/importancia_permutacion.csv`: importancia de cada variable.
- `Notebooks/Outputs/tasa_mejoria_por_variable.csv`: tasas observadas de mejoria por grupos y cuantiles.
- `Notebooks/Outputs/metadata_modelo.json`: variables, parametros, metricas y configuracion seleccionada.

El modelo seleccionado es un `RandomForestClassifier` con:

- `n_estimators=500`
- `max_depth=14`
- `min_samples_leaf=40`
- `max_features="sqrt"`
- `class_weight="balanced"`

Esta configuracion fue elegida para reducir el overfitting. En el diagnostico registrado, la brecha aproximada entre entrenamiento y validacion fue `0.071`, menor que la del bosque inicial. El notebook tambien compara `EDAD`, `EDAD_CUADRADO` y `GRUPO_ETARIO`; `EDAD` obtuvo el mejor comportamiento para los modelos evaluados.

## Salida generada

La carpeta `entregable/Salida` contiene el modelo y los resultados generados:

- `modelo_egreso_mejora.joblib`: pipeline completo con preprocesamiento y clasificador.
- `datos_preparados.csv`: datos enriquecidos y preparados.
- `datos_modelo.csv`: variables usadas por el modelo y objetivo.
- `metadata_modelo.json`: metadatos y metricas.
- Reportes EDA en formato CSV.

La carpeta `entregable/Docker` contiene el `Dockerfile`, `docker-compose.yml`, las dependencias y la documentacion de ejecucion en contenedor.

## Docker

Desde la raiz del proyecto:

```bash
docker compose -f entregable/Docker/docker-compose.yml up --build
```

La API quedara disponible en `http://localhost:8000/docs`. La imagen incluye el modelo, la ficha tecnica y el healthcheck del servicio; se ejecuta con un usuario no root.

Para construir y ejecutar manualmente:

```bash
docker build -f entregable/Docker/Dockerfile -t gentera-api:latest .
docker run --rm -p 8000:8000 gentera-api:latest
```

Verificacion rapida:

```bash
curl http://localhost:8000/health
```

Para instalar dependencias sin Docker:

```bash
pip install -r entregable/Docker/requirements.txt
```
