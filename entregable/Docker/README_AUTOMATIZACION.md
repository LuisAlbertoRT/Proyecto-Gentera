# Automatizacion del modelo

Estos scripts implementan el flujo completo para predecir egreso por mejoria.

## 1. Instalar dependencias

```bash
pip install -r entregable/Docker/requirements.txt
```

## 2. Entrenar desde un Excel

El libro debe incluir las hojas `Base de datos`, `Estatus` y `Enfermedades`. Las hojas `Ciudades` y `Gravedad` son opcionales.

```bash
python "4) entrenar_modelo.py" --input Notebooks/Inputs/Hospitales.xlsx --output entregable/Salida
```

El entrenamiento usa el Random Forest seleccionado:

- `n_estimators=500`
- `max_depth=14`
- `min_samples_leaf=40`
- `max_features='sqrt'`
- `class_weight='balanced'`

Genera el modelo `modelo_egreso_mejora.joblib`, datos preparados, metadatos y reportes EDA en CSV.


## 4. Ejecutar la API

```bash
docker compose -f entregable/Docker/docker-compose.yml up --build
```

La API expone `GET /health`, `POST /predict` y la documentacion en `/docs`.

Ejemplo de cuerpo para `POST /predict`:

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

Para usar otro modelo:

```bash
set MODEL_PATH=C:\ruta\modelo_egreso_mejora.joblib
uvicorn api_fastapi:app --host 0.0.0.0 --port 8000
```

## 5. Docker

Construir y levantar la API:

```bash
docker compose -f entregable/Docker/docker-compose.yml up --build
```

La API queda disponible en `http://localhost:8000/docs`. Para entrenar dentro de Docker:

```bash
docker compose -f entregable/Docker/docker-compose.yml run --rm api python entrenar_modelo.py --input /app/data/Hospitales.xlsx --output /app/artifacts
```
