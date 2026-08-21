FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/modelo_egreso_mejora.joblib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY modelo_utils.py entrenar_modelo.py predecir_nuevos.py api_fastapi.py ./
COPY modelo_egreso_mejora.joblib ./

EXPOSE 8000
CMD ["uvicorn", "api_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
