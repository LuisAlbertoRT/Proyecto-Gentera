FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/entregable/Salida/modelo_egreso_mejora.joblib

COPY entregable/Docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY funciones_1.py entrenar_modelo.py api_fastapi.py ./
COPY entregable/Salida/modelo_egreso_mejora.joblib ./entregable/Salida/modelo_egreso_mejora.joblib

EXPOSE 8000
CMD ["uvicorn", "api_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
