FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/entregable/Salida/modelo_egreso_mejora.joblib

COPY entregable/Docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ["funciones_1.py", "funciones_1.py"]
COPY ["4) entrenar_modelo.py", "entrenar_modelo.py"]
COPY ["5) api_fastapi.py", "api_fastapi.py"]
COPY ["README.md", "README.md"]
COPY ["MODEL_CARD.md", "MODEL_CARD.md"]
COPY entregable/Salida /app/entregable/Salida

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
