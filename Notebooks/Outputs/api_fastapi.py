"""Servicio FastAPI para clasificar pacientes con el modelo entrenado.

Inicio:
    uvicorn api_fastapi:app --host 0.0.0.0 --port 8000

Documentacion interactiva: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modelo_utils import BASE_FEATURES, model_frame

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "modelo_egreso_mejora.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
model = None
app = FastAPI(title="Clasificador de egreso por mejoria", version="1.0.0")


class Patient(BaseModel):
    dias_estancia: float | None = Field(default=None, alias="DIAS_ESTANCIA")
    edad: float | None = Field(default=None, alias="EDAD")
    genero: Any | None = Field(default=None, alias="GENERO")
    nivel_de_gravedad: float | None = Field(default=None, alias="NIVEL_DE_GRAVEDAD")
    imc: float | None = Field(default=None, alias="IMC")
    es_indigena: Any | None = Field(default=None, alias="ES_INDIGENA")
    entidad: Any | None = Field(default=None, alias="ENTIDAD")
    mes: Any | None = Field(default=None, alias="MES")


@app.get("/health")
def health() -> dict[str, str]:
    status = "ok" if MODEL_PATH.exists() else "model_missing"
    return {"status": status, "model": str(MODEL_PATH)}


@app.post("/predict")
def predict(patient: Patient) -> dict[str, Any]:
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"No existe el modelo: {MODEL_PATH}")
        try:
            loaded_model = joblib.load(MODEL_PATH)
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"No se pudo cargar el modelo: {error}") from error
    else:
        loaded_model = model
    payload = (
        patient.model_dump(by_alias=True)
        if hasattr(patient, "model_dump")
        else patient.dict(by_alias=True)
    )
    data = pd.DataFrame([payload])
    try:
        features = model_frame(data)
        probability = float(loaded_model.predict_proba(features)[0, 1])
    except (KeyError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    prediction = int(probability >= 0.5)
    return {
        "prediccion": prediction,
        "clasificacion": "egreso por mejoria" if prediction else "otro motivo",
        "probabilidad_egreso_mejoria": probability,
        "umbral": 0.5,
        "variables_modelo": BASE_FEATURES,
    }
