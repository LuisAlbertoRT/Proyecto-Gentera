"""Clasifica pacientes de un Excel usando un modelo joblib entrenado.

Uso:
    python predecir_nuevos.py --input nuevos.xlsx --model Outputs/modelo_egreso_mejora.joblib --output predicciones.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from modelo_utils import enrich_with_catalogs, model_frame, read_workbook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Excel nuevo")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--model", default=str(script_dir / "entregable" / "modelo_egreso_mejora.joblib"))
    parser.add_argument("--output", default=str(script_dir / "entregable" / "predicciones_nuevas.csv"))
    args = parser.parse_args()

    sheets = read_workbook(args.input, required_sheets={"Base de datos"})
    data = sheets["Base de datos"]
    catalog_report = {}
    optional_catalogs = {
        name: sheets[name]
        for name in ["Ciudades", "Enfermedades"]
        if name in sheets
    }
    if optional_catalogs:
        data, catalog_report = enrich_with_catalogs(data, optional_catalogs)
    features = model_frame(data)
    model = joblib.load(args.model)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")

    result = data.copy()
    result["PROBABILIDAD_EGRESO_MEJORIA"] = probabilities
    result["PREDICCION_EGRESO_MEJORIA"] = predictions
    result["CLASIFICACION"] = result["PREDICCION_EGRESO_MEJORIA"].map(
        {1: "egreso por mejoria", 0: "otro motivo"}
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Predicciones guardadas en: {output.resolve()}")
    print(f"Registros procesados: {len(result):,}")
    print(f"Codigos de enfermedad sin descripcion: {catalog_report.get('enfermedades_sin_match', 0):,}")


if __name__ == "__main__":
    main()
