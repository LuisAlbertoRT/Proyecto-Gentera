"""Entrena el modelo final desde un Excel y guarda reportes y artefactos.

Uso:
    python entrenar_modelo.py --input Inputs/Hospitales.xlsx --output Outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from funciones_1 import (
    BASE_FEATURES,
    TARGET_COLUMN,
    add_target,
    enrich_with_catalogs,
    engineer_features,
    exploratory_report,
    model_frame,
    read_workbook,
)

RANDOM_STATE = 42


def build_model() -> Pipeline:
    numeric = [
        "DIAS_ESTANCIA", "EDAD", "NIVEL_DE_GRAVEDAD", "IMC",
        "ES_INDIGENA", "ENTIDAD", "MES",
    ]
    categorical = ["GENERO"]
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])
    classifier = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=40,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def save_eda(data: pd.DataFrame, output: Path) -> None:
    report = exploratory_report(data)
    report.to_csv(output / "reporte_eda.csv", encoding="utf-8-sig")
    data[TARGET_COLUMN].value_counts(dropna=False).rename("frecuencia").to_csv(
        output / "distribucion_objetivo.csv", encoding="utf-8-sig"
    )
    numeric = data.select_dtypes(include=np.number)
    if len(numeric.columns) > 1:
        numeric.describe().T.to_csv(output / "estadistica_descriptiva.csv", encoding="utf-8-sig")
        numeric.corr().to_csv(output / "correlaciones_numericas.csv", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Excel con Base de datos y catalogos")
    parser.add_argument("--output", default="Outputs", help="Directorio de salida")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    sheets = read_workbook(args.input)
    data, catalog_report = enrich_with_catalogs(sheets["Base de datos"], sheets)
    data, improvement_code = add_target(data, sheets["Estatus"])
    data = engineer_features(data)
    save_eda(data, output)

    features = model_frame(data)
    target = data[TARGET_COLUMN]
    if target.nunique() != 2:
        raise ValueError("EGRESO_MEJORIA debe contener exactamente las clases 0 y 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_result = cross_validate(model, x_train, y_train, cv=cv, scoring="roc_auc")
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    metrics = {
        "cv_roc_auc_mean": float(cv_result["test_score"].mean()),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
    }

    data.to_csv(output / "datos_preparados.csv", index=False, encoding="utf-8-sig")
    features.assign(**{TARGET_COLUMN: target}).to_csv(
        output / "datos_modelo.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(model, output / "modelo_egreso_mejora.joblib")
    metadata = {
        "model_type": "RandomForestClassifier",
        "features": BASE_FEATURES,
        "target": TARGET_COLUMN,
        "positive_class": "egreso por mejoria",
        "improvement_status_code": str(improvement_code),
        "random_state": RANDOM_STATE,
        "rows": int(len(data)),
        "catalog_report": catalog_report,
        "metrics": metrics,
        "parameters": model.named_steps["model"].get_params(),
    }
    (output / "metadata_modelo.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"output": str(output.resolve()), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
