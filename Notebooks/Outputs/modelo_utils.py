"""Funciones compartidas para entrenar y servir el modelo de egreso por mejora."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "DIAS_ESTANCIA",
    "EDAD",
    "GENERO",
    "NIVEL_DE_GRAVEDAD",
    "IMC",
    "ES_INDIGENA",
    "ENTIDAD",
    "MES",
]
TARGET_COLUMN = "EGRESO_MEJORIA"
STATUS_COLUMN = "MOTIVO_EGRESO"


def read_workbook(
    path: str | Path,
    required_sheets: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Lee la base principal y los catálogos disponibles del libro Excel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo Excel: {path}")

    required_sheets = required_sheets or {"Base de datos", "Estatus", "Enfermedades"}
    with pd.ExcelFile(path) as workbook:
        missing = required_sheets - set(workbook.sheet_names)
        if missing:
            raise ValueError(f"Faltan hojas requeridas: {sorted(missing)}")
        return {
            sheet: pd.read_excel(workbook, sheet_name=sheet)
            for sheet in workbook.sheet_names
        }


def _first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def enrich_with_catalogs(
    base: pd.DataFrame,
    catalogs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Agrega descripciones de catálogos sin duplicar registros de la base."""
    data = base.copy()
    data.columns = [str(column).strip() for column in data.columns]
    catalog_report: dict[str, Any] = {}

    diseases = catalogs.get("Enfermedades")
    if diseases is not None and "CODIGO_ENFERMEDAD" in data.columns:
        code_column = _first_column(diseases, ["CODIGO_ENFERMEDAD"])
        if code_column:
            disease_columns = [code_column]
            for column in [
                "CODIGO_ENFERMEDAD_DESC",
                "Tipo de Problema",
                "Nivel de gravedad",
            ]:
                if column in diseases.columns:
                    disease_columns.append(column)
            disease_catalog = diseases[disease_columns].drop_duplicates(code_column)
            data = data.merge(
                disease_catalog,
                left_on="CODIGO_ENFERMEDAD",
                right_on=code_column,
                how="left",
                validate="many_to_one",
                suffixes=("", "_CATALOGO"),
            )
            if code_column != "CODIGO_ENFERMEDAD":
                data = data.drop(columns=[code_column], errors="ignore")
            data = data.rename(
                columns={
                    "Nivel de gravedad": "NIVEL_DE_GRAVEDAD",
                    "Tipo de Problema": "TIPO_DE_PROBLEMA_MEDICO",
                }
            )
            catalog_report["enfermedades_sin_match"] = int(
                data.get("CODIGO_ENFERMEDAD_DESC", pd.Series(dtype=object)).isna().sum()
            )

    if "INDIGENA" in data.columns and "ES_INDIGENA" not in data.columns:
        data["ES_INDIGENA"] = data["INDIGENA"]

    cities = catalogs.get("Ciudades")
    if cities is not None and "ENTIDAD" in data.columns and "ENTIDAD" in cities.columns:
        city_columns = [column for column in ["ENTIDAD", "NOMBRE_ENTIDAD"] if column in cities.columns]
        if len(city_columns) == 2:
            city_catalog = cities[city_columns].drop_duplicates("ENTIDAD")
            data = data.merge(
                city_catalog,
                on="ENTIDAD",
                how="left",
                validate="many_to_one",
                suffixes=("", "_CATALOGO"),
            )
            catalog_report["entidades_sin_match"] = int(data["NOMBRE_ENTIDAD"].isna().sum())

    return data, catalog_report


def find_improvement_code(status: pd.DataFrame) -> Any:
    """Obtiene desde el catalogo el codigo cuyo texto describe mejoria."""
    code_column = _first_column(status, ["MOTIVO_EGRESO"])
    description_column = _first_column(
        status, ["MOTIVO_EGRESO_DESC", "DESCRIPCION", "DESCRIPCION_MOTIVO"]
    )
    if not code_column or not description_column:
        raise ValueError(
            "El catalogo Estatus debe incluir MOTIVO_EGRESO y su descripcion."
        )
    descriptions = status[description_column].astype("string").str.strip().str.lower()
    matches = status[descriptions.str.contains("mejor", na=False)]
    if matches.empty:
        raise ValueError("No se encontro un estatus cuya descripcion contenga 'mejor'.")
    return matches.iloc[0][code_column]


def add_target(data: pd.DataFrame, status: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    data = data.copy()
    if STATUS_COLUMN not in data.columns:
        raise ValueError(f"La base debe incluir la columna {STATUS_COLUMN}.")
    improvement_code = find_improvement_code(status)
    data[TARGET_COLUMN] = (
        data[STATUS_COLUMN].astype("string").str.strip()
        == str(improvement_code).strip()
    ).astype("int8")
    return data, improvement_code


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Aplica exactamente la ingenieria de variables usada por el modelo final."""
    data = data.copy()
    missing = [column for column in BASE_FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan variables predictoras: {missing}")
    data["EDAD"] = pd.to_numeric(data["EDAD"], errors="coerce")
    return data


def model_frame(data: pd.DataFrame) -> pd.DataFrame:
    data = engineer_features(data)
    return data[BASE_FEATURES].copy()


def exploratory_report(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tipo": data.dtypes.astype(str),
            "nulos": data.isna().sum(),
            "porcentaje_nulos": (data.isna().mean() * 100).round(2),
            "valores_unicos": data.nunique(dropna=False),
        }
    ).sort_values("porcentaje_nulos", ascending=False)
