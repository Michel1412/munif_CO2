#!/usr/bin/env python3
"""
ETAPA 4 — Previsão de CO₂ per capita para 2027.

Carrega modelos treinados (Árvore ou SVM), gera features temporais
e produz previsões por país.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from utils.columns import country_to_slug
from utils.features import build_prediction_row
from utils.io import load_raw_dataset
from utils.logging_config import setup_logging
from utils.persistence import load_metadata, model_paths

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"

# Seleção manual do modelo: "tree" ou "svm"
SELECTED_MODEL: Literal["tree", "svm"] = "svm"

TARGET_YEAR = 2027


def load_country_history(country: str, country_slug: str) -> pd.DataFrame:
    """
    Carrega histórico do país até 2026.
    Prioriza CSV processado; fallback reconstrói a partir do dataset bruto.
    """
    processed_path = PROCESSED_DIR / f"{country_slug}.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        return df.sort_values("year")

    df, _ = load_raw_dataset()
    from utils.columns import detect_and_normalize_columns

    normalized = detect_and_normalize_columns(df)
    country_df = normalized[normalized["country"] == country].sort_values("year")
    return country_df.reset_index(drop=True)


def get_co2_2026(history: pd.DataFrame) -> float | None:
    """Retorna valor observado de CO₂ per capita em 2026."""
    row_2026 = history[history["year"] == 2026]
    if row_2026.empty:
        last_row = history.iloc[-1]
        logger.warning(
            "Ano 2026 ausente para %s — usando último ano %s",
            last_row.get("country", "?"),
            last_row["year"],
        )
        return float(last_row["co2_per_capita"])
    return float(row_2026.iloc[0]["co2_per_capita"])


def predict_country(
    country: str,
    model_type: str = SELECTED_MODEL,
) -> dict[str, object] | None:
    """Gera previsão 2027 para um país."""
    country_slug = country_to_slug(country)
    model_file, meta_file = model_paths(model_type, country_slug)

    if not model_file.exists():
        logger.warning("Modelo %s não encontrado para %s", model_type, country)
        return None

    metadata = load_metadata(meta_file)
    if metadata is None:
        logger.warning("Metadata ausente para %s / %s", country, model_type)
        return None

    history = load_country_history(country, country_slug)
    feature_columns = metadata.get("feature_columns", metadata.get("prediction_feature_columns", []))
    features = build_prediction_row(history, feature_columns, target_year=TARGET_YEAR)
    if features is None:
        logger.warning("Features insuficientes para previsão de %s", country)
        return None

    model = joblib.load(model_file)
    x = pd.DataFrame([features[feature_columns]])
    prediction = float(model.predict(x)[0])
    co2_2026 = get_co2_2026(history)

    return {
        "country": country,
        "co2_2026": co2_2026,
        "co2_predicted_2027": round(prediction, 4),
        "model": model_type,
    }


def list_countries() -> list[str]:
    """Lista países a partir dos arquivos processados ou metadata dos modelos."""
    countries: set[str] = set()

    for csv_path in PROCESSED_DIR.glob("*.csv"):
        df = pd.read_csv(csv_path, usecols=["country"])
        countries.add(str(df["country"].iloc[0]))

    if countries:
        return sorted(countries)

    from utils.persistence import MODELS_DIR

    for model_type in ("tree", "svm"):
        model_dir = MODELS_DIR / model_type
        if not model_dir.exists():
            continue
        for meta_file in model_dir.glob("*.meta.json"):
            metadata = load_metadata(meta_file)
            if metadata:
                countries.add(metadata["country"])

    return sorted(countries)


def run(model_type: str = SELECTED_MODEL) -> pd.DataFrame:
    """Executa previsões para todos os países."""
    if model_type not in {"tree", "svm"}:
        raise ValueError(f"SELECTED_MODEL inválido: {model_type}. Use 'tree' ou 'svm'.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    countries = list_countries()

    if not countries:
        raise FileNotFoundError(
            "Nenhum país encontrado. Execute step1_prepare.py e step2_train.py primeiro."
        )

    predictions: list[dict[str, object]] = []
    for country in countries:
        try:
            result = predict_country(country, model_type)
            if result:
                predictions.append(result)
        except Exception as exc:
            logger.error("Erro ao prever %s: %s", country, exc)

    if not predictions:
        raise RuntimeError("Nenhuma previsão gerada.")

    results_df = pd.DataFrame(predictions)
    output_cols = ["country", "co2_2026", "co2_predicted_2027"]

    csv_path = REPORTS_DIR / "predictions_2027.csv"
    xlsx_path = REPORTS_DIR / "predictions_2027.xlsx"
    results_df[output_cols].to_csv(csv_path, index=False)
    results_df[output_cols].to_excel(xlsx_path, index=False)

    logger.info(
        "Previsões 2027 (%s) salvas: %d países em %s",
        model_type,
        len(results_df),
        csv_path,
    )
    return results_df


def main() -> None:
    setup_logging(LOG_FILE)
    try:
        run(SELECTED_MODEL)
        logger.info("ETAPA 4 concluída com sucesso.")
    except Exception:
        logger.exception("Falha na ETAPA 4")
        raise


if __name__ == "__main__":
    main()
