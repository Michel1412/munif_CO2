"""Engenharia de features temporais para séries de CO₂."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TEMPORAL_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_mean_5",
    "growth_rate",
]

EXCLUDED_FROM_FEATURES = {"country", "year", "co2_per_capita", "split"}


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features temporais derivadas de co2_per_capita.
    Espera DataFrame já ordenado por year.
    """
    result = df.copy()
    co2 = result["co2_per_capita"]

    result["lag_1"] = co2.shift(1)
    result["lag_2"] = co2.shift(2)
    result["lag_3"] = co2.shift(3)
    result["rolling_mean_3"] = co2.rolling(window=3, min_periods=3).mean()
    result["rolling_mean_5"] = co2.rolling(window=5, min_periods=5).mean()
    result["growth_rate"] = co2.pct_change()

    before = len(result)
    result = result.dropna(subset=TEMPORAL_FEATURES).reset_index(drop=True)
    dropped = before - len(result)
    if dropped:
        logger.debug("Removidas %d linhas com NaN em features temporais", dropped)

    return result


def get_feature_columns(
    df: pd.DataFrame,
    include_exogenous: bool = True,
    exogenous_only_for_training: bool = True,
) -> list[str]:
    """
    Retorna colunas de features para treino.
    Por padrão inclui temporais + numéricas exógenas (exceto identificadores).
    """
    features = list(TEMPORAL_FEATURES)

    if include_exogenous:
        for col in df.columns:
            if col in EXCLUDED_FROM_FEATURES or col in features:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                features.append(col)

    if exogenous_only_for_training:
        return features

    return features


def get_prediction_feature_columns(feature_columns: list[str]) -> list[str]:
    """Features disponíveis na previsão de 2027 (somente temporais)."""
    return [col for col in feature_columns if col in TEMPORAL_FEATURES]


def build_features_for_year(
    country_df: pd.DataFrame,
    target_year: int = 2027,
) -> Optional[pd.Series]:
    """
    Constrói vetor de features temporais para um ano futuro
    usando histórico observado até target_year - 1.
    """
    history = country_df.sort_values("year").copy()
    history = history[history["year"] < target_year]

    if history.empty or history["co2_per_capita"].isna().all():
        return None

    last_year = int(history["year"].max())
    if last_year < target_year - 1:
        logger.warning(
            "Histórico incompleto para %d: último ano disponível %d",
            target_year,
            last_year,
        )

    co2_series = history.set_index("year")["co2_per_capita"]
    values = co2_series.dropna()

    if len(values) < 3:
        return None

    lag_1 = values.iloc[-1]
    lag_2 = values.iloc[-2]
    lag_3 = values.iloc[-3]
    rolling_mean_3 = values.iloc[-3:].mean()
    rolling_mean_5 = values.iloc[-5:].mean() if len(values) >= 5 else values.mean()
    growth_rate = (lag_1 - lag_2) / lag_2 if lag_2 != 0 else 0.0

    return pd.Series(
        {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_mean_3": rolling_mean_3,
            "rolling_mean_5": rolling_mean_5,
            "growth_rate": growth_rate,
        }
    )


def build_prediction_row(
    country_df: pd.DataFrame,
    feature_columns: list[str],
    target_year: int = 2027,
) -> Optional[pd.Series]:
    """
    Monta linha de features para previsão futura.
    Temporais são calculadas via histórico; exógenas usam último ano observado.
    """
    temporal = build_features_for_year(country_df, target_year=target_year)
    if temporal is None:
        return None

    history = country_df.sort_values("year")
    last_row = history[history["year"] < target_year].iloc[-1]
    row: dict[str, float] = {}

    for col in feature_columns:
        if col in temporal.index:
            row[col] = float(temporal[col])
        elif col in last_row.index and pd.notna(last_row[col]):
            row[col] = float(last_row[col])
        else:
            row[col] = 0.0

    return pd.Series(row)


def build_training_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X e y para treino/avaliação."""
    available = [col for col in feature_columns if col in df.columns]
    x = df[available].copy()
    y = df["co2_per_capita"].copy()
    return x, y
