"""Divisão temporal treino/teste."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.8


def apply_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> pd.DataFrame:
    """
    Aplica split temporal 80/20 sem embaralhar.
    Adiciona coluna 'split' com valores 'train' ou 'test'.
    """
    result = df.sort_values("year").reset_index(drop=True)
    split_idx = int(len(result) * train_ratio)

    if split_idx < 1 or split_idx >= len(result):
        raise ValueError(
            f"Dados insuficientes para split temporal: {len(result)} linhas "
            f"(split_idx={split_idx})"
        )

    result["split"] = "test"
    result.loc[: split_idx - 1, "split"] = "train"

    train_years = result.loc[result["split"] == "train", "year"]
    test_years = result.loc[result["split"] == "test", "year"]
    logger.debug(
        "Split temporal: treino %d–%d (%d), teste %d–%d (%d)",
        train_years.min(),
        train_years.max(),
        len(train_years),
        test_years.min(),
        test_years.max(),
        len(test_years),
    )
    return result
