#!/usr/bin/env python3
"""
ETAPA 1 — Preparação dos dados.

Baixa/ler planilha, detecta colunas, agrupa por país, gera features temporais,
aplica split temporal 80/20 e salva CSVs processados por país.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils.columns import country_to_slug, detect_and_normalize_columns
from utils.features import add_temporal_features
from utils.io import load_raw_dataset
from utils.logging_config import setup_logging
from utils.split import apply_temporal_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"


def group_by_country(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agrupa DataFrame normalizado por país."""
    country_tables: dict[str, pd.DataFrame] = {}
    for country, group in df.groupby("country", sort=True):
        country_name = str(country)
        sorted_group = group.sort_values("year").reset_index(drop=True)
        country_tables[country_name] = sorted_group
    return country_tables


def prepare_country(country: str, df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de preparação para um único país."""
    featured = add_temporal_features(df)
    split_df = apply_temporal_split(featured)
    split_df["country"] = country
    return split_df


def save_processed_countries(country_tables: dict[str, pd.DataFrame]) -> list[Path]:
    """Salva um CSV processado por país."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for country, df in country_tables.items():
        slug = country_to_slug(country)
        output_path = PROCESSED_DIR / f"{slug}.csv"
        prepared = prepare_country(country, df)
        prepared.to_csv(output_path, index=False)
        saved_paths.append(output_path)
        logger.info(
            "País '%s' salvo em %s (%d linhas, treino=%d, teste=%d)",
            country,
            output_path.name,
            len(prepared),
            (prepared["split"] == "train").sum(),
            (prepared["split"] == "test").sum(),
        )

    return saved_paths


def run(input_path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Executa etapa 1 completa."""
    df, source = load_raw_dataset(input_path)
    logger.info("Fonte dos dados: %s", source)

    normalized = detect_and_normalize_columns(df)
    normalized = normalized.dropna(subset=["country", "year", "co2_per_capita"])

    year_min = int(normalized["year"].min())
    year_max = int(normalized["year"].max())
    logger.info("Período dos dados: %d–%d", year_min, year_max)

    country_tables = group_by_country(normalized)
    logger.info("Total de países identificados: %d", len(country_tables))

    save_processed_countries(country_tables)
    return country_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETAPA 1 — Preparação dos dados CO₂")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Caminho opcional para CSV/XLSX local (ignora download)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(LOG_FILE)
    args = parse_args()
    try:
        run(args.input)
        logger.info("ETAPA 1 concluída com sucesso.")
    except Exception:
        logger.exception("Falha na ETAPA 1")
        raise


if __name__ == "__main__":
    main()
