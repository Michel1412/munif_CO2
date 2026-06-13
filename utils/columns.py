"""Detecção automática e normalização de colunas."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

COUNTRY_ALIASES = {"country", "pais", "país", "nation", "país_nome", "nome_pais"}
YEAR_ALIASES = {"year", "ano", "yr", "year_id", "ano_id"}
CO2_ALIASES = {
    "co2_per_capita",
    "co2_per_capita_t",
    "co2",
    "emissao",
    "emissão",
    "co2_pc",
    "co2_per_cap",
}

STANDARD_NAMES = {
    "country": "country",
    "year": "year",
    "co2_per_capita": "co2_per_capita",
}


def _normalize_header(name: str) -> str:
    """Normaliza nome de coluna para comparação."""
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^\w]", "", text)
    return text


def _match_alias(normalized: str, aliases: Iterable[str]) -> bool:
    return normalized in aliases


def _detect_column_roles(columns: list[str]) -> dict[str, str]:
    """Mapeia colunas originais para papéis standard com prioridade."""
    normalized_cols = {col: _normalize_header(col) for col in columns}
    roles: dict[str, str] = {}

    for col, norm in normalized_cols.items():
        if _match_alias(norm, COUNTRY_ALIASES):
            roles.setdefault("country", col)
        elif _match_alias(norm, YEAR_ALIASES):
            roles.setdefault("year", col)

    for col, norm in normalized_cols.items():
        if _match_alias(norm, CO2_ALIASES):
            roles.setdefault("co2_per_capita", col)
        elif "co2" in norm and "capita" in norm:
            roles.setdefault("co2_per_capita", col)

    return roles


def detect_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta colunas de país, ano e CO₂ per capita e normaliza nomes internos.
    Demais colunas são preservadas.
    """
    rename_map: dict[str, str] = {}
    detected = _detect_column_roles(list(df.columns))

    for role, original_col in detected.items():
        rename_map[original_col] = role

    missing = [key for key in ("country", "year", "co2_per_capita") if key not in detected]
    if missing:
        raise ValueError(
            f"Colunas obrigatórias não detectadas: {missing}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    result = df.rename(columns=rename_map).copy()
    # Remove colunas auxiliares vazias comuns em exportações CSV
    result = result.loc[:, ~result.columns.astype(str).str.match(r"^Unnamed")]
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    result["co2_per_capita"] = pd.to_numeric(result["co2_per_capita"], errors="coerce")

    logger.info(
        "Colunas detectadas — país: %s, ano: %s, CO₂: %s",
        detected["country"],
        detected["year"],
        detected["co2_per_capita"],
    )
    return result


def country_to_slug(country: str) -> str:
    """Converte nome do país em slug seguro para arquivos."""
    text = unicodedata.normalize("NFKD", str(country))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^\w]", "", text)
    return text or "unknown"
