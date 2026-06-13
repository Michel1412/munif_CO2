"""Leitura e download de datasets brutos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GOOGLE_SHEETS_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1CGX7KrvTQ6qM95EFzDsAMSynA7rgDbR5RsCzdTAJpZw/export?format=csv"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RAW_CSV = DEFAULT_DATA_DIR / "dataset.csv"
DEFAULT_RAW_XLSX = DEFAULT_DATA_DIR / "dataset.xlsx"


def download_dataset(
    url: str = GOOGLE_SHEETS_URL,
    dest: Path = DEFAULT_RAW_CSV,
    timeout: int = 30,
    retries: int = 3,
) -> Path:
    """Baixa a planilha Google Sheets como CSV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            logger.info("Tentativa %d/%d: baixando dados de %s", attempt, retries, url)
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            dest.write_bytes(response.content)
            logger.info("Dataset salvo em %s (%d bytes)", dest, dest.stat().st_size)
            return dest
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            logger.warning("Falha no download (tentativa %d): %s", attempt, exc)

    raise RuntimeError(f"Não foi possível baixar o dataset após {retries} tentativas") from last_error


def _read_file(path: Path) -> pd.DataFrame:
    """Lê CSV ou XLSX conforme extensão."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Formato não suportado: {path}")


def load_raw_dataset(
    path: Optional[Path] = None,
    download_url: str = GOOGLE_SHEETS_URL,
) -> tuple[pd.DataFrame, str]:
    """
    Carrega dataset bruto.

    Ordem: path explícito -> download Google Sheets -> fallback local CSV/XLSX.
    Retorna (DataFrame, origem_descricao).
    """
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {resolved}")
        df = _read_file(resolved)
        logger.info("Dataset carregado de %s (%d linhas)", resolved, len(df))
        return df, str(resolved)

    try:
        downloaded = download_dataset(download_url, DEFAULT_RAW_CSV)
        df = _read_file(downloaded)
        logger.info("Dataset baixado e carregado (%d linhas)", len(df))
        return df, str(downloaded)
    except Exception as exc:
        logger.warning("Download falhou: %s. Tentando fallback local.", exc)

    for fallback in (DEFAULT_RAW_CSV, DEFAULT_RAW_XLSX):
        if fallback.exists():
            df = _read_file(fallback)
            logger.info("Dataset carregado do fallback %s (%d linhas)", fallback, len(df))
            return df, str(fallback)

    raise FileNotFoundError(
        "Nenhuma fonte de dados disponível. "
        "Exporte a planilha para data/dataset.csv ou data/dataset.xlsx."
    )
