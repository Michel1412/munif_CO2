"""Persistência de modelos com joblib e metadata JSON."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def compute_data_fingerprint(file_path: Path) -> str:
    """Gera fingerprint SHA256 do arquivo de dados processados."""
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def model_paths(model_type: str, country_slug: str) -> tuple[Path, Path]:
    """Retorna caminhos do modelo (.joblib) e metadata (.meta.json)."""
    model_file = MODELS_DIR / model_type / f"{country_slug}.joblib"
    meta_file = MODELS_DIR / model_type / f"{country_slug}.meta.json"
    return model_file, meta_file


def load_metadata(meta_path: Path) -> Optional[dict[str, Any]]:
    """Carrega metadata JSON se existir."""
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_model_with_metadata(
    model: Any,
    model_type: str,
    country: str,
    country_slug: str,
    hyperparams: dict[str, Any],
    metrics: dict[str, Any],
    data_fingerprint: str,
    feature_columns: list[str],
    prediction_feature_columns: list[str],
) -> Path:
    """Salva modelo joblib e sidecar JSON com metadata."""
    model_file, meta_file = model_paths(model_type, country_slug)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_file)

    metadata = {
        "country": country,
        "country_slug": country_slug,
        "model_type": model_type,
        "hyperparams": hyperparams,
        "metrics": metrics,
        "data_fingerprint": data_fingerprint,
        "feature_columns": feature_columns,
        "prediction_feature_columns": prediction_feature_columns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with meta_file.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    logger.info("Modelo %s salvo: %s", model_type, model_file)
    return model_file


def load_model_if_valid(
    model_type: str,
    country_slug: str,
    data_fingerprint: str,
) -> Optional[tuple[Any, dict[str, Any]]]:
    """
    Carrega modelo existente se fingerprint dos dados coincidir.
    Retorna (modelo, metadata) ou None se retreino for necessário.
    """
    model_file, meta_file = model_paths(model_type, country_slug)

    if not model_file.exists() or not meta_file.exists():
        logger.debug("Modelo %s/%s não encontrado — treino necessário", model_type, country_slug)
        return None

    metadata = load_metadata(meta_file)
    if metadata is None:
        return None

    if metadata.get("data_fingerprint") != data_fingerprint:
        logger.info(
            "Dados de %s alterados — retreino de %s necessário",
            country_slug,
            model_type,
        )
        return None

    model = joblib.load(model_file)
    logger.info("Modelo %s/%s carregado (sem retreino)", model_type, country_slug)
    return model, metadata
