"""Utilitários compartilhados do pipeline CO₂ per capita."""

from utils.columns import detect_and_normalize_columns, country_to_slug
from utils.features import (
    TEMPORAL_FEATURES,
    add_temporal_features,
    build_features_for_year,
    get_feature_columns,
)
from utils.io import download_dataset, load_raw_dataset
from utils.metrics import compute_regression_metrics
from utils.persistence import (
    load_model_if_valid,
    save_model_with_metadata,
)
from utils.split import apply_temporal_split

__all__ = [
    "TEMPORAL_FEATURES",
    "add_temporal_features",
    "apply_temporal_split",
    "build_features_for_year",
    "compute_regression_metrics",
    "country_to_slug",
    "detect_and_normalize_columns",
    "download_dataset",
    "get_feature_columns",
    "load_model_if_valid",
    "load_raw_dataset",
    "save_model_with_metadata",
]
