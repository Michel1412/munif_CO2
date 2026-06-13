#!/usr/bin/env python3
"""
ETAPA 2 — Treinamento paralelo de modelos Árvore e SVM por país.

Treina DecisionTreeRegressor e SVR com grid search, calcula métricas
e persiste modelos via joblib com metadata.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from utils.columns import country_to_slug
from utils.features import (
    build_training_matrix,
    get_feature_columns,
    get_prediction_feature_columns,
)
from utils.logging_config import setup_logging
from utils.metrics import compute_regression_metrics, metrics_to_row
from utils.persistence import (
    compute_data_fingerprint,
    load_model_if_valid,
    save_model_with_metadata,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"
MIN_ROWS = 10

TREE_MAX_DEPTHS = [3, 5, 7, 10, None]
TREE_MIN_SAMPLES_LEAF = [1, 2, 4, 8]

SVM_KERNELS = ["rbf", "linear"]
SVM_C_VALUES = [0.1, 1, 10, 100]
SVM_GAMMA_VALUES = ["scale", "auto"]
SVM_EPSILON_VALUES = [0.01, 0.1]


@dataclass
class TrainResult:
    country: str
    country_slug: str
    model_type: str
    metrics: dict[str, Any]
    hyperparams: dict[str, Any]
    skipped: bool = False
    error: Optional[str] = None


def _evaluate_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[DecisionTreeRegressor, dict[str, Any], dict[str, Any]]:
    """Grid search manual para DecisionTreeRegressor."""
    best_model: Optional[DecisionTreeRegressor] = None
    best_metrics: dict[str, Any] = {"r2": float("-inf")}
    best_params: dict[str, Any] = {}

    for max_depth, min_samples_leaf in product(TREE_MAX_DEPTHS, TREE_MIN_SAMPLES_LEAF):
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics = compute_regression_metrics(y_test, preds)

        if metrics["r2"] is not None and metrics["r2"] > best_metrics["r2"]:
            best_model = model
            best_metrics = metrics
            best_params = {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }

    if best_model is None:
        raise RuntimeError("Nenhuma combinação válida encontrada para árvore")

    return best_model, best_metrics, best_params


def _evaluate_svm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Grid search manual para SVR com StandardScaler."""
    best_model: Optional[Pipeline] = None
    best_metrics: dict[str, Any] = {"r2": float("-inf")}
    best_params: dict[str, Any] = {}

    for kernel, c, gamma, epsilon in product(
        SVM_KERNELS, SVM_C_VALUES, SVM_GAMMA_VALUES, SVM_EPSILON_VALUES
    ):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)),
            ]
        )
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        metrics = compute_regression_metrics(y_test, preds)

        if metrics["r2"] is not None and metrics["r2"] > best_metrics["r2"]:
            best_model = pipeline
            best_metrics = metrics
            best_params = {
                "kernel": kernel,
                "C": c,
                "gamma": gamma,
                "epsilon": epsilon,
            }

    if best_model is None:
        raise RuntimeError("Nenhuma combinação válida encontrada para SVM")

    return best_model, best_metrics, best_params


def _train_single_model(
    country: str,
    country_slug: str,
    processed_path: Path,
    model_type: str,
) -> TrainResult:
    """Treina ou carrega um modelo para um país."""
    try:
        fingerprint = compute_data_fingerprint(processed_path)
        cached = load_model_if_valid(model_type, country_slug, fingerprint)
        if cached is not None:
            _, metadata = cached
            return TrainResult(
                country=country,
                country_slug=country_slug,
                model_type=model_type,
                metrics=metadata["metrics"],
                hyperparams=metadata["hyperparams"],
                skipped=True,
            )

        df = pd.read_csv(processed_path)
        if len(df) < MIN_ROWS:
            return TrainResult(
                country=country,
                country_slug=country_slug,
                model_type=model_type,
                metrics={},
                hyperparams={},
                error=f"Dados insuficientes ({len(df)} < {MIN_ROWS})",
            )

        feature_columns = get_feature_columns(df)
        prediction_features = get_prediction_feature_columns(feature_columns)

        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]

        x_train, y_train = build_training_matrix(train_df, feature_columns)
        x_test, y_test = build_training_matrix(test_df, feature_columns)

        if model_type == "tree":
            model, metrics, hyperparams = _evaluate_tree(x_train, y_train, x_test, y_test)
        elif model_type == "svm":
            model, metrics, hyperparams = _evaluate_svm(x_train, y_train, x_test, y_test)
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}")

        save_model_with_metadata(
            model=model,
            model_type=model_type,
            country=country,
            country_slug=country_slug,
            hyperparams=hyperparams,
            metrics=metrics,
            data_fingerprint=fingerprint,
            feature_columns=feature_columns,
            prediction_feature_columns=prediction_features,
        )

        return TrainResult(
            country=country,
            country_slug=country_slug,
            model_type=model_type,
            metrics=metrics,
            hyperparams=hyperparams,
        )
    except Exception as exc:
        return TrainResult(
            country=country,
            country_slug=country_slug,
            model_type=model_type,
            metrics={},
            hyperparams={},
            error=str(exc),
        )


def train_country(processed_path: Path) -> list[TrainResult]:
    """Treina árvore e SVM para um país."""
    df = pd.read_csv(processed_path)
    country = str(df["country"].iloc[0])
    country_slug = country_to_slug(country)

    results = []
    for model_type in ("tree", "svm"):
        results.append(_train_single_model(country, country_slug, processed_path, model_type))
    return results


def save_metrics(results: list[TrainResult]) -> Path:
    """Salva métricas consolidadas em reports/metrics.csv."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for result in results:
        if result.error or not result.metrics:
            logger.warning(
                "País %s / %s ignorado: %s",
                result.country,
                result.model_type,
                result.error or "sem métricas",
            )
            continue
        rows.append(metrics_to_row(result.country, result.model_type, result.metrics))

    metrics_path = REPORTS_DIR / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    logger.info("Métricas salvas em %s (%d registros)", metrics_path, len(rows))
    return metrics_path


def run(max_workers: int | None = None) -> list[TrainResult]:
    """Executa treinamento paralelo para todos os países."""
    processed_files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not processed_files:
        raise FileNotFoundError(
            f"Nenhum arquivo processado em {PROCESSED_DIR}. Execute step1_prepare.py primeiro."
        )

    all_results: list[TrainResult] = []
    workers = max_workers or min(8, len(processed_files))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(train_country, path): path for path in processed_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Treinando países"):
            path = futures[future]
            try:
                country_results = future.result()
                all_results.extend(country_results)
            except Exception as exc:
                logger.error("Falha ao treinar %s: %s", path.name, exc)

    save_metrics(all_results)
    trained = sum(1 for r in all_results if not r.error and not r.skipped)
    skipped = sum(1 for r in all_results if r.skipped)
    logger.info("Treinamento concluído: %d novos, %d reutilizados", trained, skipped)
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETAPA 2 — Treinamento de modelos CO₂")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers paralelos (padrão: min(8, n_países))",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(LOG_FILE)
    args = parse_args()
    try:
        run(max_workers=args.workers)
        logger.info("ETAPA 2 concluída com sucesso.")
    except Exception:
        logger.exception("Falha na ETAPA 2")
        raise


if __name__ == "__main__":
    main()
