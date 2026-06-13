#!/usr/bin/env python3
"""
ETAPA 3 — Resumo consolidado dos resultados.

Gera country_summary, ranking global, summary.txt e metrics.xlsx com gráficos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.logging_config import setup_logging
from utils.persistence import MODELS_DIR, load_metadata

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"

MODEL_LABELS = {"tree": "Árvore", "svm": "SVM"}


def load_metrics_csv() -> pd.DataFrame:
    """Carrega métricas geradas pelo step2."""
    metrics_path = REPORTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} não encontrado. Execute step2_train.py primeiro."
        )
    return pd.read_csv(metrics_path)


def build_country_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot métricas por país e modelo."""
    summary_rows = []

    for country in sorted(metrics_df["country"].unique()):
        row: dict[str, object] = {"country": country}
        country_metrics = metrics_df[metrics_df["country"] == country]

        for model in ("tree", "svm"):
            model_data = country_metrics[country_metrics["model"] == model]
            if model_data.empty:
                continue
            record = model_data.iloc[0]
            row[f"{model}_r2"] = record["r2"]
            row[f"{model}_mae"] = record["mae"]
            row[f"{model}_rmse"] = record["rmse"]
            row[f"{model}_mape"] = record["mape"]

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def build_best_by_country(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Determina melhor modelo por país com base no R²."""
    rows = []

    for _, record in summary_df.iterrows():
        scores: dict[str, float] = {}
        if pd.notna(record.get("tree_r2")):
            scores["tree"] = float(record["tree_r2"])
        if pd.notna(record.get("svm_r2")):
            scores["svm"] = float(record["svm_r2"])

        if not scores:
            continue

        best_model = max(scores, key=scores.get)
        rows.append(
            {
                "country": record["country"],
                "best_model": best_model,
                "best_model_label": MODEL_LABELS.get(best_model, best_model),
                "best_r2": scores[best_model],
            }
        )

    return pd.DataFrame(rows)


def build_ranking(best_df: pd.DataFrame) -> pd.DataFrame:
    """Conta vitórias por modelo."""
    if best_df.empty:
        return pd.DataFrame(columns=["model", "model_label", "wins"])

    counts = best_df["best_model"].value_counts().reindex(["tree", "svm"], fill_value=0)
    return pd.DataFrame(
        {
            "model": counts.index,
            "model_label": [MODEL_LABELS.get(m, m) for m in counts.index],
            "wins": counts.values,
        }
    )


def write_summary_text(ranking_df: pd.DataFrame) -> Path:
    """Gera resumo textual automático."""
    wins = {row["model"]: int(row["wins"]) for _, row in ranking_df.iterrows()}
    tree_wins = wins.get("tree", 0)
    svm_wins = wins.get("svm", 0)

    lines = [
        f"SVM apresentou melhor desempenho em {svm_wins} países.",
        f"Árvore apresentou melhor desempenho em {tree_wins} países.",
    ]
    if svm_wins > tree_wins:
        lines.append("SVM foi o modelo superior na maioria dos cenários.")
    elif tree_wins > svm_wins:
        lines.append("Árvore foi o modelo superior na maioria dos cenários.")
    else:
        lines.append("Árvore e SVM empataram no número de vitórias.")

    summary_path = REPORTS_DIR / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Resumo textual salvo em %s", summary_path)
    return summary_path


def save_excel_reports(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    """Salva relatórios Excel com múltiplas abas."""
    metrics_xlsx = REPORTS_DIR / "metrics.xlsx"
    summary_xlsx = REPORTS_DIR / "country_summary.xlsx"

    with pd.ExcelWriter(metrics_xlsx, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        best_df.to_excel(writer, sheet_name="best_by_country", index=False)
        ranking_df.to_excel(writer, sheet_name="ranking", index=False)

    summary_df.to_excel(summary_xlsx, index=False)
    logger.info("Relatórios Excel salvos: %s, %s", metrics_xlsx, summary_xlsx)


def generate_figures(summary_df: pd.DataFrame) -> None:
    """Gera gráficos comparativos de R² por país."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if "tree_r2" not in summary_df.columns or "svm_r2" not in summary_df.columns:
        logger.warning("Colunas R² ausentes — gráficos não gerados")
        return

    plot_df = summary_df[["country", "tree_r2", "svm_r2"]].dropna()
    if plot_df.empty:
        return

    melted = plot_df.melt(id_vars="country", var_name="model", value_name="r2")
    melted["model"] = melted["model"].map({"tree_r2": "Árvore", "svm_r2": "SVM"})

    plt.figure(figsize=(12, max(6, len(plot_df) * 0.25)))
    sns.barplot(data=melted, y="country", x="r2", hue="model")
    plt.title("Comparativo de R² por país")
    plt.xlabel("R²")
    plt.tight_layout()
    bar_path = FIGURES_DIR / "r2_by_country.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="tree_r2", y="svm_r2")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("R² Árvore")
    plt.ylabel("R² SVM")
    plt.title("Árvore vs SVM (R²)")
    plt.tight_layout()
    scatter_path = FIGURES_DIR / "tree_vs_svm_scatter.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    logger.info("Figuras salvas em %s", FIGURES_DIR)


def enrich_from_metadata(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Complementa métricas com metadata dos modelos, se necessário."""
    if not metrics_df.empty:
        return metrics_df

    rows = []
    for model_type in ("tree", "svm"):
        model_dir = MODELS_DIR / model_type
        if not model_dir.exists():
            continue
        for meta_file in model_dir.glob("*.meta.json"):
            metadata = load_metadata(meta_file)
            if metadata and "metrics" in metadata:
                rows.append(
                    {
                        "country": metadata["country"],
                        "model": model_type,
                        **metadata["metrics"],
                    }
                )

    return pd.DataFrame(rows)


def run() -> None:
    """Executa geração de relatórios."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics_csv()
    if metrics_df.empty:
        metrics_df = enrich_from_metadata(metrics_df)

    summary_df = build_country_summary(metrics_df)
    best_df = build_best_by_country(summary_df)
    ranking_df = build_ranking(best_df)

    summary_csv = REPORTS_DIR / "country_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    best_csv = REPORTS_DIR / "best_by_country.csv"
    best_df.to_csv(best_csv, index=False)
    ranking_csv = REPORTS_DIR / "ranking.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    save_excel_reports(metrics_df, summary_df, best_df, ranking_df)
    write_summary_text(ranking_df)
    generate_figures(summary_df)

    logger.info(
        "Relatório consolidado: %d países, ranking SVM=%d, Árvore=%d",
        len(summary_df),
        ranking_df.loc[ranking_df["model"] == "svm", "wins"].sum()
        if not ranking_df.empty
        else 0,
        ranking_df.loc[ranking_df["model"] == "tree", "wins"].sum()
        if not ranking_df.empty
        else 0,
    )


def main() -> None:
    setup_logging(LOG_FILE)
    try:
        run()
        logger.info("ETAPA 3 concluída com sucesso.")
    except Exception:
        logger.exception("Falha na ETAPA 3")
        raise


if __name__ == "__main__":
    main()
