#!/usr/bin/env python3
"""
Facilitador — executa o pipeline CO₂ per capita completo.

Uso:
    python run.py
    python run.py --model svm
    python run.py --setup-only
    ./run.sh
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"

STEPS = (
    ("step1_prepare.py", "Etapa 1 — Preparação dos dados"),
    ("step2_train.py", "Etapa 2 — Treinamento (Árvore + SVM)"),
    ("step3_report.py", "Etapa 3 — Relatórios consolidados"),
    ("step4_predict_2027.py", "Etapa 4 — Previsão 2027"),
)


def get_python_executable() -> Path:
    """Usa Python do venv local se existir."""
    venv_python = VENV_DIR / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def setup_venv(install: bool = True) -> Path:
    """Cria venv e instala dependências."""
    python = Path(sys.executable)
    requirements = PROJECT_ROOT / "requirements.txt"

    if not VENV_DIR.exists():
        print(">> Criando ambiente virtual (.venv)...")
        subprocess.run([str(python), "-m", "venv", str(VENV_DIR)], check=True)

    venv_python = VENV_DIR / "bin" / "python"
    if install and requirements.exists():
        print(">> Instalando dependências (requirements.txt)...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-q", "--upgrade", "pip"],
            check=True,
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-q", "-r", str(requirements)],
            check=True,
        )

    return venv_python


def run_step(
    script: str,
    python: Path,
    extra_args: list[str] | None = None,
) -> None:
    """Executa um script do pipeline."""
    cmd = [str(python), str(PROJECT_ROOT / script)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao executar {script} (código {result.returncode})")


def print_summary(model: str) -> None:
    """Exibe resumo dos artefatos gerados."""
    reports = PROJECT_ROOT / "reports"
    processed = PROJECT_ROOT / "data" / "processed"
    models_tree = PROJECT_ROOT / "models" / "tree"
    models_svm = PROJECT_ROOT / "models" / "svm"

    print("\n" + "=" * 60)
    print("  Pipeline concluído com sucesso!")
    print("=" * 60)
    print(f"  Países processados : {len(list(processed.glob('*.csv')))}")
    print(f"  Modelos Árvore     : {len(list(models_tree.glob('*.joblib')))}")
    print(f"  Modelos SVM        : {len(list(models_svm.glob('*.joblib')))}")
    print(f"  Modelo previsão    : {model.upper()}")
    print("\n  Resultados (tabelas):")
    for pattern in ("*.csv", "*.xlsx", "summary.txt"):
        for path in sorted(reports.glob(pattern)):
            print(f"    → reports/{path.name}")
    print("\n  Documentação: EXPLICACAO_RESULTADOS.md")
    print("  Notebook    : pipeline_co2.ipynb")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo de CO₂ per capita.",
    )
    parser.add_argument(
        "--model",
        choices=["tree", "svm"],
        default="svm",
        help="Modelo para previsão 2027 (padrão: svm)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Workers paralelos no treinamento (step2)",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Apenas cria venv e instala dependências",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Pula criação do venv e instalação",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Inicia a partir de uma etapa específica (1–4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    print("\n" + "=" * 60)
    print("  Pipeline ML — CO₂ per capita")
    print("=" * 60 + "\n")

    if args.skip_setup:
        python = get_python_executable()
    else:
        python = setup_venv(install=True)

    if args.setup_only:
        print(">> Setup concluído. Ambiente pronto em .venv/")
        print(">> Execute: ./run.sh")
        return

    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    steps_to_run = STEPS[args.from_step - 1 :]

    for i, (script, label) in enumerate(steps_to_run, start=args.from_step):
        print(f"\n>> [{i}/4] {label}")
        print("-" * 60)

        extra: list[str] = []
        if script == "step2_train.py" and args.workers:
            extra = ["--workers", str(args.workers)]
        if script == "step4_predict_2027.py":
            # Executa step4 via import para passar o modelo escolhido
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from step4_predict_2027 import run as run_step4

            run_step4(model_type=args.model)
            continue

        run_step(script, python, extra or None)

    elapsed = time.time() - start
    print_summary(args.model)
    print(f"Tempo total: {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        sys.exit(1)
