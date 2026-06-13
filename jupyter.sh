#!/usr/bin/env bash
# Sobe o Jupyter Notebook/Lab em localhost para visualizar pipeline_co2.ipynb
#
# Uso:
#   ./jupyter.sh              # Notebook em http://localhost:8888
#   ./jupyter.sh --lab        # JupyterLab
#   ./jupyter.sh --port 8889  # Porta customizada
#   ./jupyter.sh --no-browser # Não abre o navegador automaticamente

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT=8888
MODE="notebook"
NO_BROWSER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lab)
      MODE="lab"
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --no-browser)
      NO_BROWSER="--no-browser"
      shift
      ;;
    -h|--help)
      echo "Uso: ./jupyter.sh [--lab] [--port PORT] [--no-browser]"
      exit 0
      ;;
    *)
      echo "Opção desconhecida: $1" >&2
      exit 1
      ;;
  esac
done

if command -v python3 &>/dev/null; then
  SYSTEM_PYTHON=python3
elif command -v python &>/dev/null; then
  SYSTEM_PYTHON=python
else
  echo "Erro: Python não encontrado." >&2
  exit 1
fi

VENV_PYTHON="$ROOT/.venv/bin/python"
VENV_PIP="$ROOT/.venv/bin/pip"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo ">> Ambiente não encontrado — configurando..."
  "$SYSTEM_PYTHON" "$ROOT/run.py" --setup-only
fi

echo ">> Instalando Jupyter (se necessário)..."
"$VENV_PIP" install -q jupyter ipykernel

NOTEBOOK="$ROOT/pipeline_co2.ipynb"
if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Erro: $NOTEBOOK não encontrado." >&2
  exit 1
fi

URL="http://localhost:${PORT}"
echo ""
echo "============================================================"
echo "  Jupyter — Pipeline CO₂ per capita"
echo "============================================================"
echo "  Modo     : $MODE"
echo "  URL      : $URL"
echo "  Notebook : pipeline_co2.ipynb"
echo "============================================================"
echo ""
echo "  Pressione Ctrl+C para encerrar."
echo ""

if [[ "$MODE" == "lab" ]]; then
  exec "$VENV_PYTHON" -m jupyter lab \
    --notebook-dir="$ROOT" \
    --port="$PORT" \
    $NO_BROWSER
else
  exec "$VENV_PYTHON" -m jupyter notebook \
    "$NOTEBOOK" \
    --notebook-dir="$ROOT" \
    --port="$PORT" \
    $NO_BROWSER
fi
