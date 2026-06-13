#!/usr/bin/env bash
# Facilitador — sobe e executa o pipeline CO₂ per capita completo.
#
# Uso:
#   ./run.sh
#   ./run.sh --model tree
#   ./run.sh --setup-only
#   ./run.sh --from-step 3

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo ""
echo "============================================================"
echo "  Facilitador — Pipeline CO₂ per capita"
echo "============================================================"
echo ""

# Python do sistema (só para criar o venv)
if command -v python3 &>/dev/null; then
  SYSTEM_PYTHON=python3
elif command -v python &>/dev/null; then
  SYSTEM_PYTHON=python
else
  echo "Erro: Python não encontrado. Instale python3." >&2
  exit 1
fi

VENV_PYTHON="$ROOT/.venv/bin/python"

# Primeira execução: cria venv e instala dependências
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo ">> Primeira execução — configurando ambiente..."
  "$SYSTEM_PYTHON" "$ROOT/run.py" --setup-only
fi

# Sempre executa com o Python do venv
exec "$VENV_PYTHON" "$ROOT/run.py" "$@"
