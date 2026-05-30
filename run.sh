#!/usr/bin/env bash
# =============================================================================
#  Lancement du backend FakeGuard / FPD (Linux / macOS)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activation de l'environnement virtuel s'il existe
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Chargement des variables d'environnement
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

echo "============================================================"
echo "  FakeGuard / FPD backend — démarrage"
echo "  Python : $(python --version)"
echo "  Port   : ${PORT:-5000}"
echo "============================================================"

exec python app.py
