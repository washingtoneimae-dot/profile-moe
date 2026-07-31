#!/usr/bin/env bash
# Profile-MoE: One-command setup and verification
# Usage: bash setup.sh
set -e

echo "========================================"
echo "Profile-MoE — Setup & Verify"
echo "========================================"

# Create venv if missing
VENV_DIR="$HOME/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating Python virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Install deps
echo "[2/3] Installing dependencies..."
"$VENV_DIR/bin/python3" -m pip install -q -r requirements.txt

# Quick smoke test
echo "[3/3] Verifying installation..."
"$VENV_DIR/bin/python3" -c "import numpy, sklearn, matplotlib, openpyxl; print('  ✓ Core deps OK')"
"$VENV_DIR/bin/python3" -c "import torch; print('  ✓ PyTorch OK:', torch.__version__)" || echo "  ⚠ PyTorch not found (transformer_training.py will skip)"
echo ""
echo "Setup complete. Run: bash run_all.sh"
