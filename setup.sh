#!/usr/bin/env bash
# Profile-MoE: One-command setup and verification
# Usage: bash setup.sh
set -e

echo "========================================"
echo "Profile-MoE — Setup & Verify"
echo "========================================"

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
echo "[2/3] Installing dependencies..."
source .venv/bin/activate
pip install -q -r requirements.txt

# Quick smoke test
echo "[3/3] Verifying installation..."
python3 -c "import numpy, sklearn, matplotlib, openpyxl; print('  ✓ Core deps OK')"
python3 -c "import torch; print('  ✓ PyTorch OK:', torch.__version__)"
echo ""
echo "Setup complete. Run: bash run_all.sh"
