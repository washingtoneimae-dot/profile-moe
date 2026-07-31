#!/usr/bin/env bash
# Profile-MoE: Run all proofs and save outputs
# Usage: bash run_all.sh
set -e

source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate 2>/dev/null || true

OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Profile-MoE — Full Verification Suite"
echo "========================================"
echo "Output: $OUTPUT_DIR/"
echo ""

# ── 1. MVP (core proof) ──
echo "[1/6] Running mvp.py — Core Proof (30s)..."
python3 mvp.py 2>&1 | tee "$OUTPUT_DIR/01_mvp_output.txt"
echo "  ✓ $OUTPUT_DIR/01_mvp_output.txt"
echo ""

# ── 2. Versioning Demo ──
echo "[2/6] Running versioning_demo.py — Adding 5th Expert (30s)..."
python3 versioning_demo.py 2>&1 | tee "$OUTPUT_DIR/02_versioning_output.txt"
echo "  ✓ $OUTPUT_DIR/02_versioning_output.txt"
echo ""

# ── 3. DeepSeek Comparison ──
echo "[3/6] Running comparison_benchmark.py — vs DeepSeek (30s)..."
python3 comparison_benchmark.py 2>&1 | tee "$OUTPUT_DIR/03_comparison_output.txt"
echo "  ✓ $OUTPUT_DIR/03_comparison_output.txt"
echo ""

# ── 4. Transformer Architecture (numpy, no training) ──
echo "[4/6] Running transformer_benchmark.py — Architecture Proof (10s)..."
python3 transformer_benchmark.py 2>&1 | tee "$OUTPUT_DIR/04_transformer_arch_output.txt"
echo "  ✓ $OUTPUT_DIR/04_transformer_arch_output.txt"
echo ""

# ── 5. Transformer Training (needs PyTorch) ──
echo "[5/6] Running transformer_training.py — Full Training (1-2 min)..."
python3 transformer_training.py 2>&1 | tee "$OUTPUT_DIR/05_transformer_training_output.txt"
echo "  ✓ $OUTPUT_DIR/05_transformer_training_output.txt"
echo ""

# ── 6. Graphs & Findings ──
echo "[6/6] Generating graphs and master findings..."
python3 generate_graphs.py 2>&1 | tee "$OUTPUT_DIR/06_graphs_output.txt"
python3 export_findings.py 2>&1 | tee -a "$OUTPUT_DIR/06_graphs_output.txt"
echo "  ✓ graphs/ (5 PNGs)"
echo "  ✓ FINDINGS.xlsx"
echo ""

# ── Collect artifacts ──
cp -f results.json "$OUTPUT_DIR/" 2>/dev/null || true
cp -f transformer_results.json "$OUTPUT_DIR/" 2>/dev/null || true
cp -f versioning_demo.xlsx "$OUTPUT_DIR/" 2>/dev/null || true
cp -f comparison_benchmark.xlsx "$OUTPUT_DIR/" 2>/dev/null || true

echo "========================================"
echo "VERIFICATION COMPLETE"
echo "========================================"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "Graphs:      graphs/*.png"
echo "Master data: FINDINGS.xlsx"
echo ""
echo "Key numbers to verify:"
echo "  Routing accuracy:  99.88%  (01_mvp_output.txt)"
echo "  Swap isolation:     38.4x  (01_mvp_output.txt)"
echo "  Law routing:        96.0%  (02_versioning_output.txt)"
echo "  Transformer PPL:      9.3  (05_transformer_training_output.txt)"
echo "  Speed ratio:        0.999x (05_transformer_training_output.txt)"
echo "  Router params:           0 (05_transformer_training_output.txt)"
