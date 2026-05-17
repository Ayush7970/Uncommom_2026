#!/usr/bin/env bash
# Build the submission zip for Sigma Lab AI Forecasting Track
# Run from ai-prophet/ directory: bash forecast/scripts/make_submission.sh

set -e

SUBMISSION_NAME="uprising-prophet-forecast"
ZIP_OUT="../${SUBMISSION_NAME}.zip"

echo "Building submission: ${SUBMISSION_NAME}.zip"

# Ensure outputs exist
mkdir -p forecast/outputs

# Run final benchmark to generate outputs
echo ""
echo "Running final benchmark..."
source venv/bin/activate
python forecast/scripts/train_ml.py --all
python forecast/scripts/fit_calibrator.py
python forecast/scripts/run_replay.py --dataset small
echo "Benchmark complete."

# Build zip (exclude venv, caches, secrets, large binaries)
echo ""
echo "Packaging..."
cd ..
zip -r "${SUBMISSION_NAME}.zip" ai-prophet/forecast/ \
  --exclude "*/venv/*" \
  --exclude "*/__pycache__/*" \
  --exclude "*/.env" \
  --exclude "*/forecast_log.db" \
  --exclude "*/.DS_Store" \
  --exclude "*/node_modules/*"

echo ""
echo "Done: ${SUBMISSION_NAME}.zip"
echo ""
echo "Submission contents:"
unzip -l "${SUBMISSION_NAME}.zip" | head -40
echo ""
echo "Run command for judges:"
echo "  cd ai-prophet/forecast"
echo "  pip install -r requirements.txt"
echo "  python scripts/train_ml.py --all"
echo "  python scripts/fit_calibrator.py"
echo "  python scripts/run_replay.py --dataset small"
echo "  streamlit run dashboard/streamlit_app.py"
