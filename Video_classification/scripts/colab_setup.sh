#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel

# Core pipeline deps
python -m pip install av PyYAML matplotlib

# Optional experimental VLM backbones
python -m pip install transformers accelerate

echo "[colab_setup] Core deps installed."
echo "[colab_setup] For MMAction2 (optional), run:"
echo "  pip install -U openmim"
echo "  mim install mmengine"
echo "  mim install 'mmcv>=2.0.0'"
echo "  mim install mmaction2"
