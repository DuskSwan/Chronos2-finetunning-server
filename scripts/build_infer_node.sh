#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-./.venv/bin/python}"
DIST_DIR="${DIST_DIR:-dist}"
CLEAN="${CLEAN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 1
fi

echo "[build] install pyinstaller"
if ! "$PYTHON_EXE" -m pip --version >/dev/null 2>&1; then
  echo "[build] pip not found, running ensurepip"
  "$PYTHON_EXE" -m ensurepip --upgrade
fi

"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install pyinstaller

PYI_ARGS=(
  -m PyInstaller
  --noconfirm
  --onedir
  --name chronos_infer_node
  --distpath "$DIST_DIR"
  --collect-all chronos
  --collect-all transformers
  --collect-all accelerate
  --collect-all peft
  --collect-all tokenizers
  --hidden-import torch
  --hidden-import pandas
  --hidden-import numpy
  --hidden-import zmq
  app/cli/infer_node_cli.py
)

if [[ "$CLEAN" == "1" ]]; then
  PYI_ARGS=(
    -m PyInstaller
    --clean
    --noconfirm
    --onedir
    --name chronos_infer_node
    --distpath "$DIST_DIR"
    --collect-all chronos
    --collect-all transformers
    --collect-all accelerate
    --collect-all peft
    --collect-all tokenizers
    --hidden-import torch
    --hidden-import pandas
    --hidden-import numpy
    --hidden-import zmq
    app/cli/infer_node_cli.py
  )
fi

echo "[build] pyinstaller start"
"$PYTHON_EXE" "${PYI_ARGS[@]}"

BIN_PATH="$REPO_ROOT/$DIST_DIR/chronos_infer_node/chronos_infer_node"
if [[ ! -f "$BIN_PATH" ]]; then
  echo "Build finished but binary not found: $BIN_PATH" >&2
  exit 1
fi

echo "[build] success: $BIN_PATH"
