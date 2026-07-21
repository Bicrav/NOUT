#!/bin/bash
# NOUT — one-time installer (macOS). Double-click to run.
set -e
cd "$(dirname "$0")"
echo "==============================================="
echo "  NOUT — installation"
echo "==============================================="

# 1) Homebrew (package manager for the native dependencies)
if ! command -v brew >/dev/null 2>&1; then
  echo "→ Homebrew is required but not installed."
  echo "  Install it from https://brew.sh (copy the one-line command), then re-run this."
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi

# 2) Native dependencies (camera control, location, SVG)
echo "→ Installing native dependencies (gphoto2, cairo)…"
brew install libgphoto2 gphoto2 pkg-config cairo || true

# 3) Python 3 + virtual environment (isolated, nothing touches your system Python)
if ! command -v python3 >/dev/null 2>&1; then
  echo "→ Installing Python 3…"; brew install python
fi
echo "→ Creating the virtual environment (.venv)…"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
echo "→ Installing Python packages (this can take a few minutes)…"
pip install -r requirements.txt

echo ""
echo "✅ Installation complete."
echo "   Launch NOUT by double-clicking  run.command"
read -n 1 -s -r -p "Press any key to close."
