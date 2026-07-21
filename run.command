#!/bin/bash
# NOUT — launcher. Double-click to start the app.
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "NOUT is not installed yet — run install.command first."
  read -n 1 -s -r -p "Press any key to close."; exit 1
fi
source .venv/bin/activate
exec python sony_tether_focus.py
