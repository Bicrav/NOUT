#!/bin/bash
# NOUT — update to the latest version from GitHub. Double-click to run.
set -e
cd "$(dirname "$0")"
echo "→ Fetching the latest version…"
if [ -d ".git" ]; then
  git pull --ff-only
else
  echo "This folder is not a git clone; re-download the latest release from GitHub."
fi
echo "→ Updating Python packages…"
source .venv/bin/activate
pip install -r requirements.txt
echo "✅ Up to date. Relaunch with run.command"
read -n 1 -s -r -p "Press any key to close."
