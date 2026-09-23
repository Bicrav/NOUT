#!/bin/bash
APP="$(cd "$(dirname "$0")/../.." && pwd)"
RES="$APP/Contents/Resources"
export NOUT_BUNDLE="$APP"
export NOUT_ASTROMETRY_CFG="$RES/astrometry.cfg"
LOG="$HOME/Library/Logs/NOUT.log"
cd "$RES/app" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:$PATH"
killall ptpcamerad PTPCamera 2>/dev/null
"$RES/venv/bin/python" "$RES/app/sony_tether_focus.py" >"$LOG" 2>&1
code=$?
if [ $code -ne 0 ]; then
  /usr/bin/osascript -e "display dialog \"NOUT s'est fermé (code $code).
Journal : $LOG\" buttons {\"OK\"} with title \"NOUT\" with icon caution" >/dev/null 2>&1
fi
