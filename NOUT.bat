@echo off
title NOUT
rem Launches NOUT in WSL, with the camera (Canon 250D or Sony A7 II) if plugged in.

rem Start WSL so the camera can be attached to it.
wsl -d Ubuntu-24.04 -e true

rem Attach the camera (auto-attach keeps it attached if it sleeps or is replugged).
set "CAM="
usbipd list | findstr /i "04a9:32e9" >nul && set "CAM=04a9:32e9"
usbipd list | findstr /i "054c:0a6a" >nul && set "CAM=054c:0a6a"
if defined CAM (
  start "NOUT-camera" /min usbipd attach --wsl --auto-attach --hardware-id %CAM%
  timeout /t 5 /nobreak >nul
) else (
  echo No camera detected - NOUT starts without it.
)

rem Run NOUT.
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/NOUT && source .venv/bin/activate && python sony_tether_focus.py"
if errorlevel 1 pause

rem Give the camera back to Windows.
taskkill /fi "WINDOWTITLE eq NOUT-camera*" /f >nul 2>&1
