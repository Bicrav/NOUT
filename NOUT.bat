@echo off
setlocal EnableExtensions
title NOUT
rem NOUT.bat - installe ce qui manque, puis lance NOUT (Windows + WSL2).
rem 1re fois : WSL2, Ubuntu, usbipd-win, paquets Linux, NOUT et son venv.
rem Ensuite  : verification rapide, puis lancement.
rem Double-clic, SANS "Executer en tant qu'administrateur".

rem ================= Configuration =================
set "DISTRO=Ubuntu-24.04"
set "SONY=054c:0a6a"
set "CANON=04a9:32e9"
rem =================================================
set "WSL_UTF8=1"

rem --- 1. WSL2 ---
wsl --version >nul 2>&1 && goto :wsl_ok
echo [1/5] Installation de WSL2 (fenetre administrateur)...
powershell -NoProfile -Command "Start-Process wsl -ArgumentList '--install','--no-distribution' -Verb RunAs -Wait"
echo.
echo WSL2 installe. REDEMARREZ le PC, puis relancez NOUT.bat.
pause
exit /b 0
:wsl_ok
echo [1/5] WSL2 : OK

rem --- 2. Memoire allouee a WSL (cree .wslconfig si absent) ---
if exist "%UserProfile%\.wslconfig" goto :cfg_exist
powershell -NoProfile -Command "$r=[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB); $m=[math]::Max(4,[math]::Min(8,$r-3)); Set-Content -Encoding ascii -Path ($env:USERPROFILE+'\.wslconfig') -Value @('[wsl2]',('memory='+$m+'GB'),'swap=4GB','networkingMode=mirrored'); Write-Host ('      .wslconfig cree : '+$m+' Go pour WSL sur '+$r+' Go')"
wsl --shutdown
goto :cfg_ok
:cfg_exist
findstr /i /r /c:"^ *memory" "%UserProfile%\.wslconfig" >nul || echo [ATTENTION] .wslconfig sans ligne memory= : WSL risque de manquer de RAM.
:cfg_ok
echo [2/5] Memoire WSL : OK

rem --- 3. Distribution Ubuntu ---
wsl -l -q 2>nul | findstr /i /c:"%DISTRO%" >nul && goto :distro_ok
for /f "delims=" %%d in ('wsl -l -q 2^>nul ^| findstr /i "ubuntu"') do (set "DISTRO=%%d" & goto :distro_ok)
echo [3/5] Installation de %DISTRO%...
echo       Choisissez un nom d'utilisateur et un mot de passe Linux,
echo       puis tapez  exit  pour continuer.
wsl --install -d %DISTRO%
wsl -l -q 2>nul | findstr /i /c:"%DISTRO%" >nul && goto :distro_ok
echo Installation inachevee : redemarrez le PC si demande, puis relancez NOUT.bat.
pause
exit /b 1
:distro_ok
echo [3/5] Distribution : %DISTRO%

rem --- 4. usbipd-win (passage de l'appareil photo USB vers WSL) ---
set "USBIPD=usbipd"
where usbipd >nul 2>&1 && goto :usb_ok
set "USBIPD=%ProgramFiles%\usbipd-win\usbipd.exe"
if exist "%USBIPD%" goto :usb_ok
echo [4/5] Installation de usbipd-win...
winget install --id dorssel.usbipd-win -e --accept-source-agreements --accept-package-agreements
if exist "%USBIPD%" goto :usb_ok
echo [ERREUR] usbipd-win non installe : github.com/dorssel/usbipd-win
pause
exit /b 1
:usb_ok
echo [4/5] usbipd-win : OK

rem --- 5. Paquets Linux, NOUT, venv (script bash en fin de fichier) ---
start "NOUT-keepalive" /min wsl -d %DISTRO% -e sleep infinity
wsl -d %DISTRO% --cd "%~dp0." -e bash -c "sed -n '/^: BASH_START/,$p' '%~nx0' | tr -d '\r' > /tmp/nout.sh"
wsl -d %DISTRO% -e bash /tmp/nout.sh --setup-only
if errorlevel 1 (
  echo [ERREUR] Installation cote Linux echouee ^(voir ci-dessus^).
  pause
  goto :cleanup
)
echo [5/5] Linux, NOUT et dependances : OK

rem --- Appareil photo ---
set "HWID="
"%USBIPD%" list 2>nul | findstr /i "%SONY%"  >nul && (set "HWID=%SONY%"  & set "CAM=Sony A7 II")
"%USBIPD%" list 2>nul | findstr /i "%CANON%" >nul && (set "HWID=%CANON%" & set "CAM=Canon EOS 250D")
if defined HWID goto :cam_found
echo.
echo Aucun appareil photo detecte ^(allume, en USB, EOS Utility ferme ?^).
choice /c ON /m "Lancer NOUT quand meme, sans camera"
if errorlevel 2 goto :cleanup
goto :launch

:cam_found
echo Appareil detecte : %CAM% [%HWID%]
"%USBIPD%" list 2>nul | findstr /i "%HWID%" | findstr /i /c:"Not shared" >nul
if errorlevel 1 goto :attach
echo Partage USB de l'appareil ^(fenetre administrateur^)...
powershell -NoProfile -Command "Start-Process -FilePath '%USBIPD%' -ArgumentList 'bind','--force','--hardware-id','%HWID%' -Verb RunAs -Wait"
:attach
start "NOUT-usbipd" /min "%USBIPD%" attach --wsl --auto-attach --hardware-id %HWID%
echo Attente de la camera dans WSL...
set /a N=0
:wait
wsl -d %DISTRO% -e bash -c "gphoto2 --auto-detect 2>/dev/null | tail -n +3 | grep -q ." && goto :camok
set /a N+=1
if %N% geq 15 (
  echo [ATTENTION] Camera non vue par gphoto2. Debrancher/rebrancher peut aider.
  goto :launch
)
timeout /t 2 /nobreak >nul
goto :wait
:camok
echo Camera prete.

:launch
echo Lancement de NOUT...
wsl -d %DISTRO% -e bash /tmp/nout.sh
if errorlevel 1 (
  echo [ERREUR] NOUT s'est arrete avec une erreur ^(voir ci-dessus^).
  pause
)

:cleanup
taskkill /fi "WINDOWTITLE eq NOUT-usbipd*" /t /f >nul 2>&1
if defined HWID "%USBIPD%" detach --hardware-id %HWID% >nul 2>&1
taskkill /fi "WINDOWTITLE eq NOUT-keepalive*" /t /f >nul 2>&1
endlocal
exit /b 0

: BASH_START
# ---- Partie Linux, extraite et executee dans WSL ----
set -e
PROJ="${NOUT_DIR:-$HOME/NOUT}"
REPO=https://github.com/Bicrav/NOUT.git
PKGS="git gphoto2 libgphoto2-dev pkg-config python3-venv python3-dev build-essential libgl1 libglib2.0-0 usbutils"

missing=$(for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || echo "$p"; done)
if [ -n "$missing" ]; then
  echo "Paquets Linux a installer :" $missing
  echo "(mot de passe Linux demande par sudo)"
  sudo apt-get update && sudo apt-get install -y $missing
fi

f=$(ls "$PROJ"/sony_tether_focus.py "$PROJ"/*/sony_tether_focus.py 2>/dev/null | head -n1 || true)
if [ -z "$f" ]; then
  echo "Telechargement de NOUT dans $PROJ..."
  git clone "$REPO" "$PROJ"
  f="$PROJ/sony_tether_focus.py"
fi
cd "$(dirname "$f")"
echo "      Projet : $PWD"

v=$(ls .venv*/bin/activate 2>/dev/null | head -n1 || true)
if [ -z "$v" ]; then
  echo "Creation du venv..."
  python3 -m venv .venv
  v=.venv/bin/activate
fi
. "$v"
stamp="${v%/bin/activate}/.nout_deps_ok"
if [ ! -f "$stamp" ] || [ requirements.txt -nt "$stamp" ]; then
  echo "Installation des dependances Python..."
  pip install -r requirements.txt
  python -c "import cv2" 2>/dev/null || pip install opencv-python
  touch "$stamp"
fi

[ "$1" = "--setup-only" ] && exit 0
exec python -W ignore::DeprecationWarning sony_tether_focus.py
