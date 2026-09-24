# NOUT

Deep-sky astrophotography with a tethered camera and a star tracker. NOUT drives the
camera, stacks the exposures as they arrive, and shows the target building up on
screen — close to what an electronic telescope does, but with your own camera and your
own optics.

Named after Nut, the Egyptian sky goddess arching over the world.

## What it does

**Capture** — tethered control over USB (libgphoto2): ISO, shutter, aperture, single
shots, intervalometer, bulb sequences, bursts. Live view with a red centre reticle and
a sky-level exposure indicator that tells you whether a sub is well exposed.

**Live stacking** — every exposure is corrected for hot pixels (no darks needed),
registered on the stars (sub-pixel; 0.03 px measured against known shifts), weighted by
its own quality and integrated into a running mean with kappa-sigma rejection. Frames
spoiled by cloud, a satellite trail or lost tracking are dropped.

**Observation look** — the stack is displayed the way a smart telescope would show it:
a sky model that ignores extended objects, neutral background, star-based colour
calibration, green removal (SCNR), a colour-preserving arcsinh stretch that does not
burn galaxy cores, noise reduction and gentle local contrast. GraXpert is used for the
sky model when it is installed.

**Public view** — the picture full screen on a second display, with the target's name
and distance, for visitors.

**Mount** — Star Adventurer 2i over Wi-Fi: tracking rates (sidereal, lunar, solar,
planetary), motorised slews in RA, dithering between exposures, and a manual GoTo that
computes both angles for a rig whose axes are read from engraved dials.

**Sky map and planning** — all-sky and framing views, constellations, planets, comets,
the ISS, survey images as background, a target list ranked for the night with Moon
avoidance and framing fit, and a local horizon you draw yourself (trees, houses) that
is taken into account when ranking targets.

**Plate solving** — astrometry.net, local or online, with catalogue objects drawn over
the stack once the field is identified.

**Moon and planets** — lucky imaging: the sharpest frames are kept, aligned sub-pixel
and stacked, with multi-scale sharpening. Frames can come from the camera's live view,
from an in-camera movie, or from an HDMI capture card recorded uncompressed to SER.

## Install

Python 3.10 or newer, then:

```bash
git clone https://github.com/Bicrav/NOUT.git
cd NOUT
python3 -m venv .venv
```

**macOS**

```bash
brew install libgphoto2
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python sony_tether_focus.py
```

**Linux** (Debian/Ubuntu names; use your distribution's equivalents)

```bash
sudo apt install libgphoto2-dev pkg-config
sudo usermod -aG video $USER     # for an HDMI capture card, then log out and back in
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python sony_tether_focus.py
```

**Windows**

NOUT needs Linux, because libgphoto2 has no Windows build. The simplest way to run it
on Windows is WSL2, and `NOUT.bat` sets it up for you. Double-click it; do not use
*Run as administrator*. It checks each step, installs only what is missing, then starts
NOUT:

1. **WSL2.** It asks for administrator rights, then a reboot. Run `NOUT.bat` again afterwards.
2. **RAM for WSL.** It creates `%UserProfile%\.wslconfig` with memory sized to the PC
   (at most 8 GB), because WSL crashes on capture without enough RAM. An existing file is
   left alone.
3. **Ubuntu 24.04.** Choose a Linux user name and password, then type `exit`.
4. **[usbipd-win](https://github.com/dorssel/usbipd-win).** It passes the USB camera
   through to Linux.
5. **Linux packages, NOUT (`~/NOUT`) and its venv.** sudo asks for your Linux password.

On later runs it only checks, which takes a few seconds, then attaches the camera to
WSL and launches NOUT. Close EOS Utility or Imaging Edge first.

Other ways to get Linux on a Windows PC are a VirtualBox or VMware virtual machine with
USB passthrough (give the VM at least 8 GB of RAM), or a dual boot. Then follow the
Linux steps above.

### Optional

- [GraXpert](https://graxpert.com) — AI background extraction, found automatically on
  all three systems if installed
- [Siril](https://siril.org) — final processing, opened from the Results tab
- astrometry.net index files — for plate solving without Internet
- `ffmpeg` on PATH — so the HDMI capture card is listed by name rather than by number

## Hardware it was built against

- Sony α7 II over USB (other cameras supported by libgphoto2 should work, with fewer
  guarantees)
- Sky-Watcher Star Adventurer 2i Wi-Fi
- Evostar 72ED refractor, with and without reducer or Barlow

A camera is not required to try it: `--simulate` gives a synthetic one.

## Packaging as a macOS app

`packaging/macos/` holds a launcher script and the `Info.plist` used to wrap the code
in a `.app` bundle, including the camera and location usage descriptions macOS asks
for. Nothing there is needed to run from source.

## Safety

Never point the camera, the finder or the polar scope at the Sun without a certified
full-aperture solar filter in front of the objective. It destroys the sensor, and your
eyes.

## State of the project

A personal tool, written for one rig and one observer, growing with what that rig
needs — so expect rough edges elsewhere. Settings, target lists and horizon profiles
live in the system's own settings store, not in this repository.

## Licence

MIT — see [LICENSE](LICENSE). Use it, change it, share it.
