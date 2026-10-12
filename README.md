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

**Observing at the eyepiece** — a tab with no camera in it: the sky map, a search
over the Messier and named NGC objects and the bright stars, ranked by how high they
are now and filtered by your own horizon, and a GoTo. The right ascension is driven by
the motor; the declination, which this mount has no motor for, comes back as an angle
to set by hand on the graduated circle rather than being quietly left out.

**On your phone** — `docs/index.html` is the same pointing maths in one 34 kB page:
both dial readings with the 24 h scale drawn, Polaris' place on the polar-scope
reticle, and a calibration that works out your dials' zero and direction by syncing on
one or two stars. It needs no network, no camera and no mount connection — a manual
GoTo is pure arithmetic — so it runs offline on a phone while the computer stays at
home. Its readings match the Python to 1e-13 degrees.

**Showing faint things** — nebulosity sits at a few per cent of the sky: measured on
a real session it came to 3 % of the sky level and 1/2500 of the brightest star, at a
signal-to-noise of 1.2 per pixel, which no stretch can reveal because it is under the
noise rather than under the black point. The display separates the smooth sky from the
stars, averages only the smooth part, and puts the stars back sharp — the same stack
then shows the nebula at 21 levels of contrast instead of 12. A "nebulosity only" view
takes the stars out altogether, with StarNet++ when it is installed (3042 stars to 21,
about ten seconds) and a built-in separation otherwise.

**Focus** — a Bahtinov mask reader: the three diffraction spikes are found with a Radon
transform and the middle one's distance from the crossing of the other two is the focus
error, to a tenth of a pixel. Also live sharpness metrics and star-count focusing.

**Mount** — Star Adventurer 2i over Wi-Fi: tracking rates (sidereal, lunar, solar,
planetary), motorised slews in RA, dithering between exposures, and a manual GoTo that
computes both angles for a rig whose axes are read from engraved dials. The rate is
checked against the axis's own step counter, so a mount that quietly stopped or that is
tracking at the wrong rate for the target shows up. A measurement follows a star on the
live view for a few worm turns and reports the periodic error, what is left of the polar
alignment, and — read straight off the measured track — the longest exposure that stays
sharp, at any focal length.

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

```bat
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python sony_tether_focus.py
```

Tethered capture does not work on Windows: libgphoto2, which talks to the camera, has
no Windows build. Everything else does — planning, sky map, plate solving, stacking a
folder of exposures already on disk, lucky imaging, HDMI live view through a capture
card, and the mount over Wi-Fi. Use `--simulate` to run the app with a simulated
camera; that is how most of it is tested.

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
