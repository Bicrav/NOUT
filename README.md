# NOUT

A macOS app for deep-sky astrophotography with a tethered camera and a star tracker.
It drives the camera, stacks the exposures as they arrive, and shows the target
building up on screen — close to what an electronic telescope does, but with your own
camera and your own optics.

Named after Nut, the Egyptian sky goddess arching over the world.

## What it does

**Capture** — tethered control of the camera over USB (libgphoto2): ISO, shutter,
aperture, single shots, intervalometer, bulb sequences, burst mode. Live view with a
red centre reticle, focus aids and a sky-level exposure indicator.

**Live stacking** — every exposure is corrected for hot pixels (no darks needed),
registered on the stars (sub-pixel: 0.03 px measured), weighted by its own quality and
integrated into a running mean, with kappa-sigma rejection. Frames spoiled by cloud,
a satellite trail or lost tracking are dropped.

**Observation look** — the display of the stack is processed like a smart telescope
would: sky model that ignores extended objects, neutral background, star-based colour
calibration, green removal (SCNR), colour-preserving arcsinh stretch that does not burn
galaxy cores, noise reduction, gentle local contrast. GraXpert is used for the sky
model when it is installed.

**Public view** — the picture full screen on a second display, with the target's name
and distance, for visitors.

**Mount** — Star Adventurer 2i over Wi-Fi: tracking rates (sidereal, lunar, solar,
planetary), motorised slews on the RA axis, dithering between exposures, and a manual
GoTo that computes both axis angles for a two-axis rig read from engraved dials.

**Sky map and planning** — all-sky and framing views, constellations, planets, comets,
the ISS, survey images as background, a target list ranked for the night with Moon
avoidance and framing fit, and a local horizon you draw yourself (trees, houses) that
is taken into account when ranking targets.

**Plate solving** — astrometry.net, local or online, with catalogue objects drawn over
the stack once the field is identified.

**Moon and planets** — lucky imaging: the sharpest frames are kept, aligned sub-pixel
and stacked, with multi-scale sharpening. Video can come from the camera's live view,
from an in-camera movie, or from an HDMI capture card recorded uncompressed to SER.

## Hardware it was built against

- Sony α7 II over USB (any camera libgphoto2 supports should work, with fewer
  guarantees)
- Sky-Watcher Star Adventurer 2i Wi-Fi
- Evostar 72ED refractor, with and without reducer or Barlow

## Install

Clone this repository, then double-click **install.command** (it uses Homebrew for
libgphoto2 and builds a local `.venv`). Start the app with **run.command**, and update
it later with **update.command**.

By hand:

```bash
brew install libgphoto2
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python sony_tether_focus.py
```

`--simulate` runs the whole app without a camera, which is how most of it is tested.

Optional: [GraXpert](https://graxpert.com) for AI background extraction,
[Siril](https://siril.org) for final processing, and the astrometry.net index files if
you want to plate-solve offline.

On macOS the app can also be packaged as a `.app` bundle; `packaging/` holds the
launcher script and the `Info.plist` used for that, including the camera and location
usage descriptions macOS requires.

## Safety

Never point the camera, the finder or the polar scope at the Sun without a certified
full-aperture solar filter in front of the objective. It destroys the sensor, and your
eyes.

## State of the project

This is a personal tool, written for one rig and one observer, and it grows with what
that rig needs. Expect rough edges elsewhere. Settings, target lists and horizon
profiles are stored in the macOS preferences of the app, not in this repository.

## Licence

MIT — see [LICENSE](LICENSE). Use it, change it, share it.
