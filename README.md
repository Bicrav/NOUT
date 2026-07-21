# NOUT

Tethered capture and live astrophotography for **macOS** — built around a Sony camera
but works with most cameras supported by gphoto2.

Live view with focus assist (HFR), intervalometer, live stacking (calibration, gradient
removal, Siril-style auto-stretch), plate-solving, sky map with target GoTo, mount control
(Star Adventurer 2i and HEQ-5 Pro over Wi-Fi) with dithering, session automation, and a
lucky-imaging module for the **Moon & Sun** (sub-pixel align + multi-scale sharpening).

## Requirements

- macOS
- A camera supported by [gphoto2](http://www.gphoto.org/proj/libgphoto2/support.php),
  connected by USB
- [Homebrew](https://brew.sh) (the installer uses it for the native dependencies)

## Install

1. Get the code: green **Code** button → **Download ZIP**, then unzip — or, better,
   `git clone` it so you can update in one click.
2. Double-click **`install.command`**.
   - If macOS blocks it ("unidentified developer"): right-click → **Open** → **Open**.
   - It installs the native tools (gphoto2, cairo) and a self-contained Python environment.
     Nothing touches your system Python.
3. Double-click **`run.command`** to launch NOUT.

## Update

- **Cloned with git:** double-click **`update.command`** — or in the app, **Help → Check for
  updates**.
- **Downloaded the ZIP:** download the latest ZIP again and re-run `install.command`.

## Optional

- **GraXpert** (AI gradient removal): install the GraXpert app; NOUT detects it automatically.
- **Full LDN catalog**: run `python3 fetch_ldn.py` once to download every Lynds dark nebula
  into `catalog_ldn.csv` (appears on the Sky Map and preview overlay).

## Quick start — the Moon (best quality on Sony)

1. Point at the Moon, short exposure (1/250–1/500 s), low ISO.
2. Shooting tab → intervalometer: Pause 0, **Stability pause 2 s**, ~120 shots → Start.
3. 🪐 Planetary → **Lucky-stack full-res PHOTOS** → pick the folder → Save.

## Troubleshooting

- *"gphoto2 module not installed"* → re-run `install.command`.
- *Camera not detected* → close other apps using it (Imaging Edge, Photos), replug USB,
  set the camera to **PC Remote / PTP** mode.
- *Frequent disconnects (Sony)* → raise the **Stability pause**, use a short quality USB
  cable, battery instead of USB power, turn off in-camera Wi-Fi/LENR.

## License

MIT — see `LICENSE` (replace `<YOUR NAME>` with yours).
