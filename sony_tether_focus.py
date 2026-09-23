#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sony_tether_focus.py
====================

Tethered control (via USB cable) of a Sony A7II on macOS / Linux:

  * Real-time live view (via gphoto2 capture_preview)
  * Intervalometer (periodic triggering, saving to Mac on each shot)
  * Live sharpness curve (manual focus aid)
      -> metrics: Tenengrad (Sobel), Laplacian variance, Normalized variance (Groen)
      -> calculated on an adjustable central ROI
  * ISO / shutter speed / aperture settings read from the camera
  * Motorized focus control "best effort" (present on Canon; generally
    ABSENT on A7II — buttons self-disable if the camera doesn't expose it)

  * --simulate mode: everything works WITHOUT a camera (synthetic frames with a real
    sharpness peak), useful for testing the UI/curve before having the camera.

CAMERA REQUIREMENTS (real):
  - On the body: Menu -> USB Connection -> "PC Remote" (otherwise it mounts as
    "Mass Storage" and no control is possible).
  - macOS: no Apple "PTPCamera" should lock the port. If capture
    fails with "Could not claim the USB device", kill the process:
        killall PTPCamera

INSTALLATION:
    python3 -m venv venv && source venv/bin/activate
    pip install gphoto2 opencv-python-headless numpy PySide6 pyqtgraph
    # (libgphoto2 must be installed: `brew install libgphoto2` on macOS,
    #  `sudo apt install libgphoto2-dev` on Debian/Ubuntu)

LAUNCH:
    python sony_tether_focus.py --simulate          # without camera, to discover the UI
    python sony_tether_focus.py                     # with the A7II plugged in PC Remote
    python sony_tether_focus.py --save-dir ~/Shots --metric tenengrad
"""

import argparse
import csv
import glob
import json
import math
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

# Make sibling modules (sky_map, astro_targets, satellites…) importable no matter
# how NOUT is launched (double-clicked .app, cwd elsewhere, etc.).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot


# ---------------------------------------------------------------------------
#  Astro Translations
# ---------------------------------------------------------------------------
def _translate_target(name):
    """Translate common French target names from astro_targets catalog to English."""
    reps = {
        "Nébuleuse": "Nebula", "Galaxie": "Galaxy", "Amas": "Cluster",
        "Étoile": "Star", "du Crabe": "Crab", "de l'Aigle": "Eagle",
        "d'Orion": "Orion", "d'Andromède": "Andromeda", "Sombrero": "Sombrero",
        "du Cône": "Cone", "Rosette": "Rosette", "de la Lyre": "Ring",
        "Haltère": "Dumbbell", "Tourbillon": "Whirlpool", "du Moulinet": "Pinwheel",
        "du Triangle": "Triangulum", "Ouvert": "Open", "Globulaire": "Globular",
        "de la lagune": "Lagoon", "Trifide": "Trifid", "Oméga": "Omega",
        "d'Hercule": "Hercules", "Pléiades": "Pleiades", "Tête de Cheval": "Horsehead"
    }
    for fr, en in reps.items():
        name = name.replace(fr, en)
    return name


# ---------------------------------------------------------------------------
#  Sharpness Metrics
# ---------------------------------------------------------------------------
def _to_gray_roi(bgr, roi_frac, max_side=512):
    """Extracts a central ROI (fraction of the frame) in grayscale, downscaled."""
    h, w = bgr.shape[:2]
    rw, rh = int(w * roi_frac), int(h * roi_frac)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    roi = bgr[y0:y0 + rh, x0:x0 + rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # downscale to limit computation cost, without destroying useful high frequency
    s = max(gray.shape)
    if s > max_side:
        f = max_side / s
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return gray, (x0, y0, rw, rh)


def sharpness_tenengrad(gray):
    """Gradient energy (Sobel). Robust, good default for focusing."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def sharpness_laplacian(gray):
    """Laplacian variance. Classic, but sensitive to noise."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def sharpness_norm_var(gray):
    """Normalized variance (Groen): insensitive to lighting variations."""
    g = gray.astype(np.float32)
    mu = g.mean()
    if mu < 1e-6:
        return 0.0
    return float(((g - mu) ** 2).mean() / mu)


METRICS = {
    "tenengrad": sharpness_tenengrad,
    "laplacian": sharpness_laplacian,
    "norm_var": sharpness_norm_var,
}


# rawpy module cache (lazy import, optional)
_RAWPY = None
_RAWPY_TRIED = False


LENS_PROFILES = {
    "Sony FE 35 mm F2.8 ZA": {"fmin": 35, "fmax": 35, "amax": 2.8},
    "Tamron 70-180 mm F2.8 G2": {"fmin": 70, "fmax": 180, "amax": 2.8},
    "Sony FE 50 mm F1.8": {"fmin": 50, "fmax": 50, "amax": 1.8},
    # Telescopes: the aperture is fixed by the glass, so amax IS the focal ratio.
    # 72 mm at 420 mm is f/5.8; the 0.85x reducer shortens it to 357 mm, which at
    # the same 72 mm of aperture is f/5.0 — a full stop and a bit faster, and a
    # wider field, which is the whole point of fitting one.
    "Evostar 72ED (420 mm f/5.8)": {"fmin": 420, "fmax": 420, "amax": 5.8, "scope": True},
    "Evostar 72ED + reducer ×0.85 (357 mm f/5)": {"fmin": 357, "fmax": 357, "amax": 5.0, "scope": True},
    # 420 mm × 2.5 = 1050 mm; the Barlow multiplies the focal length but not the
    # 72 mm of glass, so the focal ratio goes the same way: f/5.8 becomes f/14.6.
    "Evostar 72ED + Barlow ×2.5 (1050 mm f/14.6)": {"fmin": 1050, "fmax": 1050,
                                                    "amax": 14.6, "scope": True},
    "Custom": None,
}


def fov_deg(focal_mm, sensor_w=35.8, sensor_h=23.9):
    """Field of view (width, height) in degrees for full frame."""
    fw = math.degrees(2 * math.atan(sensor_w / (2 * max(focal_mm, 1))))
    fh = math.degrees(2 * math.atan(sensor_h / (2 * max(focal_mm, 1))))
    return fw, fh


def fov_str(focal_mm):
    w, h = fov_deg(focal_mm)
    return "{:.1f}° × {:.1f}°".format(w, h)


def _ra_to_hms(ra_deg):
    h = (ra_deg % 360.0) / 15.0
    hh = int(h); mm = int((h - hh) * 60); ss = (h - hh - mm / 60.0) * 3600
    return "{:02d}h{:02d}m{:04.1f}s".format(hh, mm, ss)


def _dec_to_dms(dec_deg):
    sign = "-" if dec_deg < 0 else "+"
    a = abs(dec_deg); dd = int(a); mm = int((a - dd) * 60); ss = (a - dd - mm / 60.0) * 3600
    return "{}{:02d}\u00b0{:02d}'{:04.1f}\"".format(sign, dd, mm, ss)


def tracking_match(ref_gray, cur_gray, driftmax_frac=0.04):
    """Compares two frames: returns (pct_stability, drift_px).
    Measures global field offset by phase correlation (robust, without
    depending on star detection). 100% = field still; % reaches 0
    when drift equals driftmax_frac of image width (tolerance)."""
    h, w = ref_gray.shape[:2]
    ms = 700.0
    sc = ms / max(h, w) if max(h, w) > ms else 1.0
    r = cv2.resize(ref_gray, None, fx=sc, fy=sc) if sc != 1.0 else ref_gray
    c = cv2.resize(cur_gray, None, fx=sc, fy=sc) if sc != 1.0 else cur_gray
    r = r.astype(np.float32); c = c.astype(np.float32)
    try:
        win = cv2.createHanningWindow((r.shape[1], r.shape[0]), cv2.CV_32F)
        (dx, dy), _resp = cv2.phaseCorrelate(r, c, win)
    except Exception:                 # noqa: BLE001
        dx, dy = 0.0, 0.0
    drift = float((dx * dx + dy * dy) ** 0.5) / sc        # full image pixels
    driftmax = max(driftmax_frac, 0.005) * max(h, w)      # drift -> 0 %
    pct = 100.0 * max(0.0, 1.0 - drift / driftmax)
    return pct, drift


def detect_stars(gray, max_side=900):
    """Detects stars in a grayscale image (preview).
    Returns (list of (x, y, radius) in image coords, n_stars).
    Background subtraction + thresholding above noise + connected components."""
    h, w = gray.shape[:2]
    scale = 1.0
    g = gray
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # local background (median) then subtraction -> only point sources remain
    bg = cv2.medianBlur(g, 21)
    sub = cv2.subtract(g, bg)
    m, s = float(sub.mean()), float(sub.std())
    thr = max(m + 5.0 * s, 12.0)
    _, mask = cv2.threshold(sub, thr, 255, cv2.THRESH_BINARY)
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    stars = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if 1 <= area <= 120:                       # "star" size (rejects large halos)
            cx, cy = cents[i]
            r = max(2.0, (area ** 0.5))
            stars.append((cx / scale, cy / scale, r / scale))
    return stars, len(stars)


def _median_hfr(gray, stars, rad=6):
    """Median HFR (half-flux radius) on detected stars, in pixels.
    Smaller = sharper. Estimator: flux-weighted radius around each star."""
    g = gray.astype(np.float32)
    bg = float(np.median(g))
    vals = []
    H, W = g.shape[:2]
    for (cx, cy, _r) in stars:
        x0, y0 = int(cx - rad), int(cy - rad)
        x1, y1 = int(cx + rad) + 1, int(cy + rad) + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        patch = g[y0:y1, x0:x1] - bg
        patch[patch < 0] = 0
        flux = float(patch.sum())
        if flux <= 0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        hfr = float((patch * d).sum() / flux)
        if hfr > 0:
            vals.append(hfr)
    return float(np.median(vals)) if vals else 0.0


def _median_eccentricity(gray, stars, rad=6):
    """Median eccentricity of stars (0 = round, ->1 = elongated).
    High eccentricity reveals tracking defects (trailed stars)."""
    g = gray.astype(np.float32)
    bg = float(np.median(g))
    H, W = g.shape[:2]
    vals = []
    for (cx, cy, _r) in stars:
        x0, y0 = int(cx - rad), int(cy - rad)
        x1, y1 = int(cx + rad) + 1, int(cy + rad) + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        patch = g[y0:y1, x0:x1] - bg
        patch[patch < 0] = 0
        m = cv2.moments(patch)
        if m["m00"] <= 0:
            continue
        mu20 = m["mu20"] / m["m00"]
        mu02 = m["mu02"] / m["m00"]
        mu11 = m["mu11"] / m["m00"]
        common = np.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
        l1 = (mu20 + mu02 + common) / 2.0
        l2 = (mu20 + mu02 - common) / 2.0
        if l1 <= 0:
            continue
        vals.append(float(np.sqrt(max(1.0 - l2 / l1, 0.0))))
    return float(np.median(vals)) if vals else 0.0


def sharpness_hfr(gray):
    """Focus score based on HFR (stars). Returns a value where HIGHER = BETTER
    (curve peaks at best focus), like other metrics."""
    stars, n = detect_stars(gray)
    if n == 0:
        return 0.0
    hfr = _median_hfr(gray, stars)
    return 100.0 / (hfr + 0.3) if hfr > 0 else 0.0


METRICS["hfr"] = sharpness_hfr


def _sun_altitude_deg(lat, lon, dt_utc):
    """Low-precision Sun altitude (deg) for a site/time — good to ~0.3°, enough to
    detect twilight thresholds without astropy. dt_utc is a naive UTC datetime."""
    import math
    import calendar
    jd = calendar.timegm(dt_utc.timetuple()) / 86400.0 + 2440587.5
    n = jd - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst = (280.46061837 + 360.98564736629 * n) % 360.0
    ha = math.radians((gmst + lon) % 360.0) - ra
    la = math.radians(lat)
    alt = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha))
    return math.degrees(alt)


def _object_altitude_deg(lat, lon, ra_deg, dec_deg, dt_utc):
    """Altitude (deg) of a fixed RA/Dec object for a site/time (low precision, no astropy).
    RA/Dec in degrees, dt_utc a naive UTC datetime."""
    import math
    import calendar
    jd = calendar.timegm(dt_utc.timetuple()) / 86400.0 + 2440587.5
    n = jd - 2451545.0
    gmst = (280.46061837 + 360.98564736629 * n) % 360.0
    ha = math.radians((gmst + lon - ra_deg) % 360.0)
    la, dec = math.radians(lat), math.radians(dec_deg)
    alt = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha))
    return math.degrees(alt)


def _clipping_fraction(bgr):
    """Fraction of pixels close to saturation (max channel >= 250)."""
    return float((bgr.max(axis=2) >= 250).mean())


def _parse_exposure_seconds(s):
    """Converts a camera shutter speed to seconds: '30'->30, '1/60'->0.0167,
    '0.5'->0.5. Returns None if uninterpretable (e.g. 'bulb')."""
    s = str(s).strip().lower()
    if "bulb" in s or not s:
        return None
    try:
        if "/" in s:
            num, den = s.split("/")
            return float(num) / float(den)
        return float(s)
    except Exception:               # noqa: BLE001
        return None


def red_tint(bgr):
    """Keeps only the red channel (preserves night vision)."""
    out = bgr.copy()
    out[:, :, 0] = 0     # B
    out[:, :, 1] = 0     # G
    return out


_GRAXPERT_EXE = None            # resolved GraXpert executable (set by the app)
_GRAXPERT_LOG = ""              # last GraXpert stderr/error, for diagnostics


def _find_graxpert():
    """Locate a GraXpert executable (macOS app bundle or on PATH)."""
    import glob
    import shutil
    cands = []
    if sys.platform == "darwin":
        # the executable inside the bundle may have a suffix — take whatever is in MacOS/
        for base in ("/Applications/GraXpert.app/Contents/MacOS",
                     os.path.expanduser("~/Applications/GraXpert.app/Contents/MacOS")):
            cands.append(os.path.join(base, "GraXpert"))
            cands += sorted(glob.glob(os.path.join(base, "*")))
    elif os.name == "nt":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramW6432", "")):
            if base:
                cands += [os.path.join(base, "GraXpert", "GraXpert.exe"),
                          os.path.join(base, "Programs", "GraXpert", "GraXpert.exe")]
    else:
        cands += ["/usr/bin/graxpert", "/usr/local/bin/graxpert", "/opt/GraXpert/GraXpert"]
        cands += sorted(glob.glob(os.path.expanduser("~/Applications/GraXpert*")))
        cands += sorted(glob.glob(os.path.expanduser("~/.local/bin/graxpert*")))
    cands += [shutil.which("graxpert"), shutil.which("GraXpert"), shutil.which("GraXpert.exe")]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _graxpert_bg(bgr_float, exe, smoothing=0.2):
    """Run GraXpert AI background extraction (Subtraction) on an image and return the
    background-subtracted float BGR (same 0..255 scale), or None on any failure. The
    reason for a failure is stored in the module-level _GRAXPERT_LOG for diagnostics.
    CLI: `GraXpert <in> -cli -cmd background-extraction -correction Subtraction
    -smoothing s -output <base>` (output name has NO extension — GraXpert adds it)."""
    global _GRAXPERT_LOG
    if not exe or not os.path.exists(exe):
        _GRAXPERT_LOG = "GraXpert executable not found."
        return None
    import tempfile, subprocess, shutil as _sh
    d = tempfile.mkdtemp(prefix="nout_gx_")
    try:
        inp = os.path.join(d, "in.tiff")
        outbase = os.path.join(d, "out")               # no extension on purpose
        u16 = (np.clip(bgr_float / 255.0, 0, 1) * 65535.0).astype(np.uint16)
        cv2.imwrite(inp, u16, [cv2.IMWRITE_TIFF_COMPRESSION, 1])   # 1 = NONE (GraXpert's
        #                       tifffile can't decode LZW without the imagecodecs package)
        cmd = [exe, inp, "-cli", "-cmd", "background-extraction",
               "-correction", "Subtraction", "-smoothing", str(smoothing),
               "-output", outbase]
        proc = subprocess.run(cmd, timeout=900, capture_output=True, text=True, cwd=d)
        # GraXpert's output name/format varies by version — scan the temp dir for any
        # image file that isn't our input, and prefer a cv2-readable one, largest first.
        exts = (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".fits", ".fit", ".xisf")
        produced = [os.path.join(d, fn) for fn in os.listdir(d)
                    if os.path.join(d, fn) != inp and fn.lower().endswith(exts)]
        produced.sort(key=lambda p: (p.lower().endswith((".fits", ".fit", ".xisf")),
                                     -os.path.getsize(p)))
        res = produced[0] if produced else None
        if res is None:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            _GRAXPERT_LOG = "GraXpert wrote no image (exit {}). {}".format(
                proc.returncode, " ".join(tail[-4:])[:300]) or "unknown error"
            return None
        if res.lower().endswith((".fits", ".fit")):        # FITS output -> read with astropy
            try:
                from astropy.io import fits
                data = fits.getdata(res)
                im = np.asarray(data, dtype=np.float32)
                if im.ndim == 3 and im.shape[0] in (1, 3):  # (C,H,W) -> (H,W,C)
                    im = np.moveaxis(im, 0, -1)
                if im.ndim == 3 and im.shape[2] == 3:
                    # FITS planes are R, G, B; the rest of NOUT is OpenCV BGR. Without
                    # this, GraXpert's result came back with red and blue swapped
                    # (measured: a blue test patch returned as red).
                    im = np.ascontiguousarray(im[..., ::-1])
            except Exception as e:      # noqa: BLE001
                _GRAXPERT_LOG = "GraXpert FITS output unreadable: {}".format(e)[:200]
                return None
        else:
            im = cv2.imread(res, cv2.IMREAD_UNCHANGED)
            if im is None:
                _GRAXPERT_LOG = "GraXpert output could not be read ({}).".format(
                    os.path.basename(res))
                return None
            im = im.astype(np.float32)
        _GRAXPERT_LOG = "OK"
        mx = float(im.max())
        if mx > 255.0:
            im = im / 65535.0 * 255.0                  # 16-bit -> 0..255
        elif mx <= 1.5:
            im = im * 255.0                            # 0..1 float -> 0..255
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return im
    except subprocess.TimeoutExpired:
        _GRAXPERT_LOG = "GraXpert timed out (model still downloading? run it once in the GUI)."
        return None
    except Exception as e:          # noqa: BLE001
        _GRAXPERT_LOG = "GraXpert error: {}".format(e)[:300]
        return None
    finally:
        _sh.rmtree(d, ignore_errors=True)


def remove_gradient(img, strength=1.0, allow_graxpert=False, graxpert_exe=None):
    """Subtract a smooth background model (vignetting + light-pollution gradient).
    If `allow_graxpert` and GraXpert is available, use GraXpert AI (same result as in
    Siril); otherwise fall back to a robust polynomial (ABE/DBE-style) model that
    samples a sigma-clipped background over a grid and fits a degree-4 surface per
    channel. Accepts uint8 or float32; uint8 in -> uint8 out (normalised), float in ->
    float out (background-subtracted, original scale kept)."""
    was_u8 = (img.dtype == np.uint8)
    if allow_graxpert:
        gx_img = _graxpert_bg(img.astype(np.float32), graxpert_exe or _GRAXPERT_EXE)
        if gx_img is not None:
            if was_u8:
                gx_img -= gx_img.min(); gx_img = gx_img / (gx_img.max() + 1e-6) * 255.0
                return np.clip(gx_img, 0, 255).astype(np.uint8)
            return np.clip(gx_img, 0.0, None)
    f = img.astype(np.float32)
    h, w = f.shape[:2]
    ny, nx = 12, 16                                   # background sampling grid
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    deg = 4                                            # polynomial degree (radial vignette)
    gyN, gxN = 72, 108                                 # coarse grid to evaluate the model on
    yy, xx = np.mgrid[0:gyN, 0:gxN].astype(np.float32)
    xn = xx / (gxN - 1) - 0.5
    yn = yy / (gyN - 1) - 0.5
    terms = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    out = np.empty_like(f)
    for c in range(f.shape[2]):
        ch = f[:, :, c]
        cx, cy, cvv = [], [], []
        for iy in range(ny):
            for ix in range(nx):
                tile = ch[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]]
                if tile.size == 0:
                    continue
                m = float(np.median(tile))
                s = float(np.median(np.abs(tile - m))) * 1.4826 + 1e-6
                bgp = tile[tile < m + 2.0 * s]        # keep sky, drop stars/bright signal
                cvv.append(float(np.median(bgp)) if bgp.size else m)
                cx.append((xs[ix] + xs[ix + 1]) * 0.5 / w - 0.5)
                cy.append((ys[iy] + ys[iy + 1]) * 0.5 / h - 0.5)
        cx = np.asarray(cx); cy = np.asarray(cy); cvv = np.asarray(cvv)
        A = np.stack([(cx ** i) * (cy ** j) for i, j in terms], axis=-1)
        coef, *_ = np.linalg.lstsq(A, cvv, rcond=None)
        model_s = np.zeros((gyN, gxN), np.float32)
        for k, (i, j) in enumerate(terms):
            model_s += coef[k] * (xn ** i) * (yn ** j)
        model = cv2.resize(model_s, (w, h), interpolation=cv2.INTER_LINEAR)
        out[:, :, c] = ch - float(strength) * model
    if was_u8:
        out = np.clip(out, 0.0, None)
        out -= out.min()
        out = out / (out.max() + 1e-6) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
    return out                                         # float: keep sign (symmetric residual)


def _mtf(y, m):
    """Midtone Transfer Function (Siril/PixInsight), vectorised. y,m in 0..1."""
    y = np.clip(y, 0.0, 1.0)
    denom = (2.0 * m - 1.0) * y - m
    denom = np.where(np.abs(denom) < 1e-8, -1e-8, denom)
    return np.clip((m - 1.0) * y / denom, 0.0, 1.0)


def auto_stretch(img, strength=0.5, scnr=False, saturation=1.0):
    """Siril-style automatic stretch: a Midtone Transfer Function auto-adjustment
    (the "auto-ajustement" in Siril). Statistics (median + normalised MAD) set a shadow
    clip and the midtones so the sky background lands near a target level — linked
    across channels to keep colour. Optional SCNR average-neutral green removal
    (Siril's "SCNR neutre moyen"). Accepts uint8 or float32; returns uint8."""
    f = img.astype(np.float32)
    # normalise to [0,1] by a ROBUST high reference (99.95th pct), not the raw max, so a
    # single hot pixel / very bright star can't corrupt the stretch (bright pixels clip to
    # 1, exactly like Siril normalising by bit depth).
    if img.dtype == np.uint8:
        f = f / 255.0
    else:
        ref = float(np.percentile(f, 99.95)) + 1e-6
        f = np.clip(f / ref, 0.0, 1.0)
    # per-channel median + normalised MAD, then averaged -> LINKED autostretch (Siril default)
    if f.ndim == 3:
        meds = [float(np.median(f[..., c])) for c in range(f.shape[2])]
        madns = [1.4826 * float(np.median(np.abs(f[..., c] - meds[c]))) for c in range(f.shape[2])]
        med = float(np.mean(meds)); madn = float(np.mean(madns)) + 1e-8
    else:
        med = float(np.median(f)); madn = 1.4826 * float(np.median(np.abs(f - med))) + 1e-8
    target_bg = 0.1 + 0.3 * float(np.clip(strength, 0.0, 1.0))   # slider: ~0.25 at 0.5
    C = -2.8                                                     # shadow clip (MAD units)
    if med < 0.5:
        shadows = min(max(med + C * madn, 0.0), 1.0)
        hi = 1.0
        x = med - shadows
        midtones = x * (1.0 - target_bg) / (x * (1.0 - 2.0 * target_bg) + target_bg + 1e-8)
    else:
        shadows = 0.0
        hi = min(max(med - C * madn, 0.0), 1.0)
        x = hi - med
        midtones = 1.0 - (x * (1.0 - target_bg)
                          / (x * (1.0 - 2.0 * target_bg) + target_bg + 1e-8))
    midtones = float(np.clip(midtones, 1e-3, 1.0 - 1e-3))
    y = np.clip((f - shadows) / max(hi - shadows, 1e-6), 0.0, 1.0)
    out = _mtf(y, midtones)
    if scnr and out.ndim == 3:                     # average-neutral SCNR (BGR: G = idx 1)
        b, g, r = out[..., 0], out[..., 1], out[..., 2]
        out[..., 1] = np.minimum(g, 0.5 * (r + b))
    res = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    if saturation != 1.0 and res.ndim == 3:        # boost colour saturation (nebula colour)
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(saturation), 0, 255)
        res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return res


def _decode_linear(path, half=True):
    """Linear-light BGR float32 (0..255 scale) from a RAW — demosaic with NO gamma and
    NO auto-bright, exactly what a stacker like Siril integrates. Half-size by default
    for speed/memory. Returns None for non-RAW files or if rawpy is unavailable."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        return None
    global _RAWPY, _RAWPY_TRIED
    if not _RAWPY_TRIED:
        _RAWPY_TRIED = True
        try:
            import rawpy
            _RAWPY = rawpy
        except ImportError:
            _RAWPY = None
    if _RAWPY is None:
        return None
    try:
        with _RAWPY.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=half,
                                  no_auto_bright=True, output_bps=16, gamma=(1, 1))
        f = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
        return f / 65535.0 * 255.0
    except Exception:               # noqa: BLE001
        return None


def _read_fnumber_exif(path):
    """Aperture (f-number) from a file's EXIF, or None."""
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="EXIF FNumber")
        v = tags.get("EXIF FNumber")
        if v is not None:
            r = v.values[0]
            return round(float(r.num) / float(r.den) if r.den else float(r.num), 1)
    except Exception:               # noqa: BLE001
        pass
    return None


def build_master_flat(paths):
    """Median-combine flat frames into a normalised master flat (linear, per-channel
    mean ≈ 1). Returns (flat_bgr_float, fnumber) or (None, None)."""
    frames, fnum = [], None
    for p in paths:
        f = _decode_linear(p)
        if f is None:
            bgr = _decode_preview_image(p, full_demosaic=False)
            f = None if bgr is None else bgr.astype(np.float32)
        if f is None:
            continue
        frames.append(f)
        if fnum is None:
            fnum = _read_fnumber_exif(p)
    if not frames:
        return None, None
    h = min(fr.shape[0] for fr in frames); w = min(fr.shape[1] for fr in frames)
    frames = [fr[:h, :w] for fr in frames]
    master = np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)
    for c in range(master.shape[2]):                   # normalise each channel to mean 1
        master[:, :, c] /= (float(master[:, :, c].mean()) + 1e-6)
    return master, fnum


def _flat_path(flats_dir, fnum):
    return os.path.join(flats_dir, "flat_f{:.1f}.npy".format(float(fnum)))


def load_master_flat(flats_dir, fnum, tol=0.2):
    """Load the master flat whose aperture best matches `fnum` (within `tol`). Returns
    the float BGR array or None."""
    if fnum is None or not flats_dir or not os.path.isdir(flats_dir):
        return None
    best, bestd = None, 1e9
    for fn in os.listdir(flats_dir):
        if fn.startswith("flat_f") and fn.endswith(".npy"):
            try:
                val = float(fn[len("flat_f"):-len(".npy")])
            except ValueError:
                continue
            if abs(val - float(fnum)) < bestd:
                bestd, best = abs(val - float(fnum)), os.path.join(flats_dir, fn)
    if best is None or bestd > tol:
        return None
    try:
        return np.load(best).astype(np.float32)
    except Exception:               # noqa: BLE001
        return None


def apply_flat(frame, flat):
    """Flat-field a linear frame: frame / normalised_flat (per channel) — removes
    vignetting and fixed illumination. Resizes the flat to the frame if needed."""
    if flat is None:
        return frame
    h, w = frame.shape[:2]
    if flat.shape[:2] != (h, w):
        flat = cv2.resize(flat, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame.astype(np.float32) / np.clip(flat, 0.05, None)


_STD_APERTURES = [1.2, 1.4, 1.8, 2.0, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5, 5.0, 5.6, 6.3,
                  7.1, 8.0, 9.0, 10.0, 11.0, 13.0, 16.0, 22.0]


def _read_lens_exif(path):
    """Lens model from a file's EXIF, or '' if unknown."""
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        for k in ("EXIF LensModel", "Image LensModel", "MakerNote LensModel",
                  "EXIF LensSpecification"):
            v = tags.get(k)
            if v:
                return str(v).strip()
    except Exception:               # noqa: BLE001
        pass
    return ""


def _lens_norm(lens):
    return (lens or "").strip().lower()


def _flat_index_path(flats_dir):
    return os.path.join(flats_dir, "index.json")


def _load_flat_index(flats_dir):
    """Persistent library manifest: list of {lens, fnum, file, frames}."""
    p = _flat_index_path(flats_dir)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:           # noqa: BLE001
            pass
    return []


def _save_flat_index(flats_dir, index):
    try:
        with open(_flat_index_path(flats_dir), "w") as f:
            json.dump(index, f, indent=2)
    except Exception:               # noqa: BLE001
        pass


def save_master_flat(flats_dir, master, lens, focal, fnum, frames):
    """Persist a master flat keyed by (lens, focal, aperture) in the library."""
    import uuid
    os.makedirs(flats_dir, exist_ok=True)
    fname = "flat_{}.npy".format(uuid.uuid4().hex[:10])
    np.save(os.path.join(flats_dir, fname), master)
    focal = float(focal) if focal else None

    def _same(e):
        ef = e.get("focal")
        same_focal = (ef is None and focal is None) or (
            ef is not None and focal is not None and abs(float(ef) - focal) < 0.5)
        return (_lens_norm(e.get("lens")) == _lens_norm(lens)
                and abs(float(e.get("fnum", 0)) - float(fnum)) < 0.05 and same_focal)
    index = [e for e in _load_flat_index(flats_dir) if not _same(e)]
    index.append({"lens": lens or "", "focal": focal, "fnum": float(fnum),
                  "file": fname, "frames": int(frames)})
    _save_flat_index(flats_dir, index)
    return fname


def load_master_flat_for(flats_dir, lens, focal, fnum, tol_ap=0.2, tol_focal=0.25):
    """Load the master flat matching (lens, focal, aperture): same lens, focal within
    `tol_focal` (relative) and aperture within `tol_ap`. Entries saved without a focal
    (primes / legacy) match any focal. Falls back to the legacy per-aperture file."""
    if not flats_dir or not os.path.isdir(flats_dir) or fnum is None:
        return load_master_flat(flats_dir, fnum, tol_ap)
    index = _load_flat_index(flats_dir)
    ln = _lens_norm(lens)

    def ok(e):
        if abs(float(e.get("fnum", 0)) - float(fnum)) > tol_ap:
            return False
        ef = e.get("focal")
        if ef and focal and abs(float(ef) - float(focal)) / max(float(focal), 1.0) > tol_focal:
            return False
        return True

    def fdist(e):
        ef = e.get("focal")
        return abs(float(ef) - float(focal)) if (ef and focal) else 0.0

    for pool in ([e for e in index if _lens_norm(e.get("lens")) == ln] if ln else [], index):
        cands = [e for e in pool if ok(e)]
        if cands:
            best = min(cands, key=lambda e: (fdist(e), abs(float(e.get("fnum", 0)) - float(fnum))))
            try:
                return np.load(os.path.join(flats_dir, best["file"])).astype(np.float32)
            except Exception:       # noqa: BLE001
                return None
    return load_master_flat(flats_dir, fnum, tol_ap)


def _read_iso_exif(path):
    """ISO from a file's EXIF, or None."""
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="EXIF ISOSpeedRatings")
        v = tags.get("EXIF ISOSpeedRatings")
        if v is not None:
            return int(str(v.values[0]))
    except Exception:               # noqa: BLE001
        pass
    return None


def _read_exposure_exif(path):
    """Exposure time in seconds from EXIF, or None."""
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="EXIF ExposureTime")
        v = tags.get("EXIF ExposureTime")
        if v is not None:
            r = v.values[0]
            return round(float(r.num) / float(r.den) if r.den else float(r.num), 3)
    except Exception:               # noqa: BLE001
        pass
    return None


def build_master_dark(paths):
    """Median-combine dark/bias frames into a master dark (linear, NOT normalised — it's
    subtracted). Returns (master_bgr_float, iso, exposure) or (None, None, None)."""
    frames, iso, exp = [], None, None
    for p in paths:
        f = _decode_linear(p)
        if f is None:
            bgr = _decode_preview_image(p, full_demosaic=False)
            f = None if bgr is None else bgr.astype(np.float32)
        if f is None:
            continue
        frames.append(f)
        if iso is None:
            iso = _read_iso_exif(p); exp = _read_exposure_exif(p)
    if not frames:
        return None, None, None
    h = min(fr.shape[0] for fr in frames); w = min(fr.shape[1] for fr in frames)
    master = np.median(np.stack([fr[:h, :w] for fr in frames], axis=0), axis=0)
    return master.astype(np.float32), iso, exp


def _darks_index_path(flats_dir):
    return os.path.join(flats_dir, "darks_index.json")


def _load_darks_index(flats_dir):
    p = _darks_index_path(flats_dir)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:           # noqa: BLE001
            pass
    return []


def save_master_dark(flats_dir, master, iso, exp, frames):
    """Persist a master dark/bias keyed by (ISO, exposure)."""
    import uuid
    os.makedirs(flats_dir, exist_ok=True)
    fname = "dark_{}.npy".format(uuid.uuid4().hex[:10])
    np.save(os.path.join(flats_dir, fname), master)
    index = [e for e in _load_darks_index(flats_dir)
             if not (int(e.get("iso", 0)) == int(iso or 0)
                     and abs(float(e.get("exp", 0)) - float(exp or 0)) < 0.02)]
    index.append({"iso": int(iso or 0), "exp": float(exp or 0),
                  "file": fname, "frames": int(frames)})
    try:
        with open(_darks_index_path(flats_dir), "w") as f:
            json.dump(index, f, indent=2)
    except Exception:               # noqa: BLE001
        pass
    return fname


def load_master_dark_for(flats_dir, iso, exp, tol_exp=0.25):
    """Load the master dark matching ISO (exact) and exposure (within `tol_exp` relative).
    A bias is just a very short-exposure dark, so it also matches short lights."""
    if not flats_dir or not os.path.isdir(flats_dir) or iso is None:
        return None
    cands = [e for e in _load_darks_index(flats_dir) if int(e.get("iso", 0)) == int(iso)]
    if not cands:
        return None
    if exp:
        cands = [e for e in cands
                 if abs(float(e.get("exp", 0)) - float(exp)) / max(float(exp), 0.01) <= tol_exp
                 or float(e.get("exp", 0)) < 0.1]      # bias matches any exposure
    if not cands:
        return None
    best = min(cands, key=lambda e: abs(float(e.get("exp", 0)) - float(exp or 0)))
    try:
        return np.load(os.path.join(flats_dir, best["file"])).astype(np.float32)
    except Exception:               # noqa: BLE001
        return None


def apply_dark(frame, dark):
    """Subtract a master dark/bias from a linear frame (clip ≥ 0). Resizes if needed."""
    if dark is None:
        return frame
    h, w = frame.shape[:2]
    if dark.shape[:2] != (h, w):
        dark = cv2.resize(dark, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(frame.astype(np.float32) - dark, 0.0, None)


def luminance_hist(bgr, bins=128):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
    return h


def rgb_hist(bgr, bins=128):
    """B, G, R histograms (each an array of 'bins' values)."""
    return [cv2.calcHist([bgr], [c], None, [bins], [0, 256]).flatten() for c in range(3)]


def align_translation(ref_gray, cur_gray):
    """Offset (sx, sy) between two grayscale images (phase correlation).
    Used for visual live stacking: aligns each exposure on the first."""
    a = ref_gray.astype(np.float32)
    b = cur_gray.astype(np.float32)
    try:
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (sx, sy), _resp = cv2.phaseCorrelate(a, b, win)
    except Exception:               # noqa: BLE001
        return 0.0, 0.0
    return sx, sy


def cosmetic_correct(x, k=6.0):
    """Remove hot pixels without darks. A hot photosite lights ONE colour channel of
    one pixel; a star, however small, lights all three. So a pixel is fixed only when
    a single channel spikes far above all its neighbours while the other two stay at
    their local level — stars, even one pixel wide, are left alone."""
    if x.ndim != 3 or x.shape[2] != 3:
        return x
    ring = np.ones((3, 3), np.uint8); ring[1, 1] = 0
    sub = x[::4, ::4]
    out = x.copy()
    med, spike, elev = [], [], []
    for c in range(3):
        ch = np.ascontiguousarray(x[..., c])
        s = 1.4826 * float(np.median(np.abs(sub[..., c] - np.median(sub[..., c])))) + 1e-3
        m = cv2.medianBlur(ch, 3)
        nmax = cv2.dilate(ch, ring)
        med.append(m)
        spike.append((ch - nmax) > k * s)
        elev.append((ch - m) > 3.0 * s)
    for c in range(3):
        o1, o2 = [i for i in range(3) if i != c]
        hot = spike[c] & ~elev[o1] & ~elev[o2]
        if hot.any():
            out[..., c][hot] = med[c][hot]
    return out


def detection_gray(x):
    """8-bit image to FIND stars in: background-subtracted and square-root stretched,
    so faint stars (a few ADU in linear light) register instead of rounding to 0."""
    lum = x.mean(axis=2) if x.ndim == 3 else x
    sub = lum[::4, ::4]
    bg = float(np.median(sub))
    s = 1.4826 * float(np.median(np.abs(sub - bg))) + 1e-3
    y = np.clip((lum - bg) / (40.0 * s), 0.0, 1.0)
    return (np.sqrt(y) * 255.0).astype(np.uint8)


def _obs_background(f, grid=(16, 24), deg=3, iters=5):
    """Sky model per channel from sigma-clipped tile medians, where tiles lying on an
    extended object (galaxy, nebula) are rejected iteratively, so the fit only ever
    sees sky — no dark hole dug around a big galaxy."""
    h, w = f.shape[:2]
    ny, nx = grid
    ys = np.linspace(0, h, ny + 1).astype(int); xs = np.linspace(0, w, nx + 1).astype(int)
    lum = f.mean(axis=2)
    cx, cy, vals, lv = [], [], [], []
    for iy in range(ny):
        for ix in range(nx):
            sl = (slice(ys[iy], ys[iy + 1]), slice(xs[ix], xs[ix + 1]))
            t = lum[sl]
            m = float(np.median(t)); sd = 1.4826 * float(np.median(np.abs(t - m))) + 1e-6
            keep = t < m + 2.0 * sd
            if keep.sum() < 10:
                keep = np.ones_like(t, bool)
            vals.append([float(np.median(f[sl][..., c][keep])) for c in range(3)])
            lv.append(float(np.median(t[keep])))
            cx.append((xs[ix] + xs[ix + 1]) / 2.0 / w - 0.5)
            cy.append((ys[iy] + ys[iy + 1]) / 2.0 / h - 0.5)
    cx, cy, vals, lv = map(np.asarray, (cx, cy, vals, lv))
    terms = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    A = np.stack([(cx ** i) * (cy ** j) for i, j in terms], axis=-1)
    ok = np.ones(len(lv), bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[ok], lv[ok], rcond=None)
        res = lv - A @ coef
        sd = 1.4826 * float(np.median(np.abs(res[ok] - np.median(res[ok])))) + 1e-6
        new = res < 2.0 * sd                          # objects are only ever brighter
        if new.sum() < len(terms) * 2 or (new == ok).all():
            break
        ok = new
    gy, gx = np.mgrid[0:64, 0:96].astype(np.float32)
    xn = gx / 95.0 - 0.5; yn = gy / 63.0 - 0.5
    model = np.empty_like(f)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A[ok], vals[ok, c], rcond=None)
        ms = np.zeros_like(xn)
        for k, (i, j) in enumerate(terms):
            ms += coef[k] * (xn ** i) * (yn ** j)
        model[..., c] = cv2.resize(ms.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return model


def render_observation(f, flattened=None, saturation=1.05, max_side=2400):
    """Smart-telescope look for the live stack: flat neutral sky, colour-true stars,
    galaxy cores that do not burn out, clean background — made to be shown.

    `flattened`: an already background-subtracted version (GraXpert) to use instead
    of the built-in sky model. Returns BGR uint8 (at most `max_side` wide)."""
    f = np.asarray(f, np.float32)
    if f.ndim == 2:
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    if flattened is not None:
        g = np.asarray(flattened, np.float32)
        if g.shape == f.shape:
            f = g                                 # background already removed (GraXpert)
        else:
            flattened = None
    h, w = f.shape[:2]
    if max(h, w) > max_side:                      # the screen shows no more than this
        sc = max_side / float(max(h, w))
        f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    if flattened is None:
        f = f - _obs_background(f)
    # neutral sky: every channel's sky at 0
    f = f - np.array([np.median(f[::3, ::3, c]) for c in range(3)], np.float32)
    lum = f.mean(axis=2)
    # white stars on average — bounded, so a strange field can't swing the colour
    top = float(np.percentile(lum, 99.97)) + 1e-6
    thr = float(np.percentile(lum, 99.0))
    sel = (lum > thr) & (f.max(axis=2) < 0.85 * top)
    if int(sel.sum()) > 50:
        means = np.array([f[..., c][sel].mean() for c in range(3)])
        f = f * np.clip(means.mean() / (means + 1e-6), 0.8, 1.25)
        lum = f.mean(axis=2)
    # SCNR (average neutral), as Siril does for colour cameras: nothing in the deep sky
    # is truly green, so green above the red/blue average is the sensor's, not the sky's
    f[..., 1] = np.minimum(f[..., 1], 0.5 * (f[..., 0] + f[..., 2]) +
                           0.02 * np.abs(f[..., 1]))
    lum = f.mean(axis=2)
    # colour-preserving arcsinh stretch on luminance
    noise = 1.4826 * float(np.median(np.abs(lum[::3, ::3] - np.median(lum[::3, ::3])))) + 1e-6
    # Normalise on the very top of the image, not on the 99.97th percentile: on a bright
    # core (M42, a galaxy nucleus) that percentile sits well inside the object, so
    # everything above it clipped to white and the core lost every trace of detail.
    top = float(np.percentile(lum, 99.999)) + 1e-6
    n = noise / top
    lo, hi = 1.0, 1e6
    for _ in range(45):                          # sky + 2σ lands 0.10 above black
        beta = (lo * hi) ** 0.5
        if np.arcsinh(beta * 2 * n) / np.arcsinh(beta) < 0.10:
            lo = beta
        else:
            hi = beta
    x = np.clip(lum / top, 0.0, None)
    y = np.arcsinh(beta * x) / np.arcsinh(beta)
    ratio = np.where(x > 1e-6, y / np.maximum(x, 1e-6), 0.0) / top
    rgb = np.clip(f * ratio[..., None], 0.0, None)
    # highlights melt to white instead of clipping channel by channel (no core rings)
    y3 = np.clip(y, 0, 1)[..., None]
    m = np.maximum(rgb.max(axis=2, keepdims=True), 1e-6)
    rgb = np.where(m > 1.0, rgb / m, rgb)
    wmix = np.clip((y3 - 0.93) / 0.07, 0.0, 1.0)   # melt to white only at the very top
    rgb = rgb * (1 - wmix) + y3 * wmix
    black = 0.045                                 # deep sky, not crushed to pure black
    img = ((black + (1 - black) * np.clip(rgb, 0, 1)) * 255.0).astype(np.uint8)
    # denoise: luminance lightly, colour noise strongly
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    sL = 1.4826 * float(np.median(np.abs(L[::3, ::3].astype(np.float32) - np.median(L[::3, ::3]))))
    L = cv2.fastNlMeansDenoising(L, None, h=float(np.clip(1.1 * sL, 2.0, 14.0)),
                                 templateWindowSize=5, searchWindowSize=15)
    a = cv2.GaussianBlur(a, (0, 0), 2.2); b = cv2.GaussianBlur(b, (0, 0), 2.2)
    # gentle local contrast where there is signal
    Lc = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8)).apply(L)
    wgt = np.clip((L.astype(np.float32) - 40.0) / 80.0, 0.0, 1.0)
    L = (L * (1 - wgt) + Lc * wgt).astype(np.uint8)
    # saturation on the signal, not on the sky
    Lf = L.astype(np.float32)
    sat = 1.0 + (float(saturation) - 1.0) * np.clip((Lf - 25.0) / 60.0, 0.0, 1.0)
    a = np.clip(128 + (a.astype(np.float32) - 128) * sat, 0, 255).astype(np.uint8)
    b = np.clip(128 + (b.astype(np.float32) - 128) * sat, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.3)
    return cv2.addWeighted(out, 1.15, blur, -0.15, 0)   # a touch of bite, no halo


def register_affine(ref_gray, cur_gray, ref_stars=None, cur_stars=None):
    """Similarity transform (rotation + scale + translation) mapping `cur` onto `ref`,
    estimated from matched star centroids (RANSAC) — what a real stacker does (Siril's
    global star registration). Removes the field ROTATION a plain translation leaves.
    Returns a 2x3 float32 for cv2.warpAffine; falls back to translation if matching is
    unreliable."""
    if ref_stars is None:
        ref_stars, _ = detect_stars(ref_gray)
    if cur_stars is None:
        cur_stars, _ = detect_stars(cur_gray)
    try:
        sx, sy = align_translation(ref_gray, cur_gray)
    except Exception:               # noqa: BLE001
        sx = sy = 0.0
    if len(ref_stars) < 8 or len(cur_stars) < 8:
        # resolve the translation sign by star-match count even in the fallback
        return _best_translation(ref_stars, cur_stars, sx, sy)
    ref_xy = np.array([(s[0], s[1]) for s in ref_stars], np.float32)
    cur_xy = np.array([(s[0], s[1]) for s in cur_stars], np.float32)
    tol = max(6.0, 0.012 * max(ref_gray.shape))

    def match(shift):
        sh = cur_xy + np.asarray(shift, np.float32)
        src, dst = [], []
        for i in range(len(sh)):
            d2 = (ref_xy[:, 0] - sh[i, 0]) ** 2 + (ref_xy[:, 1] - sh[i, 1]) ** 2
            j = int(np.argmin(d2))
            if d2[j] <= tol * tol:
                src.append(cur_xy[i]); dst.append(ref_xy[j])
        return src, dst

    best = max([(sx, sy), (-sx, -sy), (0.0, 0.0)],
               key=lambda s: len(match(s)[0]))          # pick sign that matches most stars
    src, dst = match(best)
    if len(src) < 6:
        return np.float32([[1, 0, best[0]], [0, 1, best[1]]])
    M, inliers = cv2.estimateAffinePartial2D(
        np.array(src, np.float32), np.array(dst, np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or (inliers is not None and int(inliers.sum()) < 5):
        return np.float32([[1, 0, best[0]], [0, 1, best[1]]])
    return M.astype(np.float32)


def _best_translation(ref_stars, cur_stars, sx, sy):
    """Choose the translation sign that aligns the most stars (robust to phaseCorrelate
    sign ambiguity)."""
    if len(ref_stars) < 3 or len(cur_stars) < 3:
        return np.float32([[1, 0, sx], [0, 1, sy]])
    ref_xy = np.array([(s[0], s[1]) for s in ref_stars], np.float32)
    cur_xy = np.array([(s[0], s[1]) for s in cur_stars], np.float32)
    tol = 8.0

    def nmatch(shift):
        sh = cur_xy + np.asarray(shift, np.float32)
        n = 0
        for i in range(len(sh)):
            d2 = (ref_xy[:, 0] - sh[i, 0]) ** 2 + (ref_xy[:, 1] - sh[i, 1]) ** 2
            if float(d2.min()) <= tol * tol:
                n += 1
        return n
    best = max([(sx, sy), (-sx, -sy), (0.0, 0.0)], key=nmatch)
    return np.float32([[1, 0, best[0]], [0, 1, best[1]]])


def neutralize_background(f):
    """Make the sky background neutral gray by equalising the per-channel background
    LEVEL (Siril's background neutralisation — additive only). Each channel's sky level
    is shifted to a common small pedestal, so the background stretches to gray (not
    black) without introducing a colour cast. f: float BGR (0..255 scale)."""
    out = f.astype(np.float32).copy()
    bg, sc = [], []
    for c in range(out.shape[2]):
        ch = out[:, :, c]
        m = float(np.median(ch)); s = float(np.median(np.abs(ch - m))) * 1.4826 + 1e-6
        sky = ch[ch < m + 2.0 * s]                     # sky pixels only (drop stars)
        bg.append(float(np.median(sky)) if sky.size else m)
        sc.append(float(np.median(np.abs(sky - np.median(sky)))) * 1.4826 + 1e-6
                  if sky.size else s)
    pedestal = 8.0 * float(np.median(sc))              # keep the sky a few sigma above zero
    for c in range(out.shape[2]):
        out[:, :, c] = out[:, :, c] - bg[c] + pedestal
    return np.clip(out, 0.0, None)


def star_white_balance(f):
    """Neutralise the colour of the SIGNAL (not just the background) by scaling channels
    so the median star is gray. This is a catalog-free approximation of Siril's
    photometric colour calibration: with a rich star field, the average star is close to
    neutral, so equalising per-channel star flux removes the camera white-balance cast.
    f: float BGR (0..255 scale)."""
    out = f.astype(np.float32)
    try:
        gray = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        stars, n = detect_stars(gray)
    except Exception:               # noqa: BLE001
        return out
    if n < 20:
        return out
    h, w = gray.shape
    vals = [[], [], []]
    for (x, y, _r) in stars:
        xi, yi = int(round(x)), int(round(y))
        if 1 <= xi < w - 1 and 1 <= yi < h - 1:
            patch = out[yi - 1:yi + 2, xi - 1:xi + 2]
            for c in range(3):
                vals[c].append(float(patch[:, :, c].max()))
    means = [float(np.median(v)) if v else 1.0 for v in vals]
    target = float(np.median(means))
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (target / (means[c] + 1e-6))
    return out


def normalize_to_ref(x, ref_med, ref_mad):
    """Per-channel additive+multiplicative normalisation to a reference frame's sky
    statistics (equalises background level and noise scale across frames) so the mean
    and sigma-clip rejection behave like a real integration."""
    out = x.copy()
    for c in range(out.shape[2]):
        ch = out[:, :, c]
        m = float(np.median(ch))
        s = float(np.median(np.abs(ch - m))) + 1e-6
        out[:, :, c] = (ch - m) * (ref_mad[c] / s) + ref_med[c]
    return out


def _decode_preview_image(path, full_demosaic=False):
    """Returns a BGR preview (ndarray) of the photo, or None if undecodable.
    For a RAW (.arw): full_demosaic=True does a full-resolution demosaic (best quality,
    slower); otherwise extracts the embedded full-size JPEG (fast).
    For JPEG/PNG, direct OpenCV read."""
    global _RAWPY, _RAWPY_TRIED
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        return cv2.imread(path, cv2.IMREAD_COLOR)
    # RAW : tries rawpy
    if not _RAWPY_TRIED:
        _RAWPY_TRIED = True
        try:
            import rawpy
            _RAWPY = rawpy
        except ImportError:
            _RAWPY = None
    if _RAWPY is None:
        return None
    try:
        with _RAWPY.imread(path) as raw:
            if full_demosaic:                 # best quality from sensor data
                rgb = raw.postprocess(use_camera_wb=True, half_size=False,
                                      no_auto_bright=True, output_bps=8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            try:
                thumb = raw.extract_thumb()
                if thumb.format == _RAWPY.ThumbFormat.JPEG:
                    arr = np.frombuffer(thumb.data, np.uint8)
                    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
                # RGB bitmap thumbnail
                return cv2.cvtColor(np.asarray(thumb.data), cv2.COLOR_RGB2BGR)
            except Exception:        # noqa: BLE001 - no thumbnail -> fast demosaic
                rgb = raw.postprocess(use_camera_wb=True, half_size=True,
                                      no_auto_bright=True)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:                # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
#  Camera Backends (real gphoto2 / simulation)
# ---------------------------------------------------------------------------
class CameraError(Exception):
    pass


class SimBackend:
    """Simulated camera: high-frequency scene blurred according to a focus position.
    The 'true' point is at focus=50; moving focus towards 50 raises
    the sharpness curve, allowing testing the entire pipeline without hardware.
    """

    def __init__(self):
        self.focus = 20.0          # current position (0..100)
        self._true_focus = 50.0
        self._scene = self._build_scene()
        self.counter = 0

    def _build_scene(self, size=720):
        rng = np.random.default_rng(42)
        img = (rng.random((size, size)) * 255).astype(np.uint8)      # HF noise
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # checkerboard for sharp edges
        step = 36
        for y in range(0, size, step):
            for x in range(0, size, step):
                if ((x // step) + (y // step)) % 2 == 0:
                    img[y:y + step, x:x + step] = (40, 40, 40)
        cv2.putText(img, "SIM A7II", (40, size - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 200, 255), 6)
        for r in range(40, size // 2, 30):
            cv2.circle(img, (size // 2, size // 2), r, (0, 180, 0), 2)
        return img

    def model_name(self):
        return "Simulated A7II"

    def get_preview_jpeg(self):
        self.counter += 1
        sigma = 0.25 + 0.18 * abs(self.focus - self._true_focus)   # blur ~ distance to point
        frame = cv2.GaussianBlur(self._scene, (0, 0), sigmaX=max(sigma, 0.1))
        noise = np.random.default_rng(self.counter).normal(0, 2, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    def capture_to_file(self, path):
        root, _ = os.path.splitext(path)
        dest = root + ".jpg"            # OpenCV can't encode .arw
        frame = cv2.GaussianBlur(self._scene, (0, 0),
                                 sigmaX=max(0.25 + 0.18 * abs(self.focus - self._true_focus), 0.1))
        cv2.imwrite(dest, frame)
        return dest

    def current_shutter(self):
        return getattr(self, "_sim_shutter", "1/125")

    def ensure_not_bulb(self, preferred=None):
        if "bulb" in str(getattr(self, "_sim_shutter", "")).lower():
            self._sim_shutter = preferred or "1/125"
            return True, "Shutter was still on Bulb — set to {}.".format(self._sim_shutter)
        return False, ""

    def set_shutter_to_bulb(self):
        self._sim_shutter = "Bulb"

    def capture_bulb(self, seconds, path, should_abort=None):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            if should_abort and should_abort():
                break
            time.sleep(0.1)
        root, _ = os.path.splitext(path)
        return self.capture_to_file(root + ".jpg")

    def list_config(self, name):
        demo = {"iso": ["100", "200", "400", "800", "1600", "3200"],
                "shutterspeed": ["1/30", "1/60", "1/125", "1/250", "1/500"],
                "f-number": ["f/2.8", "f/4", "f/5.6", "f/8", "f/11"]}
        return demo.get(name, []), (demo.get(name, [""])[0] if demo.get(name) else "")

    def set_config(self, name, value):
        pass

    def focus_supported(self):
        return True

    def status_info(self, save_dir):
        import shutil
        free = None
        try:
            d = save_dir if os.path.isdir(save_dir) else "/tmp"
            free = shutil.disk_usage(d).free / 1e9
        except Exception:           # noqa: BLE001
            pass
        return {"battery": "— (sim)", "free_gb": free}

    def current_exposure_s(self):
        return 2.0                  # nominal value for simulation

    def focus_step(self, amount):
        self.focus = float(np.clip(self.focus + amount, 0, 100))
        return "sim-focus"

    def record_movie(self, seconds, path):
        import cv2 as _cv2
        out = os.path.splitext(path)[0] + ".mp4"
        vw = _cv2.VideoWriter(out, _cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 240))
        base = np.zeros((240, 320, 3), np.uint8); _cv2.circle(base, (160, 120), 90, (150, 150, 160), -1)
        for i in range(int(max(seconds, 1) * 15)):
            f = base.copy()
            if i % 3 == 0:
                f = _cv2.GaussianBlur(f, (7, 7), 2)
            vw.write(f)
        vw.release()
        return out

    def close(self):
        pass


def _find_solve_field():
    """Locate the solve-field binary even when launched as a .app (where PATH is
    minimal and excludes Homebrew). Checks PATH first, then the usual install dirs."""
    import shutil
    exe = shutil.which("solve-field")
    if exe:
        return exe
    candidates = [
        "/opt/homebrew/bin/solve-field",          # Apple-Silicon Homebrew
        "/usr/local/bin/solve-field",             # Intel Homebrew
        "/opt/local/bin/solve-field",             # MacPorts
        os.path.expanduser("~/homebrew/bin/solve-field"),
        "/usr/bin/solve-field",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class GPhoto2Backend:
    """Real camera via python-gphoto2. All methods are called
    exclusively from the worker (only one thread owns the camera object)."""

    # possible setting names depending on the body (we take the first that exists)
    CONFIG_ALIASES = {
        "iso": ["iso", "iso2"],
        "shutterspeed": ["shutterspeed", "shutterspeed2"],
        "f-number": ["f-number", "aperture"],
    }
    FOCUS_WIDGETS = ["manualfocusdrive", "focusmode", "focusdrive"]

    # macOS system daemons that claim the PTP device (to release before opening)
    _MAC_PTP_DAEMONS = ("ptpcamerad", "PTPCamera")
    _active_hammer = None          # the running kill-loop subprocess, if any

    @classmethod
    def release_usb(cls):
        """On macOS, kill the system PTP daemons holding the device — in a SINGLE
        killall call (fast). They respawn quickly, hence the continuous hammer below."""
        if platform.system() != "Darwin":
            return
        try:
            subprocess.run(["killall", *cls._MAC_PTP_DAEMONS],
                           capture_output=True, timeout=2)
        except Exception:           # noqa: BLE001
            pass

    @classmethod
    def _start_hammer(cls):
        """Spawn ONE background shell that knocks the PTP daemon down flat-out:
        SIGKILL any running instance and SIGSTOP every respawn, with no pause, so
        the USB interface stays free long enough for gp_camera_init to claim it.
        A Python loop (subprocess per iteration) is far too slow to win this race."""
        if platform.system() != "Darwin":
            return None
        names = " ".join(cls._MAC_PTP_DAEMONS)
        cmd = ("while true; do /usr/bin/killall -9 {n} 2>/dev/null; "
               "/usr/bin/killall -STOP {n} 2>/dev/null; done").format(n=names)
        try:
            proc = subprocess.Popen(["/bin/bash", "-c", cmd], start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cls._active_hammer = proc
            return proc
        except Exception as e:      # noqa: BLE001
            print("[NOUT] hammer failed to start:", e, file=sys.stderr, flush=True)
            return None

    @classmethod
    def kill_any_hammer(cls):
        """Stop a lingering kill-loop (e.g. if the thread was force-terminated)."""
        cls._stop_hammer(cls._active_hammer)
        cls._active_hammer = None

    @classmethod
    def _stop_hammer(cls, proc):
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:           # noqa: BLE001
            try:
                proc.terminate()
            except Exception:       # noqa: BLE001
                pass
        if getattr(cls, "_active_hammer", None) is proc:
            cls._active_hammer = None

    @classmethod
    def camera_present(cls):
        """Cheap USB enumeration (no claim): True if a camera is plugged and on.
        Lets us fail fast / wait efficiently instead of hammering for nothing."""
        try:
            import gphoto2 as gp
            return len(list(gp.Camera.autodetect())) > 0
        except Exception:           # noqa: BLE001 - unknown → assume present
            return True

    def __init__(self):
        try:
            import gphoto2 as gp
        except ImportError as e:
            raise CameraError(
                "The 'gphoto2' module is not installed.\n"
                "  pip install gphoto2   (and libgphoto2 via brew/apt)\n"
                "Or launch with --simulate to test without a camera."
            ) from e
        self.gp = gp
        self.camera = gp.Camera()

        # Fast presence check: avoids a pointless 12 s hammer when the body is off
        # (e.g. battery being swapped) — the reconnect loop relies on this.
        if not GPhoto2Backend.camera_present():
            raise CameraError("No camera detected. Turn it on, set USB to "
                              "“PC Remote”, or finish swapping the battery.")

        # macOS: run the kill loop in a background shell for the WHOLE open phase.
        is_mac = platform.system() == "Darwin"
        hammer = self._start_hammer() if is_mac else None
        if hammer is not None:
            print("[NOUT] camera: hammering ptpcamerad/PTPCamera to free the USB…",
                  file=sys.stderr, flush=True)
            time.sleep(0.4)            # let the loop knock the daemon down first

        last = None
        attempts = 0
        deadline = time.monotonic() + 12.0
        try:
            while time.monotonic() < deadline:
                attempts += 1
                try:
                    self.camera.init()
                    last = None
                    break
                except gp.GPhoto2Error as e:
                    last = e
                    if getattr(e, "code", 0) == -105:   # MODEL_NOT_FOUND: no camera → stop
                        break
                    time.sleep(0.05)
        finally:
            self._stop_hammer(hammer)
        if last is None:
            print("[NOUT] camera opened after {} attempt(s).".format(attempts),
                  file=sys.stderr, flush=True)
        else:
            print("[NOUT] camera FAILED after {} attempts, last error: {} (code {})".format(
                attempts, last, getattr(last, "code", "?")), file=sys.stderr, flush=True)

        if last is not None:
            raise CameraError(
                "Unable to open camera (code {}). 'Could not claim "
                "the USB device' = the device is seen but locked.\n"
                " • A previous instance of this app might still hold "
                "the camera (ps aux | grep python).\n"
                " • Close Photos / Image Capture.\n"
                " • Check USB → PC Remote on the body.\n"
                " • Terminal test to win the race:\n"
                "   ( while true; do killall ptpcamerad PTPCamera 2>/dev/null; "
                "sleep 0.05; done ) & K=$!; sleep 0.3; "
                "gphoto2 --capture-image-and-download; kill $K"
                .format(getattr(last, "code", last))
            ) from last
        self._model = self._read_model()
        self._focus_widget = self._detect_focus_widget()
        self._configure_for_tether()

    # -- internal helpers ---------------------------------------------------
    def _read_model(self):
        gp = self.gp
        try:
            cfg = self.camera.get_config()
            ok, w = gp.gp_widget_get_child_by_name(cfg, "cameramodel")
            if ok >= gp.GP_OK:
                return w.get_value()
        except gp.GPhoto2Error:
            pass
        return "Sony (gphoto2)"

    def _find_widget(self, cfg, names):
        gp = self.gp
        for n in names:
            ok, w = gp.gp_widget_get_child_by_name(cfg, n)
            if ok >= gp.GP_OK:
                return n, w
        return None, None

    def _detect_focus_widget(self):
        cfg = self.camera.get_config()
        name, _ = self._find_widget(cfg, self.FOCUS_WIDGETS)
        return name      # None if focus is not controllable (expected on A7II)

    def _configure_for_tether(self):
        """Best-effort: keep images on card to be able to retrieve them."""
        gp = self.gp
        try:
            cfg = self.camera.get_config()
            ok, w = gp.gp_widget_get_child_by_name(cfg, "capturetarget")
            if ok >= gp.GP_OK:
                for i in range(w.count_choices()):
                    if "card" in w.get_choice(i).lower():
                        w.set_value(w.get_choice(i))
                        self.camera.set_config(cfg)
                        break
        except gp.GPhoto2Error:
            pass

    # -- public API -------------------------------------------------------
    def model_name(self):
        return self._model

    def get_preview_jpeg(self):
        cam_file = self.camera.capture_preview()
        data = cam_file.get_data_and_size()
        return bytes(memoryview(data))

    def _await_file(self, path, timeout):
        """Wait for the body to announce a new file, and download it."""
        gp = self.gp
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            etype, edata = self.camera.wait_for_event(300)
            if etype == gp.GP_EVENT_FILE_ADDED:
                root, _ = os.path.splitext(path)
                _, ext = os.path.splitext(edata.name)
                dest = root + (ext if ext else ".arw")
                cam_file = self.camera.file_get(edata.folder, edata.name,
                                                gp.GP_FILE_TYPE_NORMAL)
                cam_file.save(dest)
                return dest
        return None

    def capture_trigger(self, path, timeout=30.0):
        """One exposure, timed BY THE BODY, fired by pressing and releasing the
        shutter — the same path bulb uses, which is the one this camera honours.

        gp_camera_capture() drops the connection on a Sony A7II while press/release
        works every time, so short exposures go through here. The body's own shutter
        setting decides the duration, which is the only way to reach 1/125 s: a
        software-timed bulb cannot be shorter than its own USB round trips."""
        gp = self.gp
        while self.camera.wait_for_event(10)[0] != gp.GP_EVENT_TIMEOUT:
            pass                                    # purge stale events first
        canon = self._has_widget("eosremoterelease") and not self._has_widget("bulb")
        if canon:
            self._set_widget("eosremoterelease", "Press Full")
            self._set_widget("eosremoterelease", "Release Full")
        else:
            self._set_widget("bulb", 1)
            self._set_widget("bulb", 0)
        dest = self._await_file(path, timeout)
        if dest is None:
            raise CameraError("No file after the shutter trigger — is the body in "
                              "PC Remote with a normal shutter speed set?")
        return dest

    def capture_to_file(self, path):
        """Normal exposure (≤30 s): duration is the one set on the body.

        Prefers the trigger path on bodies that expose a shutter widget, because
        that is what actually works here; gp_camera_capture() stays as the fallback
        for cameras that have no such widget."""
        gp = self.gp
        # Drain whatever the body queued while we were busy elsewhere. Several
        # seconds of stacking between frames leaves events piled up, and the next
        # capture then comes back '[-1] Unspecified error' even though the link is
        # perfectly healthy — measured on this camera.
        try:
            while self.camera.wait_for_event(10)[0] != gp.GP_EVENT_TIMEOUT:
                pass
        except Exception:               # noqa: BLE001
            pass
        last = None
        for attempt in range(2):
            try:
                fp = self.camera.capture(gp.GP_CAPTURE_IMAGE)
                cam_file = self.camera.file_get(fp.folder, fp.name,
                                                gp.GP_FILE_TYPE_NORMAL)
                root, _ = os.path.splitext(path)
                _, ext = os.path.splitext(fp.name)
                dest = root + (ext if ext else ".jpg")
                cam_file.save(dest)
                return dest
            except Exception as e:      # noqa: BLE001
                last = e
                time.sleep(0.6)         # let the body settle, then ask once more
        # Still nothing: press/release is the path bulb uses, and it works on
        # bodies where capture() is fussy.
        if self._has_widget("bulb") or self._has_widget("eosremoterelease"):
            return self.capture_trigger(path)
        raise last

    # ----- Bulb (long exposures, timed on software side) ---------------
    def _set_widget(self, name, value):
        gp = self.gp
        cfg = self.camera.get_config()
        ok, w = gp.gp_widget_get_child_by_name(cfg, name)
        if ok < gp.GP_OK:
            raise CameraError("Widget '{}' absent (put body in M).".format(name))
        w.set_value(value)
        self.camera.set_config(cfg)

    def current_shutter(self):
        """The shutter speed currently set on the body, or None."""
        cfg = self.camera.get_config()
        _, w = self._find_widget(cfg, self.CONFIG_ALIASES["shutterspeed"])
        return None if w is None else str(w.get_value())

    def ensure_not_bulb(self, preferred=None):
        """Take the body OFF Bulb before a normal exposure.

        A capture with the shutter still set to Bulb opens it and waits for a close
        that a normal capture never sends: gphoto2 blocks, the body stops answering,
        and it looks for all the world like the camera disconnected. Bulb mode sets
        this and nothing used to put it back.

        Returns (changed, message)."""
        cur = self.current_shutter()
        if cur is None or "bulb" not in cur.lower():
            return False, ""
        cfg = self.camera.get_config()
        _, w = self._find_widget(cfg, self.CONFIG_ALIASES["shutterspeed"])
        choices = [w.get_choice(i) for i in range(w.count_choices())]
        target = None
        if preferred and preferred in choices and "bulb" not in preferred.lower():
            target = preferred
        if target is None:
            usable = [c for c in choices if "bulb" not in c.lower()]
            if not usable:
                return False, "The body offers no speed other than Bulb."
            # No speed was asked for, so pick a short one rather than a long one:
            # a guessed multi-second exposure on the Moon is a ruined burst, while a
            # guessed fast one is merely dark and obviously wrong.
            fast = [c for c in usable if c.startswith("1/")]
            target = fast[len(fast) // 2] if fast else usable[-1]
        w.set_value(target)
        self.camera.set_config(cfg)
        return True, ("Shutter was still on Bulb — set to {} so the exposure can "
                      "actually end.".format(target))

    def set_shutter_to_bulb(self):
        """Sets speed to 'Bulb' (last choice in list). Do ONCE."""
        gp = self.gp
        cfg = self.camera.get_config()
        _, w = self._find_widget(cfg, self.CONFIG_ALIASES["shutterspeed"])
        if w is None:
            raise CameraError("Shutter speed setting not found.")
        target = next((w.get_choice(i) for i in range(w.count_choices())
                       if "bulb" in w.get_choice(i).lower()), None)
        if target is None:
            raise CameraError("'Bulb' value absent: put body in M mode.")
        w.set_value(target)
        self.camera.set_config(cfg)

    def is_alive(self):
        """Does the body still answer? A capture can fail with a vague error while
        the link is fine; tearing the session down then is what looked like a
        disconnect."""
        try:
            self.camera.get_config()
            return True
        except Exception:               # noqa: BLE001
            return False

    def _has_widget(self, name):
        ok, _ = self.gp.gp_widget_get_child_by_name(self.camera.get_config(), name)
        return ok >= self.gp.GP_OK

    def capture_bulb(self, seconds, path, should_abort=None):
        """Opens shutter, waits 'seconds', closes, then retrieves the file. Uses the generic
        'bulb' widget (Sony/Nikon) or Canon's 'eosremoterelease' (Press Full / Release Full).
        'should_abort': callable -> True to cleanly interrupt."""
        gp = self.gp
        # purge any pending events before exposing
        while self.camera.wait_for_event(10)[0] != gp.GP_EVENT_TIMEOUT:
            pass
        canon = self._has_widget("eosremoterelease") and not self._has_widget("bulb")
        if canon:
            self._set_widget("eosremoterelease", "Press Full")   # open shutter (Canon)
        else:
            self._set_widget("bulb", 1)
        try:
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                if should_abort and should_abort():
                    break
                time.sleep(0.1)
        finally:
            if canon:
                self._set_widget("eosremoterelease", "Release Full")   # ALWAYS close
            else:
                self._set_widget("bulb", 0)
        # file wait + download
        deadline = time.monotonic() + seconds + 20
        while time.monotonic() < deadline:
            etype, edata = self.camera.wait_for_event(300)
            if etype == gp.GP_EVENT_FILE_ADDED:
                root, _ = os.path.splitext(path)
                _, ext = os.path.splitext(edata.name)
                dest = root + (ext if ext else ".arw")
                cam_file = self.camera.file_get(edata.folder, edata.name,
                                                gp.GP_FILE_TYPE_NORMAL)
                cam_file.save(dest)
                return dest
        raise CameraError("No file received after bulb exposure (check LENR=Off).")

    def _set_widget_choice(self, name, substr):
        """Set a RADIO/MENU widget to the first choice containing `substr` (case-insensitive)."""
        gp = self.gp
        cfg = self.camera.get_config()
        ok, w = gp.gp_widget_get_child_by_name(cfg, name)
        if ok < gp.GP_OK:
            return False
        choice = next((w.get_choice(i) for i in range(w.count_choices())
                       if substr.lower() in w.get_choice(i).lower()), None)
        if choice is None:
            return False
        w.set_value(choice); self.camera.set_config(cfg); return True

    def _list_all(self, folder="/", depth=0):
        """Every file path gphoto2 can see on the camera (for diagnostics)."""
        out = []
        if depth > 8:
            return out
        try:
            files = self.camera.folder_list_files(folder)
            for i in range(files.count()):
                out.append(folder.rstrip("/") + "/" + files.get_name(i))
            folders = self.camera.folder_list_folders(folder)
            for i in range(folders.count()):
                out += self._list_all(folder.rstrip("/") + "/" + folders.get_name(i), depth + 1)
        except Exception:               # noqa: BLE001
            pass
        return out

    def _list_movies(self, folder="/"):
        """Recursively list movie files on the camera storage (path strings)."""
        exts = (".mp4", ".mov", ".m4v", ".avi", ".mts", ".xavc")
        out = []
        try:
            files = self.camera.folder_list_files(folder)
            for i in range(files.count()):
                nm = files.get_name(i)
                if nm.lower().endswith(exts):
                    out.append(folder.rstrip("/") + "/" + nm)
            folders = self.camera.folder_list_folders(folder)
            for i in range(folders.count()):
                sub = folder.rstrip("/") + "/" + folders.get_name(i)
                out += self._list_movies(sub)
        except Exception:               # noqa: BLE001
            pass
        return out

    def _download_file(self, folder, name, path):
        root, _ = os.path.splitext(path)
        _, ext = os.path.splitext(name)
        dest = root + (ext if ext else ".mp4")
        cam_file = self.camera.file_get(folder, name, self.gp.GP_FILE_TYPE_NORMAL)
        cam_file.save(dest)
        return dest

    def record_movie(self, seconds, path):
        """Trigger the camera's REAL movie recording (to card), wait, stop, then download the
        movie file. Uses 'movie'/'eosmovieswitch'. Many bodies (esp. Sony) do NOT emit a
        file-added event for movies, so we diff the card before/after and grab the new file."""
        gp = self.gp
        mw = next((n for n in ("movie", "eosmovieswitch", "recordingmedia")
                   if self._has_widget(n)), None)
        if mw is None:
            raise CameraError("this camera doesn't expose movie recording over gphoto2")
        try:                                            # make sure the movie lands on the card
            self._set_widget_choice("capturetarget", "card") or \
                self._set_widget_choice("capturetarget", "memory")
        except Exception:               # noqa: BLE001
            pass
        before = set(self._list_movies())
        while self.camera.wait_for_event(10)[0] != gp.GP_EVENT_TIMEOUT:
            pass
        self._set_widget(mw, 1)                          # START recording
        try:
            time.sleep(max(1.0, float(seconds)))
        finally:
            self._set_widget(mw, 0)                      # STOP recording
        # 1) some bodies DO emit a file-added event — take it if it comes quickly
        t_end = time.monotonic() + 8
        while time.monotonic() < t_end:
            etype, edata = self.camera.wait_for_event(500)
            if etype == gp.GP_EVENT_FILE_ADDED and edata.name.lower().endswith(
                    (".mp4", ".mov", ".m4v", ".avi", ".mts")):
                return self._download_file(edata.folder, edata.name, path)
        # 2) fallback: find the NEW movie file on the card and download it
        time.sleep(2.0)                                 # let the card finalise the file
        after = self._list_movies()
        new = [f for f in after if f not in before]
        target = (sorted(new)[-1] if new else (sorted(after)[-1] if after else None))
        if target:
            folder, name = target.rsplit("/", 1)
            return self._download_file(folder, name, path)
        # diagnostic: what CAN gphoto2 see on the card?
        allf = self._list_all()
        folders = sorted({f.rsplit("/", 1)[0] for f in allf})
        sample = ", ".join(f.rsplit("/", 1)[-1] for f in allf[:6]) or "(nothing)"
        raise CameraError(
            "recorded, but the movie wasn't retrievable. gphoto2 sees {} files in {} folders "
            "on the card. Folders: {}. Sample: {}. If movie folders (M4ROOT/AVCHD) are absent, "
            "Sony isn't exposing them — copy the file from the SD card and use 'Load a movie'.".format(
                len(allf), len(folders), (", ".join(folders[:6]) or "none"), sample))

    def list_config(self, logical_name):
        gp = self.gp
        cfg = self.camera.get_config()
        _, w = self._find_widget(cfg, self.CONFIG_ALIASES.get(logical_name, [logical_name]))
        if w is None:
            return [], ""
        choices = [w.get_choice(i) for i in range(w.count_choices())]
        return choices, w.get_value()

    def set_config(self, logical_name, value):
        gp = self.gp
        cfg = self.camera.get_config()
        name, w = self._find_widget(cfg, self.CONFIG_ALIASES.get(logical_name, [logical_name]))
        if w is None:
            raise CameraError("Setting '{}' not available".format(logical_name))
        w.set_value(value)
        self.camera.set_config(cfg)

    def focus_supported(self):
        return self._focus_widget is not None

    def current_exposure_s(self):
        """Exposure time set on body, in seconds (None if bulb/unknown)."""
        try:
            cfg = self.camera.get_config()
            _, w = self._find_widget(cfg, self.CONFIG_ALIASES["shutterspeed"])
            if w is not None:
                return _parse_exposure_seconds(w.get_value())
        except Exception:           # noqa: BLE001
            pass
        return None

    def status_info(self, save_dir):
        import shutil
        gp = self.gp
        batt = "?"
        try:
            cfg = self.camera.get_config()
            ok, w = gp.gp_widget_get_child_by_name(cfg, "batterylevel")
            if ok >= gp.GP_OK:
                batt = str(w.get_value())
        except Exception:           # noqa: BLE001
            pass
        free = None
        try:
            d = save_dir if os.path.isdir(save_dir) else (os.path.dirname(save_dir) or "/")
            free = shutil.disk_usage(d).free / 1e9
        except Exception:           # noqa: BLE001
            pass
        return {"battery": batt, "free_gb": free}

    def focus_step(self, amount):
        gp = self.gp
        if not self._focus_widget:
            raise CameraError("focus not exposed by this body via gphoto2")
        cfg = self.camera.get_config()
        ok, w = gp.gp_widget_get_child_by_name(cfg, self._focus_widget)
        if ok < gp.GP_OK:
            raise CameraError("focus widget unavailable")
        wtype = w.get_type()
        if wtype == gp.GP_WIDGET_RANGE:                 # relative drive (Nikon-style)
            w.set_value(float(amount))
        elif wtype in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):   # Near/Far steps (Canon-style)
            choices = [w.get_choice(i) for i in range(w.count_choices())]
            want = "far" if amount > 0 else "near"
            mag = str(min(3, max(1, abs(int(amount)) or 1)))
            pick = next((c for c in choices if want in c.lower() and mag in c), None) \
                or next((c for c in choices if want in c.lower()), None)
            if pick is None:
                raise CameraError("no Near/Far steps: {}".format(choices))
            w.set_value(pick)
        else:
            w.set_value(int(amount))
        self.camera.set_config(cfg)
        return self._focus_widget

    @classmethod
    def resume_daemon(cls):
        """Wake the PTP daemon (undo SIGSTOP) so macOS camera support works again."""
        if platform.system() != "Darwin":
            return
        try:
            subprocess.run(["/usr/bin/killall", "-CONT", *cls._MAC_PTP_DAEMONS],
                           capture_output=True, timeout=2)
        except Exception:           # noqa: BLE001
            pass

    def close(self):
        try:
            self.camera.exit()
        except Exception:
            pass
        self.resume_daemon()


# ---------------------------------------------------------------------------
#  Worker: owns camera, preview loop + commands (single thread)
# ---------------------------------------------------------------------------
class SkyWatcherMount:
    """Minimal SkyWatcher/SynScan motor-protocol client over WiFi (UDP port 11880), used
    for RA-only dithering on a Star Adventurer 2i. It briefly slews RA by a small angle
    then resumes sidereal tracking (tracking is never lost).

    BETA: the byte-level protocol is implemented from the public EQMOD/SkyWatcher motor
    command set but is UNTESTED here — validate on-sky and adjust the motion-mode bytes if
    the mount moves the wrong way or not at all. Every command/response is logged."""
    SIDEREAL_DAY = 86164.0905
    #: fixed cost of any move (round-trips, ramp, settle), measured on a SA 2i
    MOVE_OVERHEAD_S = 3.0

    def __init__(self, ip="192.168.4.1", port=11880, timeout=2.0, logfn=None, two_axis=False):
        import socket
        self.ip = ip; self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.cpr = None; self.freq = None
        self.two_axis = two_axis          # HEQ-5 Pro etc.: also drive the Dec axis (axis 2)
        self.cpr2 = None; self.freq2 = None
        self.motor_sign = None            # encoder counts vs the sky  } both measured,
        self.command_sign = None          # direction bit vs the sky    } never assumed
        self.measured_rate_dps = None     # real slew rate, learned from the moves
        self._rate_from_deg = 0.0         # how long the move that taught us it was
        self._rate, self._rate_dps = "sidereal", None
        self._log = logfn or (lambda m: None)

    #: Commands safe to re-send when a UDP reply goes missing. Lower case are
    #: inquiries, which change nothing; the upper case ones are idempotent setters
    #: — stop, motion mode, step count, period, position, init — so asking twice
    #: lands in the same state. 'J' (start motion) is deliberately absent: it is
    #: the one command where a second copy could restart a move already running,
    #: so it gets the careful treatment in _start_motion instead.
    _READ_CMDS = "abdefgjsKGHIEF"

    #: This protocol is request/response over a single UDP socket, so two threads
    #: must never have exchanges in flight at once or they collect each other's
    #: replies. Shared across instances: there is one mount.
    _LOCK = threading.RLock()

    def _send(self, body, retries=None):
        """Send one command over UDP, retrying reads.

        This is Wi-Fi and UDP: packets do go missing. A lost reply during a slew
        used to abort the wait, which let the caller stop the motor half way
        through a GoTo and call it finished."""
        msg = (":" + body + "\r").encode("ascii")
        if retries is None:
            retries = 5 if body[:1] in self._READ_CMDS else 0
        last = None
        for _ in range(retries + 1):
            try:
                with self._LOCK:
                    self.sock.sendto(msg, (self.ip, self.port))
                    data, _ = self.sock.recvfrom(256)
                resp = data.decode("ascii", "ignore").strip()
                self._log("SW >{}  <{}".format(body, resp))
                if resp.startswith("!"):
                    raise RuntimeError("mount error " + resp)
                return resp.lstrip("=").rstrip("\r")
            except RuntimeError:
                raise                                   # the board said no: real answer
            except OSError as e:                        # socket closed / network down
                last = e
            except Exception as e:                      # noqa: BLE001  (timeout, socket)
                last = e
        raise last

    def _start_motion(self, axis="1"):
        """Send J (start motion), which must not simply be re-sent on a lost reply.

        If the REQUEST arrived and only the reply was lost, the axis is already
        moving and a second J could restart the move. So ask the board what it is
        doing, and only re-issue if it is in fact still stopped."""
        import time
        try:
            return self._send("J" + axis, retries=0)
        except RuntimeError:
            raise
        except Exception:                               # noqa: BLE001  (reply lost)
            time.sleep(0.2)
            try:
                st = self._send("f" + axis)
                if len(st) >= 2 and (int(st[1], 16) & 0x01):
                    return ""                           # already running: it landed
            except Exception:                           # noqa: BLE001
                pass
            return self._send("J" + axis, retries=0)

    @staticmethod
    def _hex_lsb(n, digits=6):
        n = int(round(n)) & ((1 << (4 * digits)) - 1)
        s = "{:0{}X}".format(n, digits)
        return "".join(reversed([s[i:i + 2] for i in range(0, digits, 2)]))

    @staticmethod
    def _unhex_lsb(s):
        s = s.strip()
        pairs = [s[i:i + 2] for i in range(0, len(s), 2)]
        return int("".join(reversed(pairs)) or "0", 16)

    def connect(self, preserve_position=False):
        """Bring the motor board up.

        `preserve_position` keeps the axis step count instead of re-zeroing it: a
        Wi-Fi drop does not stop the board, so after a reconnect the counter is
        still valid and an axis-1 sync must survive it. Zeroing here would throw
        that away silently and send the next GoTo somewhere absurd."""
        # Opening the link is the one place where waiting is pointless: either the
        # board answers or the Mac is not on its Wi-Fi. Five retries at a two-second
        # timeout per command meant 35 s of apparent nothing before the failure showed.
        old_to = self.sock.gettimeout()
        try:
            self.sock.settimeout(0.8)
            try:
                self._send("e1", retries=1)             # version (handshake, optional)
            except Exception:           # noqa: BLE001
                pass
            try:
                self.cpr = self._unhex_lsb(self._send("a1", retries=2))   # counts / turn
                self.freq = self._unhex_lsb(self._send("b1", retries=2))  # timer frequency
            except Exception as e:      # noqa: BLE001
                raise RuntimeError(
                    "no answer from the motor board at {} — is this Mac on the mount's "
                    "Wi-Fi network, and the mount switched on? ({})".format(self.ip, e))
        finally:
            self.sock.settimeout(old_to)
        if not self.cpr or not self.freq:
            raise RuntimeError("no motor-board response")
        if not preserve_position:
            try:
                self._send("E1" + self._hex_lsb(0x800000))   # axis position := home
            except Exception:           # noqa: BLE001
                pass
        try:
            init = self._send("F1") or "ok"             # initialise & energise motor
        except Exception as e:          # noqa: BLE001
            init = "ERR " + str(e)
        if self.two_axis:                               # HEQ-5 Pro: bring up the Dec axis too
            try:
                self.cpr2 = self._unhex_lsb(self._send("a2"))
                self.freq2 = self._unhex_lsb(self._send("b2"))
                self._send("E2" + self._hex_lsb(0x800000))
                self._send("F2")
            except Exception as e:      # noqa: BLE001
                self._log("Dec axis init failed: {}".format(e))
        return "CPR={} freq={} init={}{}".format(
            self.cpr, self.freq, init, " +Dec" if self.two_axis and self.cpr2 else "")

    def slew_dec(self, arcsec):
        """Slew the Dec axis (axis 2) by `arcsec` — HEQ-5 Pro only. Dec doesn't track, so no
        resume afterwards. No-op on single-axis mounts (Star Adventurer)."""
        if not self.two_axis or not self.cpr2:
            raise RuntimeError("this mount has no Dec axis")
        steps = int(round(self.cpr2 * (abs(arcsec) / 1296000.0)))
        if steps < 1:
            return 0.0
        direction = "0" if arcsec >= 0 else "1"
        self._send("K2")
        self._send("G2" + "0" + direction)
        self._send("H2" + self._hex_lsb(steps))
        self._send("J2")
        self._wait_stopped(axis="2")
        return (steps / self.cpr2) * 1296000.0 * (1 if arcsec >= 0 else -1)

    def _stop_and_settle(self, axis="1", tmax=15.0):
        """Stop the axis and keep asking until the board says it really has.

        Two lessons, both learned the hard way on the mount:
        K only REQUESTS a decelerating stop, so any G/H/I/J sent while the axis is
        still turning is refused with '!2 motor not stopped' — which after a fast
        GoTo left the mount stopped and no longer tracking. And a single K does not
        always take: stopping a runaway slew needed a second one before the axis
        even began to slow. So K is repeated while we wait rather than sent once
        and trusted."""
        import time
        t0 = last_k = time.monotonic()
        self._send("K" + axis)
        while time.monotonic() - t0 < tmax:
            try:
                st = self._send("f" + axis)
                if len(st) >= 2 and (int(st[1], 16) & 0x01) == 0:
                    return True
            except Exception:                           # noqa: BLE001
                return False
            now = time.monotonic()
            if now - last_k > 0.8:                      # say it again
                last_k = now
                try:
                    self._send("K" + axis)
                except Exception:                       # noqa: BLE001
                    return False
            time.sleep(0.05)
        return False

    def _wait_stopped(self, tmax=8.0, axis="1", should_abort=None, stall_s=6.0,
                      progress_cb=None, total_steps=None, start_counts=None):
        """Wait for the axis to stop — on PROGRESS, not on a stopwatch.

        Measured on a Star Adventurer 2i: the board runs its GoTo at its own rate
        (~0.2°/s) and ignores the step period we send, so any duration derived
        from the commanded rate is fiction. Timing out on that fiction cuts a long
        slew off part way and then reports success. Watching the position counter
        instead is rate-agnostic: we stop waiting when the axis stops moving."""
        import time
        t0 = time.monotonic()
        last_pos, last_change = None, t0
        last_report = t0
        lost = 0
        hard_cap = max(tmax, 900.0)                     # 15 min backstop
        while True:
            now = time.monotonic()
            if should_abort is not None and should_abort():
                try:
                    # insist until the board confirms: Stop must mean stopped
                    self._stop_and_settle(axis)
                except Exception:                       # noqa: BLE001
                    pass
                return
            try:
                st = self._send("f" + axis)             # axis status nibbles
                if len(st) >= 2 and (int(st[1], 16) & 0x01) == 0:
                    return
                lost = 0
            except RuntimeError:                        # the board refused: stop asking
                return
            except Exception:                           # noqa: BLE001  (comms)
                lost += 1
                if lost > 6:                            # link really is gone
                    return
            try:
                pos = self._unhex_lsb(self._send("j" + axis))
            except Exception:                           # noqa: BLE001
                pos = None
            if pos is not None:
                if pos != last_pos:
                    last_pos, last_change = pos, now
                elif now - last_change > stall_s:
                    return                              # stopped moving: done or stuck
                # a 90° slew takes five minutes on this mount; say something
                if (progress_cb and total_steps and start_counts is not None
                        and now - last_report > 0.4):
                    last_report = now
                    moved = abs((pos - 0x800000) - start_counts)
                    frac = min(1.0, moved / float(total_steps))
                    left = (total_steps - moved) / float(self.cpr) * 360.0
                    progress_cb(frac, max(left, 0.0))
            if now - t0 > hard_cap:
                return
            time.sleep(0.1)

    def dither_ra(self, arcsec):
        """Slew RA by `arcsec` (signed), then resume sidereal tracking. Tracking is ALWAYS
        re-asserted, even if the slew errors, so a glitch can't leave the mount stopped."""
        if self.cpr is None:
            self.connect()
        steps = int(round(self.cpr * (abs(arcsec) / 1296000.0)))
        if steps < 1:
            self.resume_tracking()
            return 0.0
        direction = "0" if arcsec >= 0 else "1"
        try:
            self._stop_and_settle()                     # stop, and wait for it
            self._send("G1" + "0" + direction)         # goto mode + direction
            self._send("H1" + self._hex_lsb(steps))    # goto step increment
            self._start_motion()                            # start motion
            self._wait_stopped()
        finally:
            self.resume_tracking()                      # GUARANTEE tracking restarts
        return (steps / self.cpr) * 1296000.0 * (1 if arcsec >= 0 else -1)

    # ---- absolute positioning (closed loop on the axis encoder) --------------
    #  The motor board keeps a step counter, so a GoTo never has to be flown blind:
    #  read where the axis is, move, read again. That self-corrects a wrong slew
    #  rate, a slew that timed out, and the sky moving while tracking was paused.

    def get_counts(self, retries=None):
        """Axis-1 encoder position, in steps, relative to the 0x800000 home value.

        `retries=0` is for polling: a status read that is repeated every couple of
        seconds gains nothing from insisting, and insisting is what turns one lost
        packet into a multi-second stall."""
        return self._unhex_lsb(self._send("j1", retries=retries)) - 0x800000

    def counts_to_deg(self, counts):
        return (counts / float(self.cpr)) * 360.0

    def deg_to_counts(self, deg):
        return int(round(deg / 360.0 * self.cpr))

    def measure_directions(self, seconds=6.0, test_deg=1.0):
        """Tie BOTH conventions to the sky by experiment, not by assumption.

        Two independent signs are involved and they are not the same thing:
          * motor_sign   — which way the step counter runs as the hour angle grows.
                           Timed against tracking, which by definition follows the sky.
          * command_sign — which way the direction bit actually drives the axis.
                           Measured with one short test move.
        Assuming the second from the first silently drives a GoTo backwards on a
        mount wired the other way round."""
        import time
        self.resume_tracking()
        p0 = self.get_counts()
        time.sleep(max(2.0, seconds))
        p1 = self.get_counts()
        if p1 == p0:
            raise RuntimeError("the axis counter did not move while tracking — "
                               "is the motor actually running?")
        self.motor_sign = 1.0 if p1 > p0 else -1.0

        before = self.get_counts()
        self.goto_delta_deg(abs(test_deg), rate_mult=64)     # nominal "forward" bit
        moved = self.get_counts() - before
        if moved == 0:
            raise RuntimeError("the axis did not move on a test GoTo — check the "
                               "motor-board wiring or the slew rate.")
        self.command_sign = 1.0 if (self.motor_sign * moved) > 0 else -1.0
        # Put it back. You centred a star to get here; leaving the axis a degree
        # off would send you hunting for it again, and it makes the test visible
        # as a there-and-back rather than a mysterious drift.
        try:
            self.goto_delta_deg(-abs(test_deg), rate_mult=64)
        except Exception:               # noqa: BLE001
            pass
        return self.motor_sign, self.command_sign

    def max_slew_deg_per_s(self, rate_mult=64):
        """Slew rate to expect, in degrees of sky per second.

        Prefers a rate actually measured on this mount: the commanded period is
        only an upper bound the board is free to ignore, and on a Star Adventurer
        2i it does exactly that."""
        measured = getattr(self, "measured_rate_dps", None)
        if measured:
            return measured
        # A Star Adventurer 2i runs its GoTo at its own pace and ignores the period
        # we send, so the computed figure below is an upper bound, not a promise.
        period = max(int(round(self.freq * self.SIDEREAL_DAY / self.cpr / max(rate_mult, 1))), 8)
        return (self.freq / float(period)) / float(self.cpr) * 360.0

    def goto_delta_deg(self, deg, rate_mult=64, settle=0.4, should_abort=None,
                       progress_cb=None):
        """Turn axis 1 by `deg` (signed, sky degrees) and come back tracking.

        Unlike dither_ra this sets the slew speed explicitly and waits a timeout
        scaled to the distance — an 8 s cap is fine for a dither but would abort a
        real GoTo half way."""
        import time
        if self.cpr is None:
            self.connect()
        steps = self.deg_to_counts(abs(deg))
        if steps < 1:
            self.resume_tracking()
            return 0.0
        before = self.get_counts()
        direction = "0" if deg >= 0 else "1"
        period = max(int(round(self.freq * self.SIDEREAL_DAY / self.cpr / max(rate_mult, 1))), 8)
        # The period floor caps the real slew rate well below what a big rate_mult
        # asks for, so time the wait from the period actually sent — not from the
        # rate requested, or a long GoTo is cut off part way.
        est = steps / (self.freq / float(period))
        elapsed = 0.0
        try:
            self._stop_and_settle()                       # stop, and wait for it
            self._send("G1" + "0" + direction)            # goto mode + direction
            self._send("H1" + self._hex_lsb(steps))       # how far
            self._send("I1" + self._hex_lsb(period))      # how fast (dither_ra omits this)
            self._start_motion()                              # go
            t0 = time.monotonic()
            self._wait_stopped(tmax=est * 2.0 + 12.0, should_abort=should_abort,
                               progress_cb=progress_cb, total_steps=steps,
                               start_counts=before)
            elapsed = time.monotonic() - t0
            time.sleep(settle)
        finally:
            self.resume_tracking()                        # tracking is never left off
        done = self.counts_to_deg(self.get_counts() - before)
        if elapsed > 1.0 and abs(done) > 0.5:
            # Learn the STEADY rate: every move also pays a fixed ~3 s of command
            # round-trips, motor ramp and settle, which makes a short slew look far
            # slower than the mount really is. Keeping the best rate seen converges
            # on the asymptote instead of the short-move penalty.
            # A long move measures the steady rate far better than a short one,
            # where the fixed overhead dominates — so trust the longest seen.
            r = abs(done) / max(elapsed - self.MOVE_OVERHEAD_S, 0.5)
            if abs(done) >= getattr(self, "_rate_from_deg", 0.0):
                self._rate_from_deg = abs(done)
                self.measured_rate_dps = r
        return done

    def reconnect(self):
        """Re-open the UDP socket and re-initialise — used to recover from a Wi-Fi drop."""
        import socket
        try:
            self.sock.close()
        except Exception:               # noqa: BLE001
            pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)
        self.cpr = None; self.freq = None
        return self.connect(preserve_position=True)     # keep the count: see connect()

    def ensure_tracking(self):
        """Make sure the mount is tracking; if a command fails (Wi-Fi drop), reconnect once
        and retry. Called between exposures so a silent stop is self-healed."""
        try:
            self.resume_tracking()
        except Exception:               # noqa: BLE001
            self.reconnect()
            self.resume_tracking()

    SOLAR_DAY = 86400.0
    LUNAR_DAY = 86164.0905 / 0.97633     # mean lunar rate, ~14.685″/s

    def set_rate(self, rate):
        """Tracking rate: 'sidereal', 'solar', 'lunar', or a number in degrees of
        hour angle per second (what body_track_rate returns for a planet)."""
        if isinstance(rate, (int, float)):
            self._rate, self._rate_dps = "custom", float(rate)
        else:
            self._rate = rate if rate in ("solar", "lunar", "sidereal") else "sidereal"
            self._rate_dps = None

    def tracking_omega(self):
        """Degrees of hour angle per second the axis should turn at."""
        if getattr(self, "_rate_dps", None):
            return self._rate_dps
        day = {"solar": self.SOLAR_DAY, "lunar": self.LUNAR_DAY}.get(
            getattr(self, "_rate", "sidereal"), self.SIDEREAL_DAY)
        return 360.0 / day

    def resume_tracking(self, omega_dps=None):
        if self.cpr is None or self.freq is None:
            self.connect()
        omega = float(omega_dps or self.tracking_omega())
        # timer ticks per motor step = ticks per second / steps per second
        period = int(round(self.freq * 360.0 / (self.cpr * omega)))
        import time
        for attempt in (0, 1):
            self._stop_and_settle()                     # K1 alone is not enough
            try:
                self._send("G110")                      # low-speed (tracking), forward
                self._send("I1" + self._hex_lsb(period))
                self._start_motion()
                return
            except RuntimeError:
                if attempt:
                    raise
                time.sleep(1.5)                         # still decelerating: let it

    def slew(self, direction, rate_mult=128):
        """Continuous manual slew in RA at rate_mult × sidereal (visible rotation). Use
        stop() to halt. direction: +1 forward, -1 reverse."""
        if self.cpr is None:
            self.connect()
        period = int(round(self.freq * self.SIDEREAL_DAY / self.cpr / max(rate_mult, 1)))
        period = max(period, 8)
        d = "0" if direction >= 0 else "1"
        self._stop_and_settle()
        self._send("G1" + "1" + d)                      # slew (continuous) mode, direction
        self._send("I1" + self._hex_lsb(period))
        self._start_motion()

    def stop(self):
        if self.cpr is None:
            self.connect()
        self._stop_and_settle()                         # stop RA, and check it took

    def close(self):
        try:
            self.sock.close()
        except Exception:               # noqa: BLE001
            pass


class _MountOp(QtCore.QThread):
    """One long mount operation, run off the camera thread.

    A GoTo can take five minutes. Run on the camera thread it freezes the live
    view for all of it — and worse, that thread is also the only one draining the
    command queue, so the Stop button lands in a queue nobody is reading until the
    very move it was meant to interrupt has finished.

    A plain thread is the right tool rather than a second process: this work is
    almost entirely waiting on a UDP socket, which releases the GIL, so the
    preview loop keeps its slice. There is no computation to parallelise."""

    def __init__(self, fn=None, kw=None, parent=None):
        super().__init__(parent)
        self._fn, self._kw = fn, kw or {}

    def set_job(self, fn, kw):
        """Re-arm this thread for another job instead of building a new one.

        Creating a QThread every couple of seconds and dropping the old one is a
        trap: if the object is collected while its thread is still winding down,
        Qt prints 'Destroyed while thread is still running' and calls abort() —
        a SIGABRT, exit code 134. Reusing one object removes the race entirely."""
        self._fn, self._kw = fn, kw or {}

    def run(self):
        if self._fn is not None:
            self._fn(self._kw)


#: Threads that outlived their owner. Letting Python collect a QThread that is
#: still running makes Qt print 'Destroyed while thread is still running' and call
#: abort() — a SIGABRT. Parking the reference here costs nothing and cannot crash.
_LINGERING_THREADS = []


def _save_axis1_sync(zero):
    """Keep the axis-1 sync across restarts: it stays valid as long as the mount is
    not switched off, and re-doing it costs a star and twenty seconds."""
    try:
        QtCore.QSettings("NOUT", "NOUT").setValue(
            "axis1_sync", json.dumps({"counts": int(zero[0]), "ha": float(zero[1]),
                                      "sign": int(zero[2]), "t": time.time()}))
    except Exception:                   # noqa: BLE001
        pass


def _load_axis1_sync(max_age_h=24.0):
    try:
        d = json.loads(QtCore.QSettings("NOUT", "NOUT").value("axis1_sync", "", type=str))
        if time.time() - float(d.get("t", 0)) > max_age_h * 3600:
            return None
        return (int(d["counts"]), float(d["ha"]), int(d["sign"]))
    except Exception:                   # noqa: BLE001
        return None


def _retire_thread(op, timeout_ms=15000):
    """Wait for a worker thread, and never drop it while it is still alive."""
    if op is None:
        return
    try:
        if op.wait(timeout_ms):
            return
    except Exception:                   # noqa: BLE001
        pass
    _LINGERING_THREADS.append(op)


class _MountQueue(QtCore.QThread):
    """Short mount commands, run off the camera thread and strictly in order.

    Connecting, starting tracking or changing rate all end in _stop_and_settle or a
    handshake, which poll the mount over UDP for seconds. Run on the camera thread
    they freeze the live view — the same fault that spoiled the GoTo and then the
    tracking probe. One long-lived thread avoids both that and the QThread churn
    that made the app abort."""

    def __init__(self, handler, parent=None):
        super().__init__(parent)
        self._handler = handler
        self._q = queue.Queue()
        self._go = True

    def put(self, kind, kw):
        self._q.put((kind, kw))

    def shutdown(self):
        """Stop after the item in flight, dropping whatever is still queued."""
        self._go = False
        while True:                     # pending work is pointless once we are closing
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put((None, None))

    def run(self):
        while self._go:
            kind, kw = self._q.get()
            if kind is None or not self._go:
                break
            try:
                self._handler(kind, kw)
            except Exception:           # noqa: BLE001  (never kill the queue)
                pass


def gallery_dir():
    """Where the gallery keeps its copies: a folder you can browse and back up."""
    base = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.PicturesLocation) or os.path.expanduser("~")
    d = os.path.join(base, "NOUT Gallery")
    os.makedirs(d, exist_ok=True)
    return d


def load_gallery():
    """{object id: {"file", "date", "frames", "integ", "note"}}."""
    try:
        with open(os.path.join(gallery_dir(), "gallery.json"), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:                   # noqa: BLE001
        return {}


def save_gallery(d):
    try:
        with open(os.path.join(gallery_dir(), "gallery.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except Exception as e:              # noqa: BLE001
        print("[NOUT] gallery not saved:", e, file=sys.stderr, flush=True)


def gallery_slots():
    """Every slot of the collection: the 110 Messier first, then the named NGC/IC
    objects NOUT knows a distance for — the ones people actually go after."""
    out = [("M{}".format(i), "Messier") for i in range(1, 111)]
    try:
        import csv as _csv
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dso_info.csv"),
                  newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                oid = (row.get("id") or "").strip()
                if oid and not re.match(r"(?i)^m\d+$", oid):
                    out.append((oid, "NGC / IC"))
    except Exception:                   # noqa: BLE001
        pass
    return out


class _FrameProcessor(QtCore.QThread):
    """Decodes, measures and stacks saved frames, off the capture thread.

    The camera thread only shoots and downloads; everything that takes time happens
    here, so the interval between exposures is the exposure plus the download, not
    the exposure plus the download plus the stack."""
    def __init__(self, worker):
        super().__init__()
        self._w = worker
        self._q = queue.Queue()
        self._running = True

    def submit(self, job):
        self._q.put(job)

    def backlog(self):
        return self._q.qsize()

    def idle(self):
        return self._q.empty() and not getattr(self, "_busy", False)

    def wait_idle(self, timeout_s=60.0):
        end = time.monotonic() + timeout_s
        while not self.idle() and time.monotonic() < end:
            time.sleep(0.05)
        return self.idle()

    def stop(self):
        self._running = False
        self._q.put(None)

    def run(self):
        while self._running:
            job = self._q.get()
            if job is None:
                break
            self._busy = True
            try:
                self._w._process_frame(*job)
            except Exception as e:      # noqa: BLE001
                print("[NOUT] frame processing failed:", e, file=sys.stderr, flush=True)
            finally:
                self._busy = False


class CameraWorker(QtCore.QThread):
    frame_ready = Signal(object, float, object)  # (BGR frame, sharpness|nan, stars|None)
    capture_saved = Signal(str)
    capture_image_ready = Signal(object, str)  # (BGR preview ndarray | None, caption)
    interval_running = Signal(bool)
    interval_paused = Signal(bool)         # the series is held, not finished          # True when intervalometer is running
    interval_progress = Signal(int, int, float)  # (done, total|0=unlimited, avg sec/shot)
    cam_status = Signal(str)                  # battery + free disk
    seq_started = Signal(int, str, str)       # (id, type, mode)
    seq_progress = Signal(int, int, float)    # (id, num shots, integration s)
    seq_ended = Signal(int, int, float, float)  # (id, num shots, integration s, duration s)
    seq_result = Signal(int, object, str)     # (id, result BGR image|None, caption)
    stack_ready = Signal(object, int, float)  # (stacked BGR image, num shots, integration s)
    track_point = Signal(int, float, int, float)  # (shot index, HFR, num stars, eccentricity)
    config_loaded = Signal(dict)             # {logical_name: (choices, current)}
    status = Signal(str)
    mount_status = Signal(str)                # persistent mount connection/tracking state
    dither_point = Signal(float, int)         # (RA offset arcsec, shot number) per dither
    mount_goto_done = Signal(float)           # emitted after a targeted RA goto completes
    mount_axis1_result = Signal(bool, str, float)  # (ok, message, plate angle to set)
    mount_axis1_progress = Signal(float, float, float)  # (fraction, deg left, s left)
    mount_track_state = Signal(float, float, float)    # (measured, commanded, cpr) steps/s
    movie_ready = Signal(str)                 # emitted with the path of a recorded movie
    failed = Signal(str)

    def __init__(self, backend_factory, metric="tenengrad", roi_frac=0.5):
        super().__init__()
        self._make_backend = backend_factory
        self.metric = metric
        self.roi_frac = roi_frac
        self._cmd = queue.Queue()
        self._running = True
        # intervalometer state
        self._interval = 0.0
        self._remaining = 0
        self._total = 0
        self._done = 0
        self._next_shot = None
        self._paused_at = None         # (next shot time, frames left) while held
        self._speed_recovery = 1.2   # USB/camera recovery pause after each speed-mode shot
        self._frame_times = deque(maxlen=8)   # recent durations per shot (for ETA)
        self._save_dir = os.path.expanduser("~/SonyTether")
        # exposure mode
        self._bulb = False           # False = body speed (≤30s) ; True = timed bulb
        self._bulb_seconds = 60.0
        self._abort_exposure = False
        # activations (battery / CPU saving)
        self._live_enabled = True
        self._metric_enabled = True
        self._stars_enabled = False
        # auto cull bad shots + session log
        self._cull_enabled = False
        self._star_hist = deque(maxlen=10)
        self._log_path = None
        # visual live stacking (Welford running average + kappa-sigma rejection)
        self._stack_enabled = False
        self._raw_full = False        # full-resolution demosaic for the capture preview
        self._stack_linear = True      # stack linear light from RAW (Siril-like) when possible
        self._stack_full = False       # half-size RAW by default: 4x less memory and work
        self._shutter_before_bulb = None   # so a bulb series can hand the speed back
        self._apply_flats = False      # divide frames by a master flat (vignetting)
        self._flats_dir = None
        self._master_flat = None       # cached master flat for the current aperture
        self._flat_fnum = None
        self._apply_darks = False      # subtract a master dark/bias (camera signal)
        self._master_dark = None
        self._dark_key = None
        self._dither_on = False         # RA dithering via the mount (Star Adventurer 2i)
        self._dither_mount = None
        self._axis1_zero = None         # (encoder counts, hour angle, motor sign)
        self._axis1_abort = False
        self._mount_op = None           # long mount moves run on their own thread
        self._probe_op = None           # so does the two-second tracking probe
        self._cap_retries = 0           # consecutive retries of the current frame
        self._mount_q = None            # slow mount commands, off the camera thread
        self._track_sample = None       # (t, counts) for measuring the tracking rate
        self._track_spec = None         # (body, lat, lon) for a non-sidereal rate
        self._dither_every = 4; self._dither_amp = 150.0; self._dither_settle = 4.0
        self._shots_since_dither = 0; self._dither_pos = 0.0
        self._proc = None              # frame-processing thread (decode/measure/stack)
        self._calibration = False      # flats/darks/bias: nothing to display or stack
        self._stack_reset_req = False  # reset asked for while the processor is running
        self._stack_ref_gray = None
        self._stack_wpx = None
        self._stack_crop = None
        self._stack_qhist = deque(maxlen=12)
        self._stack_mean = None
        self._stack_M2 = None
        self._stack_count = 0
        self._kappa_enabled = False
        self._kappa = 2.5
        # whole-frame rejection during live stacking (clouds, trails, bad tracking)
        self._stack_reject = False
        self._stack_ref_stars = None      # reference star count (median of first frames)
        self._stack_starhist = []
        self._stack_starwin = deque(maxlen=12)  # rolling star counts (adaptive)
        self._stack_consec_rej = 0        # consecutive rejects (anti-freeze safety)
        self._stack_rejected = 0
        # auto stop at astronomical dawn (Sun above -18°)
        self._dawn_stop = False
        self._stop_sun_alt = None      # stop when Sun rises above this altitude (deg) or None
        self._stop_fixed_min = None    # stop at this local time (minutes since midnight) or None
        self._stop_target_alt = None   # stop when target drops below this altitude (deg) or None
        self._target_ra = None; self._target_dec = None
        self._site_lat = 48.8566
        self._site_lon = 2.3522
        # auto stop on low battery / disk
        self._autostop = False
        self._batt_thr = 15
        self._disk_thr = 2.0
        # sequences (bursts) + integration time
        self._seq_id = 0
        self._seq_count = 0
        self._seq_integ = 0.0
        self._seq_start = None
        self._seq_type = ""
        self._last_preview = None

    # -- commands (called from UI) ----------------------------------
    def post(self, kind, **kw):
        self._cmd.put((kind, kw))

    def stop(self):
        self._running = False
        self._axis1_abort = True
        proc = getattr(self, "_proc", None)
        if proc is not None:
            try:
                proc.stop()
            except Exception:               # noqa: BLE001
                pass
        q = getattr(self, "_mount_q", None)
        if q is not None:
            try:
                q.shutdown()
            except Exception:               # noqa: BLE001
                pass
            _retire_thread(q)
            self._mount_q = None
        for name in ("_mount_op", "_probe_op"):
            _retire_thread(getattr(self, name, None))
            setattr(self, name, None)
        self._cmd.put(("__stop__", {}))

    # -- loop ------------------------------------------------------------
    def run(self):
        try:
            backend = self._make_backend()
        except CameraError as e:
            msg = str(e)
            if "not installed" in msg or "--simulate" in msg:
                self.failed.emit(msg)         # gphoto2 missing: nothing to wait for
                return
            backend = self._reconnect(None)   # camera off/busy: wait for it to appear
            if backend is None:
                return
        else:
            self.status.emit("Connected: {}".format(backend.model_name()))
            self._emit_config(backend)

        while self._running:
            self._drain_commands(backend)
            if not self._running:
                break

            # intervalometer
            if self._next_shot is not None and time.monotonic() >= self._next_shot:
                stop_reason = self._should_autostop()
                if stop_reason:
                    self._next_shot = None
                    self._end_sequence()
                    self.interval_running.emit(False)
                    if self._dither_mount is not None:   # also stop the mount
                        try:
                            self._dither_mount.stop()
                            self.mount_status.emit("🌅 {}: tracking stopped".format(stop_reason))
                        except Exception:     # noqa: BLE001
                            pass
                    self.status.emit("🌅 Auto stop rule « {} » — sequence ended "
                                     "({} shots).".format(stop_reason, self._done))
                else:
                    try:
                        self._do_capture(backend)
                        self._cap_retries = 0
                    except Exception as e:        # noqa: BLE001
                        self.status.emit("Capture error: {}".format(str(e)[:160]))
                        alive = False
                        try:
                            alive = backend.is_alive()
                        except Exception:             # noqa: BLE001
                            alive = False
                        if alive and self._cap_retries < 3:
                            # the body still answers: this was a glitch, not a drop
                            self._cap_retries += 1
                            self.status.emit(
                                "… camera still responding — retrying this frame "
                                "({}/3).".format(self._cap_retries))
                            self.msleep(900)
                            continue
                        self._cap_retries = 0
                        # Whether the body answers is the real test. Error codes lie:
                        # '[-1] Unspecified error' shows up with a perfectly healthy
                        # link, and a dead link can report almost anything.
                        if not alive or self._is_disconnect(e):
                            backend = self._reconnect(backend)
                            if backend is None:
                                break
                            continue              # retry this shot, keep the sequence
                        raise
                    if self._next_shot is None:         # stop triggered during capture
                        pass
                    else:
                        if not self._bulb and self._running:
                            # camera-speed mode fires fast; give the USB/camera a breather
                            # after each RAW download to avoid dropouts.
                            self.msleep(int(self._speed_recovery * 1000))
                        if self._total <= 0:            # unlimited: until manual Stop
                            self._maybe_dither()
                            self._next_shot = time.monotonic() + self._interval
                        else:                           # finite length series
                            self._remaining -= 1
                            self._maybe_dither()
                            if self._remaining <= 0:
                                self._next_shot = None
                                self._end_sequence()
                                self.interval_running.emit(False)
                                self.status.emit("Intervalometer finished ({} shots).".format(
                                    self._done))
                            else:
                                self._next_shot = time.monotonic() + self._interval

            # live view ONLY at idle, if enabled; during intervalometer we
            # give way to the last photo (and save battery + USB bus).
            if self._next_shot is None and self._live_enabled:
                try:
                    jpeg = backend.get_preview_jpeg()
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._live_fail = 0
                        val = float("nan")
                        if self._metric_enabled:
                            gray, _ = _to_gray_roi(frame, self.roi_frac)
                            val = METRICS[self.metric](gray)
                        stars = None
                        if self._stars_enabled:
                            g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            stars, _ = detect_stars(g2)
                        self.frame_ready.emit(frame, val, stars)
                except Exception as e:        # noqa: BLE001 - survive USB glitches
                    self._live_fail = getattr(self, "_live_fail", 0) + 1
                    # a real disconnect, or many glitches in a row → reconnect
                    if self._is_disconnect(e) or self._live_fail >= 8:
                        backend = self._reconnect(backend)
                        if backend is None:
                            break
                        self._live_fail = 0
                    else:
                        self._say("Live view: {}".format(e), key="preview", min_s=5.0)
                        self.msleep(200)
            else:
                self.msleep(60)               # intervalometer running or live view off

        backend.close()
        _retire_thread(getattr(self, "_proc", None), 5000)

    # -- reconnection -------------------------------------------------------
    @staticmethod
    def _is_disconnect(e):
        """True if an error means the camera link dropped (battery swap, unplug,
        power cycle) rather than a benign one-off glitch.

        CameraError used to count on its own, which was far too broad: it is this
        app's own error type and covers 'widget absent', 'no file received' and
        'Bulb value absent' — none of which mean the USB link went away. Treating
        them as a drop tore down a perfectly healthy session and started hammering
        PTPCamera, which is what a disconnect looked like from the outside. Judge a
        CameraError by what it says, like any other."""
        code = getattr(e, "code", None)
        # -1 'unspecified' is deliberately absent: measured on this camera, it comes
        # back from a capture while the body is still answering normally.
        if code in (-7, -52, -53, -105, -108):       # IO / claim / not-found
            return True
        s = str(e).lower()
        needles = ("could not claim", "find the requested device", "no camera",
                   "not connected", "i/o", "input/output", "could not find")
        return any(n in s for n in needles)

    def _say(self, msg, key=None, min_s=4.0):
        """Status message that does not repeat itself. The preview loop used to emit
        the same failure twenty times a second, which made the status bar flicker."""
        key = key or msg.split(":")[0]
        last = getattr(self, "_say_last", None)
        if last is None:
            last = self._say_last = {}
        now = time.monotonic()
        prev_t, prev_msg = last.get(key, (0.0, None))
        if msg == prev_msg and now - prev_t < min_s:
            return
        last[key] = (now, msg)
        self.status.emit(msg)

    def _pump_stop(self):
        """While blocked reconnecting, honour a Stop/Quit; keep other commands."""
        kept = []
        while True:
            try:
                item = self._cmd.get_nowait()
            except queue.Empty:
                break
            if item[0] == "__stop__":
                self._running = False
            else:
                kept.append(item)
        for it in kept:
            self._cmd.put(it)

    #: Commands that poll the mount for seconds. Nothing here needs to finish
    #: before the next preview frame, so none of them belongs on the camera thread.
    _SLOW_MOUNT = ("mount_connect", "mount_track_start", "mount_track_stop",
                   "mount_set_track_rate", "mount_slew", "mount_slew_stop",
                   "mount_goto_ra", "set_dither")

    def _mount_cmd(self, kind, kw):
        """Entry point: hand the slow ones to the mount queue, run the rest here."""
        if kind in self._SLOW_MOUNT:
            q = getattr(self, "_mount_q", None)
            if q is None or not q.isRunning():
                q = self._mount_q = _MountQueue(self._mount_cmd_sync)
                q.start()
            q.put(kind, kw)
            return
        self._mount_cmd_sync(kind, kw)

    def _mount_cmd_sync(self, kind, kw):
        """Handle mount commands (connect/track/dither). Runs even with no camera."""
        if kind == "set_dither":
            self._dither_on = kw["on"]
            self._dither_every = int(kw.get("every", 4))
            self._dither_amp = float(kw.get("amp", 150.0))
            self._dither_settle = float(kw.get("settle", 4.0))
            self._shots_since_dither = 0; self._dither_pos = 0.0
            if kw["on"]:
                self._mount_connect(kw.get("ip", "192.168.4.1"), kw.get("two_axis", False))
            elif self._dither_mount is not None:
                self._dither_mount.close(); self._dither_mount = None
        elif kind == "mount_connect":
            self._mount_connect(kw.get("ip", "192.168.4.1"), kw.get("two_axis", False),
                                preserve=bool(kw.get("preserve")))
        elif kind == "mount_track_start":
            if self._dither_mount is None:
                self._mount_connect(kw.get("ip", "192.168.4.1"), kw.get("two_axis", False))
            if self._dither_mount is not None:
                try:
                    self._dither_mount.resume_tracking()
                    self.mount_status.emit("🟢 Mount: tracking (sidereal)")
                except Exception as e:      # noqa: BLE001
                    self.mount_status.emit("Mount: start-tracking failed: {}".format(e))
        elif kind == "mount_track_stop":
            if self._dither_mount is not None:
                try:
                    self._dither_mount.stop()
                    self.mount_status.emit("⏸ Mount: tracking stopped")
                except Exception as e:      # noqa: BLE001
                    self.mount_status.emit("Mount: stop failed: {}".format(e))
        elif kind == "mount_slew":
            if self._dither_mount is None:
                self._mount_connect(kw.get("ip", "192.168.4.1"), kw.get("two_axis", False))
            if self._dither_mount is not None:
                try:
                    self._dither_mount.slew(kw.get("dir", 1), kw.get("rate", 128))
                    self.mount_status.emit("⏩ Mount: slewing RA {}".format(
                        "+" if kw.get("dir", 1) >= 0 else "−"))
                except Exception as e:      # noqa: BLE001
                    self.mount_status.emit("Mount: slew failed: {}".format(e))
        elif kind == "mount_slew_stop":
            if self._dither_mount is not None:
                try:
                    self._dither_mount.stop()
                    self.mount_status.emit("⏹ Mount: slew stopped")
                except Exception as e:      # noqa: BLE001
                    self.mount_status.emit("Mount: stop failed: {}".format(e))
        elif kind == "mount_axis1_sync":
            self._start_mount_op(self._axis1_sync, kw)
        elif kind == "mount_axis1_goto":
            self._start_mount_op(self._axis1_goto, kw)
        elif kind == "mount_axis1_stop":
            self._axis1_abort = True
            # do not only raise the flag and hope the slew notices: halt the axis
            # here and now, from the thread that received the click
            m = self._dither_mount
            if m is not None:
                try:
                    m._stop_and_settle()
                    self.mount_status.emit("⏹ Axis 1 halted.")
                except Exception as e:      # noqa: BLE001
                    self.mount_status.emit("Stop failed: {}".format(e))
        elif kind == "mount_set_track_rate":
            self._set_track_rate(kw)
        elif kind == "mount_track_probe":
            # Off the camera thread, always. This is a UDP round trip to the mount,
            # and a lost packet here used to stall the live view for seconds. If a
            # probe is still in flight we skip this one rather than queue another:
            # a missed sample costs nothing, a backlog costs the preview.
            op = getattr(self, "_probe_op", None)
            if op is None:
                op = self._probe_op = _MountOp()
            elif op.isRunning():
                return                      # still polling: skip this one
            else:
                op.wait(2000)               # fully finished before re-arming
            op.set_job(self._track_probe, kw)
            op.start()
        elif kind == "mount_goto_ra":
            if self._dither_mount is None:
                self._mount_connect(kw.get("ip", "192.168.4.1"), kw.get("two_axis", False))
            if self._dither_mount is not None:
                try:
                    dec_as = kw.get("dec_arcsec")
                    if dec_as is not None and getattr(self._dither_mount, "two_axis", False):
                        try:
                            dmoved = self._dither_mount.slew_dec(float(dec_as))
                            self.mount_status.emit("↕ Slewed Dec {:+.0f}\"".format(dmoved))
                        except Exception as e:      # noqa: BLE001
                            self.mount_status.emit("Dec slew skipped: {}".format(e))
                    moved = self._dither_mount.dither_ra(float(kw["arcsec"]))
                    self.mount_status.emit("🎯 Slewed RA {:+.0f}\" toward goal — tracking".format(
                        moved))
                    self.mount_goto_done.emit(float(moved))
                except Exception as e:      # noqa: BLE001
                    self.mount_status.emit("Mount: goto failed: {}".format(e))
                    self.mount_goto_done.emit(0.0)

    def _track_probe(self, kw):
        """Measure what the axis is REALLY doing, from the step counter.

        One reading per call, the rate coming from the gap to the previous one —
        so nothing blocks, and the answer is a measurement rather than a repeat of
        what was commanded. A mount that silently stopped, or that is tracking at
        the wrong rate for the target, both show up here and nowhere else."""
        import time
        m = self._dither_mount
        if m is None or m.cpr is None:
            self.mount_track_state.emit(float("nan"), 0.0, 0.0)
            return
        try:
            now, counts = time.monotonic(), m.get_counts(retries=0)
        except Exception:               # noqa: BLE001  (Wi-Fi hiccup: skip this sample)
            self.mount_track_state.emit(float("nan"), 0.0, 0.0)
            # The 2i's Wi-Fi drops now and then, and a mount that stops answering is
            # usually still tracking — it is the link that died. After ~12 s of
            # silence, reopen it, keeping the step counter so an axis-1 sync survives.
            self._probe_fails = getattr(self, "_probe_fails", 0) + 1
            if self._probe_fails == 6 or (self._probe_fails > 6 and self._probe_fails % 15 == 0):
                try:
                    info = m.reconnect()
                    self._probe_fails = 0
                    self.mount_status.emit("🔄 Wi-Fi dropped — link re-opened ({})".format(info))
                except Exception as e:  # noqa: BLE001
                    self.mount_status.emit("⚠ Mount unreachable: {}".format(e))
            return
        self._probe_fails = 0
        prev = getattr(self, "_track_sample", None)
        self._track_sample = (now, counts)
        if prev is None or now - prev[0] < 1.5:
            return                      # too short a gap to mean anything yet
        rate = (counts - prev[1]) / (now - prev[0])
        try:
            commanded = m.cpr * m.tracking_omega() / 360.0
        except Exception:               # noqa: BLE001
            commanded = 0.0
        self.mount_track_state.emit(rate, commanded, float(m.cpr))

    def _set_track_rate(self, kw):
        """Apply the tracking rate a given target actually needs.

        Recomputed from the ephemeris each time rather than stored: the Moon's
        rate wanders by a few percent over a month, and a planet's flips either
        side of sidereal as it goes in and out of retrograde."""
        import manual_goto
        from datetime import datetime
        if self._dither_mount is None:
            self._mount_connect(kw.get("ip", "192.168.4.1"))
        m = self._dither_mount
        if m is None:
            self.mount_status.emit("Rate: mount not connected."); return
        body = (kw.get("body") or "sidereal").lower()
        lat, lon = float(kw.get("lat", 0.0)), float(kw.get("lon", 0.0))
        self._track_spec = (body, lat, lon)
        try:
            omega = manual_goto.body_track_rate(body, datetime.utcnow(), lat, lon)
            m.set_rate(omega)
            m.resume_tracking()
            self.mount_status.emit("🛰 " + manual_goto.rate_summary(
                body, datetime.utcnow(), lat, lon))
        except Exception as e:          # noqa: BLE001
            self.mount_status.emit("Rate change failed: {}".format(e))

    @staticmethod
    def _eta_str(seconds):
        """A slew of several minutes deserves a readable estimate."""
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return "{:.0f} s".format(seconds)
        return "{:.0f} min {:02.0f} s".format(seconds // 60, seconds % 60)

    def _start_mount_op(self, fn, kw):
        """Hand a long mount move to its own thread and return at once, so the
        preview keeps running and Stop stays answerable."""
        op = getattr(self, "_mount_op", None)
        if op is not None and op.isRunning():
            self.mount_status.emit("Mount is already busy — stop that move first.")
            return
        if op is None:
            op = self._mount_op = _MountOp()
        else:
            op.wait(5000)                   # never drop a thread that is still ending
        self._axis1_abort = False
        op.set_job(fn, kw)
        op.start()

    def _axis1_ref(self, mount):
        """Both axis conventions, measured once per connection rather than assumed."""
        if getattr(mount, "command_sign", None) is None:
            self.mount_status.emit("⏱ Learning the axis directions "
                                   "(timing the counter, then one test move)…")
            ms, cs = mount.measure_directions()
            self.mount_status.emit("Axis directions measured: encoder {:+.0f}, "
                                   "commands {:+.0f}".format(ms, cs))
        return mount.motor_sign

    def _axis1_sync(self, kw):
        """Tie the encoder to the sky: the camera is on this target now, so this
        encoder count means this hour angle. Every GoTo follows from that."""
        import manual_goto
        from datetime import datetime
        if self._dither_mount is None:
            self._mount_connect(kw.get("ip", "192.168.4.1"))
        m = self._dither_mount
        if m is None:
            self.mount_axis1_result.emit(False, "Mount not connected.", 0.0)
            return
        try:
            # Take the reference FIRST. This instant — camera centred on the target —
            # is the only one we can anchor to, and measuring the axis directions
            # involves a small test move that would otherwise invalidate it. The
            # encoder keeps counting through that move, so the anchor stays good.
            now = datetime.utcnow()
            g = manual_goto.ManualGoto(lat=float(kw.get("lat", 0.0)), lon=kw["lon"])
            ra_d, dec_d = g.resolve(kw.get("name", ""), kw["ra"], kw["dec"], now)
            counts0 = m.get_counts()
            ha = manual_goto.wrap180(manual_goto.lst_deg(now, kw["lon"]) - ra_d)
            # Which side of the axis the L-plate is on matters: on the mirrored side
            # the camera looks the opposite way round the pole, so the same encoder
            # count means an hour angle half a turn away. Getting this wrong puts
            # every later GoTo out by exactly 12 h.
            if float(kw.get("plate", 0.0)) < 0:
                ha = manual_goto.wrap180(ha + 180.0)
            self._axis1_ref(m)          # may move the axis; the anchor is already taken
            self._axis1_zero = (counts0, ha, m.motor_sign)
            _save_axis1_sync(self._axis1_zero)
            self.mount_axis1_result.emit(
                True, "Axis 1 synced on {} — encoder tied to hour angle {:+.3f}°{}. "
                      "GoTo is live.".format(
                          kw.get("name", "target"), ha,
                          " (L-plate on its negative side)"
                          if float(kw.get("plate", 0.0)) < 0 else ""), 0.0)
        except Exception as e:          # noqa: BLE001
            self.mount_axis1_result.emit(False, "Axis-1 sync failed: {}".format(e), 0.0)

    def _axis1_goto(self, kw):
        """Drive axis 1 onto the target, correcting until it is actually there.

        Re-solving the wanted hour angle on every pass is what makes this exact:
        the sky keeps turning while the mount slews with tracking paused, and the
        loop absorbs that instead of trying to predict it."""
        import manual_goto
        from datetime import datetime, timedelta
        if self._dither_mount is None:
            self._mount_connect(kw.get("ip", "192.168.4.1"))
        m = self._dither_mount
        if m is None:
            self.mount_axis1_result.emit(False, "Mount not connected.", 0.0)
            return
        ref = getattr(self, "_axis1_zero", None)
        if not ref:
            self.mount_axis1_result.emit(
                False, "Sync axis 1 on a known star first — without that the mount has "
                       "no idea where it is pointing.", 0.0)
            return
        counts0, ha0, msign = ref
        tol = float(kw.get("tol_deg", 0.15))
        limit = float(kw.get("max_deg", 180.0))
        plate = 0.0
        try:
            for it in range(int(kw.get("max_iter", 4))):
                if self._axis1_abort:
                    self.mount_axis1_result.emit(False, "GoTo stopped.", plate)
                    return
                now = datetime.utcnow()
                g = manual_goto.ManualGoto(lat=kw["lat"], lon=kw["lon"])
                have = ha0 + msign * m.counts_to_deg(m.get_counts() - counts0)
                rate = m.max_slew_deg_per_s(int(kw.get("rate", 64))) or 0.30
                # Aim where the target WILL BE on arrival, not where it is now.
                # This mount is capped at 0.30 deg/s, so a 90 deg swing takes five
                # minutes with tracking paused, during which the sky turns 1.25 deg.
                # Chasing that afterwards costs a whole extra pass; predicting it
                # lands first time. The estimate feeds back into itself, so iterate:
                # it converges fast, the correction being small next to the move.
                when, delta, plate, travel = now, 0.0, 0.0, 0.0
                for _ in range(4):
                    res = g.solve(kw.get("name", "target"), kw["ra"], kw["dec"], when)
                    # the mirrored position is a real mechanical choice, not a detail:
                    # it is the same sky with the camera on the other side of the pole
                    sol = res.alternate if kw.get("flip") else res.primary
                    plate = sol.axis2.reading
                    delta = manual_goto.wrap180(sol.axis1.sky_deg - have)
                    travel = m.MOVE_OVERHEAD_S + abs(delta) / max(rate, 0.01)
                    when = now + timedelta(seconds=travel)
                if abs(delta) > limit:
                    self.mount_axis1_result.emit(
                        False, "That would swing the axis {:.0f}° — refused. Re-sync, or "
                               "raise the limit deliberately.".format(abs(delta)), plate)
                    return
                # how far the axis would end up from where it was synced: this is
                # what runs the camera into a tripod leg, not the size of one move
                from_sync = abs(m.counts_to_deg(m.get_counts() - counts0)
                                + msign * delta)
                if from_sync > float(kw.get("max_from_sync", 120.0)):
                    self.mount_axis1_result.emit(
                        False, "That would leave the axis {:.0f}° from where you synced "
                               "it — refused, to keep the camera clear of the tripod. "
                               "Re-position and re-sync, or raise the travel limit."
                               .format(from_sync), plate)
                    return
                if abs(delta) <= tol:
                    self.mount_axis1_result.emit(
                        True, "Axis 1 on target — {:+.2f}° residual after {} pass(es). "
                              "Now set the L-plate to {:+.1f}°.".format(delta, it, plate),
                        plate)
                    return
                self.mount_status.emit(
                    "🔭 Axis 1: {:+.2f}° to go, about {} (pass {})…"
                    .format(delta, self._eta_str(travel), it + 1))

                last_text = [0.0]

                def _prog(frac, left):
                    import time as _t
                    secs = left / max(rate, 0.01)
                    self.mount_axis1_progress.emit(frac, left, secs)
                    if _t.monotonic() - last_text[0] > 5.0:   # the status bar, sparingly
                        last_text[0] = _t.monotonic()
                        self.mount_status.emit(
                            "🔭 Axis 1: {:.0f}% — {:.2f}° left, "
                            "about {} remaining".format(frac * 100.0, left,
                                                        self._eta_str(secs)))

                # delta is in hour-angle degrees; command_sign maps that onto the
                # direction bit this particular mount actually honours
                m.goto_delta_deg(delta * m.command_sign,
                                 rate_mult=int(kw.get("rate", 64)),
                                 should_abort=lambda: self._axis1_abort,
                                 progress_cb=_prog)
            self.mount_axis1_result.emit(
                False, "Axis 1 did not settle in {} passes — check the encoder direction "
                       "and the slew rate.".format(kw.get("max_iter", 4)), plate)
        except Exception as e:          # noqa: BLE001
            try:
                m.ensure_tracking()
            except Exception:           # noqa: BLE001
                pass
            self.mount_axis1_result.emit(False, "Axis-1 GoTo failed: {}".format(e), plate)

    def _mount_connect(self, ip, two_axis=False, preserve=False):
        # a fresh connect re-zeroes the board's step counter, so any axis-1 sync
        # taken against the old origin is now meaningless — unless the mount kept
        # running and we deliberately keep its count.
        if not preserve:
            self._axis1_zero = None
        self.mount_status.emit("🔌 Connecting to the mount at {}…".format(ip))
        try:
            self._dither_mount = SkyWatcherMount(ip, logfn=lambda m: self.status.emit(m),
                                                 two_axis=two_axis)
            info = self._dither_mount.connect(preserve_position=preserve)
            if preserve:
                saved = _load_axis1_sync()
                if saved is not None:
                    self._axis1_zero = saved
                    self.mount_status.emit(
                        "🔌 Mount connected ({}) — axis-1 sync restored from the last "
                        "session; check one target before trusting a GoTo.".format(info))
                    return
            self.mount_status.emit("🔌 Mount connected ({})".format(info))
        except Exception as e:              # noqa: BLE001
            self._dither_mount = None
            self.mount_status.emit("❌ Mount connect failed: {}".format(e))

    def _drain_mount_only(self):
        """Process only mount commands (keep the rest queued) — used while waiting for the
        camera so mount connect/track/dither still work with no camera attached."""
        keep = []
        while True:
            try:
                kind, kw = self._cmd.get_nowait()
            except queue.Empty:
                break
            if kind in ("set_dither", "mount_connect", "mount_track_start", "mount_track_stop",
                        "mount_slew", "mount_slew_stop", "mount_goto_ra",
                        "mount_axis1_sync", "mount_axis1_goto", "mount_axis1_stop",
                        "mount_set_track_rate", "mount_track_probe"):
                self._mount_cmd(kind, kw)
            elif kind == "__stop__":
                self._running = False
            else:
                keep.append((kind, kw))
        for item in keep:
            self._cmd.put(item)

    def _reconnect(self, old):
        """Camera dropped: keep the whole session, wait for it to come back, reopen."""
        try:
            if old is not None:
                old.close()
        except Exception:             # noqa: BLE001
            pass
        self.status.emit("⚠ Camera disconnected — reconnecting… "
                         "(swap the battery / check USB; your session is kept)")
        self.cam_status.emit("⚠ reconnecting…")
        is_sim = getattr(self._make_backend, "__name__", "") == "SimBackend"
        delay_ms = 600
        waited = said = 0.0
        while self._running:
            self._pump_stop()
            self._drain_mount_only()          # mount connect/track/dither work w/o camera
            if not self._running:
                return None
            if is_sim or GPhoto2Backend.camera_present():
                try:
                    b = self._make_backend()
                    self.status.emit("✓ Camera reconnected: {} — session continues."
                                     .format(b.model_name()))
                    self.cam_status.emit("")
                    self._emit_config(b)
                    return b
                except Exception:     # noqa: BLE001 - not ready yet, keep waiting
                    pass
            for _ in range(0, delay_ms, 100):      # responsive to Stop/Quit
                if not self._running:
                    return None
                self.msleep(100)
            delay_ms = min(int(delay_ms * 1.4), 4000)
            waited += delay_ms / 1000.0
            if waited - said > 20.0:           # a word every 20 s, not on every try
                said = waited
                self._say("⏳ Still waiting for the camera ({:.0f} s) — USB set to PC "
                          "Remote, body awake?".format(waited), key="wait", min_s=0.0)
        return None

    # -- internals ----------------------------------------------------------
    def _drain_commands(self, backend):
        while True:
            try:
                kind, kw = self._cmd.get_nowait()
            except queue.Empty:
                return
            try:
                if kind == "__stop__":
                    self._running = False
                elif kind == "set_metric":
                    self.metric = kw["name"]
                elif kind == "set_roi":
                    self.roi_frac = kw["frac"]
                elif kind == "set_live":
                    self._live_enabled = kw["on"]
                elif kind == "set_metric_enabled":
                    self._metric_enabled = kw["on"]
                elif kind == "set_stars":
                    self._stars_enabled = kw["on"]
                elif kind == "set_cull":
                    self._cull_enabled = kw["on"]
                elif kind == "set_stack":
                    self._stack_enabled = kw["on"]
                    if not kw["on"]:
                        self._stack_reset_req = True
                elif kind == "set_stack_full":
                    self._stack_full = bool(kw["on"])
                    self._stack_reset_req = True   # accumulator size changes: start over
                    self.status.emit(
                        "Live stack: {} resolution — restarting the stack.".format(
                            "FULL 24 MP" if self._stack_full else "half"))
                elif kind == "set_stack_reject":
                    self._stack_reject = kw["on"]
                elif kind == "set_dawn_stop":
                    self._dawn_stop = kw["on"]
                elif kind == "set_autostop":
                    # Two panels feed this: the dawn/time/altitude rules, and the
                    # battery/disk guard. They used to be two 'set_autostop' branches in
                    # this same chain, so the second was dead code and the guard never
                    # reached the worker. Update only the keys that were sent.
                    if "sun_alt" in kw or "fixed_min" in kw or "target_alt" in kw:
                        self._stop_sun_alt = kw.get("sun_alt")
                        self._stop_fixed_min = kw.get("fixed_min")
                        self._stop_target_alt = kw.get("target_alt")
                        self._target_ra = kw.get("ra"); self._target_dec = kw.get("dec")
                    if "on" in kw:
                        self._autostop = bool(kw["on"])
                        self._batt_thr = kw.get("batt", self._batt_thr)
                        self._disk_thr = kw.get("disk", self._disk_thr)
                elif kind == "set_site":
                    self._site_lat = kw["lat"]; self._site_lon = kw["lon"]
                elif kind == "set_raw_full":
                    self._raw_full = kw["on"]
                elif kind == "set_flats":
                    self._apply_flats = kw["on"]; self._flats_dir = kw.get("flats_dir")
                    self._master_flat = None; self._flat_fnum = None   # reload on next frame
                elif kind == "set_darks":
                    self._apply_darks = kw["on"]; self._flats_dir = kw.get("flats_dir")
                    self._master_dark = None; self._dark_key = None    # reload on next frame
                elif kind in ("set_dither", "mount_connect", "mount_track_start",
                              "mount_track_stop", "mount_slew", "mount_slew_stop",
                              "mount_goto_ra", "mount_axis1_sync", "mount_axis1_goto",
                              "mount_axis1_stop", "mount_set_track_rate",
                              "mount_track_probe"):
                    self._mount_cmd(kind, kw)
                elif kind == "reset_stack":
                    self._stack_reset_req = True          # the processor does it
                    self.status.emit("Stacking reset.")
                elif kind == "set_kappa":
                    self._kappa_enabled = kw["on"]
                    self._kappa = kw.get("kappa", self._kappa)
                elif kind == "set_config":
                    backend.set_config(kw["name"], kw["value"])
                    self._emit_config(backend)
                elif kind == "focus":
                    backend.focus_step(kw["amount"])
                elif kind == "record_movie":
                    # switch the mount to SOLAR rate for solar imaging, then record the movie
                    if self._dither_mount is not None:
                        try:
                            self._dither_mount.set_rate("solar")
                            self._dither_mount.resume_tracking()
                            self.mount_status.emit("☀ Mount: solar tracking rate")
                        except Exception as e:      # noqa: BLE001
                            self.status.emit("Solar rate failed: {}".format(e))
                    try:
                        self.status.emit("🎥 Recording movie ({:.0f}s)…".format(kw["seconds"]))
                        dest = os.path.join(kw.get("save_dir") or "/tmp",
                                            "nout_movie_{}".format(int(time.time())))
                        path = backend.record_movie(kw["seconds"], dest)
                        self.status.emit("🎥 Movie saved: {}".format(path))
                        self.movie_ready.emit(path)
                    except Exception as e:          # noqa: BLE001
                        self.status.emit("Movie recording failed: {}".format(e))
                elif kind == "set_exposure_mode":
                    self._bulb = kw["bulb"]
                    self._bulb_seconds = kw.get("seconds", self._bulb_seconds)
                elif kind == "capture":
                    self._save_dir = kw.get("save_dir", self._save_dir)
                    self._do_capture(backend)
                elif kind == "start_interval":
                    self._save_dir = kw.get("save_dir", self._save_dir)
                    self._interval = kw["interval"]
                    self._remaining = kw["count"]      # 0 = unlimited
                    self._bulb = kw.get("bulb", self._bulb)
                    self._bulb_seconds = kw.get("seconds", self._bulb_seconds)
                    self._abort_exposure = False
                    self._total = self._remaining
                    self._done = 0
                    self._frame_times.clear()
                    self._star_hist.clear()
                    self._reset_stack()              # new stack per series
                    self._init_session_log()      # session CSV log
                    if self._bulb:
                        try:
                            try:                       # so we can hand it back later
                                cur = backend.current_shutter()
                                if cur and "bulb" not in cur.lower():
                                    self._shutter_before_bulb = cur
                            except Exception:          # noqa: BLE001
                                pass
                            backend.set_shutter_to_bulb()
                            self.status.emit("Bulb mode: speed set to Bulb.")
                        except CameraError as e:
                            self.status.emit(str(e))
                            self._next_shot = None
                            continue
                    else:
                        try:    # never shoot 'speed' while the body is still on Bulb.
                            # Prefer the speed showing in the UI: that is what the user
                            # asked for. Falling back to a guess once meant a burst
                            # firing multi-second exposures nobody asked for.
                            want = kw.get("shutter") or getattr(
                                self, "_shutter_before_bulb", None)
                            if want and "bulb" in str(want).lower():
                                want = None
                            changed, msg = backend.ensure_not_bulb(want)
                            if changed:
                                self.status.emit("⏱ " + msg)
                            # ensure_not_bulb only acts when the body reads Bulb. If the
                            # body is on some other speed (left over from a test shot or
                            # a previous series), the series used to fire THAT speed —
                            # measured: 1/200 frames in a series the UI said was 30 s.
                            cur = backend.current_shutter()
                            if want and cur is not None and str(cur) != str(want):
                                backend.set_config("shutterspeed", want)
                                self.status.emit("⏱ Shutter set to {} for this series "
                                                 "(body was on {}).".format(want, cur))
                        except Exception as e:         # noqa: BLE001
                            self.status.emit("Could not check the shutter speed: {}".format(e))
                    self._next_shot = time.monotonic()  # first shot immediate
                    self._paused_at = None
                    self.interval_paused.emit(False)
                    self.interval_running.emit(True)    # -> switch to review mode
                    # new sequence (burst)
                    self._seq_id += 1
                    self._seq_count = 0
                    self._seq_integ = 0.0
                    self._seq_start = time.monotonic()
                    seq_type = os.path.basename(self._save_dir) or "lights"
                    self._seq_type = seq_type
                    # flats / darks / bias go straight to disk: no preview, no stars,
                    # no stacking — that work only makes sense on light frames.
                    self._calibration = seq_type.lower() in ("flats", "darks", "bias",
                                                             "dark", "flat", "offset")
                    if self._calibration:
                        self.status.emit(
                            "📁 {}: frames are saved only — no preview, no stacking, so "
                            "the series runs at the camera's own pace.".format(seq_type))
                    seq_mode = ("bulb {:.0f}s".format(self._bulb_seconds)
                                if self._bulb else "camera speed")
                    self.seq_started.emit(self._seq_id, seq_type, seq_mode)
                    # initial time/shot estimation for ETA
                    est = (self._bulb_seconds if self._bulb else 3.0) + self._interval
                    self.interval_progress.emit(0, self._total, est)
                    self.status.emit("Intervalometer started{}.".format(
                        "" if self._total else " (unlimited)"))
                elif kind == "pause_interval":
                    # Hold the series where it is: same sequence number, same count,
                    # same stack. Starting a new series instead would open a second
                    # burst and throw the integration away.
                    if self._next_shot is not None:
                        self._paused_at = (self._next_shot, self._remaining)
                        self._next_shot = None
                        self.interval_paused.emit(True)
                        self.status.emit(
                            "⏸ Series paused after the current frame — {} left of {}. "
                            "Resume when you are ready.".format(self._remaining, self._total)
                            if self._total else "⏸ Series paused.")
                elif kind == "resume_interval":
                    if getattr(self, "_paused_at", None) is not None:
                        self._next_shot = time.monotonic()
                        self._paused_at = None
                        self.interval_paused.emit(False)
                        self.status.emit("▶ Series resumed — same sequence, same stack.")
                elif kind == "stop_interval":
                    was_running = (self._next_shot is not None
                                   or getattr(self, "_paused_at", None) is not None)
                    self._paused_at = None
                    self.interval_paused.emit(False)
                    self._next_shot = None
                    self._abort_exposure = True        # interrupts ongoing bulb exposure
                    if was_running:
                        self._end_sequence()
                        self.interval_running.emit(False)
                    self.status.emit("Intervalometer stopped.")
                elif kind == "set_count":
                    # change the target number of shots on the fly (finite series only)
                    if self._total > 0 and self._next_shot is not None:
                        taken = self._total - self._remaining
                        new_total = max(int(kw["count"]), taken)
                        self._total = new_total
                        self._remaining = new_total - taken
                        self.interval_progress.emit(taken, self._total, 0.0)
                        self.status.emit("Target updated: {} shots ({} done, {} to go)."
                                         .format(new_total, taken, self._remaining))
                        if self._remaining <= 0:       # already reached -> wrap up
                            self._next_shot = None
                            self._end_sequence()
                            self.interval_running.emit(False)
            except CameraError as e:
                self.status.emit(str(e))
            except Exception as e:           # noqa: BLE001
                self.status.emit("Cmd {}: {}".format(kind, e))

    def _init_session_log(self):
        """Creates the session CSV log in the capture folder."""
        try:
            os.makedirs(self._save_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = os.path.join(self._save_dir, "session_{}.csv".format(ts))
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["n", "file", "datetime", "mode", "exposure_s", "stars", "hfr", "state"])
        except Exception:               # noqa: BLE001
            self._log_path = None

    def _log_row(self, n, fname, mode, pose_s, stars, hfr, state):
        if not self._log_path:
            return
        try:
            with open(self._log_path, "a", newline="") as f:
                csv.writer(f).writerow([n, fname, datetime.now().isoformat(timespec="seconds"),
                                        mode, pose_s, stars, hfr, state])
        except Exception:               # noqa: BLE001
            pass

    def _reset_stack(self):
        self._stack_ref_gray = None
        self._stack_mean = None
        self._stack_M2 = None
        self._stack_count = 0
        self._stack_ref_stars = None
        self._stack_starhist = []
        self._stack_starwin = deque(maxlen=12)
        self._stack_consec_rej = 0
        self._stack_rejected = 0
        self._stack_ref_xy = None          # reference star centroids (for registration)
        self._stack_ref_med = None         # reference per-channel background level
        self._stack_ref_mad = None         # reference per-channel noise scale
        self._stack_wpx = None             # per-pixel accumulated weight
        self._stack_crop = None            # region covered by every frame (ref coords)
        self._stack_qhist = deque(maxlen=12)   # (stars, HFR) of recent frames -> weights

    def _is_dawn(self):
        """True once the Sun rises above -18° (end of astronomical night)."""
        try:
            from datetime import datetime
            return _sun_altitude_deg(self._site_lat, self._site_lon,
                                     datetime.utcnow()) > -18.0
        except Exception:             # noqa: BLE001
            return False

    def _should_autostop(self):
        """Return a short reason string if any active auto-stop condition is met, else ''.
        Conditions: Sun above a chosen altitude, a fixed local time, or the target dropping
        below a chosen altitude. `_dawn_stop` keeps the legacy -18° behaviour."""
        from datetime import datetime
        try:
            if self._dawn_stop and _sun_altitude_deg(
                    self._site_lat, self._site_lon, datetime.utcnow()) > -18.0:
                return "astronomical dawn"
            if self._stop_sun_alt is not None and _sun_altitude_deg(
                    self._site_lat, self._site_lon, datetime.utcnow()) > self._stop_sun_alt:
                return "Sun above {:.0f}\u00b0".format(self._stop_sun_alt)
            if self._stop_fixed_min is not None:
                now = datetime.now()
                cur = now.hour * 60 + now.minute
                if self._stop_fixed_min <= cur < self._stop_fixed_min + 180:
                    return "{:02d}:{:02d}".format(self._stop_fixed_min // 60,
                                                  self._stop_fixed_min % 60)
            if (self._stop_target_alt is not None and self._target_ra is not None):
                alt = _object_altitude_deg(self._site_lat, self._site_lon,
                                           self._target_ra, self._target_dec, datetime.utcnow())
                if alt < self._stop_target_alt:
                    return "target below {:.0f}\u00b0".format(self._stop_target_alt)
        except Exception:             # noqa: BLE001
            return ""
        return ""

    def _accumulate_stack(self, frame):
        """Stack into a per-pixel weighted running mean (float32, linear light).

        Each frame: hot pixels removed (no darks needed), stars found on a stretched
        copy so faint ones count, registered on the reference with Lanczos (keeps
        stars fine), and added with a weight from its quality — sharper, richer frames
        count more. Pixels a frame does not cover get no weight from it, so drift and
        dither leave no mirrored edges; the shown stack is cropped to the part every
        frame covers. Optional kappa-sigma rejection against the running spread."""
        x0 = frame.astype(np.float32)
        if x0.ndim == 2:
            x0 = cv2.cvtColor(x0, cv2.COLOR_GRAY2BGR)
        x0 = cosmetic_correct(x0)
        h, w = x0.shape[:2]
        gray = detection_gray(x0)
        try:
            cur_xy, nst = detect_stars(gray)
        except Exception:               # noqa: BLE001
            cur_xy, nst = [], None
        # whole-frame quality gate: drop frames with far fewer stars than RECENT frames.
        # The reference is a rolling median (adapts to the target lowering / transparency
        # changing over a long night) — not a fixed value, which used to freeze the stack.
        if self._stack_reject and nst is not None:
            ref = float(np.median(self._stack_starwin)) if self._stack_starwin else None
            if (ref and ref > 6 and nst < 0.5 * ref and self._stack_count >= 1
                    and self._stack_consec_rej < 6):        # anti-freeze: cap consecutive
                self._stack_rejected += 1
                self._stack_consec_rej += 1
                self.status.emit("⛔ Frame rejected ({} stars vs ~{:.0f}) — "
                                 "kept {}, dropped {}".format(
                                     nst, ref, self._stack_count, self._stack_rejected))
                if self._stack_mean is not None:
                    self.stack_ready.emit(self._stack_view(), self._stack_count,
                                          self._seq_integ)
                return
            if self._stack_consec_rej >= 6:
                self._stack_starwin.clear()   # conditions changed for good: re-baseline
                self.status.emit("↺ Re-baselining stack reference (conditions changed).")
            self._stack_consec_rej = 0
            self._stack_starwin.append(nst)
        # frame quality -> weight (1 for the first frames, then relative to recent ones)
        try:
            hfr = float(_median_hfr(gray, cur_xy)) if nst else 0.0
        except Exception:               # noqa: BLE001
            hfr = 0.0
        wq = 1.0
        if nst and hfr > 0 and len(self._stack_qhist) >= 3:
            rn = float(np.median([q[0] for q in self._stack_qhist]))
            rh = float(np.median([q[1] for q in self._stack_qhist]))
            wq = float(np.clip((nst / max(rn, 1.0)) * (rh / hfr) ** 2, 0.3, 1.5))
        if nst and hfr > 0:
            self._stack_qhist.append((nst, hfr))
        if self._stack_ref_gray is None:
            self._stack_ref_gray = gray
            self._stack_ref_xy = cur_xy
            self._stack_ref_med = [float(np.median(x0[::2, ::2, c])) for c in range(3)]
            self._stack_ref_mad = [float(np.median(np.abs(x0[::2, ::2, c] - self._stack_ref_med[c])))
                                   + 1e-6 for c in range(3)]
            M = np.float32([[1, 0, 0], [0, 1, 0]])
        else:
            try:
                M = register_affine(self._stack_ref_gray, gray, self._stack_ref_xy, cur_xy)
            except Exception:           # noqa: BLE001
                M = np.float32([[1, 0, 0], [0, 1, 0]])
        x = cv2.warpAffine(x0, M, (w, h), flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))   # drop Lanczos edge ringing
        x = np.clip(x, 0.0, None)
        if self._stack_count > 0:
            try:
                x = normalize_to_ref(x, self._stack_ref_med, self._stack_ref_mad)
            except Exception:           # noqa: BLE001
                pass
        wv = valid * wq
        if self._stack_mean is None or self._stack_mean.shape != x.shape:
            self._stack_mean = x * valid[..., None]
            self._stack_M2 = np.zeros_like(x)
            self._stack_wpx = wv.copy()
            self._stack_count = 1
        else:
            wpx = self._stack_wpx
            if self._kappa_enabled and self._stack_count >= 4:
                sigma = np.sqrt(self._stack_M2 / np.maximum(wpx, 1e-6)[..., None]) + 1e-3
                out = (np.abs(x - self._stack_mean) > (self._kappa * sigma)) & \
                    (wpx > 0)[..., None]
                x = np.where(out, self._stack_mean, x)          # outlier -> current mean
            wnew = wpx + wv
            coef = np.where(wnew > 0, wv / np.maximum(wnew, 1e-6), 0.0)[..., None]
            delta = x - self._stack_mean
            self._stack_mean += coef * delta
            self._stack_M2 += wv[..., None] * delta * (x - self._stack_mean)
            self._stack_wpx = wnew
            self._stack_count += 1
        # the part of the sky every accepted frame covers (only ever shrinks)
        pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        try:
            inv = cv2.invertAffineTransform(M)                  # M maps frame -> reference
            q = pts @ M[:, :2].T + M[:, 2]
        except Exception:               # noqa: BLE001
            q = pts
        rect = (max(q[0, 0], q[2, 0]), max(q[0, 1], q[1, 1]),
                min(q[1, 0], q[3, 0]), min(q[2, 1], q[3, 1]))
        c = self._stack_crop
        self._stack_crop = rect if c is None else (max(c[0], rect[0]), max(c[1], rect[1]),
                                                   min(c[2], rect[2]), min(c[3], rect[3]))
        self.stack_ready.emit(self._stack_view(), self._stack_count, self._seq_integ)

    def _stack_view(self):
        """The stack cropped to the region every frame covered (a copy, safe to hand
        to the GUI thread). Falls back to the whole frame if the crop looks wrong."""
        m = self._stack_mean
        if m is None:
            return None
        h, w = m.shape[:2]
        c = self._stack_crop
        if c is not None:
            x0, y0 = int(np.ceil(c[0])) + 3, int(np.ceil(c[1])) + 3
            x1, y1 = int(np.floor(c[2])) - 3, int(np.floor(c[3])) - 3
            if x1 - x0 > 0.6 * w and y1 - y0 > 0.6 * h:
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
                self._stack_view_rect = (x0, y0, x1, y1, w, h)
                return m[y0:y1, x0:x1].astype(np.float32)
        self._stack_view_rect = (0, 0, w, h, w, h)
        return m.astype(np.float32)

    def _emit_cam_status(self, backend):
        try:
            info = backend.status_info(self._save_dir)
            free = info.get("free_gb")
            batt = info.get("battery", "?")
            self.cam_status.emit("Battery {}  ·  {} free".format(
                batt, "{:.1f} GB".format(free) if free is not None else "?"))
            # auto stop on low battery/disk (during a series)
            if self._autostop and self._next_shot is not None:
                low = []
                m = re.search(r"(\d+)", str(batt))
                if m and int(m.group(1)) <= self._batt_thr:
                    low.append("battery {}%".format(m.group(1)))
                if free is not None and free <= self._disk_thr:
                    low.append("disk {:.1f} GB".format(free))
                if low:
                    self._next_shot = None
                    self._end_sequence()
                    self.interval_running.emit(False)
                    self.status.emit("⚠ Auto stop: " + ", ".join(low))
        except Exception:               # noqa: BLE001
            pass

    def _end_sequence(self):
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.isRunning() and not proc.idle():
            self.status.emit("Finishing the last frames…")
            proc.wait_idle(120.0)
        """Finalizes the ongoing sequence (burst) and emits its summary + result image."""
        if self._seq_start is None:
            return
        dur = time.monotonic() - self._seq_start
        result = None
        if self._stack_enabled and self._stack_mean is not None and self._stack_count > 0:
            # emit the raw float (linear) mean; the UI flattens + stretches it once,
            # at full precision, and stores it as an already-processed result.
            result = self._stack_view()
        elif self._last_preview is not None:
            result = self._last_preview
        cap = "Burst {} · {} · {} shots · {}".format(
            self._seq_id, self._seq_type, self._seq_count, _fmt_dur(self._seq_integ))
        self.seq_ended.emit(self._seq_id, self._seq_count, self._seq_integ, dur)
        self.seq_result.emit(self._seq_id, result, cap)
        self._seq_start = None

    def _maybe_dither(self):
        """Between exposures, keep the mount healthy: re-assert sidereal tracking every shot
        (self-heals a silent stop / Wi-Fi drop), and every N shots add a small random RA
        nudge to break walking noise. Never leaves the mount stopped."""
        if not self._dither_on or self._dither_mount is None:
            return
        self._shots_since_dither += 1
        do_dither = self._shots_since_dither >= self._dither_every
        try:
            if do_dither:
                self._shots_since_dither = 0
                import random
                target = random.uniform(-self._dither_amp, self._dither_amp)
                move = target - self._dither_pos      # relative move within the bounded box
                self._dither_pos = target
                moved = self._dither_mount.dither_ra(move)
                self.status.emit("🎲 Dither: RA {:+.0f}\" · settling {:.0f}s…".format(
                    moved, self._dither_settle))
                self.dither_point.emit(float(self._dither_pos), int(self._done))
                time.sleep(self._dither_settle)
            else:
                self._dither_mount.ensure_tracking()  # keepalive: guarantee it's still tracking
        except Exception as e:          # noqa: BLE001
            self.status.emit("Mount hiccup — recovering tracking… ({})".format(e))
            try:
                self._dither_mount.reconnect()
                self._dither_mount.resume_tracking()
                self.mount_status.emit("🔄 Mount reconnected — tracking resumed")
            except Exception as e2:     # noqa: BLE001
                self.mount_status.emit("⚠ Mount recovery failed: {}".format(e2))

    def _do_capture(self, backend):
        """Take one frame and hand it over. Nothing heavy happens here: decoding,
        measuring and stacking run on their own thread, so the next exposure starts
        as soon as the file is down instead of waiting for the picture to be
        processed — which used to add seconds to every cycle."""
        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        dest = os.path.join(self._save_dir, "shot_{}.arw".format(ts))
        t0 = time.monotonic()
        try:
            if self._bulb:
                self.status.emit("Bulb exposure {:.0f}s in progress…".format(self._bulb_seconds))
                saved = backend.capture_bulb(self._bulb_seconds, dest,
                                             should_abort=lambda: self._abort_exposure)
            else:
                saved = backend.capture_to_file(dest)
            self._done += 1
            self._frame_times.append(time.monotonic() - t0 + self._interval)
            exp_s = self._bulb_seconds if self._bulb else (backend.current_exposure_s() or 0.0)
            self.capture_saved.emit(saved)
            self.status.emit("Saved: {}".format(os.path.basename(saved)))
            self._emit_cam_status(backend)
            avg = (sum(self._frame_times) / len(self._frame_times)
                   if self._frame_times else 0.0)
            self.interval_progress.emit(self._done, self._total, avg)
            self._post_frame(saved, self._done, exp_s)
        except Exception as e:               # noqa: BLE001
            if self._is_disconnect(e):
                raise                         # let run() reconnect and retry the shot
            self.status.emit("Capture failed: {}".format(e))

    # -- processing, on its own thread -----------------------------------------
    def _post_frame(self, saved, idx, exp_s):
        """Queue a saved frame for processing (or skip it when nothing is needed)."""
        if self._calibration:
            # Flats, darks and bias are never displayed or stacked: they are only
            # ever averaged later, by the master builder. Measuring stars on a dark
            # frame is meaningless, and it cost seconds per frame.
            self._log_row(idx, os.path.basename(saved),
                          "bulb{:.0f}s".format(self._bulb_seconds) if self._bulb else "speed",
                          self._bulb_seconds if self._bulb else "", -1, 0.0, "calibration")
            self.capture_image_ready.emit(None, "{}  ·  #{}".format(
                os.path.basename(saved), idx))
            return
        if self._proc is None or not self._proc.isRunning():
            self._proc = _FrameProcessor(self)
            self._proc.start()
        light = self._proc.backlog() >= 3     # falling behind: keep the frame, skip the stack
        self._proc.submit((saved, idx, exp_s, light))

    def _process_frame(self, saved, idx, exp_s, light=False):
        """Decode, measure, stack and display one saved frame. Runs on the processing
        thread; only this thread touches the stack accumulators."""
        try:
            if self._stack_reset_req:
                self._stack_reset_req = False
                self._reset_stack()
            preview = _decode_preview_image(saved, full_demosaic=self._raw_full)
            self._last_preview = preview
            scount, hfr, ecc = -1, 0.0, 0.0
            pgray = None
            if preview is not None:
                try:
                    pgray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                    pstars, scount = detect_stars(pgray)
                    hfr = _median_hfr(pgray, pstars)
                    ecc = _median_eccentricity(pgray, pstars)
                except Exception:           # noqa: BLE001
                    scount, hfr, ecc = -1, 0.0, 0.0
            rejected = False
            if self._cull_enabled and scount >= 0 and len(self._star_hist) >= 4:
                med = float(np.median(self._star_hist))
                if med > 0 and scount < 0.5 * med:     # sharp drop = cloud / trail / plane
                    rejected = True
            if scount >= 0 and not rejected:
                self._star_hist.append(scount)
            if rejected:                    # move rejected exposure to rejected/
                try:
                    rej = os.path.join(self._save_dir, "rejected")
                    os.makedirs(rej, exist_ok=True)
                    newp = os.path.join(rej, os.path.basename(saved))
                    os.replace(saved, newp); saved = newp
                except Exception:           # noqa: BLE001
                    pass
            if not rejected and scount >= 0:
                self.track_point.emit(idx, hfr, scount, ecc)
            if not rejected:
                self._seq_count += 1
                self._seq_integ += exp_s
                self.seq_progress.emit(self._seq_id, self._seq_count, self._seq_integ)

            # visual live stacking — integrate LINEAR light from the RAW (like Siril);
            # fall back to the 8-bit preview if the RAW/rawpy isn't available.
            if (self._stack_enabled and not light and preview is not None
                    and pgray is not None and not rejected):
                stack_src = (_decode_linear(saved, half=not self._stack_full)
                             if self._stack_linear else None)
                if stack_src is None:
                    stack_src = preview.astype(np.float32)
                if self._apply_darks:                       # subtract dark/bias (camera signal)
                    if self._master_dark is None and self._dark_key is None:
                        self._dark_key = (_read_iso_exif(saved), _read_exposure_exif(saved))
                        self._master_dark = load_master_dark_for(
                            self._flats_dir, self._dark_key[0], self._dark_key[1])
                    if self._master_dark is not None:
                        stack_src = apply_dark(stack_src, self._master_dark)
                if self._apply_flats:                       # flat-field (vignetting)
                    if self._master_flat is None and self._flat_fnum is None:
                        self._flat_fnum = _read_fnumber_exif(saved) or 0.0
                        self._master_flat = load_master_flat_for(
                            self._flats_dir, _read_lens_exif(saved),
                            _read_focal_exif(saved), self._flat_fnum)
                    if self._master_flat is not None:
                        stack_src = apply_flat(stack_src, self._master_flat)
                self._accumulate_stack(stack_src)
            elif light and self._stack_enabled and not rejected:
                self.status.emit("⏩ Processing is behind — frame {} saved but not "
                                 "stacked.".format(idx))

            mode = "bulb{:.0f}s".format(self._bulb_seconds) if self._bulb else "speed"
            self._log_row(idx, os.path.basename(saved), mode,
                          self._bulb_seconds if self._bulb else "", scount,
                          round(hfr, 2), "rejected" if rejected else "ok")
            star_txt = "  ·  ★{}".format(scount) if scount >= 0 else ""
            rej_txt = "  ·  ⚠ discarded" if rejected else ""
            cap = "{}  ·  {}{}{}".format(
                os.path.basename(saved),
                "{}/{}".format(idx, self._total) if self._total else "#{}".format(idx),
                star_txt, rej_txt)
            self.capture_image_ready.emit(preview, cap)
        except Exception as e:               # noqa: BLE001
            self.status.emit("Processing failed on {}: {}".format(
                os.path.basename(saved), e))

    def _emit_config(self, backend):
        out = {}
        for name in ("iso", "shutterspeed", "f-number"):
            try:
                out[name] = backend.list_config(name)
            except Exception:                # noqa: BLE001
                out[name] = ([], "")
        out["__focus__"] = backend.focus_supported()
        self.config_loaded.emit(out)


# ---------------------------------------------------------------------------
#  HDMI live view (capture card) + SER recording
# ---------------------------------------------------------------------------
# The A7II's USB live view is fixed at 1024×680 JPEG. Its micro-HDMI output is a
# clean 1080p stream at 50/60 fps; an HDMI→USB capture card turns it into an
# ordinary UVC webcam that macOS and OpenCV read with no driver.

def _ser_ticks(dt):
    """SER timestamps: 100 ns ticks since 0001-01-01 00:00."""
    d = dt - datetime(1, 1, 1)
    return d.days * 864000000000 + d.seconds * 10_000_000 + d.microseconds * 10


class SerWriter:
    """Uncompressed SER movie (the planetary-imaging standard read by AutoStakkert!,
    Siril, PIPP, RegiStax). Unlike the mp4v NOUT used to write, nothing is thrown
    away between the sensor and the stacker: lucky imaging lives on exactly the
    fine detail a video codec removes first."""
    HEADER = 178
    MONO, RGB, BGR = 0, 100, 101

    def __init__(self, path, width, height, mono):
        import struct
        self._struct = struct
        self.path = path
        self.w, self.h = int(width), int(height)
        self.color = self.MONO if mono else self.BGR
        self.n = 0
        self._stamps = []
        self._t_local = _ser_ticks(datetime.now())
        self._t_utc = _ser_ticks(datetime.utcnow())
        self.f = open(path, "wb")
        self._header(0)

    def _header(self, count):
        pad = lambda s: s.encode("ascii", "replace")[:40].ljust(40, b"\0")   # noqa: E731
        hdr = self._struct.pack(
            "<14s7i40s40s40sqq", b"LUCAM-RECORDER", 0, self.color, 0,
            self.w, self.h, 8, int(count),
            pad("NOUT"), pad("Sony A7II HDMI"), pad(""), self._t_local, self._t_utc)
        assert len(hdr) == self.HEADER
        self.f.seek(0); self.f.write(hdr); self.f.seek(0, 2)

    def write(self, bgr):
        if bgr.shape[0] != self.h or bgr.shape[1] != self.w:
            return False                     # the source changed format mid-recording
        if self.color == self.MONO:
            img = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = bgr if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        self.f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        self._stamps.append(_ser_ticks(datetime.utcnow()))
        self.n += 1
        return True

    def close(self):
        if self.f is None:
            return
        try:
            if self._stamps:                  # optional trailer: one UTC stamp per frame
                self.f.write(self._struct.pack("<%dq" % len(self._stamps), *self._stamps))
            self._header(self.n)
        finally:
            self.f.close(); self.f = None


def read_ser_header(path):
    import struct
    with open(path, "rb") as f:
        raw = f.read(SerWriter.HEADER)
    if len(raw) < SerWriter.HEADER or not raw.startswith(b"LUCAM-RECORDER"):
        return None
    _id, color, _le, w, h, depth, n = struct.unpack("<7i", raw[14:42])
    return {"color": color, "w": w, "h": h, "depth": depth, "n": n}


def open_video_frames(path):
    """(frame_count, factory) where factory() yields BGR uint8 frames from the start.
    Reads SER (NOUT's HDMI recordings, FireCapture, SharpCap) as well as anything
    OpenCV opens. Returns (0, None) if the file cannot be read."""
    if path.lower().endswith(".ser"):
        hd = read_ser_header(path)
        if hd is None:
            return 0, None
        planes = 3 if hd["color"] in (SerWriter.RGB, SerWriter.BGR) else 1
        bpp = 2 if hd["depth"] > 8 else 1
        fsize = hd["w"] * hd["h"] * planes * bpp
        avail = (os.path.getsize(path) - SerWriter.HEADER) // fsize
        n = min(hd["n"], avail) if hd["n"] > 0 else avail

        def gen():
            with open(path, "rb") as f:
                f.seek(SerWriter.HEADER)
                for _ in range(n):
                    buf = f.read(fsize)
                    if len(buf) < fsize:
                        return
                    a = np.frombuffer(buf, dtype=np.uint16 if bpp == 2 else np.uint8)
                    if bpp == 2:
                        a = (a >> max(0, hd["depth"] - 8)).clip(0, 255).astype(np.uint8)
                    if planes == 3:
                        a = a.reshape(hd["h"], hd["w"], 3)
                        if hd["color"] == SerWriter.RGB:
                            a = a[:, :, ::-1]
                        yield np.ascontiguousarray(a)
                    else:                     # mono (and raw Bayer, read as mono)
                        yield cv2.cvtColor(a.reshape(hd["h"], hd["w"]), cv2.COLOR_GRAY2BGR)
        return n, gen

    probe = cv2.VideoCapture(path)
    if not probe.isOpened():
        return 0, None
    n = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    probe.release()

    def gen():
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    return
                yield f
        finally:
            cap.release()
    return n, gen


def video_backend():
    """The OpenCV capture backend that actually works on this system."""
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if os.name == "nt":
        return cv2.CAP_DSHOW              # DirectShow: what UVC cards expose on Windows
    return cv2.CAP_V4L2                   # Linux


def _ffmpeg_devices(fmt, dummy):
    """Device names as ffmpeg lists them for an input format (macOS / Windows)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-f", fmt,
                              "-list_devices", "true", "-i", dummy],
                             capture_output=True, text=True, timeout=8).stderr
    except Exception:                   # noqa: BLE001 - no ffmpeg
        return None
    return out


def list_video_devices():
    """[(index, name)] of this machine's video inputs, numbered the way the capture
    backend numbers them. Falls back to bare indexes when the names cannot be read."""
    plain = [(i, "Video device {}".format(i)) for i in range(4)]
    if sys.platform == "darwin":
        out = _ffmpeg_devices("avfoundation", "")
        if out is None:
            return plain
        devs, in_video = [], False
        for line in out.splitlines():
            if "video devices" in line:
                in_video = True; continue
            if "audio devices" in line:
                break
            m = re.search(r"\[(\d+)\]\s+(.+)$", line)
            if in_video and m and "capture screen" not in m.group(2).lower():
                devs.append((int(m.group(1)), m.group(2).strip()))
        return devs or plain
    if os.name == "nt":
        out = _ffmpeg_devices("dshow", "dummy")
        if out is None:
            return plain
        devs = []
        for line in out.splitlines():
            m = re.search(r'"(.+)"\s*\(video\)', line)
            if m:
                devs.append((len(devs), m.group(1).strip()))
        return devs or plain
    devs = []                                        # Linux: /dev/videoN + its name
    for path in sorted(glob.glob("/dev/video*")):
        try:
            idx = int(re.sub(r"\D", "", os.path.basename(path)))
        except ValueError:
            continue
        name = "Video device {}".format(idx)
        try:
            with open("/sys/class/video4linux/video{}/name".format(idx)) as f:
                name = f.read().strip() or name
        except OSError:
            pass
        devs.append((idx, name))
    return devs or plain


def _is_builtin_camera(name):
    n = (name or "").lower()
    return any(k in n for k in ("facetime", "macbook", "desk view", "iphone", "continuity"))


class HdmiSource(QtCore.QThread):
    """Reads a UVC capture card (the camera's HDMI output) on its own thread.

    Every frame goes to the SER file while recording; the screen only gets
    DISPLAY_FPS of them, so a 60 fps card never floods the GUI. Frames go out on
    the same signal the USB live view uses, so the reticle, sharpness curve, HFR,
    drift and lucky stacking all work unchanged."""
    frame_ready = Signal(object, float, object)
    status = Signal(str)
    opened = Signal(int, int, float)          # width, height, measured fps
    rec_progress = Signal(int, float)          # frames written, seconds elapsed
    rec_done = Signal(str, int, float)         # path, frames, measured fps

    DISPLAY_FPS = 25.0

    def __init__(self, source, size=(1920, 1080), fps=60.0):
        super().__init__()
        self.source = source                   # int index, or a file path (tests)
        self.size, self.fps = size, fps
        self._running = True
        self.paused = False                    # stop feeding the screen, keep the card open
        self.metric, self.roi_frac = "tenengrad", 0.5
        self.metric_on, self.stars_on = True, False
        self._lock = threading.Lock()
        self._rec_req = None
        self._rec_stop = False
        self._rec = None

    def stop(self):
        self._running = False

    def start_recording(self, path, seconds, mono):
        with self._lock:
            self._rec_req = (path, float(seconds), bool(mono)); self._rec_stop = False

    def stop_recording(self):
        with self._lock:
            self._rec_stop = True

    @property
    def recording(self):
        return self._rec is not None or self._rec_req is not None

    def _open(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, video_backend())
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                cap.set(cv2.CAP_PROP_FPS, self.fps)
        else:
            cap = cv2.VideoCapture(self.source)
        return cap if cap.isOpened() else None

    def _rec_step(self, frame, now):
        with self._lock:
            req, stop = self._rec_req, self._rec_stop
            self._rec_req = None; self._rec_stop = False
        if req is not None and self._rec is None:
            path, secs, mono = req
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                w = SerWriter(path, frame.shape[1], frame.shape[0], mono)
                self._rec = {"w": w, "t0": now, "until": now + secs}
                self.status.emit("🎥 Recording HDMI → {}".format(os.path.basename(path)))
            except Exception as e:      # noqa: BLE001
                self.status.emit("Cannot record: {}".format(e))
        r = self._rec
        if r is None:
            return
        r["w"].write(frame)
        if r["w"].n % 30 == 0:
            self.rec_progress.emit(r["w"].n, now - r["t0"])
        if stop or now >= r["until"] or not self._running:
            self._finish_rec(now)

    def _finish_rec(self, now):
        r, self._rec = self._rec, None
        if r is None:
            return
        try:
            r["w"].close()
        except Exception as e:          # noqa: BLE001
            self.status.emit("Recording not closed cleanly: {}".format(e))
        dur = max(now - r["t0"], 1e-3)
        self.rec_done.emit(r["w"].path, r["w"].n, r["w"].n / dur)

    def run(self):
        cap = self._open()
        if cap is None:
            self.status.emit(
                "HDMI source unavailable. Is the capture card plugged in, and has NOUT "
                "been allowed to use the camera (System Settings → Privacy & Security "
                "→ Camera)?")
            return
        is_file = not isinstance(self.source, int)
        last_emit, fails, first, n_meas, t_meas = 0.0, 0, True, 0, time.monotonic()
        try:
            while self._running:
                ok, frame = cap.read()
                now = time.monotonic()
                if not ok or frame is None:
                    if is_file:
                        break
                    fails += 1
                    if fails == 30:
                        self.status.emit("HDMI: no signal — camera off, asleep, or "
                                         "HDMI cable out? Waiting…")
                    if fails % 150 == 0:          # a stalled card: reopen it
                        cap.release(); self.msleep(500)
                        cap = self._open()
                        if cap is None:
                            self.msleep(1500); cap = self._open()
                            if cap is None:
                                self.status.emit("HDMI source lost."); return
                    self.msleep(10)
                    continue
                if fails >= 30:
                    self.status.emit("HDMI signal back.")
                fails = 0
                n_meas += 1
                if first and n_meas >= 30:
                    first = False
                    self.opened.emit(frame.shape[1], frame.shape[0],
                                     n_meas / max(now - t_meas, 1e-3))
                self._rec_step(frame, now)
                if self.paused or now - last_emit < 1.0 / self.DISPLAY_FPS:
                    if is_file:
                        self.msleep(int(1000 / max(self.fps, 1)))
                    continue
                last_emit = now
                val, stars = float("nan"), None
                try:
                    if self.metric_on:
                        gray, _ = _to_gray_roi(frame, self.roi_frac)
                        val = float(METRICS[self.metric](gray))
                    if self.stars_on:
                        stars, _ = detect_stars(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                except Exception:       # noqa: BLE001 - a metric glitch must not kill the feed
                    pass
                self.frame_ready.emit(frame, val, stars)
                if is_file:
                    self.msleep(int(1000 / max(self.fps, 1)))
        finally:
            self._finish_rec(time.monotonic())
            cap.release()


# ---------------------------------------------------------------------------
#  Graphical Interface
# ---------------------------------------------------------------------------
_DSO_INFO = None


def dso_distance_ly(oid):
    """Distance (light-years) for a catalogue id, from dso_info.csv — the usual
    reference values, rounded: right for telling visitors, not for science."""
    global _DSO_INFO
    if _DSO_INFO is None:
        _DSO_INFO = {}
        try:
            import csv as _csv
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dso_info.csv"),
                      newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    try:
                        _DSO_INFO[(row.get("id") or "").upper().replace(" ", "")] = \
                            float(row.get("distance_ly") or "nan")
                    except ValueError:
                        pass
        except Exception:               # noqa: BLE001
            pass
    return _DSO_INFO.get((oid or "").upper().replace(" ", ""), float("nan"))


def _num(x, dec=0):
    return ("{:,.%df}" % dec).format(x)


def fmt_distance(ly=None, km=None):
    """'2.5 million light-years', '22,200 light-years', '403,100 km', '1.3 billion km'."""
    def big(v, unit):
        if v >= 1e9:
            return "{} billion {}".format(_num(v / 1e9, 1).rstrip("0").rstrip("."), unit)
        if v >= 1e6:
            m = v / 1e6
            return "{} million {}".format(_num(m, 1).rstrip("0").rstrip(".") if m < 10
                                          else _num(m), unit)
        return None
    if km is not None:
        return big(km, "km") or "{} km".format(_num(round(km, -2)))
    if ly is None or ly != ly:
        return ""
    return big(ly, "light-years") or "{} light-years".format(_num(ly))


_TYPE_EN = (("globular", "Globular cluster"), ("open cluster", "Open cluster"),
            ("planetary", "Planetary nebula"), ("supernova", "Supernova remnant"),
            ("galax", "Galaxy"), ("nebula", "Nebula"), ("cluster", "Star cluster"),
            ("star", "Star"))

# solar-system names a target may carry (English UI, French aliases from older sessions)
_BODY_EN = {"sun": "Sun", "moon": "Moon", "mercury": "Mercury", "venus": "Venus",
            "mars": "Mars", "jupiter": "Jupiter", "saturn": "Saturn", "uranus": "Uranus",
            "neptune": "Neptune", "soleil": "Sun", "lune": "Moon", "mercure": "Mercury",
            "vénus": "Venus", "saturne": "Saturn"}


def horizon_profiles():
    """{name: {"lat", "lon", "alt": [180 heights, az 0..358 step 2]}} from settings."""
    try:
        d = json.loads(QtCore.QSettings("NOUT", "NOUT").value("horizon_profiles", "{}", type=str))
        return d if isinstance(d, dict) else {}
    except Exception:                   # noqa: BLE001
        return {}


def save_horizon_profiles(d):
    QtCore.QSettings("NOUT", "NOUT").setValue("horizon_profiles", json.dumps(d))


class _HorizonCanvas(QtWidgets.QWidget):
    """Draw the skyline seen from the mount: azimuth across (N E S W N), height up.
    Drag to trace trees and roofs, right-drag to erase."""
    changed = Signal()
    MAX_ALT = 60.0

    def __init__(self, profile=None, min_alt=None):
        super().__init__()
        self.alt = list(profile) if profile else [0.0] * 180
        self.min_alt = min_alt
        self.setMinimumSize(720, 300)
        self.setMouseTracking(True)
        self._last = None; self._hover = None

    def _to_az_alt(self, pos):
        m = 34
        w, h = self.width() - 2 * m, self.height() - 2 * m
        az = (pos.x() - m) / max(w, 1) * 360.0
        alt = (self.height() - m - pos.y()) / max(h, 1) * self.MAX_ALT
        return max(0.0, min(359.99, az)), max(0.0, min(self.MAX_ALT, alt))

    def _xy(self, az, alt):
        m = 34
        w, h = self.width() - 2 * m, self.height() - 2 * m
        return m + az / 360.0 * w, self.height() - m - alt / self.MAX_ALT * h

    def _paint_at(self, az, alt):
        i = int(round(az / 2.0)) % 180
        if self._last is not None:                 # fill the bins between two samples
            j, a0 = self._last
            steps = abs(i - j)
            if 0 < steps < 90:
                for k in range(1, steps):
                    b = j + (k if i > j else -k)
                    self.alt[b % 180] = a0 + (alt - a0) * k / steps
        self.alt[i] = alt
        self._last = (i, alt)
        self.changed.emit(); self.update()

    def mousePressEvent(self, ev):
        self._last = None
        az, alt = self._to_az_alt(ev.position())
        self._paint_at(az, 0.0 if ev.button() == Qt.RightButton else alt)

    def mouseMoveEvent(self, ev):
        az, alt = self._to_az_alt(ev.position())
        self._hover = (az, alt)
        if ev.buttons() & Qt.LeftButton:
            self._paint_at(az, alt)
        elif ev.buttons() & Qt.RightButton:
            self._paint_at(az, 0.0)
        self.update()

    def mouseReleaseEvent(self, ev):
        self._last = None

    def paintEvent(self, ev):
        q = QtGui.QPainter(self); q.setRenderHint(QtGui.QPainter.Antialiasing)
        g = QtGui.QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0, QtGui.QColor(8, 11, 26)); g.setColorAt(1, QtGui.QColor(26, 38, 70))
        q.fillRect(self.rect(), g)
        q.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 28), 1))
        f = QtGui.QFont(self.font()); f.setPixelSize(11); q.setFont(f)
        for alt in range(0, int(self.MAX_ALT) + 1, 10):
            x0, y = self._xy(0, alt); x1, _ = self._xy(360, alt)
            q.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x1, y))
            q.setPen(QtGui.QColor(150, 165, 195)); q.drawText(QtCore.QPointF(4, y + 4), "{}°".format(alt))
            q.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 28), 1))
        for az, lab in ((0, "N"), (45, "NE"), (90, "E"), (135, "SE"), (180, "S"),
                        (225, "SW"), (270, "W"), (315, "NW"), (360, "N")):
            x, y0 = self._xy(az, self.MAX_ALT); _, y1 = self._xy(az, 0)
            q.drawLine(QtCore.QPointF(x, y0), QtCore.QPointF(x, y1))
            q.setPen(QtGui.QColor(220, 230, 250)); q.drawText(QtCore.QPointF(x - 6, y1 + 18), lab)
            q.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 28), 1))
        if self.min_alt:                            # the "h min" of the targets list
            x0, y = self._xy(0, self.min_alt); x1, _ = self._xy(360, self.min_alt)
            q.setPen(QtGui.QPen(QtGui.QColor(232, 178, 58, 150), 1, Qt.DashLine))
            q.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x1, y))
        path = QtGui.QPainterPath()
        x, y = self._xy(0, 0); path.moveTo(x, y)
        for i in range(181):
            x, y = self._xy(i * 2.0, self.alt[i % 180]); path.lineTo(x, y)
        x, y = self._xy(360, 0); path.lineTo(x, y); path.closeSubpath()
        q.fillPath(path, QtGui.QColor(18, 26, 18, 240))
        q.setPen(QtGui.QPen(QtGui.QColor(110, 150, 110), 1.6)); q.drawPath(path)
        if self._hover:
            az, alt = self._hover
            q.setPen(QtGui.QColor(230, 235, 250))
            q.drawText(QtCore.QPointF(self.width() - 150, 18), "az {:.0f}°   h {:.0f}°".format(az, alt))
        q.end()


class HorizonEditor(QtWidgets.QDialog):
    """Create or edit a named local horizon (e.g. 'Home') for a site."""
    def __init__(self, parent, name, profile, lat, lon, min_alt):
        super().__init__(parent)
        self.setWindowTitle("Local horizon")
        v = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit(name or "Home"); top.addWidget(self.name, 1)
        self.where = QtWidgets.QLabel("Site {:.4f}, {:.4f}".format(lat, lon))
        self.where.setObjectName("cap"); top.addWidget(self.where)
        top.addWidget(_HelpButton(
            "Draw the skyline as seen from where the mount stands: drag along the tops of "
            "trees and roofs; right-drag erases. Azimuth runs N → E → S → W → N.<br><br>"
            "To measure a few points: a phone compass + inclinometer app aimed at the top "
            "of the tree, or note at which height stars disappear behind it."))
        v.addLayout(top)
        self.canvas = _HorizonCanvas(profile, min_alt); v.addWidget(self.canvas, 1)
        b = QtWidgets.QHBoxLayout()
        clr = QtWidgets.QPushButton("Clear"); clr.clicked.connect(self._clear)
        self.delete_btn = QtWidgets.QPushButton("Delete profile")
        self.delete_btn.clicked.connect(lambda: self.done(2))
        b.addWidget(clr); b.addWidget(self.delete_btn); b.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        ok = QtWidgets.QPushButton("Save"); ok.setObjectName("primary"); ok.clicked.connect(self.accept)
        b.addWidget(cancel); b.addWidget(ok); v.addLayout(b)
        self.resize(860, 420)

    def _clear(self):
        self.canvas.alt = [0.0] * 180; self.canvas.update()


class ObsRenderWorker(QtCore.QThread):
    """Renders the Observation look off the GUI thread (denoising takes ~0.5 s)."""
    done = Signal(object)

    def __init__(self, src, flat, saturation, max_side=2400):
        super().__init__()
        self.src, self.flat, self.sat, self.max_side = src, flat, saturation, max_side

    def run(self):
        try:
            self.done.emit(render_observation(self.src, self.flat, saturation=self.sat,
                                              max_side=self.max_side))
        except Exception as e:          # noqa: BLE001
            print("[NOUT] observation render failed:", e, file=sys.stderr, flush=True)
            self.done.emit(None)


class PublicView(QtWidgets.QWidget):
    """Full-screen presentation for visitors: the picture, large, on black — on the
    second screen when there is one — with the target and the time gathered so far,
    and a slow cross-fade each time a new frame joins the stack."""
    def __init__(self):
        super().__init__(None, Qt.Window)
        self.setWindowTitle("NOUT — Public view")
        self.setStyleSheet("background:#000;")
        self.setCursor(Qt.BlankCursor)
        self._cur = None; self._prev = None; self._t = 1.0
        self.title, self.sub, self.dist = "", "", ""
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setStartValue(0.0); self._anim.setEndValue(1.0); self._anim.setDuration(1600)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        self._anim.valueChanged.connect(self._tick)
        self._banner = _NutBanner()
        self._banner.setParent(self)

    def set_image(self, bgr):
        if bgr is None:
            return
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        pm = QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w,
                                                  QtGui.QImage.Format_RGB888).copy())
        self._prev, self._cur = self._cur, pm
        if self._prev is None or self._prev.size() != pm.size():
            self._prev = None; self._t = 1.0; self.update()
        else:
            self._t = 0.0; self._anim.stop(); self._anim.start()

    def set_caption(self, title, sub, dist=""):
        self.title, self.sub, self.dist = title, sub, dist; self.update()

    def _tick(self, v):
        self._t = float(v); self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._banner.move(self.width() - self._banner.width() - 28,
                          self.height() - self._banner.height() - 22)

    def paintEvent(self, ev):
        q = QtGui.QPainter(self)
        q.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        q.fillRect(self.rect(), Qt.black)

        def draw(pm, alpha):
            if pm is None:
                return
            sc = min(self.width() / pm.width(), self.height() / pm.height())
            w, h = int(pm.width() * sc), int(pm.height() * sc)
            q.setOpacity(alpha)
            q.drawPixmap(QtCore.QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h), pm)
        draw(self._prev, 1.0)
        draw(self._cur, self._t)
        q.setOpacity(1.0)
        if self.title or self.sub or self.dist:
            H = self.height()
            g = QtGui.QLinearGradient(0, H - 190, 0, H)
            g.setColorAt(0, QtGui.QColor(0, 0, 0, 0)); g.setColorAt(1, QtGui.QColor(0, 0, 0, 180))
            q.fillRect(0, H - 190, self.width(), 190, g)
            tw = self.width() - 400
            f = QtGui.QFont(self.font()); f.setPixelSize(max(26, H // 26))
            f.setWeight(QtGui.QFont.DemiBold); q.setFont(f)
            q.setPen(QtGui.QColor("#f3ead8"))
            q.drawText(QtCore.QRect(40, H - 160, tw, 60), Qt.AlignLeft | Qt.AlignBottom, self.title)
            f1 = QtGui.QFont(self.font()); f1.setPixelSize(max(18, H // 40)); q.setFont(f1)
            q.setPen(QtGui.QColor("#e8b23a"))
            q.drawText(QtCore.QRect(42, H - 96, tw, 36), Qt.AlignLeft | Qt.AlignVCenter, self.dist)
            f2 = QtGui.QFont(self.font()); f2.setPixelSize(max(15, H // 56)); q.setFont(f2)
            q.setPen(QtGui.QColor("#c9b99a"))
            q.drawText(QtCore.QRect(42, H - 56, tw, 34), Qt.AlignLeft | Qt.AlignTop, self.sub)
        q.end()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Escape, Qt.Key_V):
            self.close()
        else:
            super().keyPressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self.close()


class _HelpButton(QtWidgets.QToolButton):
    """Small round '?' (the macOS help button). The explanation stays out of the
    way until someone asks for it, then opens in a popover under the button."""
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setText("?"); self.setObjectName("help")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(22, 22)
        self.setFocusPolicy(Qt.NoFocus)
        self._text = text
        self.clicked.connect(self._show)

    def _show(self):
        pop = QtWidgets.QFrame(self, Qt.Popup)
        pop.setObjectName("helpPop")
        pop.setAttribute(Qt.WA_DeleteOnClose)
        lay = QtWidgets.QVBoxLayout(pop); lay.setContentsMargins(14, 12, 14, 12)
        lb = QtWidgets.QLabel(self._text); lb.setWordWrap(True)
        lb.setFixedWidth(320); lb.setTextFormat(Qt.AutoText)
        lay.addWidget(lb)
        pop.adjustSize()
        g = self.mapToGlobal(QtCore.QPoint(self.width(), self.height() + 6))
        x, y = g.x() - pop.width(), g.y()
        scr = QtGui.QGuiApplication.screenAt(g)
        if scr is not None:
            r = scr.availableGeometry()
            x = max(r.left() + 8, min(x, r.right() - pop.width() - 8))
            if y + pop.height() > r.bottom():
                y = self.mapToGlobal(QtCore.QPoint(0, 0)).y() - pop.height() - 6
        pop.move(x, y); pop.show()


def _hint(text):
    """A right-aligned '?' that holds what used to be a paragraph of instructions."""
    w = QtWidgets.QWidget(); w.setObjectName("hintRow")
    h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
    h.addStretch(1); h.addWidget(_HelpButton(text))
    return w


class _ElideLabel(QtWidgets.QLabel):
    """One-line status: shortened with an ellipsis rather than widening its panel.
    The whole text stays in the tooltip."""
    def __init__(self, text=""):
        super().__init__()
        self._full = ""
        self.setMinimumWidth(40)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text):
        self._full = text or ""
        self.setToolTip(self._full)
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight,
                                                     max(self.width() - 2, 40)))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight,
                                                      max(self.width() - 2, 40)))


class _AutoHideLabel(QtWidgets.QLabel):
    """A status line that takes no room while it has nothing to say."""
    def __init__(self, text=""):
        super().__init__(text)
        self.setVisible(bool(text))

    def setText(self, text):
        super().setText(text)
        self.setVisible(bool(text))


class _AspectBox(QtWidgets.QWidget):
    """Keeps its one child at the image's aspect ratio (3:2 for the A7II's
    6000×4000 sensor and its 1024×680 live view), centred, so the visor frame
    hugs the picture instead of enclosing black bands."""
    def __init__(self, child, ratio=1.5):
        super().__init__()
        self.ratio = float(ratio)
        self._c = None
        self.set_child(child)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(240, 160)

    def set_child(self, child):
        self._c = child
        child.setParent(self); child.show()
        self._relayout()

    def set_ratio(self, r):
        if r > 0.1 and abs(r - self.ratio) > 0.005:
            self.ratio = float(r); self._relayout()

    def resizeEvent(self, ev):
        self._relayout(); super().resizeEvent(ev)

    def _relayout(self):
        c = self._c
        if c is None or c.parent() is not self:
            return
        W, H = self.width(), self.height()
        w = min(W, int(H * self.ratio)); h = int(round(w / self.ratio))
        c.setGeometry((W - w) // 2, (H - h) // 2, w, h)


class _GearButton(QtWidgets.QToolButton):
    """A ⚙ that opens a small settings popover holding `content` — for settings
    you set once and then leave alone, so they stay off the main panel."""
    def __init__(self, content, tip="", width=330):
        super().__init__()
        self.setObjectName("gear"); self.setToolTip(tip)
        self.setIcon(_make_icon("gear", "#eef2f7")); self.setIconSize(QtCore.QSize(16, 16))
        self.setCursor(Qt.PointingHandCursor); self.setFocusPolicy(Qt.NoFocus)
        self._pop = QtWidgets.QFrame(self, Qt.Popup); self._pop.setObjectName("helpPop")
        lay = QtWidgets.QVBoxLayout(self._pop); lay.setContentsMargins(12, 10, 12, 10)
        content.setObjectName(content.objectName() or "plain")
        lay.addWidget(content); self._pop.setFixedWidth(width)
        self.clicked.connect(self._show)

    def _show(self):
        pop = self._pop; pop.adjustSize()
        g = self.mapToGlobal(QtCore.QPoint(self.width(), self.height() + 6))
        x, y = g.x() - pop.width(), g.y()
        scr = QtGui.QGuiApplication.screenAt(g)
        if scr is not None:
            r = scr.availableGeometry()
            x = max(r.left() + 8, min(x, r.right() - pop.width() - 8))
            if y + pop.height() > r.bottom():
                y = self.mapToGlobal(QtCore.QPoint(0, 0)).y() - pop.height() - 6
        pop.move(x, y); pop.show()


class _Section(QtWidgets.QWidget):
    """A macOS-style disclosure section: a grey caption with a chevron that folds
    a rounded card of controls away."""
    expandedChanged = Signal(object, bool)

    def __init__(self, title, expanded=True):
        super().__init__()
        self._title = title
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
        self.head = QtWidgets.QPushButton(); self.head.setObjectName("secHead")
        self.head.setCheckable(True); self.head.setChecked(expanded)
        self.head.setFlat(True)
        self.head.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.head.setCursor(Qt.PointingHandCursor); self.head.setFocusPolicy(Qt.NoFocus)
        self.body = QtWidgets.QFrame(); self.body.setObjectName("secBody")
        self.lay = QtWidgets.QVBoxLayout(self.body)
        self.lay.setContentsMargins(10, 8, 10, 8); self.lay.setSpacing(5)
        v.addWidget(self.head); v.addWidget(self.body)
        self.head.toggled.connect(self._apply)
        self._apply(expanded, emit=False)

    def _apply(self, on, emit=True):
        self.head.setText(("▾   " if on else "▸   ") + self._title)
        self.body.setVisible(on)
        if emit:
            self.expandedChanged.emit(self, on)

    def expanded(self):
        return self.head.isChecked()

    def set_expanded(self, on):
        if self.head.isChecked() != bool(on):
            self.head.setChecked(bool(on))

    def row(self, *items, stretch_last=False):
        """Add a horizontal row of widgets (strings become labels; None = stretch)."""
        h = QtWidgets.QHBoxLayout(); h.setSpacing(6)
        for i, it in enumerate(items):
            if it is None:
                h.addStretch(1)
            elif isinstance(it, str):
                lb = QtWidgets.QLabel(it); lb.setObjectName("cap"); h.addWidget(lb)
            elif isinstance(it, QtWidgets.QLayout):
                h.addLayout(it)
            else:
                h.addWidget(it, 1 if (stretch_last and i == len(items) - 1) else 0)
        self.lay.addLayout(h)
        return h


class _SectionColumn(QtWidgets.QScrollArea):
    """Stack of _Sections that never needs scrolling: opening a section that does
    not fit folds the sections opened longest ago until everything shows."""
    def __init__(self):
        super().__init__()
        self.setObjectName("rightPanel")
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._w = QtWidgets.QWidget(); self._w.setObjectName("rightPanelBody")
        self._v = QtWidgets.QVBoxLayout(self._w)
        self._v.setContentsMargins(0, 0, 4, 0); self._v.setSpacing(8)
        self._v.addStretch(1)
        self.setWidget(self._w)
        self._w.installEventFilter(self)          # content grew? fold, never scroll
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._secs, self._order = [], []
        self._auto_folded = []          # folded by fit(), not by the user: reopen when room returns
        self._fitting = False

    def add(self, sec):
        self._v.insertWidget(self._v.count() - 1, sec)
        self._secs.append(sec)
        if sec.expanded():
            self._order.append(sec)
        sec.expandedChanged.connect(self._toggled)
        return sec

    def _toggled(self, sec, on):
        if not self._fitting and sec in self._auto_folded:
            self._auto_folded.remove(sec)            # the user took it over
        if sec in self._order:
            self._order.remove(sec)
        if on:
            self._order.append(sec)
            if not self._fitting:
                QtCore.QTimer.singleShot(0, lambda: self.fit(keep=sec))

    def _need(self):
        h = 0
        for sec in self._secs:
            sec.layout().invalidate()
            h += sec.sizeHint().height()
        return h + self._v.spacing() * max(len(self._secs) - 1, 0)

    def fit(self, keep=None):
        avail = self.viewport().height()
        for _ in range(len(self._secs)):
            if self._need() <= avail:
                return
            # fold the least important first (sticky sections last), oldest first
            victims = sorted((s for s in self._order if s is not keep),
                             key=lambda s: (getattr(s, "sticky", 0), self._order.index(s)))
            if not victims:
                return
            self._fitting = True
            try:
                victims[0].set_expanded(False)
            finally:
                self._fitting = False
            self._auto_folded.append(victims[0])

    def _reopen(self):
        """Give back what fit() folded, most recent first, as long as it fits."""
        avail = self.viewport().height()
        for sec in list(reversed(self._auto_folded)):
            self._fitting = True
            try:
                sec.set_expanded(True)
            finally:
                self._fitting = False
            if self._need() > avail:
                self._fitting = True
                try:
                    sec.set_expanded(False)
                finally:
                    self._fitting = False
                break
            self._auto_folded.remove(sec)

    def eventFilter(self, obj, ev):
        """A longer status line can make a section taller between two exposures.
        Refit then too, so the panel folds instead of growing a scroll bar."""
        if ev.type() == QtCore.QEvent.Type.LayoutRequest and not self._fitting:
            QtCore.QTimer.singleShot(0, self._refit)
        return False

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        QtCore.QTimer.singleShot(0, self._refit)

    def _refit(self):
        self.fit(self._order[-1] if self._order else None)
        self._reopen()


class _NutBanner(QtWidgets.QWidget):
    """The app title, drawn after the logo: Nout, the sky goddess, arching her
    body over the name the way she arches over the world — shoulders and hips
    bending down into arms and legs, her head at the eastern end, stars along her
    back and a faint band of Milky Way behind."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(260, 50)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._gold, self._gold2 = QtGui.QColor("#f5c451"), QtGui.QColor("#c98a22")
        self._star, self._band = QtGui.QColor("#eef2f7"), QtGui.QColor("#8b7fd6")
        import random
        rnd = random.Random(7)
        self._stars = [(rnd.uniform(0.04, 0.96), rnd.uniform(0.08, 0.92),
                        rnd.uniform(0.5, 1.3), rnd.uniform(0.25, 0.9)) for _ in range(34)]

    def set_palette(self, pal, night=False):
        if night:
            self._gold, self._gold2 = QtGui.QColor(pal["accent"]), QtGui.QColor(pal["accent_dim"])
            self._star, self._band = QtGui.QColor(pal["text"]), QtGui.QColor(pal["accent_dim"])
        else:
            self._gold, self._gold2 = QtGui.QColor("#f5c451"), QtGui.QColor("#c98a22")
            self._star, self._band = QtGui.QColor("#eef2f7"), QtGui.QColor("#8b7fd6")
        self.update()

    def paintEvent(self, ev):
        self._paint_arch()

    def _paint_arch(self):
        q = QtGui.QPainter(self); q.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # Nout as the logo draws her: a thick arched body between two round shoulders,
        # thinner arms and legs slanting down to the horizon, and her head held on a
        # short neck from the eastern shoulder.
        gold = QtGui.QLinearGradient(0, 0, 0, h)
        gold.setColorAt(0.0, self._gold); gold.setColorAt(1.0, self._gold2)
        brush = QtGui.QBrush(gold)
        xl, xr, sy, top = 46.0, w - 46.0, 21.0, 4.5
        def stroke(width):
            pen = QtGui.QPen(brush, width)
            pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
            q.setPen(pen); q.setBrush(Qt.NoBrush)
        stroke(1.9)                                       # arms and legs
        q.drawLine(QtCore.QPointF(xl, sy), QtCore.QPointF(xl - 24, h - 2))
        q.drawLine(QtCore.QPointF(xr, sy), QtCore.QPointF(xr + 24, h - 2))
        stroke(2.4)                                       # neck
        head = QtCore.QPointF(xl - 11.0, sy - 9.5)
        q.drawLine(QtCore.QPointF(xl, sy), head)
        body = QtGui.QPainterPath()                      # the arched body, thickest
        body.moveTo(xl, sy)
        body.cubicTo(xl + 4, top - 4, xr - 4, top - 4, xr, sy)
        stroke(4.6)
        q.drawPath(body)
        q.setPen(Qt.NoPen); q.setBrush(brush)
        for x in (xl, xr):                                # rounded shoulders
            q.drawEllipse(QtCore.QPointF(x, sy), 3.1, 3.1)
        q.drawEllipse(head, 4.6, 4.6)                     # head
        q.drawEllipse(QtCore.QPointF(head.x(), head.y() - 7.6), 1.9, 1.9)   # crown
        # a small sun under the arch, as in the icon, then the name beneath her
        glow = QtGui.QRadialGradient(w / 2, 20, 11)
        gc = QtGui.QColor(self._gold); gc.setAlpha(110); glow.setColorAt(0, gc)
        gc2 = QtGui.QColor(self._gold); gc2.setAlpha(0); glow.setColorAt(1, gc2)
        q.setBrush(QtGui.QBrush(glow)); q.drawEllipse(QtCore.QPointF(w / 2, 20), 11, 11)
        q.setBrush(QtGui.QBrush(gold)); q.drawEllipse(QtCore.QPointF(w / 2, 20), 3.2, 3.2)
        f = QtGui.QFont(self.font()); f.setPixelSize(15); f.setWeight(QtGui.QFont.DemiBold)
        f.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 7)
        q.setFont(f); q.setPen(QtGui.QPen(QtGui.QBrush(gold), 1))
        r = QtCore.QRectF(0, 25, w + 7, h - 25)        # +7: trailing letter-space
        q.drawText(r, Qt.AlignHCenter | Qt.AlignVCenter, "NOUT")
        q.end()


class _TopBar(QtWidgets.QWidget):
    """Top bar whose centre widget sits at the true middle of the window, whatever
    the widths of the left and right groups."""
    def __init__(self):
        super().__init__()
        self._center = None
        self.left_widgets, self.right_widget = [], None
        self._star, self._band = QtGui.QColor("#eef2f7"), QtGui.QColor("#8b7fd6")
        import random
        rnd = random.Random(11)
        # stars in fractions of the bar, denser near the middle where Nout arches
        self._stars = []
        for _ in range(170):
            fx = min(max(rnd.gauss(0.5, 0.28), 0.0), 1.0)
            self._stars.append((fx, rnd.uniform(0.06, 0.94), rnd.uniform(0.45, 1.25),
                                rnd.uniform(0.15, 0.85)))

    def set_palette(self, pal, night=False):
        self._star = QtGui.QColor(pal["text"] if night else "#eef2f7")
        self._band = QtGui.QColor(pal["accent_dim"] if night else "#8b7fd6")
        self.update()

    def paintEvent(self, ev):
        opt = QtWidgets.QStyleOption(); opt.initFrom(self)
        q = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, q, self)
        q.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # Milky Way: one soft band crossing the whole bar, brightest behind the title
        band = QtGui.QLinearGradient(0, 0, w, 0)
        for t, a in ((0.0, 0), (0.25, 14), (0.5, 44), (0.75, 14), (1.0, 0)):
            c = QtGui.QColor(self._band); c.setAlpha(a); band.setColorAt(t, c)
        path = QtGui.QPainterPath()
        path.moveTo(0, h * 0.95)
        path.cubicTo(w * 0.35, h * 0.62, w * 0.62, h * 0.40, w, h * 0.02)
        path.lineTo(w, h * 0.42)
        path.cubicTo(w * 0.62, h * 0.78, w * 0.35, h * 0.95, 0, h * 1.25)
        path.closeSubpath()
        q.fillPath(path, QtGui.QBrush(band))
        q.setPen(Qt.NoPen)
        for fx, fy, r, a in self._stars:
            c = QtGui.QColor(self._star); c.setAlphaF(a * 0.7)
            q.setBrush(c); q.drawEllipse(QtCore.QPointF(fx * w, fy * h), r, r)
        q.end()

    def set_center(self, w):
        self._center = w; w.setParent(self); w.show(); self._place()

    def resizeEvent(self, ev):
        super().resizeEvent(ev); self._place()

    def showEvent(self, ev):
        super().showEvent(ev); QtCore.QTimer.singleShot(0, self._place)

    def _place(self):
        """Centre the title; if the side groups leave no room there, slide it into
        the free gap between them, and hide it rather than draw over them."""
        c = self._center
        if c is None:
            return
        lo = max([w.geometry().right() + 12 for w in self.left_widgets
                  if w.isVisible() and w.width() > 0] or [0])
        hi = (self.right_widget.geometry().left() - 12
              if self.right_widget is not None and self.right_widget.isVisible()
              else self.width())
        x = (self.width() - c.width()) // 2
        x = max(lo, min(x, hi - c.width()))
        c.setVisible(hi - lo >= c.width())
        c.move(x, (self.height() - c.height()) // 2)
        c.raise_()


class _HudOverlay(QtWidgets.QWidget):
    """Transparent HUD over the preview (EVA helmet visor): anti-aliased rounded-corner
    cover, subtle internal vignette, corner brackets + centre reticle, and small HUD
    readouts in the four corners. Mouse-transparent."""
    def __init__(self, parent, color="#e8b23a", bg="#000000", reticle="#ff3b30"):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._c = QtGui.QColor(color)
        self._bg = QtGui.QColor(bg)
        self._rc = QtGui.QColor(reticle)   # centre reticle: deliberately NOT the visor gold
        self._ra = 1.0                     # reticle opacity, 0..1
        self._info = {}
        self._mono = QtGui.QFont("Menlo", 10)
        self._mono.setStyleHint(QtGui.QFont.Monospace)

    def set_color(self, c):
        self._c = QtGui.QColor(c); self.update()

    def set_bg(self, c):
        self._bg = QtGui.QColor(c); self.update()

    def set_reticle_color(self, c):
        self._rc = QtGui.QColor(c); self.update()

    def set_reticle_opacity(self, a):
        """0 hides the reticle entirely, 1 is fully opaque."""
        self._ra = max(0.0, min(1.0, float(a))); self.update()

    def set_info(self, info):
        self._info = info or {}; self.update()

    def _draw_reticle(self, q, cx, cy):
        """Centre reticle. Drawn in its own colour (red by default) so it never
        melts into the gold visor, and twice: a dark outline first, then the
        colour on top, so it stays readable over the Moon as well as over a
        black sky or the red-filtered night-vision preview. Opacity is the
        user's, and it fades the dark outline with it so a faint reticle does
        not leave a grey ghost behind."""
        if self._ra <= 0.005:
            return
        gap, arm, ring = 4, 13, 8
        segs = [((cx - gap - arm, cy), (cx - gap, cy)),
                ((cx + gap, cy), (cx + gap + arm, cy)),
                ((cx, cy - gap - arm), (cx, cy - gap)),
                ((cx, cy + gap), (cx, cy + gap + arm))]
        col = QtGui.QColor(self._rc); col.setAlphaF(self._ra)
        shade = QtGui.QColor(0, 0, 0, int(150 * self._ra))
        for pen, ext in ((QtGui.QPen(shade, 3.0), 0), (QtGui.QPen(col, 1.4), 1)):
            pen.setCapStyle(Qt.RoundCap)
            q.setPen(pen); q.setBrush(Qt.NoBrush)
            for (x1, y1), (x2, y2) in segs:
                q.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
            q.drawEllipse(QtCore.QPointF(cx, cy), ring, ring)
            if ext:                       # centre dot, colour pass only
                q.setBrush(QtGui.QBrush(col))
                q.drawEllipse(QtCore.QPointF(cx, cy), 1.1, 1.1)
                q.setBrush(Qt.NoBrush)

    def paintEvent(self, ev):
        q = QtGui.QPainter(self); q.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        rad = 19
        # 1) anti-aliased rounded corners: fill the area OUTSIDE a rounded rect with bg
        outer = QtGui.QPainterPath(); outer.addRect(0, 0, w, h)
        inner = QtGui.QPainterPath(); inner.addRoundedRect(0, 0, w, h, rad, rad)
        q.fillPath(outer.subtracted(inner), self._bg)
        # 2) subtle internal vignette
        g = QtGui.QRadialGradient(w / 2, h / 2, max(w, h) * 0.72)
        g.setColorAt(0.0, QtGui.QColor(0, 0, 0, 0))
        g.setColorAt(0.72, QtGui.QColor(0, 0, 0, 0))
        g.setColorAt(1.0, QtGui.QColor(0, 0, 0, 120))
        q.fillPath(inner, QtGui.QBrush(g))
        # 3) corner brackets + reticle
        m, L = 16, 28
        q.setPen(QtGui.QPen(self._c, 2))
        for x, y, dx, dy in ((m, m, 1, 1), (w - m, m, -1, 1),
                             (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            q.drawLine(x, y, x + dx * L, y); q.drawLine(x, y, x, y + dy * L)
        self._draw_reticle(q, w // 2, h // 2)
        # 4) HUD readouts in the corners
        q.setFont(self._mono); q.setPen(QtGui.QPen(self._c))
        pad = m + 6
        tl, tr = self._info.get("tl", ""), self._info.get("tr", "")
        bl, br = self._info.get("bl", ""), self._info.get("br", "")
        if tl:
            q.drawText(pad, pad + 8, tl)
        if tr:
            q.drawText(w - pad - q.fontMetrics().horizontalAdvance(tr), pad + 8, tr)
        if bl:
            q.drawText(pad, h - pad, bl)
        if br:
            q.drawText(w - pad - q.fontMetrics().horizontalAdvance(br), h - pad, br)
        # 5) gold visor border ON TOP — same rounded path, so it covers the corner seam
        q.setBrush(Qt.NoBrush)
        q.setPen(QtGui.QPen(self._c, 3))
        q.drawRoundedRect(QtCore.QRectF(1.5, 1.5, w - 3, h - 3), rad, rad)


class ZoomableView(QtWidgets.QGraphicsView):
    """Image view with zoom (scroll wheel), pan (drag) and fit
    (double-click). Exposes setPixmap()/setText() to remain compatible with
    the rest of the code. Zoom level is kept from one frame to the next."""
    aspect_changed = Signal(float)              # width / height of the image shown

    def __init__(self, hud=False):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = self._scene.addPixmap(QtGui.QPixmap())
        self._pix_item.setTransformationMode(Qt.SmoothTransformation)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QColor("#111"))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMinimumSize(360, 240)
        self.setToolTip("Mouse wheel: zoom · drag: pan · double-click: fit")
        self._overlay = QtWidgets.QLabel("Waiting for live view…", self)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setWordWrap(True)
        self._overlay.setStyleSheet("color:#888;background:transparent;")
        self._hud = _HudOverlay(self) if hud else None
        self._has_image = False
        self._last_size = None

    def set_hud_color(self, c):
        if self._hud is not None:
            self._hud.set_color(c)

    def set_reticle_color(self, c):
        if self._hud is not None:
            self._hud.set_reticle_color(c)

    def set_reticle_opacity(self, a):
        if self._hud is not None:
            self._hud.set_reticle_opacity(a)

    def set_hud_info(self, info):
        if self._hud is not None:
            self._hud.set_info(info)

    def resizeEvent(self, ev):
        self._overlay.setGeometry(self.viewport().rect())
        if self._hud is not None:
            self._hud.setGeometry(self.viewport().rect()); self._hud.raise_()
        super().resizeEvent(ev)
        if self._has_image and self.transform().m11() <= self._fit_scale() * 1.001:
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)   # stay fitted, not zoomed

    def _fit_scale(self):
        r = self._pix_item.boundingRect()
        vp = self.viewport().rect()
        if r.width() <= 0 or r.height() <= 0:
            return 1.0
        return min(vp.width() / r.width(), vp.height() / r.height())

    def setPixmap(self, pix):
        if pix is None or pix.isNull():
            return
        self._overlay.hide()
        new_size = (pix.width(), pix.height())
        first = (not self._has_image) or (new_size != self._last_size)
        self._pix_item.setPixmap(pix)
        self._scene.setSceneRect(QtCore.QRectF(pix.rect()))
        self._last_size = new_size
        self._has_image = True
        if first:                       # new image/format -> fit (otherwise keep zoom)
            self.aspect_changed.emit(pix.width() / max(pix.height(), 1))
            self.resetTransform()
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)

    def setText(self, text):
        self._has_image = False
        self._pix_item.setPixmap(QtGui.QPixmap())
        self._overlay.setGeometry(self.viewport().rect())
        self._overlay.setText(text)
        self._overlay.show()

    def fit(self):
        if self._has_image:
            self.resetTransform()
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)

    def wheelEvent(self, ev):
        if not self._has_image:
            return
        factor = 1.25 if ev.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, ev):
        self.fit()
        super().mouseDoubleClickEvent(ev)


def _fmt_dur(seconds):
    """Readable duration: mm:ss, or h:mm:ss beyond one hour."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "{}:{:02d}:{:02d}".format(h, m, s) if h else "{:02d}:{:02d}".format(m, s)


class AstroWorker(QtCore.QThread):
    """Calculates night targets via astro_targets (astropy), outside UI thread."""
    done = Signal(object, str)        # (payload | None, error message)

    def __init__(self, lat, lon, height, name, tz, when, focal, min_alt, limit, max_mag=None,
                 horizon=None):
        super().__init__()
        self.args = (lat, lon, height, name, tz, when, focal, min_alt, limit, max_mag)
        self.horizon = horizon

    def run(self):
        lat, lon, height, name, tz, when, focal, min_alt, limit, max_mag = self.args
        try:
            import astro_targets as at
            site = at.Site(lat, lon, height, name, tz)
            night, recos = at.plan(site, when, focal_mm=focal,
                                   min_alt=min_alt, limit=limit, max_mag=max_mag,
                                   horizon=self.horizon)
            if night.kind == "none" or night.start is None:
                hdr = "No astronomical night on this date (Sun never below -18°)."
            else:
                mid = night.start + (night.end - night.start) / 2
                moon = at.moon_state(site, mid)
                ns, ne = at._to_local(night.start, site), at._to_local(night.end, site)
                hdr = ("·   {} night {:%H:%M} → {:%H:%M}   ·   "
                       "Moon {:.0f}% lit, altitude {:+.0f}°{}").format(
                    night.kind.capitalize(), ns, ne, moon["illum"] * 100, moon["alt"],
                    "  🌑 dark sky" if moon["alt"] < 0 else "")
            fitmap = {True: "✓ fits", False: "✗ wide", None: "—"}
            rows = [(r.target.id, r.target.name, r.max_alt,
                     r.transit_local.strftime("%Hh%M"), r.moon_sep, fitmap[r.fits],
                     float(r.target.coord.ra.deg), float(r.target.coord.dec.deg),
                     r.target.type, getattr(r, "moon_av", None),
                     getattr(r, "clear_h", None), float(r.target.size_arcmin or 0.0))
                    for r in recos]
            self.done.emit((hdr, rows), "")
        except Exception as e:        # noqa: BLE001
            self.done.emit(None, "{}: {}".format(type(e).__name__, e))


def _read_focal_exif(path):
    """Reads focal length (mm) from file EXIF; None if unavailable."""
    try:
        import exifread
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False, stop_tag="FocalLength")
        v = tags.get("EXIF FocalLength")
        if v is not None and getattr(v, "values", None):
            r = v.values[0]
            return float(r.num) / float(r.den) if hasattr(r, "num") else float(r)
    except Exception:               # noqa: BLE001
        pass
    return None


def _prep_solve_image(image=None, image_path=None, max_side=1600):
    """Grayscale image, flattened background, ready for solve-field/nova.
    For a RAW file, uses the embedded full-size JPEG (fast) — the RAW is left untouched —
    and only falls back to a demosaic if no embedded preview is available."""
    g = None
    if image is not None:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    elif image_path:
        # fast path: embedded JPEG (RAW) or direct decode (JPEG/PNG/TIFF)
        img = _decode_preview_image(image_path)
        if img is not None:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        else:                          # fallback: full demosaic (slower)
            ext = os.path.splitext(image_path)[1].lower()
            if ext in (".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".rw2"):
                try:
                    import rawpy
                    with rawpy.imread(image_path) as raw:
                        rgb = raw.postprocess(use_camera_wb=True, half_size=True,
                                              no_auto_bright=True, output_bps=8)
                    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                except Exception:       # noqa: BLE001
                    g = None
    if g is None:
        return None
    h, w = g.shape[:2]
    if max(h, w) > max_side:
        sc = max_side / max(h, w)
        g = cv2.resize(g, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    gf = g.astype(np.float32)
    # fast background flatten: estimate the gradient on a tiny image, then upscale
    # (≈10× faster than a large-sigma GaussianBlur on the full frame).
    hh, ww = gf.shape[:2]
    small = cv2.resize(gf, (max(ww // 8, 1), max(hh // 8, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=3.0)
    bg = cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
    f = gf - bg
    f -= f.min()
    return np.clip(f / (f.max() + 1e-6) * 255, 0, 255).astype(np.uint8)


class SolveWorker(QtCore.QThread):
    """Plate-solving via astrometry.net: local (solve-field) or online (nova).
    Reads focal from EXIF, flattens background, widens scale bounds."""
    done = Signal(str, object, float, float, float, float)  # (msg, annot, ra, dec, rot, fov_w)

    def __init__(self, focal_mm, image_path=None, image=None, sensor_w=35.8,
                 online=False, api_key=""):
        super().__init__()
        self.image_path = image_path
        self.image = image
        self.focal_mm = focal_mm
        self.sensor_w = sensor_w
        self.online = online
        self.api_key = api_key

    def _objects_in_field(self, ra, dec, fov_w):
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            import astro_targets as at
            center = SkyCoord(ra * u.deg, dec * u.deg)
            radius = fov_w / 2.0
            inside = []
            nearest, nsep = None, 1e9
            for t in at.load_catalog():
                sep = float(t.coord.separation(center).deg)
                if sep < nsep:
                    nsep, nearest = sep, t
                if sep <= radius:
                    inside.append((sep, t))
            if inside:
                inside.sort(key=lambda x: x[0])
                names = ", ".join("{} ({})".format(t.id, _translate_target(t.name)) for _, t in inside[:6])
                return "  ·  in field: " + names
            if nearest:
                return "  ·  near {} ({}) at {:.1f}°".format(nearest.id, _translate_target(nearest.name), nsep)
        except Exception:             # noqa: BLE001
            pass
        return ""

    def _nearest_target(self, ra, dec, fov_w):
        return self._objects_in_field(ra, dec, fov_w)

    def run(self):
        import tempfile
        nan = float("nan")
        img = _prep_solve_image(image=self.image, image_path=self.image_path)
        if img is None:
            self.done.emit("Image unreadable for plate-solving.", None, nan, nan, nan, nan)
            return
        focal = self.focal_mm
        if self.image_path:
            f = _read_focal_exif(self.image_path)
            if f:
                focal = f                      # real focal from EXIF
        fov_w = float(np.degrees(2 * np.arctan(self.sensor_w / (2 * max(focal, 1)))))
        tmp = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(tmp, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        try:
            if self.online:
                msg, annot, ra, dec, rot = self._solve_online(tmp, fov_w, focal)
            else:
                msg, annot, ra, dec, rot = self._solve_local(tmp, fov_w, focal)
            self.done.emit(msg, annot, ra, dec, rot, fov_w)
        except Exception as e:        # noqa: BLE001
            self.done.emit("Plate-solving: {}".format(e), None, nan, nan, nan, fov_w)
        finally:
            try:
                os.remove(tmp)
            except Exception:         # noqa: BLE001
                pass

    def _solve_local(self, png, fov_w, focal):
        import glob
        import re
        import subprocess
        exe = _find_solve_field()
        if not exe:
            return ("solve-field not found. Install astrometry.net "
                    "(brew install astrometry-net) then wide-field index files covering "
                    "~{:.0f}° (series index-4107…4119, or Gaia 5200), "
                    "or use the « online » mode.".format(fov_w), None, float("nan"), float("nan"), float("nan"))
        # solve-field calls sibling tools (astrometry-engine, image2xy…) that live
        # in the same bin dir — make sure that dir is on PATH for the subprocess.
        solve_env = dict(os.environ)
        solve_env["PATH"] = os.path.dirname(exe) + os.pathsep + solve_env.get("PATH", "")
        base = os.path.splitext(png)[0]
        text = ""
        annot = None
        try:
            # no --no-plots: we want the annotated image (<base>-ngc.png)
            cmd = [exe, "--overwrite", "--downsample", "2",
                   "--depth", "20,40",
                   "--scale-units", "degwidth",
                   "--scale-low", "{:.2f}".format(fov_w * 0.5),
                   "--scale-high", "{:.2f}".format(fov_w * 2.0),
                   "--cpulimit", "120"]
            # use the bundled index/config if NOUT ships one (self-contained .app)
            cfg = os.environ.get("NOUT_ASTROMETRY_CFG")
            if not cfg:
                here = os.path.dirname(os.path.abspath(__file__))
                cand = os.path.join(os.path.dirname(here), "astrometry.cfg")
                if os.path.exists(cand):
                    cfg = cand
            if cfg and os.path.exists(cfg):
                cmd += ["--config", cfg]
            cmd.append(png)
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=240,
                                 env=solve_env)
            text = out.stdout + "\n" + out.stderr
            for cand in (base + "-ngc.png", base + "-objs.png"):
                if os.path.exists(cand):
                    annot = cv2.imread(cand)
                    if cand.endswith("-ngc.png"):
                        break
        except subprocess.TimeoutExpired:
            return "Plate-solving: timeout exceeded.", None, float("nan"), float("nan"), float("nan")
        finally:
            for p in glob.glob(base + "*"):
                if p != png:
                    try:
                        os.remove(p)
                    except Exception:     # noqa: BLE001
                        pass
        m = re.search(r"Field center: \(RA,Dec\) = \(\s*([-\d.]+),\s*([-\d.]+)\)", text)
        if not m:
            hint = ""
            if "index" in text.lower() or "no index" in text.lower():
                hint = " — no index covers this field. Install wide series."
            elif "solving" in text.lower():
                hint = " — try online mode (field {:.0f}°, focal {:.0f} mm).".format(
                    fov_w, focal)
            return ("Field not solved (focal {:.0f} mm → ~{:.0f}°){}".format(focal, fov_w, hint),
                    None, float("nan"), float("nan"), float("nan"))
        ra, dec = float(m.group(1)), float(m.group(2))
        mr = re.search(r"Field rotation angle: up is\s+([-\d.]+)\s+degrees", text)
        rot = float(mr.group(1)) if mr else float("nan")
        rtxt = "  ·  rot {:.1f}°".format(rot) if rot == rot else ""
        return ("Field solved: RA {:.3f}°  Dec {:+.3f}°{}{}".format(
            ra, dec, rtxt, self._nearest_target(ra, dec, fov_w)), annot, ra, dec, rot)

    def _solve_online(self, png, fov_w, focal):
        import time as _t
        import urllib.request
        import urllib.parse
        if not self.api_key:
            return ("Online mode: enter your nova.astrometry.net API key "
                    "(free, in your profile on the site).", None, float("nan"), float("nan"), float("nan"))
        base = "http://nova.astrometry.net/api"

        def post_json(url, payload):
            data = urllib.parse.urlencode({"request-json": json.dumps(payload)}).encode()
            req = urllib.request.Request(url, data=data)
            return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())

        r = post_json(base + "/login", {"apikey": self.api_key})
        if r.get("status") != "success":
            return "nova: connection refused (invalid API key?).", None, float("nan"), float("nan"), float("nan")
        session = r["session"]
        sub = self._nova_upload(base + "/upload", png, {
            "session": session, "scale_units": "degwidth", "scale_type": "ul",
            "scale_lower": fov_w * 0.5, "scale_upper": fov_w * 2.0,
            "publicly_visible": "n", "allow_modifications": "n",
            "allow_commercial_use": "n"})
        if sub.get("status") != "success":
            return "nova: image upload failed.", None, float("nan"), float("nan"), float("nan")
        subid = sub["subid"]
        jobid = None
        for _ in range(60):
            s = json.loads(urllib.request.urlopen(
                base + "/submissions/%d" % subid, timeout=60).read().decode())
            jobs = s.get("jobs") or []
            if jobs and jobs[0]:
                jobid = jobs[0]; break
            _t.sleep(5)
        if not jobid:
            return "nova: processing too long, try again later.", None, float("nan"), float("nan"), float("nan")
        for _ in range(60):
            j = json.loads(urllib.request.urlopen(
                base + "/jobs/%d" % jobid, timeout=60).read().decode())
            st = j.get("status")
            if st == "success":
                break
            if st == "failure":
                return "nova: field not solved.", None, float("nan"), float("nan"), float("nan")
            _t.sleep(5)
        cal = json.loads(urllib.request.urlopen(
            base + "/jobs/%d/calibration" % jobid, timeout=60).read().decode())
        ra, dec = float(cal.get("ra")), float(cal.get("dec"))
        # annotated image from nova server
        annot = None
        try:
            raw = urllib.request.urlopen(
                "http://nova.astrometry.net/annotated_display/%d" % jobid, timeout=60).read()
            annot = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception:             # noqa: BLE001
            pass
        try:
            rot = float(cal.get("orientation"))
        except Exception:             # noqa: BLE001
            rot = float("nan")
        rtxt = "  ·  rot {:.1f}°".format(rot) if rot == rot else ""
        return ("Field solved (nova): RA {:.3f}°  Dec {:+.3f}°{}{}".format(
            ra, dec, rtxt, self._nearest_target(ra, dec, fov_w)), annot, ra, dec, rot)

    def _nova_upload(self, url, filepath, payload):
        import urllib.request
        import uuid
        boundary = "===astro" + uuid.uuid4().hex
        with open(filepath, "rb") as f:
            filedata = f.read()
        lines = []
        lines.append("--" + boundary)
        lines.append('Content-Type: text/plain')
        lines.append('MIME-Version: 1.0')
        lines.append('Content-disposition: form-data; name="request-json"')
        lines.append('')
        lines.append(json.dumps(payload))
        head = ("\r\n".join(lines) + "\r\n").encode()
        fhead = ("--" + boundary + "\r\n"
                 'Content-Type: application/octet-stream\r\n'
                 'MIME-Version: 1.0\r\n'
                 'Content-disposition: form-data; name="file"; filename="solve.png"\r\n'
                 "\r\n").encode()
        tail = ("\r\n--" + boundary + "--\r\n").encode()
        body = head + fhead + filedata + tail
        req = urllib.request.Request(url, data=body)
        req.add_header("Content-Type",
                       'multipart/form-data; boundary="%s"' % boundary)
        return json.loads(urllib.request.urlopen(req, timeout=180).read().decode())


def _sky_cache_dir():
    d = os.path.join(os.path.expanduser("~"), ".nout", "sky_cache")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:                 # noqa: BLE001
        pass
    return d


def _sky_cache_key(hips, ra, dec, fov, size):
    import hashlib
    raw = "{}|{:.3f}|{:.3f}|{:.4f}|{}".format(hips, ra, dec, fov, size)
    return os.path.join(_sky_cache_dir(), hashlib.md5(raw.encode()).hexdigest() + ".jpg")


_FOV_LADDER = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0]
_BG_TILE_SIZE = 768


def _snap_fov(fov):
    lf = math.log(max(fov, 0.1))
    return min(_FOV_LADDER, key=lambda f: abs(math.log(f) - lf))


def _bg_tile_params(ra, dec, fov):
    """Snap a view to a stable tile (center, fetch-fov, size) so the SAME area
    always maps to the SAME cache entry — which is what makes offline reuse work."""
    fq = _snap_fov(fov)
    step = max(fq / 3.0, 0.02)
    ra_q = (round(ra / step) * step) % 360.0
    dec_q = max(-89.0, min(89.0, round(dec / step) * step))
    fetch_fov = fq * 1.5                       # margin so the tile covers small offsets
    return round(ra_q, 4), round(dec_q, 4), round(fetch_fov, 4), _BG_TILE_SIZE


def _light_tile_params(ra, dec, fov):
    """Coarse single-coverage tiling for the whole-sky light backdrop (no oversampling,
    so the all-sky download stays small)."""
    step = max(fov, 0.05)                       # tiles tile the sky once (fetch-fov adds overlap)
    ra_q = (round(ra / step) * step) % 360.0
    dec_q = max(-89.0, min(89.0, round(dec / step) * step))
    return round(ra_q, 4), round(dec_q, 4), round(fov * 1.5, 4), _BG_TILE_SIZE


def _fetch_and_cache(hips, ra, dec, fov, size, timeout=25):
    """Return jpeg bytes for a hips2fits view, using the on-disk cache (offline-safe)."""
    cache = _sky_cache_key(hips, ra, dec, fov, size)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return f.read()
    import urllib.parse
    import urllib.request
    params = {"hips": hips, "width": int(size), "height": int(size),
              "fov": "{:.4f}".format(max(fov, 0.02)), "projection": "TAN",
              "coordsys": "icrs", "ra": "{:.5f}".format(ra),
              "dec": "{:.5f}".format(dec), "format": "jpg"}
    url = ("https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"
           + urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={"User-Agent": "NOUT/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    try:
        tmp = cache + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, cache)        # atomic: a killed thread never leaves a half file
    except Exception:                 # noqa: BLE001
        pass
    return data


class DSOImageLoader(QtCore.QThread):
    """Fetches a real sky image (CDS hips2fits) with on-disk cache (cached views work offline)."""
    done = Signal(object, str)        # (jpeg bytes | None, error)

    def __init__(self, ra, dec, fov_deg, size=240, hips="CDS/P/DSS2/color"):
        super().__init__()
        self.ra = ra; self.dec = dec; self.fov = fov_deg
        self.size = int(size); self.hips = hips

    def run(self):
        try:
            self.done.emit(_fetch_and_cache(self.hips, self.ra, self.dec, self.fov, self.size), "")
        except Exception as e:        # noqa: BLE001
            self.done.emit(None, str(e))


class PreloadWorker(QtCore.QThread):
    """Pre-fetches and caches a set of tiles so a zone (or the whole sky) is available offline."""
    progress = Signal(int, int, int)  # done, total, cached_ok

    def __init__(self, jobs, throttle=0.0):
        super().__init__()
        self.jobs = jobs              # list of (hips, ra, dec, fov, size)
        self.throttle = throttle
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import time
        n = len(self.jobs); ok = 0
        for i, (hips, ra, dec, fov, size) in enumerate(self.jobs):
            if self._stop:
                self.progress.emit(n, n, ok); return
            key = _sky_cache_key(hips, ra, dec, fov, size)
            if os.path.exists(key):           # already cached: no network, no wait
                ok += 1; self.progress.emit(i + 1, n, ok); continue
            try:
                _fetch_and_cache(hips, ra, dec, fov, size, timeout=30)
                ok += 1
            except Exception:                 # noqa: BLE001
                pass
            if self.throttle:
                time.sleep(self.throttle)     # be polite to the CDS server
            self.progress.emit(i + 1, n, ok)


class ClickableLabel(QtWidgets.QLabel):
    clicked = Signal()

    def mousePressEvent(self, ev):
        self.clicked.emit()
        super().mousePressEvent(ev)


class DialSpin(QtWidgets.QDoubleSpinBox):
    """Entry box for a setting-circle reading.

    In hour mode it also accepts the way the graduation is actually read out loud
    — 23h30, 23:30, 23h30m — instead of forcing a mental conversion to 23.5 in
    the dark at the eyepiece."""

    _HM = re.compile(r"^\s*(-?)\s*(\d{1,3})\s*[hH:]\s*(\d{1,2})?\s*[mM']?\s*$")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hours = False
        self.setDecimals(3)
        self.setRange(-360.0, 360.0)

    def set_hours(self, on):
        self._hours = bool(on)
        self.setToolTip("Hours. Type 23,5 or 23.5 — or straight off the dial: "
                        "23h30 / 23:30." if on else "Degrees, e.g. 70 or 70,5.")

    @staticmethod
    def _norm(text):
        """Accept both decimal separators. This machine's locale wants a comma, so a
        typed '23.5' would otherwise be rejected and silently fall back to zero."""
        dp = QtCore.QLocale().decimalPoint()
        return text.replace(".", dp).replace(",", dp)

    def validate(self, text, pos):
        if self._hours and (self._HM.match(text) or text.strip() in ("", "-")):
            return (QtGui.QValidator.Acceptable, text, pos)
        state, _t, p = super().validate(self._norm(text), pos)
        return (state, text, p)

    def valueFromText(self, text):
        m = self._HM.match(text) if self._hours else None
        if m:
            v = float(m.group(2)) + float(m.group(3) or 0) / 60.0
            return -v if m.group(1) == "-" else v
        return super().valueFromText(self._norm(text))


class PolarReticle(QtWidgets.QWidget):
    """The Star Adventurer polar-scope reticle, drawn as you will see it.

    Faithful to the engraved glass: a 12-division clock face, an inner ring group
    at 36/40/44′ for Polaris and an outer one at 60/65/70′ for σ Octantis in the
    south. The twelve divisions span the whole 24 h, so one division is 2 h of
    right ascension — measuring the published reticle confirms 30° per division,
    not 15°, which is what makes the scale unambiguous over a full day.

    The reticle turns with the RA axis, so it is almost never upright when you
    look through it: `rotation_h` turns the engraved scale by that much RA while
    Polaris stays where the sky actually puts it, which is exactly the view in
    the eyepiece.
    """
    R_EDGE = 76.0                     # arcminutes at the drawn field edge
    INNER = (36.0, 40.0, 44.0)        # Polaris rings
    OUTER = (60.0, 65.0, 70.0)        # σ Octantis rings
    R_NUM = 52.0                      # where the clock numbers sit

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(330, 330)
        self.ha_hours = 0.0            # Polaris' true hour angle
        self.pole_arcmin = 37.5        # its polar distance
        self.rotation_h = 0.0          # RA the reticle has turned from upright
        self.flip = False
        self.mirror = False
        self.numbers_ccw = False       # which way the engraved hours are numbered
        self.night = False

    def set_state(self, ha_hours, pole_arcmin, rotation_h=0.0, flip=False,
                  mirror=False, numbers_ccw=False, night=False):
        self.ha_hours = float(ha_hours); self.pole_arcmin = float(pole_arcmin)
        self.rotation_h = float(rotation_h)
        self.flip = bool(flip); self.mirror = bool(mirror)
        self.numbers_ccw = bool(numbers_ccw); self.night = bool(night)
        self.update()

    def clock_reading(self):
        """Which engraved number Polaris falls on, in clock hours out of 12.

        Facing north the sky turns ANTICLOCKWISE around the pole, while a clock
        face is numbered clockwise — so by default the two run against each other
        and the reading decreases as the hour angle grows. A reticle engraved the
        other way round reverses that, which is what `numbers_ccw` is for."""
        d = (self.ha_hours - self.rotation_h) / 2.0
        return (d if self.numbers_ccw else -d) % 12.0

    def _num_angle(self, n):
        """Sky angle at which the engraved number `n` currently sits."""
        a = n * 30.0 if self.numbers_ccw else -n * 30.0
        return a + self.rotation_h * 15.0

    # -- geometry ---------------------------------------------------------
    def _pt(self, cx, cy, scale, r_arcmin, sky_deg):
        # anticlockwise with increasing angle: that is how the sky turns around
        # the pole when you are facing north
        t = math.radians(sky_deg)
        x, y = -scale * r_arcmin * math.sin(t), -scale * r_arcmin * math.cos(t)
        if self.flip:
            x, y = -x, -y
        if self.mirror:
            x = -x
        return cx + x, cy + y

    def paintEvent(self, ev):            # noqa: N802  (Qt naming)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        scale = (min(w, h) / 2.0 - 16) / self.R_EDGE

        if self.night:
            bg, line, faint, mark, txt = "#120000", "#a02020", "#4a0f0f", "#ff4d4d", "#c05555"
        else:
            bg, line, faint, mark, txt = "#06120a", "#39d353", "#14532d", "#eab308", "#5fe37a"
        p.fillRect(self.rect(), QtGui.QColor(bg))
        pen = lambda c, wd=1: p.setPen(QtGui.QPen(QtGui.QColor(c), wd))  # noqa: E731

        # the engraved scale turns with the axis
        off = self.rotation_h * 15.0

        p.setBrush(Qt.NoBrush)
        for r in (self.R_EDGE,) + self.OUTER + self.INNER:
            pen(line if r in (self.R_EDGE, 44.0, 36.0, 70.0, 60.0) else faint, 1)
            p.drawEllipse(QtCore.QPointF(cx, cy), scale * r, scale * r)

        # cross-hairs, through the whole field
        pen(line, 1)
        for a in (0.0, 90.0, 180.0, 270.0):
            x1, y1 = self._pt(cx, cy, scale, 4.0, a + off)
            x2, y2 = self._pt(cx, cy, scale, self.R_EDGE, a + off)
            p.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

        # ticks across both ring bands: one per hour of RA, longer every 2 h
        for i in range(24):
            a = i * 15.0 + off
            major = (i % 2 == 0)
            for r_in, r_out in ((self.INNER[0], self.INNER[2]),
                                (self.OUTER[0], self.OUTER[2])):
                mid = (r_in + r_out) / 2.0
                lo = r_in if major else mid - 1.2
                hi = r_out if major else mid + 1.2
                pen(line if major else faint, 1)
                x1, y1 = self._pt(cx, cy, scale, lo, a)
                x2, y2 = self._pt(cx, cy, scale, hi, a)
                p.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

        # clock numbers, 12 at the top of the engraved scale when it is upright
        f = QtGui.QFont(); f.setPointSize(13); f.setBold(True); p.setFont(f)
        pen(txt)
        for n in range(1, 13):
            x, y = self._pt(cx, cy, scale, self.R_NUM, self._num_angle(n % 12))
            p.drawText(QtCore.QRectF(x - 15, y - 12, 30, 24), Qt.AlignCenter, str(n))

        # Ring labels mark radii, not angles, so they do NOT turn with the scale —
        # and they are fanned out because 4′ apart is only a few pixels, which had
        # them printing on top of each other.
        f2 = QtGui.QFont(); f2.setPointSize(8); p.setFont(f2); pen(txt)
        for group, base in ((self.INNER, 292.0), (self.OUTER, 292.0)):
            for i, r in enumerate(group):
                x, y = self._pt(cx, cy, scale, r, base + i * 13.0)
                p.drawText(QtCore.QRectF(x - 17, y - 8, 34, 16), Qt.AlignCenter,
                           "{:.0f}'".format(r))

        # the pole: the point the axis must aim at
        pen(line, 2)
        p.drawEllipse(QtCore.QPointF(cx, cy), scale * 3.5, scale * 3.5)

        # Polaris, at its real place in the sky, on its real radius
        px, py = self._pt(cx, cy, scale, self.pole_arcmin, self.ha_hours * 15.0)
        p.setPen(QtGui.QPen(QtGui.QColor(mark), 1, Qt.DashLine))
        p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(px, py))
        p.setPen(QtGui.QPen(QtGui.QColor(mark), 3))
        p.drawLine(QtCore.QPointF(px - 7, py - 7), QtCore.QPointF(px + 7, py + 7))
        p.drawLine(QtCore.QPointF(px - 7, py + 7), QtCore.QPointF(px + 7, py - 7))
        f3 = QtGui.QFont(); f3.setPointSize(9); f3.setBold(True); p.setFont(f3)
        p.setPen(QtGui.QColor(mark))
        # sideways rather than further out: straight out lands on the clock numbers
        n = math.hypot(px - cx, py - cy) or 1.0
        ux, uy = (px - cx) / n, (py - cy) / n
        p.drawText(QtCore.QRectF(px - uy * 26 - 30, py + ux * 26 - 9, 60, 18),
                   Qt.AlignCenter, "Polaris")

        # readout, bottom left like the reticle apps
        f4 = QtGui.QFont(); f4.setPointSize(10); p.setFont(f4)
        p.setPen(QtGui.QColor(txt))
        c = self.clock_reading()
        ch = int(c) or 12
        cm = (c - int(c)) * 60.0
        p.drawText(QtCore.QRectF(8, h - 42, w - 16, 18), Qt.AlignLeft,
                   "{:.1f}'".format(self.pole_arcmin))
        p.drawText(QtCore.QRectF(8, h - 24, w - 16, 18), Qt.AlignLeft,
                   "cadran {}:{:04.1f}".format(ch, cm))
        p.end()


class GeolocateWorker(QtCore.QThread):
    """Location via macOS CoreLocation (GPS) if available, else IP (city-level)."""
    done = Signal(object, str)        # (dict | None, error)

    @staticmethod
    def _coreloc():
        try:
            import time
            import CoreLocation
            from Foundation import NSDate, NSRunLoop
            mgr = CoreLocation.CLLocationManager.alloc().init()
            mgr.requestWhenInUseAuthorization()
            mgr.startUpdatingLocation()
            deadline = time.time() + 8
            while time.time() < deadline:
                loc = mgr.location()
                if loc is not None:
                    c = loc.coordinate()
                    if c.latitude or c.longitude:
                        return float(c.latitude), float(c.longitude)
                NSRunLoop.currentRunLoop().runUntilDate_(
                    NSDate.dateWithTimeIntervalSinceNow_(0.3))
        except Exception:             # noqa: BLE001
            return None
        return None

    def run(self):
        import json
        import urllib.request
        gps = self._coreloc()
        if gps:
            self.done.emit({"lat": gps[0], "lon": gps[1], "city": "GPS",
                            "country": "", "source": "gps"}, "")
            return
        try:
            req = urllib.request.Request("http://ip-api.com/json/",
                                         headers={"User-Agent": "NOUT/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            if d.get("status") == "success":
                d["source"] = "ip"
                self.done.emit(d, "")
            else:
                self.done.emit(None, d.get("message", "failed"))
        except Exception as e:        # noqa: BLE001
            self.done.emit(None, str(e))


class MosaicTile(QtWidgets.QLabel):
    """Resizable 3:2 tile for the mosaic grid; keeps its source image and refits on resize."""
    clicked = Signal(int, int)

    def __init__(self, r, c):
        super().__init__()
        self.r = r
        self.c = c
        self._src = None          # full QPixmap source (None = empty)
        self._active = False
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self._apply_style()
        self.setText(f"{r+1}×{c+1}")

    def _apply_style(self):
        if self._active:
            self.setStyleSheet("border:3px solid #f59e0b; background:#222; color:#999;")
        else:
            self.setStyleSheet("border:1px solid #444; background:#0c0c0c; color:#666;")

    def set_active(self, on):
        self._active = on
        self._apply_style()

    def set_source(self, pix):
        self._src = pix
        if pix is not None:
            self.setText("")
        self.refit()

    def clear_source(self):
        self._src = None
        self.setPixmap(QtGui.QPixmap())
        self.setText(f"{self.r+1}×{self.c+1}")

    def refit(self):
        if self._src is None or self.width() < 4 or self.height() < 4:
            return
        w, h = self.width(), self.height()
        scaled = self._src.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        # crop centré pour remplir exactement la tuile (3:2)
        x = max(0, (scaled.width() - w) // 2)
        y = max(0, (scaled.height() - h) // 2)
        self.setPixmap(scaled.copy(x, y, w, h))

    def mousePressEvent(self, ev):
        self.clicked.emit(self.r, self.c)
        super().mousePressEvent(ev)


class MosaicCanvas(QtWidgets.QWidget):
    """Lays out 3:2 tiles, overlapping by a fraction, filling the available area."""

    def __init__(self):
        super().__init__()
        self.rows = 1
        self.cols = 1
        self.overlap = 0.20
        self.tiles = {}
        self.setMinimumSize(240, 160)

    def set_grid(self, rows, cols, tiles):
        self.rows, self.cols, self.tiles = rows, cols, tiles
        self._relayout()

    def set_overlap(self, ov):
        self.overlap = max(0.0, min(0.6, ov))
        self._relayout()

    def resizeEvent(self, ev):
        self._relayout()
        super().resizeEvent(ev)

    def _relayout(self):
        if not self.tiles:
            return
        cw, ch = self.width(), self.height()
        ov = self.overlap
        cols, rows = self.cols, self.rows
        fx = 1 + (cols - 1) * (1 - ov)         # largeur totale en unités de tuile
        fy = 1 + (rows - 1) * (1 - ov)
        # tuile : largeur w, hauteur h = w*2/3 (3:2) ; on prend la plus grande qui tient
        w = min(cw / fx, ch / ((2.0 / 3.0) * fy))
        w = max(20.0, w)
        h = w * 2.0 / 3.0
        step_x = w * (1 - ov)
        step_y = h * (1 - ov)
        total_w = w * fx
        total_h = h * fy
        ox = (cw - total_w) / 2.0
        oy = (ch - total_h) / 2.0
        for (r, c), tile in self.tiles.items():
            tile.setGeometry(int(ox + c * step_x), int(oy + r * step_y), int(w), int(h))
            tile.refit()
        # ordre de superposition cohérent (haut-gauche dessous, bas-droite dessus)
        for r in range(rows):
            for c in range(cols):
                t = self.tiles.get((r, c))
                if t:
                    t.raise_()


class _SortItem(QtWidgets.QTableWidgetItem):
    """Table item that sorts on a supplied key (numeric or string) rather than text."""
    def __init__(self, text, sortkey):
        super().__init__(str(text))
        self._k = sortkey

    def __lt__(self, other):
        try:
            return self._k < other._k
        except Exception:             # noqa: BLE001 - mixed types
            return super().__lt__(other)


_THEME_DEEPSPACE = dict(
    bg="#0d1026", bar="#141a33", panel="#141a33", panel2="#1c2342", chip="#1c2342",
    border="#272f54", border2="#34406e", accent="#f5b942", accent_dim="#b9892b",
    onaccent="#1a1205", text="#e8eaf2", muted="#8b93b5", sel="#2a3360", radius=9,
    visor="#f5b942", reticle="#ff3b30")
_THEME_NIGHT = dict(
    bg="#0a0303", bar="#160606", panel="#160606", panel2="#220a0a", chip="#220a0a",
    border="#3a1212", border2="#511a1a", accent="#e0524a", accent_dim="#a23a34",
    onaccent="#160303", text="#d98b80", muted="#9a5b55", sel="#3a1010", radius=9,
    visor="#e0524a", reticle="#ff7a6e")
# "EVA" — extra-vehicular activity suit: space-black outside, graphite suit panels,
# NASA-orange piping, gold helmet visor around the preview.
_THEME_EVA = dict(
    bg="#05070c", bar="#12161d", panel="#12161d", panel2="#1b2129", chip="#1b2129",
    border="#2a323d", border2="#3b4653", accent="#ff7a29", accent_dim="#c85f1f",
    onaccent="#180a02", text="#eef2f7", muted="#8a95a4", sel="#28313d", radius=12,
    visor="#e8b23a", reticle="#ff3b30")
_THEME_EVA_NIGHT = dict(
    bg="#070302", bar="#160a06", panel="#160a06", panel2="#20100a", chip="#20100a",
    border="#3a1e12", border2="#512a18", accent="#e0602a", accent_dim="#a2461f",
    onaccent="#160803", text="#d99b80", muted="#9a6b55", sel="#3a1e10", radius=12,
    visor="#c8863a", reticle="#ff7a6e")


def _indicator_icons(p):
    """Checkmark and combo/spin arrows as small PNGs (QSS can only take images).
    Cached per colour, so switching day/night just points at other files."""
    base = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.CacheLocation) or os.path.expanduser("~/.cache")
    d = os.path.join(base, "NOUT", "qss")
    os.makedirs(d, exist_ok=True)
    out = {}

    def make(name, color, draw, size=28):
        path = os.path.join(d, "{}_{}.png".format(name, color.lstrip("#")))
        if not os.path.exists(path):
            pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
            q = QtGui.QPainter(pm); q.setRenderHint(QtGui.QPainter.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(color), size * 0.12)
            pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
            q.setPen(pen); draw(q, size); q.end(); pm.save(path)
        return path.replace("\\", "/")

    out["check"] = make("check", p["onaccent"], lambda q, s: q.drawPolyline(QtGui.QPolygonF([
        QtCore.QPointF(s * .24, s * .52), QtCore.QPointF(s * .43, s * .70),
        QtCore.QPointF(s * .77, s * .31)])))
    out["down"] = make("down", p["muted"], lambda q, s: q.drawPolyline(QtGui.QPolygonF([
        QtCore.QPointF(s * .30, s * .40), QtCore.QPointF(s * .50, s * .60),
        QtCore.QPointF(s * .70, s * .40)])))
    out["up"] = make("up", p["muted"], lambda q, s: q.drawPolyline(QtGui.QPolygonF([
        QtCore.QPointF(s * .30, s * .60), QtCore.QPointF(s * .50, s * .40),
        QtCore.QPointF(s * .70, s * .60)])))
    return out


def _theme_qss(p):
    """macOS-like: system font, quiet surfaces, hairline separators, grey section
    captions above rounded cards, and the accent kept for what is active —
    selected tab, primary button, checked boxes, slider fill, focus ring."""
    p = dict(p)
    try:
        p.update(_indicator_icons(p))
    except Exception:                   # noqa: BLE001 - no cache dir: plain indicators
        p.update(check="", down="", up="")
    return """
QWidget {{ background:{bg}; color:{text}; font-size:13px; }}
QLabel, QCheckBox, QRadioButton, QAbstractScrollArea > QWidget > QWidget {{ background:transparent; }}
QScrollArea {{ background:transparent; border:none; }}
QToolTip {{ background:{panel2}; color:{text}; border:1px solid {border2}; padding:5px 7px; }}

/* top bar -------------------------------------------------------------- */
#topbar {{ background:{bg}; border-bottom:1px solid {border}; }}
#brand {{ color:{accent}; font-size:15px; font-weight:600; letter-spacing:1px; }}
#topsep {{ color:{border}; }}
#topTarget {{ color:{text}; font-weight:600; font-size:13px; }}
#topTargetSub {{ color:{muted}; }}
#topbar QPushButton {{ background:transparent; border:1px solid transparent; padding:4px 10px; }}
#topbar QPushButton:hover {{ background:{chip}; border:1px solid {border}; }}
#topbar QPushButton:checked {{ background:{chip}; color:{accent}; border:1px solid {border}; }}

/* tabs: centred segmented control ------------------------------------------ */
QTabWidget::pane {{ border:none; border-top:1px solid {border}; background:{bg}; top:-1px; }}
QTabWidget::tab-bar {{ alignment:center; }}
QTabBar {{ background:transparent; }}
QTabBar::tab {{ background:transparent; color:{muted}; padding:5px 16px; margin:6px 2px;
  border:1px solid transparent; border-radius:7px; font-size:13px; }}
QTabBar::tab:selected {{ color:{text}; background:{chip}; border:1px solid {border2}; }}
QTabBar::tab:hover:!selected {{ color:{text}; }}

/* cards ------------------------------------------------------------------ */
QGroupBox {{ background:{panel}; border:1px solid {border}; border-radius:10px;
  margin-top:20px; padding:10px 10px 8px 10px; font-weight:400; }}
QGroupBox::title {{ subcontrol-origin:margin; subcontrol-position:top left; left:6px;
  top:2px; padding:0 2px; color:{muted}; font-size:11px; font-weight:600; background:transparent; }}
QToolBox {{ background:transparent; }}
QToolBox > QWidget, QToolBox QScrollArea > QWidget > QWidget {{ background:transparent; }}
QToolBox::tab {{ background:transparent; color:{text}; border:none;
  border-bottom:1px solid {border}; border-radius:0; padding:0 4px; min-height:30px;
  font-weight:600; }}
QToolBox::tab:selected {{ color:{accent}; }}
QToolBox::tab:hover {{ color:{accent}; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color:{border}; background:{border}; max-height:1px; border:none; }}

/* buttons ---------------------------------------------------------------- */
QPushButton, QToolButton {{ background:{chip}; color:{text}; border:1px solid {border2};
  border-radius:7px; padding:5px 12px; min-height:18px; }}
QPushButton:hover, QToolButton:hover {{ background:{panel2}; }}
QPushButton:pressed, QToolButton:pressed {{ background:{sel}; }}
QPushButton:disabled, QToolButton:disabled {{ color:{muted}; background:{panel}; border:1px solid {border}; }}
QPushButton:checked, QToolButton:checked {{ background:{accent}; color:{onaccent}; border:1px solid {accent}; }}
QPushButton#primary {{ background:{accent}; color:{onaccent}; border:1px solid {accent};
  font-weight:600; padding:6px 16px; }}
QPushButton#primary:hover {{ background:{accent_dim}; border:1px solid {accent_dim}; }}
QToolButton#help {{ background:{chip}; color:{muted}; border:1px solid {border2};
  border-radius:11px; padding:0; min-height:0; font-size:12px; font-weight:600; }}
QToolButton#help:hover {{ color:{accent}; border:1px solid {accent}; }}
#hintRow {{ background:transparent; }}
QToolButton::menu-indicator {{ image:none; width:0; height:0; }}
QToolButton#gear {{ padding:2px 7px; min-height:18px; }}
#rightPanel, #rightPanelBody {{ background:transparent; }}
#rightPanelBody QWidget {{ font-size:12px; }}
#rightPanelBody QComboBox, #rightPanelBody QAbstractSpinBox, #rightPanelBody QLineEdit {{
  padding:2px 6px; min-height:18px; }}
#rightPanelBody QPushButton, #rightPanelBody QToolButton {{ padding:3px 10px; min-height:18px; }}
QPushButton#secHead {{ background:transparent; border:none; color:{muted}; font-size:11px;
  font-weight:600; padding:2px 2px; text-align:left; min-height:0; }}
QPushButton#secHead:hover {{ color:{text}; }}
QPushButton#secHead:checked {{ background:transparent; color:{muted}; border:none; }}
QPushButton#secHead:checked:hover {{ color:{text}; }}
#secBody {{ background:{panel}; border:1px solid {border}; border-radius:10px; }}
QWidget#plain {{ background:transparent; }}
#cap {{ color:{muted}; font-size:11px; }}
#fixedVal {{ color:{text}; font-weight:600; padding:3px 6px; }}
QGroupBox#flatGroup {{ background:transparent; border:none; margin:0; padding:0; }}
#helpPop {{ background:{panel2}; border:1px solid {border2}; border-radius:8px; }}
#helpPop QLabel {{ color:{text}; font-size:12px; }}

/* inputs ----------------------------------------------------------------- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit,
QAbstractSpinBox, QPlainTextEdit, QTextEdit {{
  background:{chip}; color:{text}; border:1px solid {border}; border-radius:6px;
  padding:4px 8px; min-height:18px; selection-background-color:{accent}; selection-color:{onaccent}; }}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
  border:1px solid {accent}; }}
QComboBox::drop-down {{ border:none; width:20px; }}
QComboBox::down-arrow {{ image:url({down}); width:14px; height:14px; }}
QComboBox QAbstractItemView {{ background:{panel2}; color:{text}; border:1px solid {border2};
  selection-background-color:{accent}; selection-color:{onaccent}; outline:none; padding:4px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ border:none; width:16px; background:transparent; }}
QAbstractSpinBox::up-arrow {{ image:url({up}); width:12px; height:12px; }}
QAbstractSpinBox::down-arrow {{ image:url({down}); width:12px; height:12px; }}
QDateEdit::drop-down {{ border:none; width:18px; }}

QCheckBox, QRadioButton {{ spacing:7px; padding:2px 6px 2px 0; }}
QCheckBox::indicator, QRadioButton::indicator {{ width:14px; height:14px;
  border:1px solid {border2}; border-radius:4px; background:{chip}; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border:1px solid {accent_dim}; }}
QCheckBox::indicator:checked {{ background:{accent}; border:1px solid {accent}; image:url({check}); }}
QRadioButton::indicator {{ border-radius:8px; }}
QRadioButton::indicator:checked {{ border:4px solid {accent}; background:{onaccent}; width:8px; height:8px; }}

QSlider {{ background:transparent; }}
QSlider::groove:horizontal {{ height:4px; background:{border2}; border-radius:2px; }}
QSlider::sub-page:horizontal {{ background:{accent}; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{text}; border:1px solid {border2}; width:14px; height:14px;
  margin:-6px 0; border-radius:8px; }}
QProgressBar {{ background:{chip}; border:none; border-radius:5px; text-align:center;
  color:{text}; min-height:10px; font-size:11px; }}
QProgressBar::chunk {{ background:{accent}; border-radius:5px; }}

/* lists and tables ------------------------------------------------------- */
QHeaderView {{ background:transparent; }}
QHeaderView::section {{ background:{bg}; color:{muted}; padding:5px 8px; border:none;
  border-bottom:1px solid {border}; font-size:12px; font-weight:600; }}
QTableWidget, QTableView, QTreeView, QListView, QListWidget {{ background:{bg}; color:{text};
  gridline-color:{bg}; border:1px solid {border}; border-radius:8px;
  selection-background-color:{sel}; selection-color:{text}; alternate-background-color:{panel}; }}
QTableWidget::item, QTableView::item {{ padding:3px 6px; }}
QTableWidget::item:selected, QTableView::item:selected {{ background:{sel}; color:{accent}; }}
QTableCornerButton::section {{ background:{bg}; border:none; }}

QScrollBar:vertical {{ background:transparent; width:8px; margin:2px; }}
QScrollBar::handle:vertical {{ background:{border2}; border-radius:3px; min-height:28px; }}
QScrollBar:horizontal {{ background:transparent; height:8px; margin:2px; }}
QScrollBar::handle:horizontal {{ background:{border2}; border-radius:3px; min-width:28px; }}
QScrollBar::handle:hover {{ background:{muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}

QMenuBar {{ background:{bar}; }}
QMenu {{ background:{panel2}; color:{text}; border:1px solid {border2}; padding:4px; }}
QMenu::item {{ padding:5px 18px; border-radius:5px; }}
QMenu::item:selected {{ background:{accent}; color:{onaccent}; }}
QMenu::separator {{ height:1px; background:{border}; margin:4px 6px; }}
QStatusBar {{ background:{bar}; color:{muted}; border-top:1px solid {border}; font-size:12px; }}
QStatusBar QLabel {{ color:{muted}; }}
QSplitter::handle {{ background:{border}; }}
QDialog {{ background:{bg}; }}
#visor {{ background:#000; border:none; border-radius:18px; }}
""".format(**p)


def _make_icon(name, color="#e8eaf2", size=20):
    """Tiny vector line-icons drawn with QPainter (no external assets)."""
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    q = QtGui.QPainter(pm); q.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color), 1.7); pen.setJoinStyle(Qt.RoundJoin)
    pen.setCapStyle(Qt.RoundCap); q.setPen(pen); q.setBrush(Qt.NoBrush)
    s = size
    if name == "gallery":                  # a sheet of little pictures
        for x, y in ((3, 3), (11, 3), (3, 11), (11, 11)):
            q.drawRoundedRect(QtCore.QRectF(x, y, 6, 6), 1.2, 1.2)
    elif name == "public":                 # a screen, and an audience
        q.drawRoundedRect(QtCore.QRectF(2.5, 2.5, s - 5, s * 0.52), 1.5, 1.5)
        for cx in (s * 0.36, s * 0.64):
            q.drawEllipse(QtCore.QPointF(cx, s * 0.72), 1.6, 1.6)
            q.drawArc(QtCore.QRectF(cx - 3.2, s * 0.80, 6.4, 5.0), 0, 180 * 16)
    elif name == "gear":                   # settings: ring + eight teeth
        c = QtCore.QPointF(s / 2, s / 2)
        for k in range(8):
            a = math.radians(k * 45)
            q.drawLine(QtCore.QPointF(c.x() + 5.2 * math.cos(a), c.y() + 5.2 * math.sin(a)),
                       QtCore.QPointF(c.x() + 7.8 * math.cos(a), c.y() + 7.8 * math.sin(a)))
        q.drawEllipse(c, 5.2, 5.2)
        q.drawEllipse(c, 2.0, 2.0)
    elif name == "fullscreen":
        for x, y, dx, dy in ((3, 3, 1, 1), (s - 3, 3, -1, 1), (3, s - 3, 1, -1), (s - 3, s - 3, -1, -1)):
            q.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + 5 * dx, y))
            q.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x, y + 5 * dy))
    elif name == "camera":
        q.drawRoundedRect(2, 6, s - 4, s - 9, 2, 2)
        q.drawLine(7, 6, 9, 3); q.drawLine(9, 3, s - 6, 3)
        q.drawEllipse(QtCore.QPointF(s / 2, s / 2 + 1.5), 3.2, 3.2)
    elif name == "grid":
        q.drawRoundedRect(3, 3, s - 6, s - 6, 2, 2)
        q.drawLine(s / 2, 3, s / 2, s - 3); q.drawLine(3, s / 2, s - 3, s / 2)
    elif name == "image":
        q.drawRoundedRect(3, 4, s - 6, s - 8, 2, 2)
        q.drawEllipse(QtCore.QPointF(7, 8), 1.6, 1.6)
        q.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(4, s - 5), QtCore.QPointF(9, 10),
                       QtCore.QPointF(13, 13), QtCore.QPointF(s - 4, 7)]))
    elif name == "target":
        q.drawEllipse(QtCore.QPointF(s / 2, s / 2), 6, 6)
        q.drawEllipse(QtCore.QPointF(s / 2, s / 2), 2, 2)
        q.drawLine(s / 2, 1, s / 2, 4); q.drawLine(s / 2, s - 4, s / 2, s - 1)
        q.drawLine(1, s / 2, 4, s / 2); q.drawLine(s - 4, s / 2, s - 1, s / 2)
    elif name == "map":  # sparkle / sky
        cx, cy = s / 2, s / 2
        q.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(cx, 2), QtCore.QPointF(cx + 2, cy - 2),
            QtCore.QPointF(s - 2, cy), QtCore.QPointF(cx + 2, cy + 2), QtCore.QPointF(cx, s - 2),
            QtCore.QPointF(cx - 2, cy + 2), QtCore.QPointF(2, cy), QtCore.QPointF(cx - 2, cy - 2)]))
    elif name == "capture":
        q.setBrush(QtGui.QColor(color)); q.drawEllipse(QtCore.QPointF(s / 2, s / 2), 5, 5)
    elif name == "search":
        q.drawEllipse(QtCore.QPointF(s / 2 - 1, s / 2 - 1), 4.5, 4.5)
        q.drawLine(s / 2 + 2.5, s / 2 + 2.5, s - 3, s - 3)
    elif name == "moon":
        path = QtGui.QPainterPath(); path.addEllipse(QtCore.QPointF(s / 2, s / 2), 6, 6)
        cut = QtGui.QPainterPath(); cut.addEllipse(QtCore.QPointF(s / 2 + 3, s / 2 - 1), 6, 6)
        q.setBrush(QtGui.QColor(color)); q.setPen(Qt.NoPen); q.drawPath(path.subtracted(cut))
    elif name == "location":
        path = QtGui.QPainterPath(); path.moveTo(s / 2, s - 2)
        path.cubicTo(2, s / 2, 4, 2, s / 2, 2); path.cubicTo(s - 4, 2, s - 2, s / 2, s / 2, s - 2)
        q.drawPath(path); q.drawEllipse(QtCore.QPointF(s / 2, s / 2 - 1), 1.8, 1.8)
    elif name == "calc":
        q.drawRoundedRect(4, 2, s - 8, s - 4, 2, 2)
        q.drawLine(6, 7, s - 6, 7)
        for yy in (10, 13, 16):
            for xx in (6, 9, 12):
                q.drawPoint(int(xx), int(yy))
    q.end()
    return QtGui.QIcon(pm)


def _artificial_to_sqm(art_ucd_m2):
    """World-Atlas artificial brightness (µcd/m²) -> zenith SQM (mag/arcsec²).
    Natural background taken at ~22.0 mag/arcsec² (1.7e-4 cd/m²). Calibrated against
    lightpollutionmap.info: 496 µcd/m² -> SQM 20.52 (Bortle 4)."""
    import math
    L_nat = 1.7e-4                                  # cd/m²
    L_art = max(0.0, float(art_ucd_m2)) * 1e-6      # µcd/m² -> cd/m²
    return 12.58 - 2.5 * math.log10(L_nat + L_art)


def _sqm_to_bortle_index(sqm):
    """SQM -> Bortle index 0..8 (B1..B9), using the lightpollutionmap.info bins."""
    bins = [(21.99, 0), (21.89, 1), (21.69, 2), (20.49, 3),
            (19.50, 4), (18.94, 5), (18.38, 6), (17.80, 7)]
    for thr, i in bins:
        if sqm >= thr:
            return i
    return 8


def _parse_first_float(text):
    import re as _re
    m = _re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text or "")
    return float(m.group(0)) if m else None


class GeoBortleWorker(QtCore.QThread):
    """Queries lightpollutionmap.info (World Atlas 2015) for the artificial sky
    brightness at a point, then derives SQM + Bortle. Needs a free API key."""
    done = Signal(object, object, str)        # (sqm|None, bortle_idx|None, error)

    def __init__(self, lat, lon, key):
        super().__init__()
        self.lat = lat; self.lon = lon; self.key = key

    def run(self):
        try:
            import urllib.request, urllib.parse
            params = {"ql": "wa_2015", "qt": "point",
                      "qd": "{:.6f},{:.6f}".format(self.lon, self.lat),  # lon,lat order
                      "key": self.key}
            url = ("https://www.lightpollutionmap.info/QueryRaster/?"
                   + urllib.parse.urlencode(params))
            req = urllib.request.Request(url, headers={"User-Agent": "NOUT/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                txt = r.read().decode("utf-8", "replace").strip()
            val = _parse_first_float(txt)
            if val is None:
                self.done.emit(None, None, "Unexpected response: " + txt[:90])
                return
            sqm = _artificial_to_sqm(val)
            self.done.emit(sqm, _sqm_to_bortle_index(sqm), "")
        except Exception as e:                # noqa: BLE001
            self.done.emit(None, None, str(e))


class ExposureCalcDialog(QtWidgets.QDialog):
    """Estimate the optimal single sub-exposure so sky-background shot noise swamps
    read noise (Glover/SharpCap approach). Values are estimates — the relative
    behaviour (f-ratio, sky, read noise) is what matters most."""
    # Bortle -> sky brightness (mag/arcsec²)
    _BORTLE = [("1 — pristine (22.0)", 22.0), ("2 — dark (21.7)", 21.7),
               ("3 — rural (21.5)", 21.5), ("4 — rural/suburban (21.0)", 21.0),
               ("5 — suburban (20.4)", 20.4), ("6 — bright suburban (19.4)", 19.4),
               ("7 — suburban/urban (18.7)", 18.7), ("8 — city (18.2)", 18.2),
               ("9 — inner city (17.8)", 17.8)]
    # Sony A7 II (ILCE-7M2) input-referred read noise (e-) by ISO — Photons to Photos
    # / sensorgen reference. Dual conversion gain drops the read noise sharply at
    # ISO 800 (the astro sweet spot of this sensor).
    _ISO_RN = [("100", 4.3), ("200", 3.9), ("400", 3.5), ("640", 3.4),
               ("800 ★ (dual-gain)", 2.5), ("1600", 1.8), ("3200", 1.6),
               ("6400", 1.5), ("12800", 1.5)]
    _S0 = 7.0e7    # calibration constant for the sky electron rate model

    def __init__(self, parent=None, focal=135.0, fnumber=2.8, lat=None, lon=None):
        super().__init__(parent)
        self.setWindowTitle("Sub-exposure calculator")
        self.setMinimumWidth(440)
        self._lat = lat; self._lon = lon; self._geo = None
        form = QtWidgets.QFormLayout(self)

        self.focal = QtWidgets.QDoubleSpinBox(); self.focal.setRange(8, 4000)
        self.focal.setValue(focal); self.focal.setSuffix(" mm")
        self.fnum = QtWidgets.QDoubleSpinBox(); self.fnum.setRange(0.95, 64)
        self.fnum.setSingleStep(0.1); self.fnum.setValue(fnumber); self.fnum.setPrefix("f/")
        self.pixel = QtWidgets.QDoubleSpinBox(); self.pixel.setRange(1.0, 12.0)
        self.pixel.setSingleStep(0.01); self.pixel.setValue(5.97); self.pixel.setSuffix(" µm")
        self.iso = QtWidgets.QComboBox()
        for label, _rn in self._ISO_RN:
            self.iso.addItem(label)
        self.iso.setCurrentIndex(4)        # ISO 800 (dual-gain sweet spot)
        self.bortle = QtWidgets.QComboBox()
        for label, _m in self._BORTLE:
            self.bortle.addItem(label)
        self.bortle.setCurrentIndex(3)
        self.qe = QtWidgets.QDoubleSpinBox(); self.qe.setRange(0.1, 1.0)
        self.qe.setSingleStep(0.05); self.qe.setValue(0.5)
        self.noise_pct = QtWidgets.QComboBox()
        for s in ("1 % (very conservative)", "5 % (recommended)", "10 % (shorter subs)"):
            self.noise_pct.addItem(s)
        self.noise_pct.setCurrentIndex(1)
        self.total = QtWidgets.QDoubleSpinBox(); self.total.setRange(1, 1200)
        self.total.setValue(120); self.total.setSuffix(" min")

        form.addRow("Focal length", self.focal)
        form.addRow("Aperture (f-number)", self.fnum)
        form.addRow("Pixel size", self.pixel)
        form.addRow("ISO (read noise)", self.iso)
        form.addRow("Sky (Bortle)", self.bortle)

        # auto-detect Bortle from the light-pollution map at the site location
        detect_row = QtWidgets.QHBoxLayout()
        self.detect_btn = QtWidgets.QPushButton("📍 Detect from my location")
        self.detect_btn.setToolTip("Look up the artificial sky brightness at your site "
                                   "(lightpollutionmap.info) and set the Bortle class.")
        self.detect_btn.clicked.connect(self._detect_bortle)
        detect_row.addWidget(self.detect_btn)
        self.detect_lbl = QtWidgets.QLabel("")
        self.detect_lbl.setStyleSheet("color:#94a3b8;")
        detect_row.addWidget(self.detect_lbl, 1)
        form.addRow("", detect_row)
        self.lp_key = QtWidgets.QLineEdit()
        self.lp_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lp_key.setPlaceholderText("lightpollutionmap.info API key (free, see ?)")
        self.lp_key.setText(QtCore.QSettings("NOUT", "NOUT").value("lp_api_key", "", type=str))
        key_help = QtWidgets.QToolButton(); key_help.setText("?")
        key_help.setToolTip("A free key (~500 requests/day) is given by the map's author.\n"
                            "Email starej@t-2.net (a small donation is appreciated).")
        key_help.clicked.connect(lambda: QtWidgets.QMessageBox.information(
            self, "API key",
            "Auto-detection uses lightpollutionmap.info, which needs a free API key.\n\n"
            "Request one by email from the site's author (Jurij Stare): starej@t-2.net "
            "(~500 requests/day; a small donation is appreciated).\n\n"
            "Paste the key here once and NOUT remembers it. Without a key you can still "
            "pick the Bortle class manually."))
        key_row = QtWidgets.QHBoxLayout()
        key_row.addWidget(self.lp_key, 1); key_row.addWidget(key_help)
        form.addRow("LP map key", key_row)
        form.addRow("Quantum efficiency", self.qe)
        form.addRow("Allowed noise increase", self.noise_pct)
        form.addRow("Target total integration", self.total)

        self.result = QtWidgets.QLabel(""); self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.RichText)
        self.result.setStyleSheet("padding:8px;")
        form.addRow(self.result)
        form.addRow(_hint("Estimate. The dominant factors are f-ratio, sky brightness "
                          "and read noise; the absolute value is approximate, the "
                          "relative guidance is reliable."))

        for wdg in (self.focal, self.fnum, self.pixel, self.qe, self.total):
            wdg.valueChanged.connect(self._compute)
        for cb in (self.iso, self.bortle, self.noise_pct):
            cb.currentIndexChanged.connect(self._compute)
        self._compute()

    def _detect_bortle(self):
        if self._lat is None or self._lon is None:
            self.detect_lbl.setText("No site set — set lat/lon in the Targets tab.")
            return
        key = self.lp_key.text().strip()
        if not key:
            self.detect_lbl.setText("API key required — click “?” to get one.")
            return
        QtCore.QSettings("NOUT", "NOUT").setValue("lp_api_key", key)
        self.detect_btn.setEnabled(False)
        self.detect_lbl.setText("Querying light-pollution map…")
        self._geo = GeoBortleWorker(self._lat, self._lon, key)
        self._geo.done.connect(self._on_bortle)
        self._geo.start()

    def _on_bortle(self, sqm, idx, err):
        self.detect_btn.setEnabled(True)
        if err or sqm is None:
            self.detect_lbl.setText("Couldn't detect ({}). Pick Bortle manually."
                                    .format((err or "no data")[:60]))
            return
        self.bortle.setCurrentIndex(int(idx))
        cls = self._BORTLE[int(idx)][0].split(" — ")[0]
        self.detect_lbl.setText("SQM ≈ {:.2f} → Bortle {}".format(sqm, cls))
        self._compute()

    def _compute(self, *args):
        import math
        focal = self.focal.value(); fnum = self.fnum.value(); px = self.pixel.value()
        qe = self.qe.value()
        rn = self._ISO_RN[self.iso.currentIndex()][1]
        skymag = self._BORTLE[self.bortle.currentIndex()][1]
        p = (0.01, 0.05, 0.10)[self.noise_pct.currentIndex()]
        factor = 1.0 / ((1.0 + p) ** 2 - 1.0)
        scale = 206.265 * px / focal                      # arcsec / px
        sky_e_s = self._S0 * (10 ** (-0.4 * skymag)) * (px ** 2) * qe / (fnum ** 2)
        if sky_e_s <= 0:
            self.result.setText("—"); return
        t_opt = factor * (rn ** 2) / sky_e_s
        subs = max(1, math.ceil(self.total.value() * 60.0 / t_opt))
        # round the suggestion to a friendly value
        if t_opt < 30:
            disp = "{:.0f} s".format(round(t_opt / 5) * 5 or 5)
        elif t_opt < 120:
            disp = "{:.0f} s".format(round(t_opt / 15) * 15)
        else:
            disp = "{:.1f} min".format(t_opt / 60.0)
        self.result.setText(
            "<b>Recommended sub-exposure ≈ {}</b><br>"
            "≈ {} subs for {:.0f} min total<br>"
            "<span style='color:#94a3b8'>pixel scale {:.2f} \"/px · sky rate "
            "≈ {:.2f} e⁻/s/px · read noise {:.1f} e⁻</span>".format(
                disp, subs, self.total.value(), scale, sky_e_s, rn))


class CometFetchWorker(QtCore.QThread):
    """Download the MPC comet orbital-element catalogue, compute each comet's current
    magnitude, and keep only those brighter than a limit (i.e. capturable)."""
    done = Signal(object, str)            # (rows or None, error)

    def __init__(self, mag_limit, out_path):
        super().__init__(); self.mag_limit = mag_limit; self.out_path = out_path

    def run(self):
        try:
            import urllib.request, gzip, json as _json
            from datetime import datetime
            import sky_map
            url = "https://www.minorplanetcenter.net/Extended_Files/cometels.json.gz"
            req = urllib.request.Request(url, headers={"User-Agent": "NOUT/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            try:
                txt = gzip.decompress(raw).decode("utf-8", "replace")
            except Exception:             # noqa: BLE001 - maybe already plain json
                txt = raw.decode("utf-8", "replace")
            data = _json.loads(txt)
            now = datetime.utcnow(); jd = sky_map._jd_utc(now)
            rows = []
            for c in data:
                try:
                    q = float(c["Perihelion_dist"]); e = float(c["e"])
                    i = float(c["i"]); node = float(c["Node"])
                    peri = float(c["Argument_of_perihelion"])
                    yy = int(c["Year_of_perihelion"]); mm = int(c["Month_of_perihelion"])
                    dd = float(c["Day_of_perihelion"])
                    tp = sky_map._jd_utc(datetime(yy, mm, max(1, int(dd)))) + (dd - int(dd))
                    H = c.get("H"); G = c.get("G")
                    if H in (None, ""):
                        continue
                    H = float(H); G = float(G) if G not in (None, "") else 4.0
                    name = (c.get("Designation_and_name") or "").strip()
                    el = {"q": q, "e": e, "i": i, "node": node, "peri": peri, "tp": tp}
                    _ra, _dec, r_au, delta = sky_map.comet_state(el, jd)
                    mag = sky_map.comet_apparent_mag(H, G, r_au, delta)
                    if mag is None or mag > self.mag_limit:
                        continue
                    rows.append((name, q, e, i, node, peri, yy, mm, dd, round(mag, 1)))
                except Exception:         # noqa: BLE001
                    continue
            rows.sort(key=lambda x: x[9])         # brightest first
            # write comets.csv
            import csv as _csv
            with open(self.out_path, "w", newline="") as f:
                wr = _csv.writer(f)
                wr.writerow(["name", "q_au", "e", "i_deg", "node_deg", "peri_deg", "tp", "mag"])
                for (name, q, e, i, node, peri, yy, mm, dd, mag) in rows:
                    tp_iso = "{:04d}-{:02d}-{:08.5f}".format(yy, mm, dd)
                    wr.writerow([name, q, e, i, node, peri, tp_iso, mag])
            self.done.emit(rows, "")
        except Exception as ex:           # noqa: BLE001
            self.done.emit(None, str(ex))


class TLEFetchWorker(QtCore.QThread):
    """Download a fresh TLE from Celestrak (needs internet)."""
    done = Signal(str, str)              # (tle_text, error)

    def __init__(self, catnr=25544):
        super().__init__(); self.catnr = catnr

    def run(self):
        try:
            import urllib.request
            url = ("https://celestrak.org/NORAD/elements/gp.php?CATNR={}&FORMAT=TLE"
                   .format(self.catnr))
            req = urllib.request.Request(url, headers={"User-Agent": "NOUT/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                txt = r.read().decode("utf-8", "replace").strip()
            if "1 " in txt and "2 " in txt:
                self.done.emit(txt, "")
            else:
                self.done.emit("", "Unexpected response from Celestrak.")
        except Exception as e:           # noqa: BLE001
            self.done.emit("", str(e))


_DEFAULT_ISS_TLE = (
    "ISS (ZARYA)\n"
    "1 25544U 98067A   24287.51782528  .00016717  00000-0  30074-3 0  9993\n"
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.50125623 12345")


class SatellitePassDialog(QtWidgets.QDialog):
    """Predict and list upcoming satellite passes (ISS by default) from the site."""
    def __init__(self, parent=None, lat=48.8566, lon=2.3522):
        super().__init__(parent)
        self.setWindowTitle("Satellite passes (ISS)")
        self.setMinimumWidth(640)
        self.lat = lat; self.lon = lon; self._fetch = None
        v = QtWidgets.QVBoxLayout(self)

        trow = QtWidgets.QHBoxLayout(); trow.addWidget(QtWidgets.QLabel("TLE"))
        trow.addWidget(_hint("Paste the TLE of any satellite, or fetch the latest ISS."))
        v.addLayout(trow)
        self.tle = QtWidgets.QPlainTextEdit()
        self.tle.setMaximumHeight(80)
        s = QtCore.QSettings("NOUT", "NOUT")
        self.tle.setPlainText(s.value("iss_tle", _DEFAULT_ISS_TLE, type=str))
        v.addWidget(self.tle)

        row = QtWidgets.QHBoxLayout()
        self.fetch_btn = QtWidgets.QPushButton("⟳ Fetch latest ISS TLE")
        self.fetch_btn.clicked.connect(self._do_fetch)
        row.addWidget(self.fetch_btn)
        row.addWidget(QtWidgets.QLabel("Days:"))
        self.days = QtWidgets.QSpinBox(); self.days.setRange(1, 10); self.days.setValue(3)
        row.addWidget(self.days)
        row.addWidget(QtWidgets.QLabel("Min alt:"))
        self.minalt = QtWidgets.QSpinBox(); self.minalt.setRange(5, 60); self.minalt.setValue(10)
        self.minalt.setSuffix("°"); row.addWidget(self.minalt)
        self.vis_only = QtWidgets.QCheckBox("Visible only"); self.vis_only.setChecked(True)
        self.vis_only.setToolTip("Only passes where the satellite is sunlit while your sky "
                                 "is dark (naked-eye visible).")
        row.addWidget(self.vis_only)
        self.go = QtWidgets.QPushButton("Compute"); self.go.clicked.connect(self._compute)
        row.addWidget(self.go)
        v.addLayout(row)

        self.msg = QtWidgets.QLabel(""); self.msg.setStyleSheet("color:#94a3b8;")
        v.addWidget(self.msg)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Date", "Rise (local)", "Max alt", "Max dir", "Set (local)", "Visible"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v.addWidget(self.table)
        self._compute()

    def _do_fetch(self):
        self.fetch_btn.setEnabled(False); self.msg.setText("Fetching ISS TLE from Celestrak…")
        self._fetch = TLEFetchWorker(25544)
        self._fetch.done.connect(self._on_fetch)
        self._fetch.start()

    def _on_fetch(self, txt, err):
        self.fetch_btn.setEnabled(True)
        if err or not txt:
            self.msg.setText("Fetch failed ({}). Paste a TLE manually.".format((err or "")[:60]))
            return
        self.tle.setPlainText(txt)
        QtCore.QSettings("NOUT", "NOUT").setValue("iss_tle", txt)
        self.msg.setText("TLE updated.")
        self._compute()

    def _compute(self):
        try:
            import satellites
        except Exception as e:            # noqa: BLE001
            self.msg.setText("satellites module unavailable: {}".format(e)); return
        tle = self.tle.toPlainText().strip()
        if not tle:
            self.msg.setText("Paste a TLE or fetch the ISS one."); return
        QtCore.QSettings("NOUT", "NOUT").setValue("iss_tle", tle)
        try:
            passes = satellites.predict_passes(
                tle, self.lat, self.lon, hours=self.days.value() * 24,
                min_alt=float(self.minalt.value()), visible_only=self.vis_only.isChecked())
        except Exception as e:            # noqa: BLE001
            self.msg.setText("Couldn't compute: {}".format(e)); return
        self.table.setRowCount(len(passes))
        # UTC -> local offset
        import time as _t
        off = -(_t.altzone if _t.daylight and _t.localtime().tm_isdst else _t.timezone)
        from datetime import timedelta
        for i, p in enumerate(passes):
            rl = p["rise"] + timedelta(seconds=off)
            sl = p["set"] + timedelta(seconds=off)
            vals = [rl.strftime("%a %d %b"),
                    "{:%H:%M:%S} {}".format(rl, p["rise_dir"]),
                    "{:.0f}°".format(p["max_alt"]), p["max_dir"],
                    "{:%H:%M:%S} {}".format(sl, p["set_dir"]),
                    "👁 yes" if p["visible"] else "—"]
            for j, txt in enumerate(vals):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()
        self.msg.setText("{} pass(es) in the next {} day(s).".format(
            len(passes), self.days.value())
            + ("" if passes else "  (try unchecking “Visible only” or more days.)"))


class FolderStackWorker(QtCore.QThread):
    """Stacks images already on disk (a folder) through the same pipeline as the live
    stacker: linear-light RAW decode, translation alignment, Welford mean with
    kappa-sigma + weak-frame rejection. Emits the running float mean so the UI shows
    the stack build up live. Lets you test the stacker or resume a crashed session."""
    progress = Signal(int, int, str)          # done, total, filename
    stack_ready = Signal(object, int, float)  # float32 mean, count, integration s
    done = Signal(object, object, int, int, bool)   # flat_mean, raw_mean, kept, rejected, gx_used
    failed = Signal(str)

    _RAW_EXT = (".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".rw2", ".orf", ".pef")
    _IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    def __init__(self, folder, linear=True, kappa=True, kappa_k=2.5, reject_weak=True,
                 use_graxpert=False, graxpert_exe=None, apply_flats=False, flats_dir=None,
                 apply_darks=False):
        super().__init__()
        self.folder = folder; self.linear = linear
        self.kappa = kappa; self.kappa_k = kappa_k; self.reject_weak = reject_weak
        self.use_graxpert = use_graxpert; self.graxpert_exe = graxpert_exe
        self.apply_flats = apply_flats; self.flats_dir = flats_dir
        self.apply_darks = apply_darks
        self._abort = False

    def stop(self):
        self._abort = True

    def _list_files(self):
        try:
            names = sorted(os.listdir(self.folder))
        except OSError:
            return []
        raws = [n for n in names if os.path.splitext(n)[1].lower() in self._RAW_EXT]
        pick = raws if raws else [n for n in names
                                  if os.path.splitext(n)[1].lower() in self._IMG_EXT]
        return [os.path.join(self.folder, n) for n in pick]

    @staticmethod
    def _exposure_s(path):
        try:
            import exifread
            with open(path, "rb") as f:
                tags = exifread.process_file(f, details=False,
                                             stop_tag="EXIF ExposureTime")
            v = tags.get("EXIF ExposureTime")
            if v is not None:
                r = v.values[0]
                return float(r.num) / float(r.den) if r.den else float(r.num)
        except Exception:               # noqa: BLE001
            pass
        return 0.0

    def run(self):
        files = self._list_files()
        if not files:
            self.failed.emit("No RAW or image files found in that folder.")
            return
        mean = M2 = None
        ref_gray = None
        count = rejected = 0
        integ = 0.0
        ref_stars = None; starhist = []
        total = len(files)
        master_flat = None
        master_dark = None
        if self.apply_darks and files:
            master_dark = load_master_dark_for(
                self.flats_dir, _read_iso_exif(files[0]), _read_exposure_exif(files[0]))
        if self.apply_flats and files:
            master_flat = load_master_flat_for(
                self.flats_dir, _read_lens_exif(files[0]),
                _read_focal_exif(files[0]), _read_fnumber_exif(files[0]))
        for i, path in enumerate(files):
            if self._abort or self.isInterruptionRequested():
                break
            self.progress.emit(i + 1, total, os.path.basename(path))
            frame = _decode_linear(path) if self.linear else None
            if frame is None:
                bgr = _decode_preview_image(path, full_demosaic=False)
                if bgr is None:
                    continue
                frame = bgr.astype(np.float32)
            if master_dark is not None:
                frame = apply_dark(frame, master_dark)          # subtract camera signal
            if master_flat is not None:
                frame = apply_flat(frame, master_flat)          # correct vignetting
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            try:
                cur_stars, nst = detect_stars(gray)
            except Exception:           # noqa: BLE001
                cur_stars, nst = [], None
            # weak-frame rejection (clouds / trails / lost tracking)
            if self.reject_weak and nst is not None:
                if ref_stars is None:
                    starhist.append(nst)
                    if len(starhist) >= 3:
                        ref_stars = float(np.median(starhist))
                if ref_stars and ref_stars > 6 and nst < 0.5 * ref_stars and count >= 1:
                    rejected += 1
                    continue
            if ref_gray is None:
                ref_gray = gray; reg_ref_xy = cur_stars
                reg_ref_med = [float(np.median(frame[:, :, c])) for c in range(frame.shape[2])]
                reg_ref_mad = [float(np.median(np.abs(frame[:, :, c] - reg_ref_med[c]))) + 1e-6
                               for c in range(frame.shape[2])]
                x = frame
            else:
                try:
                    M = register_affine(ref_gray, gray, reg_ref_xy, cur_stars)
                    x = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
                    x = normalize_to_ref(x, reg_ref_med, reg_ref_mad)
                except Exception:       # noqa: BLE001
                    x = frame
            if mean is None:
                mean = x.copy(); M2 = np.zeros_like(x); count = 1
            else:
                if self.kappa and count >= 4:
                    sigma = np.sqrt(M2 / count) + 1e-3
                    x = np.where(np.abs(x - mean) > self.kappa_k * sigma, mean, x)
                count += 1
                delta = x - mean
                mean += delta / count
                M2 += delta * (x - mean)
            integ += self._exposure_s(path)
            self.stack_ready.emit(mean.astype(np.float32), count, integ)
        if mean is None:
            self.failed.emit("Could not decode any file in that folder.")
            return
        # flatten the background (worker thread): GraXpert if enabled+available, else polynomial
        used_gx = False
        flat = None
        if self.use_graxpert and self.graxpert_exe:
            self.progress.emit(total, total, "GraXpert: extracting background…")
            gx = _graxpert_bg(mean.astype(np.float32), self.graxpert_exe, 0.2)
            if gx is not None:
                flat = np.clip(gx, 0.0, None); used_gx = True
        if flat is None:
            try:
                flat = remove_gradient(mean.astype(np.float32), 1.0)   # polynomial fallback
            except Exception:           # noqa: BLE001
                flat = mean.astype(np.float32)
        self.done.emit(flat.astype(np.float32), mean.astype(np.float32),
                       count, rejected, used_gx)


class GraXpertWorker(QtCore.QThread):
    """Runs GraXpert AI background extraction on an image off the UI thread, so it can
    be used on the LIVE stack (GraXpert takes several seconds). Emits the flattened
    float BGR, or None on failure."""
    done = Signal(object)

    def __init__(self, img_float, exe, smoothing=0.2):
        super().__init__()
        self._img = img_float; self._exe = exe; self._smoothing = smoothing

    def run(self):
        try:
            self.done.emit(_graxpert_bg(self._img, self._exe, self._smoothing))
        except Exception:               # noqa: BLE001
            self.done.emit(None)


class LibraryDialog(QtWidgets.QDialog):
    """Manage the persistent calibration library: list master flats and darks/bias, and
    delete the ones you no longer want."""
    def __init__(self, parent, flats_dir):
        super().__init__(parent)
        self.setWindowTitle("Calibration library")
        self.setMinimumSize(560, 460)
        self.flats_dir = flats_dir
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel("<b>Master flats</b> (lens · focal · aperture)"))
        self.flats_list = QtWidgets.QListWidget(); v.addWidget(self.flats_list, 1)
        fr = QtWidgets.QHBoxLayout()
        del_flat = QtWidgets.QPushButton("🗑 Delete selected flat")
        del_flat.clicked.connect(self._del_flat); fr.addWidget(del_flat); fr.addStretch(1)
        v.addLayout(fr)
        v.addWidget(QtWidgets.QLabel("<b>Master darks / bias</b> (ISO · exposure)"))
        self.darks_list = QtWidgets.QListWidget(); v.addWidget(self.darks_list, 1)
        dr = QtWidgets.QHBoxLayout()
        del_dark = QtWidgets.QPushButton("🗑 Delete selected dark")
        del_dark.clicked.connect(self._del_dark); dr.addWidget(del_dark); dr.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close"); close_btn.clicked.connect(self.accept)
        dr.addWidget(close_btn)
        v.addLayout(dr)
        self._refresh()

    def _refresh(self):
        self.flats_list.clear()
        for e in _load_flat_index(self.flats_dir):
            foc = "{:.0f}mm · ".format(float(e["focal"])) if e.get("focal") else ""
            it = QtWidgets.QListWidgetItem("{} · {}f/{:.1f}   ({} frames)".format(
                e.get("lens") or "any lens", foc, float(e.get("fnum", 0)), e.get("frames", 0)))
            it.setData(Qt.UserRole, e.get("file"))
            self.flats_list.addItem(it)
        self.darks_list.clear()
        for e in _load_darks_index(self.flats_dir):
            it = QtWidgets.QListWidgetItem("ISO {} · {:.4g}s   ({} frames)".format(
                e.get("iso"), float(e.get("exp", 0)), e.get("frames", 0)))
            it.setData(Qt.UserRole, e.get("file"))
            self.darks_list.addItem(it)

    def _delete(self, index_loader, index_saver, listw):
        item = listw.currentItem()
        if item is None:
            return
        fname = item.data(Qt.UserRole)
        if QtWidgets.QMessageBox.question(
                self, "Delete", "Delete this master frame permanently?") \
                != QtWidgets.QMessageBox.Yes:
            return
        idx = [e for e in index_loader(self.flats_dir) if e.get("file") != fname]
        index_saver(self.flats_dir, idx)
        try:
            os.remove(os.path.join(self.flats_dir, fname))
        except OSError:
            pass
        self._refresh()

    def _del_flat(self):
        self._delete(_load_flat_index, _save_flat_index, self.flats_list)

    def _del_dark(self):
        def _save_darks(d, idx):
            try:
                with open(_darks_index_path(d), "w") as f:
                    json.dump(idx, f, indent=2)
            except Exception:           # noqa: BLE001
                pass
        self._delete(_load_darks_index, _save_darks, self.darks_list)


class MasterFlatDialog(QtWidgets.QDialog):
    """Build a master flat for a (lens, focal, aperture) triplet and add it to the
    persistent library. On a zoom the focal length matters (vignetting changes), so it's
    selectable; on a prime it's fixed. Already-registered (focal, aperture) pairs for the
    selected lens are greyed out."""
    def __init__(self, parent, flats_dir, current_lens="", current_focal=None):
        super().__init__(parent)
        self.setWindowTitle("Create master flat")
        self.setMinimumWidth(480)
        self.flats_dir = flats_dir
        self.index = _load_flat_index(flats_dir)
        self._cur_focal = current_focal
        v = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        app_lenses = [k for k in LENS_PROFILES.keys() if k != "Custom"]
        idx_lenses = [e.get("lens", "") for e in self.index if e.get("lens")]
        lenses = list(dict.fromkeys(app_lenses + idx_lenses))
        self.lens = QtWidgets.QComboBox(); self.lens.setEditable(True)
        self.lens.addItems(lenses)
        self.lens.setCurrentText(current_lens if current_lens else (lenses[0] if lenses else ""))
        self.lens.setToolTip("Pick one of your lenses (from the app) or type another. The "
                             "name is matched to the lens EXIF for automatic flat application.")
        form.addRow("Lens", self.lens)
        self.focal = QtWidgets.QSpinBox(); self.focal.setRange(8, 2000); self.focal.setSuffix(" mm")
        self.focal.setToolTip("Focal length. On a zoom, vignetting changes with focal, so "
                              "build one flat per focal you use.")
        form.addRow("Focal length", self.focal)
        self.ap = QtWidgets.QComboBox()
        for a in _STD_APERTURES:
            self.ap.addItem("f/{:.1f}".format(a), a)
        self.ap.setCurrentIndex(_STD_APERTURES.index(2.8))
        form.addRow("Aperture", self.ap)
        v.addLayout(form)
        self.reg_lbl = QtWidgets.QLabel(""); self.reg_lbl.setWordWrap(True)
        self.reg_lbl.setStyleSheet("color:#94a3b8;")
        v.addWidget(self.reg_lbl)
        self.status = QtWidgets.QLabel(""); self.status.setStyleSheet("color:#cbd5e1;")
        v.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.build_btn = QtWidgets.QPushButton("📂 Select flat frames && build")
        self.build_btn.clicked.connect(self._build)
        row.addWidget(self.build_btn)
        close_btn = QtWidgets.QPushButton("Close"); close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        v.addLayout(row)
        self.lens.editTextChanged.connect(self._on_lens)
        self.focal.valueChanged.connect(self._refresh)
        self._on_lens()

    def _on_lens(self, *args):
        """Set the focal field from the lens profile: fixed for a prime, editable for a zoom."""
        prof = LENS_PROFILES.get(self.lens.currentText())
        if prof:
            fmin, fmax = prof.get("fmin"), prof.get("fmax")
            if fmin and fmax:
                self.focal.setRange(int(fmin), int(fmax))
                if fmin == fmax:                       # prime -> fixed focal
                    self.focal.setValue(int(fmin)); self.focal.setEnabled(False)
                else:                                  # zoom -> selectable
                    self.focal.setEnabled(True)
                    self.focal.setValue(int(self._cur_focal) if self._cur_focal
                                        else int(fmin))
                self._refresh(); return
        self.focal.setRange(8, 2000); self.focal.setEnabled(True)
        if self._cur_focal:
            self.focal.setValue(int(self._cur_focal))
        self._refresh()

    def _registered(self, lens):
        """{(focal_or_None, aperture)} already saved for this lens."""
        ln = _lens_norm(lens)
        out = set()
        for e in self.index:
            if _lens_norm(e.get("lens")) == ln:
                ef = e.get("focal")
                out.add((round(float(ef)) if ef else None, round(float(e["fnum"]), 1)))
        return out

    def _refresh(self, *args):
        regs = self._registered(self.lens.currentText())
        foc = int(self.focal.value())
        model = self.ap.model()
        for i in range(self.ap.count()):
            a = float(self.ap.itemData(i))
            item = model.item(i)
            already = (foc, round(a, 1)) in regs or (None, round(a, 1)) in regs
            if item is not None:
                item.setEnabled(not already)
                item.setText("f/{:.1f}".format(a) + ("   ✓ saved" if already else ""))
        if self.index:
            self.reg_lbl.setText("Library: " + " · ".join(
                "{} {}f/{:.1f}".format(
                    e.get("lens") or "any",
                    "{:.0f}mm ".format(float(e["focal"])) if e.get("focal") else "",
                    float(e["fnum"])) for e in self.index))
        else:
            self.reg_lbl.setText("Library empty. Build your first master flat below.")

    def _build(self):
        lens = self.lens.currentText().strip()
        a = float(self.ap.currentData())
        foc = int(self.focal.value())
        if (foc, round(a, 1)) in self._registered(lens):
            QtWidgets.QMessageBox.information(
                self, "Already in library",
                "A master flat already exists for {} {}mm f/{:.1f}.".format(lens or "this lens", foc, a))
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder of flat frames (evenly lit) for {} {}mm f/{:.1f}".format(
                lens or "this lens", foc, a))
        if not folder:
            return
        exts = (".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".rw2", ".orf", ".pef",
                ".jpg", ".jpeg", ".png", ".tif", ".tiff")
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(exts)]
        if not paths:
            self.status.setText("No image files in that folder."); return
        if not lens:
            lens = _read_lens_exif(paths[0])
            if lens:
                self.lens.setEditText(lens)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status.setText("Building master flat from {} frames…".format(len(paths)))
        QtWidgets.QApplication.processEvents()
        try:
            master, _exif_fn = build_master_flat(paths)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if master is None:
            self.status.setText("Could not decode the flat frames."); return
        save_master_flat(self.flats_dir, master, lens, foc, a, len(paths))
        self.index = _load_flat_index(self.flats_dir)
        self._refresh()
        self.status.setText("✓ Saved master flat for {} {}mm f/{:.1f} ({} frames).".format(
            lens or "any lens", foc, a, len(paths)))


class DarkDialog(QtWidgets.QDialog):
    """Build a master dark/bias for an (ISO, exposure) pair and add it to the persistent
    library. A dark only depends on the camera (ISO + exposure time), not the lens.
    A bias is just a very-short-exposure dark. Already-registered pairs are greyed out."""
    _ISOS = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    _EXPS = [("Bias (1/8000 s)", 0.000125), ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0),
             ("10 s", 10.0), ("15 s", 15.0), ("20 s", 20.0), ("30 s", 30.0), ("60 s", 60.0)]

    def __init__(self, parent, flats_dir, current_iso=None, current_exp=None):
        super().__init__(parent)
        self.setWindowTitle("Create master dark / bias")
        self.setMinimumWidth(460)
        self.flats_dir = flats_dir
        self.index = _load_darks_index(flats_dir)
        v = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.iso = QtWidgets.QComboBox()
        for i in self._ISOS:
            self.iso.addItem("ISO {}".format(i), i)
        if current_iso in self._ISOS:
            self.iso.setCurrentIndex(self._ISOS.index(current_iso))
        form.addRow("ISO", self.iso)
        self.exp = QtWidgets.QComboBox()
        for label, val in self._EXPS:
            self.exp.addItem(label, val)
        form.addRow("Exposure", self.exp)
        v.addLayout(form)
        self.reg_lbl = QtWidgets.QLabel(""); self.reg_lbl.setWordWrap(True)
        self.reg_lbl.setStyleSheet("color:#94a3b8;")
        v.addWidget(self.reg_lbl)
        self.status = QtWidgets.QLabel(""); self.status.setStyleSheet("color:#cbd5e1;")
        v.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.build_btn = QtWidgets.QPushButton("📂 Select dark/bias frames && build")
        self.build_btn.clicked.connect(self._build)
        row.addWidget(self.build_btn)
        close_btn = QtWidgets.QPushButton("Close"); close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        v.addLayout(row)
        self.iso.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def _registered(self, iso):
        return {round(float(e.get("exp", 0)), 3) for e in self.index
                if int(e.get("iso", 0)) == int(iso)}

    def _refresh(self, *args):
        regs = self._registered(self.iso.currentData())
        model = self.exp.model()
        for i in range(self.exp.count()):
            val = float(self.exp.itemData(i))
            item = model.item(i)
            already = round(val, 3) in regs
            if item is not None:
                item.setEnabled(not already)
                base = self._EXPS[i][0]
                item.setText(base + ("   ✓ saved" if already else ""))
        if self.index:
            self.reg_lbl.setText("Library: " + " · ".join(
                "ISO {} / {:.4g}s".format(e.get("iso"), float(e.get("exp", 0)))
                for e in self.index))
        else:
            self.reg_lbl.setText("Library empty. Build your first master dark/bias below.")

    def _build(self):
        iso = int(self.iso.currentData())
        exp = float(self.exp.currentData())
        if round(exp, 3) in self._registered(iso):
            QtWidgets.QMessageBox.information(
                self, "Already in library",
                "A master dark already exists for ISO {} / {:.4g}s.".format(iso, exp))
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder of dark/bias frames (lens capped) for ISO {} / {:.4g}s".format(iso, exp))
        if not folder:
            return
        exts = (".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".rw2", ".orf", ".pef",
                ".jpg", ".jpeg", ".png", ".tif", ".tiff")
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(exts)]
        if not paths:
            self.status.setText("No image files in that folder."); return
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status.setText("Building master dark from {} frames…".format(len(paths)))
        QtWidgets.QApplication.processEvents()
        try:
            master, exif_iso, exif_exp = build_master_dark(paths)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if master is None:
            self.status.setText("Could not decode the dark frames."); return
        save_master_dark(self.flats_dir, master, iso, exp, len(paths))
        self.index = _load_darks_index(self.flats_dir)
        self._refresh()
        self.status.setText("✓ Saved master dark for ISO {} / {:.4g}s ({} frames).".format(
            iso, exp, len(paths)))


class MainWindow(QtWidgets.QMainWindow):
    HIST = 300  # number of points kept in the curve
    _LIGHT_HIPS = "CDS/P/Mellinger/color"   # whole-sky optical, used for the light backdrop
    _LIGHT_LEVELS = (40.0, 30.0, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0)   # coarse→fine, ~2.5–3 GB total
    _SURVEYS = [("None", ""),
                ("DSS2 color", "CDS/P/DSS2/color"),
                ("DSS2 red", "CDS/P/DSS2/red"),
                ("DSS2 blue", "CDS/P/DSS2/blue"),
                ("2MASS color", "CDS/P/2MASS/color"),
                ("AllWISE color", "CDS/P/allWISE/color"),
                ("Mellinger (optical)", "CDS/P/Mellinger/color"),
                ("Fermi (gamma)", "CDS/P/Fermi/color")]

    def __init__(self, worker, save_dir, roi_frac):
        super().__init__()
        self.worker = worker
        self.roi_frac = roi_frac
        self.save_dir = save_dir
        self.hist = deque(maxlen=self.HIST)
        self.session_peak = 0.0
        self.review_mode = False
        self._last_capture_bgr = None
        self._stack_float = None        # float (linear) live-stack mean, for full-precision stretch
        self._folder_stacking = False   # True while stacking a folder from disk
        self._iv_running = False         # True while an intervalometer burst is running
        self._gx_busy = False            # GraXpert live job running
        self._gx_flat = None             # cached GraXpert-flattened live-stack float
        self._gx_done_id = None
        self._gx_pending_id = None
        self._graxpert_exe = _find_graxpert()    # resolved once; GraXpert used when present
        self._last_capture_path = None
        self._solve_wcs = None          # (ra, dec, rot, fov_w) from the last plate-solve
        # planetary / lucky-imaging mode
        self._plan_active = False
        self._plan_mean = None          # running mean of kept aligned crops (float32)
        self._plan_count = 0            # frames kept
        self._plan_total = 0            # frames seen
        self._plan_scores = deque(maxlen=200)   # recent sharpness scores (percentile gate)
        self._plan_keep_pct = 30        # keep best X%
        self._plan_roi = 0              # crop size (px) around the object; 0 = full frame
        self._plan_ref = None           # reference gray for sub-pixel phase-correlation align
        self._plan_hann = None          # Hanning window for phase correlation
        self._plan_sharpen_amt = 0.6    # multi-scale sharpening strength
        self._plan_readout_lbl = None
        self._plan_dlg = None
        self._plan_recording = False    # NOUT is recording the live-view stream to a file
        self._plan_rec_writer = None
        self._plan_rec_path = None
        self._autocenter = {"active": False, "iter": 0, "phase": None}
        self._ghost_img = None
        self._track_x = deque(maxlen=500)
        self._track_y = deque(maxlen=500)
        self._seq_rows = {}          # seq_id -> table row
        self._seq_counts = {}        # seq_id -> num shots
        self._seq_integs = {}        # seq_id -> integration (s)
        self.night_mode = False
        self._night = False
        self._await_done = False
        self._cam_aper_choices = []
        self._track_ref_gray = None
        self._track_ref_stars = None
        self._track_warned = False
        self._track_popup = None
        self._drift_active = False
        self._drift_ref_gray = None
        self._star_hist = []          # recent star counts (transparency)
        self._cloud_warned = False
        self._sb = {"integ": "", "hfr": "", "batt": ""}

        self.setWindowTitle("NOUT")
        _logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nout_1024.png")
        if os.path.exists(_logo_path):
            self.setWindowIcon(QtGui.QIcon(_logo_path))
        self.resize(1240, 820)
        self.setMinimumSize(840, 560)
        # settings needed by the UI (night mode + notifications live in the top bar)
        _s = QtCore.QSettings("SonyTether", "SonyTether")
        self._night = _s.value("night", False, type=bool)
        self.chk_notify = QtWidgets.QCheckBox("🔔")
        self.chk_notify.setToolTip("macOS notification (with sound) at end of burst, "
                                   "low battery/disk, or degraded tracking.")
        self.chk_notify.setChecked(_s.value("notify", True, type=bool))
        self.awake_mode = QtWidgets.QComboBox()
        self.awake_mode.addItem("☕ Awake: always", "always")
        self.awake_mode.addItem("☕ Awake: during capture", "capture")
        self.awake_mode.addItem("☕ Awake: off", "off")
        self.awake_mode.setToolTip("Keep the machine awake while capturing "
                                   "(caffeinate, systemd-inhibit, or the Windows "
                                   "execution-state flags).")
        _i = self.awake_mode.findData(_s.value("keep_awake_mode", "always", type=str))
        self.awake_mode.setCurrentIndex(max(0, _i))
        self._caffeinate = None
        self.awake_mode.currentIndexChanged.connect(self._apply_awake_mode)
        self._build_ui()

        worker.frame_ready.connect(self.on_frame)
        worker.capture_saved.connect(self.on_saved)
        worker.capture_image_ready.connect(self.on_capture_image)
        worker.interval_running.connect(self.on_interval_state)
        worker.interval_paused.connect(self.on_interval_paused)
        worker.interval_progress.connect(self.on_interval_progress)
        worker.interval_progress.connect(self._plan_burst_progress)
        worker.cam_status.connect(self.on_cam_status)
        worker.stack_ready.connect(self.on_stack_ready)
        worker.track_point.connect(self.on_track_point)
        worker.seq_started.connect(self.on_seq_started)
        worker.seq_progress.connect(self.on_seq_progress)
        worker.seq_ended.connect(self.on_seq_ended)
        worker.seq_result.connect(self.on_seq_result)
        worker.config_loaded.connect(self.on_config)
        worker.status.connect(lambda m: self.statusBar().showMessage(m, 6000))
        worker.mount_status.connect(self._on_mount_status)
        worker.dither_point.connect(self._on_dither_point)
        worker.mount_goto_done.connect(self._ac_after_slew)
        worker.mount_axis1_result.connect(self._on_axis1_result)
        worker.mount_axis1_progress.connect(self._on_axis1_progress)
        worker.mount_track_state.connect(self._on_track_state)
        # the probe lives here, not in the GoTo dialog: closing that window must not
        # blind you to a mount that has quietly stopped
        self._track_timer = QtCore.QTimer(self)
        self._track_timer.setInterval(2000)
        self._track_timer.timeout.connect(lambda: self.worker.post("mount_track_probe"))
        self._track_timer.start()
        worker.movie_ready.connect(self._on_movie_ready)
        self._dither_hist = []                            # (shot, RA offset arcsec) trail
        self._ac_active = False       # auto-center-on-goal loop running
        self._ac_iter = 0; self._ac_max = 6; self._ac_phase = None
        self._ac_ip = "192.168.4.1"; self._ac_tol_deg = 0.12
        worker.failed.connect(self.on_failed)

        # --- 1. Barre de statut (sans doublons) ---
        s = QtCore.QSettings("SonyTether", "SonyTether")

        self.status_perm = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.status_perm)

        # --- 2. Raccourcis clavier : limités à la vue image (jamais dans un champ) ---
        # Whole window, so they work without clicking the picture first. Text fields
        # still take their keys: Qt gives a focused QLineEdit/spin box first claim.
        ctx = Qt.ShortcutContext.WindowShortcut
        for keyseq, slot in [("Space", self._single_capture),
                             ("S", self._toggle_interval),
                             ("F", lambda: self.view.fit()),
                             ("T", self._toggle_stretch_shortcut),
                             ("P", self._toggle_fullscreen_preview),
                             ("V", self._toggle_public_view),
                             ("X", self._mount_stop_now),
                             ("Escape", self._exit_fullscreen_preview)]:
            sc = QtGui.QShortcut(QtGui.QKeySequence(keyseq), self, activated=slot)
            sc.setContext(ctx)
        self._fs_preview = False
        self._hud_info = {}
        self._sky_timer = QtCore.QTimer(self); self._sky_timer.setInterval(120000)
        self._sky_timer.timeout.connect(self._update_sky_window); self._sky_timer.start()


        # --- 3. Chargement des paramètres ---
        self._load_settings()      # restore settings from previous session
        self._apply_awake_mode()   # start keeping the Mac awake per the chosen mode
        self._apply_lens(self.lens_combo.currentText())   # focal/aperture based on lens
        self._update_mosaic()
        self._update_tfov()
    # -- settings persistence ------------------------------------------
    def _settings(self):
        return QtCore.QSettings("SonyTether", "SonyTether")

    def _save_settings(self):
        s = self._settings()
        vals = {
            "dir": self.dir_edit.text(), "type": self.type_combo.currentText(),
            "exp_mode": self.exp_mode.currentIndex(), "bulb_secs": self.bulb_secs.value(),
            "iv_interval": self.iv_interval.value(), "iv_count": self.iv_count.value(),
            "unlimited": self.chk_unlimited.isChecked(), "metric": self.metric_cb.currentText(),
            "roi": self.roi_slider.value(), "reticle_alpha": self.ret_slider.value(),
            "obs_look": self.chk_obs.isChecked(),
            "live_source": (self.src_combo.currentData() or (None, ""))[1] or "",
            "live": self.chk_live.isChecked(),
            "metric_on": self.chk_metric.isChecked(), "stars": self.chk_stars.isChecked(),
            "stretch": self.chk_stretch.isChecked(),
            "cull": self.chk_cull.isChecked(), "stack": self.chk_stack.isChecked(),
            "autostop": self.chk_autostop_pd.isChecked(), "batt_thr": self.batt_thr.value(),
            "disk_thr": self.disk_thr.value(), "ghost_alpha": self.ghost_alpha.value(),
            "ghost_dx": self.ghost_dx.value(), "ghost_dy": self.ghost_dy.value(),
            "kappa": self.chk_kappa.isChecked(), "kappa_v": self.kappa_val.value(),
            "dawn_stop": self.chk_dawn.isChecked(),
            "frame_reject": self.chk_frame_reject.isChecked(),
            "integ_on": self.chk_integ_target.isChecked(), "integ_v": self.integ_target.value(),
            "solve_focal": self.solve_focal.value(), "solve_mode": self.solve_mode.currentText(),
            "solve_key": self.solve_key.text(),
            "night": getattr(self, "_night", False),
            "t_lat": self.t_lat.value(), "t_lon": self.t_lon.value(),
            "t_focal": self.t_focal.value(), "t_minalt": self.t_minalt.value(),
            "lens": self.lens_combo.currentText(), "lens_focal": self.lens_focal.value(),
            "track_on": self.chk_track.isChecked(), "track_thr": self.track_thresh.value(),
            "track_tol": self.track_tol.value(),
            "notify": self.chk_notify.isChecked(),
            "mos_rows": self.mos_rows.value(),
            "mos_cols": self.mos_cols.value()
        }
        for k, v in vals.items():
            s.setValue(k, v)
        # Save Mosaic paths
        str_paths = {f"{r},{c}": p for (r, c), p in self._mosaic_paths.items()}
        s.setValue("mosaic_paths", json.dumps(str_paths))

    def _load_settings(self):
        s = self._settings()
        if not s.contains("dir"):
            return                       # first use: keep defaults
        self.dir_edit.setText(s.value("dir", self.dir_edit.text(), type=str))
        self.type_combo.setCurrentText(s.value("type", "lights", type=str))
        self.exp_mode.setCurrentIndex(s.value("exp_mode", 0, type=int))
        self.bulb_secs.setValue(s.value("bulb_secs", 60.0, type=float))
        self.iv_interval.setValue(s.value("iv_interval", 2.0, type=float))
        self.iv_count.setValue(s.value("iv_count", 20, type=int))
        self.chk_unlimited.setChecked(s.value("unlimited", False, type=bool))
        self.metric_cb.setCurrentText(s.value("metric", "tenengrad", type=str))
        self.roi_slider.setValue(s.value("roi", int(self.roi_frac * 100), type=int))
        self.ret_slider.setValue(s.value("reticle_alpha", 85, type=int))
        self.chk_obs.setChecked(s.value("obs_look", True, type=bool))
        self._on_reticle_alpha(self.ret_slider.value())   # slider may not have moved
        # Come back on the HDMI card if it was in use and is plugged in again; never
        # switch to the Mac's own webcam on our own.
        _src = s.value("live_source", "", type=str)
        if _src and not _is_builtin_camera(_src):
            QtCore.QTimer.singleShot(1500, lambda n=_src: self._refresh_video_sources(restore=n))
        else:
            QtCore.QTimer.singleShot(1500, lambda: self._refresh_video_sources(restore=""))
        self.chk_live.setChecked(s.value("live", True, type=bool))
        self.chk_metric.setChecked(False)          # removed from the UI
        self.chk_stars.setChecked(False)           # removed from the UI
        self.chk_stretch.setChecked(s.value("stretch", False, type=bool))
        self.chk_cull.setChecked(s.value("cull", False, type=bool))
        self.chk_stack.setChecked(s.value("stack", False, type=bool))
        self.chk_autostop_pd.setChecked(s.value("autostop", False, type=bool))
        self.batt_thr.setValue(s.value("batt_thr", 15, type=int))
        self.disk_thr.setValue(s.value("disk_thr", 2.0, type=float))
        self.ghost_alpha.setValue(s.value("ghost_alpha", 40, type=int))
        self.ghost_dx.setValue(s.value("ghost_dx", 0, type=int))
        self.ghost_dy.setValue(s.value("ghost_dy", 0, type=int))
        self.chk_kappa.setChecked(s.value("kappa", False, type=bool))
        self.chk_dawn.setChecked(False)            # retired; never restored
        self.chk_frame_reject.setChecked(s.value("frame_reject", False, type=bool))
        self._push_site()
        self.kappa_val.setValue(s.value("kappa_v", 2.5, type=float))
        self.chk_integ_target.setChecked(s.value("integ_on", False, type=bool))
        self.integ_target.setValue(s.value("integ_v", 60.0, type=float))
        self.solve_focal.setValue(s.value("solve_focal", 135.0, type=float))
        self.solve_mode.setCurrentText(s.value("solve_mode", "Online", type=str))
        self.solve_key.setText(s.value("solve_key", "", type=str))
        self.t_lat.setValue(s.value("t_lat", 48.8566, type=float))
        self.t_lon.setValue(s.value("t_lon", 2.3522, type=float))
        self.t_focal.setValue(s.value("t_focal", 135.0, type=float))
        self.t_minalt.setValue(s.value("t_minalt", 30.0, type=float))
        self.lens_combo.setCurrentText(s.value("lens", "Sony FE 50 mm F1.8", type=str))
        self.lens_focal.setValue(s.value("lens_focal", 50.0, type=float))
        self.chk_track.setChecked(s.value("track_on", True, type=bool))
        self.track_thresh.setValue(s.value("track_thr", 70, type=int))
        self.track_tol.setValue(s.value("track_tol", 4.0, type=float))

        # Load Mosaic
        self.mos_rows.setValue(s.value("mos_rows", 2, type=int))
        self.mos_cols.setValue(s.value("mos_cols", 2, type=int))
        self._reset_mosaic(load_paths=s.value("mosaic_paths", "{}", type=str))

    # -- UI construction ----------------------------------------------
    def _check_updates(self):
        """One-click update: git-pull the app folder if it's a clone, else point to GitHub."""
        import subprocess
        appdir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(os.path.join(appdir, ".git")):
            QtWidgets.QMessageBox.information(
                self, "Update",
                "This copy isn't a git clone.\n\nTo enable one-click updates, install NOUT by "
                "cloning it from GitHub. Otherwise download the latest release and re-run "
                "install.command.")
            return
        try:
            out = subprocess.run(["git", "-C", appdir, "pull", "--ff-only"],
                                 capture_output=True, text=True, timeout=60)
        except Exception as e:          # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Update", "Update failed: {}".format(e))
            return
        msg = (out.stdout or "") + (out.stderr or "")
        if "Already up to date" in msg or "Already up-to-date" in msg:
            QtWidgets.QMessageBox.information(self, "Update", "NOUT is already up to date.")
        elif out.returncode == 0:
            QtWidgets.QMessageBox.information(
                self, "Update",
                "Updated to the latest version.\n\nQuit and relaunch NOUT to apply "
                "(and run update.command once if new packages were added).")
        else:
            QtWidgets.QMessageBox.warning(
                self, "Update", "Could not update automatically:\n\n{}".format(msg.strip()))

    def _build_ui(self):
        help_menu = self.menuBar().addMenu("Help")
        act_upd = help_menu.addAction("Check for updates…")
        act_upd.triggered.connect(self._check_updates)
        act_about = help_menu.addAction("About NOUT")
        act_about.triggered.connect(lambda: QtWidgets.QMessageBox.information(
            self, "NOUT", "NOUT — tethered capture + live astro stacking.\n"
            "Update: Help → Check for updates, or run update.command."))
        self.tabs = QtWidgets.QTabWidget()
        central = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(central); cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(0)
        self.top_target_bar = _TopBar(); self.top_target_bar.setObjectName("topbar")
        tb = QtWidgets.QHBoxLayout(self.top_target_bar); tb.setContentsMargins(12, 6, 12, 6)
        # app title in the true middle of the bar: Nout arching over her name
        self.nut_banner = _NutBanner()
        self.top_target_bar.setMinimumHeight(56)
        self.top_target_bar.set_center(self.nut_banner)
        self.top_target_label = QtWidgets.QLabel("🎯  No target selected")
        self.top_target_label.setObjectName("topTarget")
        tb.addWidget(self.top_target_label)
        self.top_target_sub = QtWidgets.QLabel(""); self.top_target_sub.setObjectName("topTargetSub")
        tb.addWidget(self.top_target_sub); tb.addStretch(1)
        self.top_target_bar.left_widgets = [self.top_target_label, self.top_target_sub]
        tb.addSpacing(10)
        self.mount_btn = QtWidgets.QPushButton("🔭 Mount")
        self.mount_btn.setToolTip("Everything about the mount in one window: the two dial "
                                  "readings for the target, the once-a-session setup, and "
                                  "the control itself — connection, tracking, slews, "
                                  "dithering. (Ctrl+G)")
        self.mount_btn.setShortcut("Ctrl+G")
        self.mount_btn.clicked.connect(lambda: self._open_manual_goto("point"))
        tb.addWidget(self.mount_btn)
        self.top_target_bar.right_widget = self.mount_btn
        self.goto_btn = self.mount_btn          # older call sites still work
        self.planet_btn = QtWidgets.QPushButton("🪐 Planetary")
        self.planet_btn.setToolTip("Lucky imaging for the Moon & planets: keep the sharpest "
                                   "live-view frames and stack them.")
        self.planet_btn.clicked.connect(self._open_planetary_dialog)
        tb.addWidget(self.planet_btn)
        self.track_pill = ClickableLabel("⚪  mount off")
        self.track_pill.setCursor(Qt.PointingHandCursor)
        self.track_pill.setToolTip(
            "What the mount is actually doing, measured from its step counter every "
            "two seconds — not a repeat of what was commanded.\n"
            "Being on the mount's Wi-Fi is not the same as NOUT having opened it: "
            "click here to connect.")
        self.track_pill.clicked.connect(self._track_pill_clicked)
        self.track_pill.setContextMenuPolicy(Qt.CustomContextMenu)
        self.track_pill.customContextMenuRequested.connect(self._track_pill_menu)
        self.track_pill.setStyleSheet("color:#94a3b8; font-weight:bold; padding:0 8px;")
        tb.addWidget(self.track_pill)
        self.night_btn = QtWidgets.QPushButton(); self.night_btn.setCheckable(True)
        self.night_btn.setFixedWidth(40); self.night_btn.setToolTip(
            "Night-vision mode (deep red, preserves dark adaptation)")
        self.night_btn.toggled.connect(self._toggle_night)
        tb.addWidget(self.night_btn)
        # Notifications and keep-awake live in one ⚙ menu instead of two widgets.
        # The original controls stay (hidden): they hold the state and its handlers.
        self.chk_notify.setParent(self.top_target_bar); self.chk_notify.hide()
        self.awake_mode.setParent(self.top_target_bar); self.awake_mode.hide()
        self.prefs_btn = QtWidgets.QToolButton(); self.prefs_btn.setObjectName("gear")
        self.prefs_btn.setIconSize(QtCore.QSize(16, 16))
        self.prefs_btn.setToolTip("Notifications · keep the Mac awake")
        self.prefs_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        pm = QtWidgets.QMenu(self.prefs_btn)
        a_notif = pm.addAction("Notifications"); a_notif.setCheckable(True)
        a_notif.setChecked(self.chk_notify.isChecked())
        a_notif.toggled.connect(self.chk_notify.setChecked)
        self.chk_notify.toggled.connect(a_notif.setChecked)
        pm.addSeparator()
        sub = pm.addMenu("Keep Mac awake")
        grp = QtGui.QActionGroup(sub); grp.setExclusive(True)
        for i in range(self.awake_mode.count()):
            a = sub.addAction(self.awake_mode.itemText(i).replace("☕ Awake: ", "").capitalize())
            a.setCheckable(True); a.setChecked(i == self.awake_mode.currentIndex())
            grp.addAction(a)
            a.triggered.connect(lambda _c=False, k=i: self.awake_mode.setCurrentIndex(k))
        self.awake_mode.currentIndexChanged.connect(
            lambda k: grp.actions()[k].setChecked(True))
        self.prefs_btn.setMenu(pm)
        tb.addWidget(self.prefs_btn)
        cv.addWidget(self.top_target_bar)
        cv.addWidget(self.tabs)
        self.setCentralWidget(central)
        _shoot = self._build_shoot_tab()
        _results = self._build_results_tab()
        _targets = self._build_targets_tab()
        _skymap = self._build_skymap_tab()
        _mosaic = self._build_mosaic_tab()
        _gallery = self._build_gallery_tab()
        self.tabs.addTab(_shoot, "Shooting")
        self.tabs.addTab(_mosaic, "Mosaic View")
        self.tabs.addTab(_results, "Results")
        self.tabs.addTab(_targets, "Targets")
        self.tabs.addTab(_skymap, "Sky Map")
        self.tabs.addTab(_gallery, "Gallery")
        self._shoot_index, self._mosaic_index, self._results_index, \
            self._targets_index, self._skymap_index, self._gallery_index = 0, 1, 2, 3, 4, 5
        self._tab_icon_names = ["camera", "grid", "image", "target", "map", "gallery"]
        night0 = getattr(self, "_night", False)
        if hasattr(self, "night_btn"):
            self.night_btn.blockSignals(True)
            self.night_btn.setChecked(night0)
            self.night_btn.blockSignals(False)
        self._apply_theme()                 # Deep Space (or Night) theme + icons
        # intervalometer countdown
        self._eta_deadline = None
        self._elapsed_start = None
        self._last_progress = (0, 0)
        self._prog_timer = QtCore.QTimer(self)
        self._prog_timer.setInterval(1000)
        self._prog_timer.timeout.connect(self._tick_progress)
        self._iss_timer = QtCore.QTimer(self)     # live ISS position (~1°/4s)
        self._iss_timer.setInterval(2000)
        self._iss_timer.timeout.connect(self._tick_iss)
        self.statusBar().showMessage("Starting…")

    def _apply_theme(self):
        pal = _THEME_EVA_NIGHT if getattr(self, "_night", False) else _THEME_EVA
        self.setStyleSheet(_theme_qss(pal))
        # tab icons in the accent colour
        if hasattr(self, "tabs"):
            for i, nm in enumerate(getattr(self, "_tab_icon_names", [])):
                self.tabs.setTabIcon(i, _make_icon(nm, pal["accent"]))
            self.tabs.setIconSize(QtCore.QSize(18, 18))
        # icons on key buttons (name -> icon)
        btn_icons = [("shot_btn", "capture"), ("iv_start", "capture"), ("t_btn", "calc"),
                     ("t_locate", "location"), ("t_search_btn", "search"),
                     ("solve_btn", "target"), ("solve_res_btn", "target"),
                     ("sky_img_btn", "image")]
        for attr, icon in btn_icons:
            b = getattr(self, attr, None)
            if isinstance(b, QtWidgets.QAbstractButton):
                b.setIcon(_make_icon(icon, pal["text"]))
                # the drawn icon replaces any emoji the label started with — two
                # pictograms side by side read as clutter and crowd the text
                b.setText(" " + re.sub(r"^[^\w(\[]+\s*", "", b.text()).strip())
        for gb in self.findChildren(QtWidgets.QToolButton, "gear"):
            nm = ("fullscreen" if gb is getattr(self, "fs_btn", None) else
                  "public" if gb is getattr(self, "pub_btn", None) else "gear")
            gb.setIcon(_make_icon(nm, pal["text"]))
        if getattr(self, "nut_banner", None) is not None:
            self.nut_banner.set_palette(pal, night=getattr(self, "_night", False))
            self.top_target_bar.set_palette(pal, night=getattr(self, "_night", False))
        if hasattr(self, "shot_btn"):
            self.shot_btn.setIcon(_make_icon("capture", pal["onaccent"]))
        # night toggle glyph
        if hasattr(self, "night_btn"):
            self.night_btn.setIcon(_make_icon("moon", pal["accent"]))
        # the sky map repaints with its own palette; nudge it
        if hasattr(self, "view") and hasattr(self.view, "set_hud_color"):
            self.view.set_hud_color(pal["visor"])
            self.view.set_reticle_color(pal.get("reticle", "#ff3b30"))
            sl = getattr(self, "ret_slider", None)
            if sl is not None:
                self.view.set_reticle_opacity(sl.value() / 100.0)
        if getattr(self, "skymap", None):
            self.skymap.update()

    def _toggle_night(self, on):
        on = bool(on)
        self._night = on
        self.night_mode = on              # also drives the red-channel filter on the live view
        for w in (getattr(self, "chk_night", None), getattr(self, "night_btn", None)):
            if w is not None and w.isChecked() != on:
                w.blockSignals(True); w.setChecked(on); w.blockSignals(False)
        self._apply_theme()

    def _build_shoot_tab(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(tab)
        root.setContentsMargins(8, 8, 10, 8); root.setSpacing(10)

        # --- left column: live view (zoomable) + (stars) + curve ---
        left = QtWidgets.QVBoxLayout(); left.setSpacing(6)
        self.view = ZoomableView(hud=True)
        self.view.setObjectName("visor")               # helmet-visor frame (EVA theme)
        self.rec_label = QtWidgets.QLabel("● REC — sequence in progress")
        self.rec_label.setAlignment(Qt.AlignCenter)
        self.rec_label.setStyleSheet("color:#fff;background:#b91c1c;font-weight:bold;padding:3px;")
        self.rec_label.setVisible(False)
        left.addWidget(self.rec_label)
        self.view_box = _AspectBox(self.view, 1.5)          # 3:2, like the sensor
        self.view.aspect_changed.connect(self.view_box.set_ratio)
        left.addWidget(self.view_box, stretch=8)
        self._left_layout = left
        self.done_btn = QtWidgets.QPushButton("✓ Done — return to live view")
        self.done_btn.setStyleSheet("background:#166534;color:#fff;font-weight:bold;padding:6px;")
        self.done_btn.clicked.connect(self._return_to_live)
        self.done_btn.setVisible(False)
        left.addWidget(self.done_btn)
        self.fs_btn = QtWidgets.QToolButton(); self.fs_btn.setObjectName("gear")
        self.fs_btn.setIconSize(QtCore.QSize(16, 16))
        self.fs_btn.setToolTip("Fullscreen preview (P). Esc to exit.")
        self.fs_btn.clicked.connect(self._toggle_fullscreen_preview)
        fsrow = QtWidgets.QHBoxLayout(); fsrow.setContentsMargins(0, 0, 0, 0)
        fsrow.addStretch(1); fsrow.addWidget(self.fs_btn)
        left.addLayout(fsrow); self._fsrow = fsrow

        self.readout = QtWidgets.QLabel("")
        f = self.readout.font(); f.setPointSize(13); f.setBold(True)
        self.readout.setFont(f)
        self.readout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._fsrow.insertWidget(0, self.readout, 1)

        self.star_label = _AutoHideLabel("")
        self.star_label.setAlignment(Qt.AlignCenter)
        self.star_label.setStyleSheet("color:#94a3b8;")
        left.addWidget(self.star_label)

        self.clip_label = _AutoHideLabel("")
        self.clip_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.clip_label)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#161616")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Sharpness")
        self.plot.setLabel("bottom", "Frame")
        self.curve = self.plot.plot(pen=pg.mkPen("#2dd4bf", width=2))
        self.peak_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#f59e0b", style=Qt.DashLine))
        self.plot.addItem(self.peak_line)
        left.addWidget(self.plot, stretch=2)

        # RGB histogram + HFR tracking, side by side (shown in review mode) to save height
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setBackground("#161616")
        self.hist_plot.setLabel("bottom", "Level (per channel)")
        self.hist_plot.setMaximumHeight(120)
        self.hist_curve_r = self.hist_plot.plot(pen=pg.mkPen("#ef4444"))
        self.hist_curve_g = self.hist_plot.plot(pen=pg.mkPen("#22c55e"))
        self.hist_curve_b = self.hist_plot.plot(pen=pg.mkPen("#3b82f6"))
        self.hist_plot.setVisible(False)

        self.track_plot = pg.PlotWidget()
        self.track_plot.setBackground("#161616")
        self.track_plot.showGrid(x=True, y=True, alpha=0.3)
        self.track_plot.setLabel("left", "HFR (px)")
        self.track_plot.setLabel("bottom", "Exposure")
        self.track_plot.setMaximumHeight(120)
        self.track_curve = self.track_plot.plot(
            pen=pg.mkPen("#f59e0b", width=2), symbol="o", symbolSize=4,
            symbolBrush="#f59e0b")
        self.track_plot.setVisible(False)
        plots_row = QtWidgets.QHBoxLayout()
        plots_row.addWidget(self.hist_plot); plots_row.addWidget(self.track_plot)
        left.addLayout(plots_row)
        self.track_info = _AutoHideLabel("")
        self.track_info.setAlignment(Qt.AlignCenter)
        self.track_info.setStyleSheet("color:#94a3b8;")
        left.addWidget(self.track_info)
        self.track_match = _AutoHideLabel("")
        self.track_match.setAlignment(Qt.AlignCenter)
        fM = self.track_match.font(); fM.setBold(True); self.track_match.setFont(fM)
        left.addWidget(self.track_match)
        root.addLayout(left, stretch=3)

        # --- right column: segmented options (accordion) in scroll area ---
        self.opt_box = QtWidgets.QToolBox()

        # 1) Exposure
        page_exp = QtWidgets.QGroupBox("Exposure"); le = QtWidgets.QFormLayout(page_exp)
        self.cb_iso = QtWidgets.QComboBox()
        self.cb_shutter = QtWidgets.QComboBox()
        self.cb_aper = QtWidgets.QComboBox()
        for cb in (self.cb_iso, self.cb_shutter, self.cb_aper):
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            cb.setMinimumWidth(110)
            cb.view().setMinimumWidth(110)
        self.cb_iso.activated.connect(lambda: self._set("iso", self.cb_iso))
        self.cb_shutter.activated.connect(lambda: self._set("shutterspeed", self.cb_shutter))
        self.cb_aper.activated.connect(lambda: self._set("f-number", self.cb_aper))
        le.addRow("ISO", self.cb_iso)
        le.addRow("Speed", self.cb_shutter)
        le.addRow("Aperture", self.cb_aper)

        # 2) Display & Focus
        page_disp = QtWidgets.QGroupBox("Display && Focus"); dl = QtWidgets.QVBoxLayout(page_disp)
        self.chk_live = QtWidgets.QCheckBox("Live view"); self.chk_live.setChecked(True)
        self.chk_metric = QtWidgets.QCheckBox("Sharpness measurement + curve"); self.chk_metric.setChecked(True)
        self.chk_stars = QtWidgets.QCheckBox("Star detection")
        self.chk_stars.setToolTip("Counts stars in field: confirms pointing, framing, "
                                  "and roughly focus.")
        self.chk_live.toggled.connect(self._on_live_toggle)
        self.chk_metric.toggled.connect(self._on_metric_toggle)
        self.chk_stars.toggled.connect(lambda on: self.worker.post("set_stars", on=on))
        dl.addWidget(self.chk_live); dl.addWidget(self.chk_metric); dl.addWidget(self.chk_stars)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Live source"))
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.setToolTip(
            "USB: the camera's own live view, 1024×680 JPEG.\n"
            "HDMI: the camera's micro-HDMI output through a capture card — 1080p, "
            "50/60 fps, uncompressed with a good card. Photos still go over USB.")
        self.src_combo.addItem("USB — 1024×680", None)
        self.src_combo.activated.connect(lambda _i: self._on_live_source())
        srow.addWidget(self.src_combo, 1)
        src_ref = QtWidgets.QToolButton(); src_ref.setText("↻")
        src_ref.setToolTip("Look for capture cards again (plug the card in first).")
        src_ref.clicked.connect(lambda: self._refresh_video_sources())
        srow.addWidget(src_ref)
        dl.addLayout(srow)
        self.src_info = QtWidgets.QLabel("")
        self.src_info.setStyleSheet("color:#94a3b8;"); self.src_info.setWordWrap(True)
        dl.addWidget(self.src_info)
        self._hdmi = None
        for sig in (self.chk_metric.toggled, self.chk_stars.toggled):
            sig.connect(lambda *_: self._sync_hdmi())
        dl.addSpacing(6)
        dl.addWidget(QtWidgets.QLabel("Sharpness Metric"))
        self.metric_cb = QtWidgets.QComboBox()
        self.metric_cb.addItems(["tenengrad", "laplacian", "norm_var", "hfr"])
        self.metric_cb.setToolTip("hfr = half-flux radius of stars (astro focus); "
                                  "curve peaks at best focus.")
        self.metric_cb.setCurrentText(self.worker.metric)
        self.metric_cb.activated.connect(
            lambda: self.worker.post("set_metric", name=self.metric_cb.currentText()))
        self.metric_cb.activated.connect(lambda *_: self._sync_hdmi())
        dl.addWidget(self.metric_cb)
        self.roi_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.roi_slider.setRange(10, 100); self.roi_slider.setValue(int(self.roi_frac * 100))
        self.roi_slider.valueChanged.connect(self._on_roi)
        self.roi_label = QtWidgets.QLabel("Analysis zone: {}%".format(self.roi_slider.value()))
        dl.addWidget(self.roi_label); dl.addWidget(self.roi_slider)

        self.ret_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.ret_slider.setRange(0, 100); self.ret_slider.setValue(85)
        self.ret_slider.setToolTip("Opacity of the red centre reticle on the live view. "
                                   "0 hides it completely.")
        self.ret_label = QtWidgets.QLabel("85%")
        self.ret_slider.valueChanged.connect(self._on_reticle_alpha)
        dl.addWidget(self.ret_label); dl.addWidget(self.ret_slider)

        # 3) Capture & interval
        page_cap = QtWidgets.QGroupBox("Capture && Interval"); cl = QtWidgets.QVBoxLayout(page_cap)
        dirrow = QtWidgets.QHBoxLayout()
        self.dir_edit = QtWidgets.QLineEdit(self.save_dir)
        browse = QtWidgets.QPushButton("…"); browse.setFixedWidth(32)
        browse.clicked.connect(self._browse)
        openf = QtWidgets.QPushButton("📂"); openf.setFixedWidth(32)
        openf.setToolTip("Open session folder in Finder.")
        openf.clicked.connect(self._open_session_folder)
        dirrow.addWidget(QtWidgets.QLabel("Folder")); dirrow.addWidget(self.dir_edit)
        dirrow.addWidget(browse); dirrow.addWidget(openf)
        cl.addLayout(dirrow)
        typerow = QtWidgets.QHBoxLayout()
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["lights", "flats", "darks", "bias"]); self.type_combo.setEditable(True)
        self.type_combo.setToolTip("Photos go into <Folder>/<type> to sort directly.")
        typerow.addWidget(QtWidgets.QLabel("Type")); typerow.addWidget(self.type_combo)
        cl.addLayout(typerow)
        self.shot_btn = QtWidgets.QPushButton("Single Shot")
        self.shot_btn.setObjectName("primary")
        self.shot_btn.clicked.connect(self._single_capture)
        cl.addWidget(self.shot_btn)
        moderow = QtWidgets.QHBoxLayout()
        self.exp_mode = QtWidgets.QComboBox()
        self.exp_mode.addItems(["Camera speed (≤30s)", "Bulb (timed)"])
        self.exp_mode.setToolTip(
            "Camera speed: duration is set on camera (max 30s).\n"
            "Bulb: long exposure timed by software — body in M, LENR disabled.")
        self.bulb_secs = QtWidgets.QDoubleSpinBox()
        self.bulb_secs.setRange(1, 1800); self.bulb_secs.setValue(60.0); self.bulb_secs.setSuffix(" s")
        self.bulb_secs.setEnabled(False)
        self.exp_mode.currentIndexChanged.connect(lambda i: self.bulb_secs.setEnabled(i == 1))
        moderow.addWidget(self.exp_mode)
        moderow.addWidget(QtWidgets.QLabel("Exposure")); moderow.addWidget(self.bulb_secs)
        cl.addLayout(moderow)
        ivrow = QtWidgets.QHBoxLayout()
        self.iv_interval = QtWidgets.QDoubleSpinBox()
        self.iv_interval.setRange(0.0, 3600); self.iv_interval.setValue(2.0); self.iv_interval.setSuffix(" s")
        self.iv_interval.setToolTip("Pause between end of one exposure and start of next.")
        self.iv_count = QtWidgets.QSpinBox(); self.iv_count.setRange(1, 100000); self.iv_count.setValue(20)
        self.iv_count.setToolTip("Number of shots. You can change this DURING a burst to "
                                 "extend or shorten it on the fly.")
        self.iv_count.valueChanged.connect(self._on_count_changed)
        self.chk_unlimited = QtWidgets.QCheckBox("∞")
        self.chk_unlimited.setToolTip("Unlimited: series runs until Stop.")
        self.chk_unlimited.toggled.connect(lambda on: self.iv_count.setEnabled(not on))
        ivrow.addWidget(QtWidgets.QLabel("Pause")); ivrow.addWidget(self.iv_interval)
        ivrow.addWidget(QtWidgets.QLabel("Num")); ivrow.addWidget(self.iv_count); ivrow.addWidget(self.chk_unlimited)
        cl.addLayout(ivrow)
        btnrow = QtWidgets.QHBoxLayout()
        self.iv_start = QtWidgets.QPushButton("▶ Start")
        self.iv_stop = QtWidgets.QPushButton("⏹ Stop")
        self.iv_start.clicked.connect(self._start_interval)
        self.iv_stop.clicked.connect(lambda: self.worker.post("stop_interval"))
        btnrow.addWidget(self.iv_start); btnrow.addWidget(self.iv_stop)
        cl.addLayout(btnrow)
        # The old hidden "stop at astronomical dawn" box is retired: left checked in
        # saved settings, it ended a series the moment it started whenever the Sun was
        # above -18° — i.e. every early evening. Automation replaces it.
        self.chk_dawn = QtWidgets.QCheckBox("Stop at astronomical dawn")
        self.chk_dawn.setVisible(False); self.chk_dawn.setChecked(False)
        self.worker.post("set_dawn_stop", on=False)
        auto_group = QtWidgets.QGroupBox("Automation")
        ag = QtWidgets.QVBoxLayout(auto_group)
        self.polaris_lbl = QtWidgets.QLabel("🧭 Polaris — waiting for dusk")
        self.polaris_lbl.setStyleSheet("color:#94a3b8;")
        ag.addWidget(self.polaris_lbl)
        sr = QtWidgets.QHBoxLayout()
        self.chk_autostart = QtWidgets.QCheckBox("Auto-start")
        self.autostart_when = QtWidgets.QComboBox()
        self.autostart_when.addItem("Nautical dusk (−12°)", -12.0)
        self.autostart_when.addItem("Astro night (−18°)", -18.0)
        self.autostart_when.addItem("Fixed time", None)
        self.autostart_time = QtWidgets.QTimeEdit(); self.autostart_time.setDisplayFormat("HH:mm")
        self.autostart_time.setTime(QtCore.QTime(22, 0)); self.autostart_time.setEnabled(False)
        sr.addWidget(self.chk_autostart); sr.addWidget(self.autostart_when)
        sr.addWidget(self.autostart_time)
        ag.addLayout(sr)
        sr2 = QtWidgets.QHBoxLayout()
        sr2.addWidget(QtWidgets.QLabel("target above"))
        self.autostart_alt = QtWidgets.QSpinBox(); self.autostart_alt.setRange(0, 80)
        self.autostart_alt.setValue(20); self.autostart_alt.setSuffix("°")
        sr2.addWidget(self.autostart_alt); sr2.addStretch(1)
        ag.addLayout(sr2)
        tr = QtWidgets.QHBoxLayout()
        self.chk_autostop = QtWidgets.QCheckBox("Auto-stop")
        self.autostop_when = QtWidgets.QComboBox()
        self.autostop_when.addItem("Astro dawn (−18°)", -18.0)
        self.autostop_when.addItem("Nautical dawn (−12°)", -12.0)
        self.autostop_when.addItem("Civil dawn (−6°)", -6.0)
        self.autostop_when.addItem("Fixed time", None)
        self.autostop_time = QtWidgets.QTimeEdit(); self.autostop_time.setDisplayFormat("HH:mm")
        self.autostop_time.setTime(QtCore.QTime(5, 0)); self.autostop_time.setEnabled(False)
        tr.addWidget(self.chk_autostop); tr.addWidget(self.autostop_when)
        tr.addWidget(self.autostop_time)
        ag.addLayout(tr)
        tr2 = QtWidgets.QHBoxLayout()
        self.chk_stop_target = QtWidgets.QCheckBox("or target below")
        self.autostop_alt = QtWidgets.QSpinBox(); self.autostop_alt.setRange(0, 80)
        self.autostop_alt.setValue(15); self.autostop_alt.setSuffix("°")
        tr2.addWidget(self.chk_stop_target); tr2.addWidget(self.autostop_alt); tr2.addStretch(1)
        ag.addLayout(tr2)
        cl.addWidget(auto_group)
        self.autostart_when.currentIndexChanged.connect(
            lambda i: self.autostart_time.setEnabled(self.autostart_when.currentData() is None))
        self.autostop_when.currentIndexChanged.connect(
            lambda i: self.autostop_time.setEnabled(self.autostop_when.currentData() is None))
        for wdg in (self.chk_autostop, self.chk_stop_target):
            wdg.toggled.connect(self._push_autostop)
        self.chk_autostart.toggled.connect(self._on_autostart_toggled)
        self.autostop_when.currentIndexChanged.connect(self._push_autostop)
        self.autostop_time.timeChanged.connect(self._push_autostop)
        self.autostop_alt.valueChanged.connect(self._push_autostop)
        self._auto_timer = QtCore.QTimer(self); self._auto_timer.setInterval(30000)
        self._auto_timer.timeout.connect(self._automation_tick)
        self._auto_timer.start()
        QtCore.QTimer.singleShot(1500, self._automation_tick)
        self.exp_calc_btn = QtWidgets.QPushButton("🧮 Sub-exposure calculator")
        self.exp_calc_btn.setToolTip("Estimate the optimal single-exposure length for your "
                                     "f-ratio, ISO and sky brightness.")
        self.exp_calc_btn.clicked.connect(self._open_exposure_calc)
        cl.addWidget(self.exp_calc_btn)
        self.prog_label = QtWidgets.QLabel("—"); self.prog_label.setStyleSheet("color:#cbd5e1;")
        cl.addWidget(self.prog_label)
        self.cam_status_label = QtWidgets.QLabel("State: —"); self.cam_status_label.setStyleSheet("color:#94a3b8;")
        cl.addWidget(self.cam_status_label)

        # 4) Processing (Astro)
        page_rev = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(page_rev)
        self.chk_stretch = QtWidgets.QCheckBox("Auto-stretch")
        self.chk_stretch.setToolTip("Non-destructive automatic screen stretch of the preview, "
                                    "equivalent to Siril's linked auto-adjustment (no level to "
                                    "set — the optimal stretch is computed from the data).")
        self.chk_stretch.toggled.connect(self._restretch)
        rl.addWidget(self.chk_stretch)
        self.chk_obs = QtWidgets.QCheckBox("Observation look")
        self.chk_obs.setToolTip("Smart-telescope rendering of the live stack: flat neutral "
                                "sky, true star colours, protected galaxy cores, denoised "
                                "background. Off: the plain Siril-style stretch.")
        self.chk_obs.setChecked(True)
        self.chk_obs.toggled.connect(self._restretch)
        self.chk_objs = QtWidgets.QCheckBox("Show sky objects (needs plate-solve) — beta")
        self.chk_objs.setToolTip("After a plate-solve, overlay catalog DSOs on the stack so "
                                 "you can see where the first faint details should appear.")
        self.chk_objs.toggled.connect(self._restretch)
        rl.addWidget(self.chk_objs)
        srow = QtWidgets.QHBoxLayout()
        self.sat_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.sat_slider.setRange(100, 250); self.sat_slider.setValue(140)   # ×1.0 – ×2.5
        self.sat_slider.setToolTip("Colour saturation of the stretched preview — brings out "
                                   "nebula and star colour. ×1.0 = neutral.")
        self.sat_slider.valueChanged.connect(self._restretch)
        srow.addWidget(QtWidgets.QLabel("saturation")); srow.addWidget(self.sat_slider)
        rl.addLayout(srow)
        self.chk_cull = QtWidgets.QCheckBox("Discard bad exposures (→ rejected/)")
        self.chk_cull.setToolTip("Moves to rejected/ the shots where star count drops.")
        self.chk_cull.toggled.connect(lambda on: self.worker.post("set_cull", on=on))
        rl.addWidget(self.chk_cull)
        self.chk_stack = QtWidgets.QCheckBox("Live stacking")
        self.chk_stack.setToolTip("Aligns and stacks exposures to SEE the target emerge. "
                                  "Preview only (no darks, no bias, no flats).")
        self.chk_stack.toggled.connect(lambda on: self.worker.post("set_stack", on=on))
        rl.addWidget(self.chk_stack)
        self.chk_degrad = QtWidgets.QCheckBox("Remove gradient")
        self.chk_degrad.setToolTip("Remove background gradient — GraXpert AI if installed, "
                                   "else built-in.")
        self.chk_degrad.setChecked(True)               # astro default: flatten the background
        self.chk_degrad.toggled.connect(self._restretch)
        rl.addWidget(self.chk_degrad)
        self.chk_flats = QtWidgets.QCheckBox("Apply flats")
        self.chk_flats.setToolTip("Divide by a master flat matched to lens/focal/aperture — "
                                  "removes vignetting.")
        self.chk_flats.setChecked(QtCore.QSettings("NOUT", "NOUT").value("apply_flats", False, type=bool))
        self.chk_flats.toggled.connect(self._push_flats)
        rl.addWidget(self.chk_flats)
        self.chk_darks = QtWidgets.QCheckBox("Apply darks/bias")
        self.chk_darks.setToolTip("Subtract a master dark/bias matched to ISO + exposure.")
        self.chk_darks.setChecked(QtCore.QSettings("NOUT", "NOUT").value("apply_darks", False, type=bool))
        self.chk_darks.toggled.connect(self._push_darks)
        rl.addWidget(self.chk_darks)
        self.chk_dither = QtWidgets.QCheckBox("Dither (mount, RA)")
        self.chk_dither.setToolTip("RA nudge between shots to break walking noise. "
                                   "walking/fixed-pattern noise. Tracking is auto-started when "
                                   "a capture begins. Open “Mount control…” to connect/slew.")
        self.chk_dither.toggled.connect(self._push_dither)
        rl.addWidget(self.chk_dither)
        drow = QtWidgets.QHBoxLayout()
        self.dither_every = QtWidgets.QSpinBox(); self.dither_every.setRange(1, 50); self.dither_every.setValue(4)
        self.dither_every.setToolTip("Dither every N shots")
        self.dither_amp = QtWidgets.QSpinBox(); self.dither_amp.setRange(10, 1200); self.dither_amp.setValue(150)
        self.dither_amp.setSuffix("\""); self.dither_amp.setToolTip("Max dither amplitude (arcsec)")
        self.dither_settle = QtWidgets.QSpinBox(); self.dither_settle.setRange(0, 30); self.dither_settle.setValue(4)
        self.dither_settle.setSuffix("s"); self.dither_settle.setToolTip("Settle time after dither")
        for wdg in (self.dither_every, self.dither_amp, self.dither_settle):
            wdg.valueChanged.connect(self._push_dither)
        drow.addWidget(QtWidgets.QLabel("every")); drow.addWidget(self.dither_every)
        drow.addWidget(self.dither_amp); drow.addWidget(self.dither_settle)
        rl.addLayout(drow)
        self._mount_dlg_lbl = None
        self._mount_status_text = "not connected"
        self.stack_reset = QtWidgets.QPushButton("↺ Reset stacking")
        self.stack_reset.clicked.connect(lambda: self.worker.post("reset_stack"))
        rl.addWidget(self.stack_reset)

        # tracking control between exposures
        trow = QtWidgets.QHBoxLayout()
        self.chk_track = QtWidgets.QCheckBox("Tracking control")
        self.chk_track.setChecked(True)
        self.chk_track.setToolTip("Compares each exposure to the first of the burst: "
                                  "% of star match + drift. Alerts if too low.")
        trow.addWidget(self.chk_track)
        trow.addWidget(QtWidgets.QLabel("alert under"))
        self.track_thresh = QtWidgets.QSpinBox()
        self.track_thresh.setRange(10, 99); self.track_thresh.setValue(70); self.track_thresh.setSuffix(" %")
        trow.addWidget(self.track_thresh)
        rl.addLayout(trow)
        trow2 = QtWidgets.QHBoxLayout()
        trow2.addWidget(QtWidgets.QLabel("Tolerance (drift = 0 %)"))
        self.track_tol = QtWidgets.QDoubleSpinBox()
        self.track_tol.setRange(0.5, 20.0); self.track_tol.setSingleStep(0.5)
        self.track_tol.setValue(4.0); self.track_tol.setSuffix(" % of field")
        self.track_tol.setToolTip("Drift (in % of image width) for which tracking "
                                  "drops to 0%. Larger = more tolerant.")
        trow2.addWidget(self.track_tol)
        rl.addLayout(trow2)

        krow = QtWidgets.QHBoxLayout()
        self.chk_kappa = QtWidgets.QCheckBox("Kappa-sigma rejection")
        self.chk_kappa.setToolTip("Discards aberrant pixels from stacking "
                                  "(planes, satellites, cosmic rays).")
        self.kappa_val = QtWidgets.QDoubleSpinBox()
        self.kappa_val.setRange(1.0, 5.0); self.kappa_val.setSingleStep(0.5); self.kappa_val.setValue(2.5)
        self.kappa_val.setPrefix("κ ")
        self.chk_kappa.toggled.connect(self._push_kappa)
        self.kappa_val.valueChanged.connect(self._push_kappa)
        krow.addWidget(self.chk_kappa); krow.addWidget(self.kappa_val)
        rl.addLayout(krow)
        self.chk_stack_full = QtWidgets.QCheckBox("Live-stack at FULL resolution (24 MP)")
        self.chk_stack_full.setToolTip(
            "On, the Observation render also keeps the full size, so zooming in shows "
            "every pixel (about 5 s per render instead of 1 s — never blocks the app). "
            "For small objects; for an audience, leave it off.\n"
            "Off, the live stack integrates the RAW at half size — 3000×2000 — which is "
            "four times less memory and work and is why it keeps up between exposures.\n"
            "On, it integrates the full 6000×4000. Expect around 0.9 GB of accumulators "
            "and noticeably longer per frame; worth it when the target is small in the "
            "frame and you want every pixel. Changing this restarts the stack.")
        self.chk_stack_full.toggled.connect(
            lambda on: self.worker.post("set_stack_full", on=on))
        self.chk_stack_full.toggled.connect(self._restretch)
        rl.addWidget(self.chk_stack_full)
        self.chk_frame_reject = QtWidgets.QCheckBox("Reject bad frames (clouds / trails / lost tracking)")
        self.chk_frame_reject.setToolTip("During live stacking, skip whole frames whose star "
                                         "count drops far below the reference (passing clouds, "
                                         "big satellite trails, tracking glitches).")
        self.chk_frame_reject.toggled.connect(
            lambda on: self.worker.post("set_stack_reject", on=on))
        rl.addWidget(self.chk_frame_reject)
        self.chk_autostop_pd = QtWidgets.QCheckBox("Auto stop if battery/disk low")
        self.chk_autostop_pd.toggled.connect(self._push_autostop_pd)
        rl.addWidget(self.chk_autostop_pd)
        asrow = QtWidgets.QHBoxLayout()
        self.batt_thr = QtWidgets.QSpinBox(); self.batt_thr.setRange(1, 90)
        self.batt_thr.setValue(15); self.batt_thr.setSuffix(" %")
        self.disk_thr = QtWidgets.QDoubleSpinBox(); self.disk_thr.setRange(0.2, 100)
        self.disk_thr.setValue(2.0); self.disk_thr.setSuffix(" GB")
        self.batt_thr.valueChanged.connect(self._push_autostop_pd)
        self.disk_thr.valueChanged.connect(self._push_autostop_pd)
        asrow.addWidget(QtWidgets.QLabel("thresholds")); asrow.addWidget(self.batt_thr); asrow.addWidget(self.disk_thr)
        rl.addLayout(asrow)
        self.solve_btn = QtWidgets.QPushButton("🔭 Identify field (last photo)")
        self.solve_btn.clicked.connect(self._run_solve)
        rl.addWidget(self.solve_btn)
        srow2 = QtWidgets.QHBoxLayout()
        self.solve_focal = QtWidgets.QDoubleSpinBox()
        self.solve_focal.setRange(8, 2000); self.solve_focal.setValue(135); self.solve_focal.setSuffix(" mm")
        self.solve_focal.setToolTip("Focal length (read automatically from file EXIF).")
        self.solve_mode = QtWidgets.QComboBox(); self.solve_mode.addItems(["Local", "Online"])
        self.solve_mode.setCurrentText("Online")
        self.solve_mode.setToolTip("« Online » = nova.astrometry.net: no index to install, "
                                   "requires Internet + API key.")
        srow2.addWidget(QtWidgets.QLabel("focal")); srow2.addWidget(self.solve_focal)
        srow2.addWidget(self.solve_mode)
        rl.addLayout(srow2)
        self.solve_key = QtWidgets.QLineEdit("")
        self.solve_key.setPlaceholderText("nova API key (online mode)")
        rl.addWidget(self.solve_key)
        self.solve_label = QtWidgets.QLabel(""); self.solve_label.setWordWrap(True)
        self.solve_label.setStyleSheet("color:#cbd5e1;")
        rl.addWidget(self.solve_label)

        # Drift Group inside Processing
        drift_group = QtWidgets.QGroupBox("Polar Alignment (Drift)")
        d_layout = QtWidgets.QVBoxLayout(drift_group)
        
        d_layout.addWidget(_hint(
            "Aim at a star, keep Live view on, start measurement.<br><br>"
            "Azimuth: aim near meridian + equator. Altitude: aim low, east/west. "
            "Null the Dec drift with the matching screw."))

        db = QtWidgets.QHBoxLayout()
        self.drift_btn = QtWidgets.QPushButton("▶ Start measurement")
        self.drift_btn.clicked.connect(self._toggle_drift)
        self.drift_reset = QtWidgets.QPushButton("↺ New reference")
        self.drift_reset.clicked.connect(self._reset_drift_ref)
        db.addWidget(self.drift_btn); db.addWidget(self.drift_reset)
        d_layout.addLayout(db)
        self.drift_out = QtWidgets.QLabel("—"); self.drift_out.setWordWrap(True)
        fD = self.drift_out.font(); fD.setBold(True); self.drift_out.setFont(fD)
        self.drift_out.setStyleSheet("color:#2dd4bf;")
        d_layout.addWidget(self.drift_out)
        

        rl.addStretch(1)
        # Polar Alignment gets its own accordion page (it's a setup step, not processing)
        page_align = QtWidgets.QWidget(); al = QtWidgets.QVBoxLayout(page_align)
        al.addWidget(drift_group); al.addStretch(1)
        # Drift alignment is not used on this rig (polar scope + GoTo dialog do it).
        # The page is still built, so the code that drives it keeps working, but it is
        # kept out of the accordion; holding a reference stops Qt deleting it.
        self._page_align = page_align; page_align.hide()
        self.opt_box.addItem(page_rev, "Processing (Astro)")

        # 5) Framing / Mosaic (ghost)
        page_ghost = QtWidgets.QWidget(); gl = QtWidgets.QVBoxLayout(page_ghost)
        grow = QtWidgets.QHBoxLayout()
        self.chk_ghost = QtWidgets.QCheckBox("Show reference")
        load_ghost = QtWidgets.QPushButton("Load an image…")
        load_ghost.clicked.connect(self._load_ghost)
        grow.addWidget(self.chk_ghost); grow.addWidget(load_ghost)
        gl.addLayout(grow)
        gl.addWidget(QtWidgets.QLabel("opacity"))
        self.ghost_alpha = QtWidgets.QSlider(Qt.Horizontal)
        self.ghost_alpha.setRange(10, 90); self.ghost_alpha.setValue(40)
        gl.addWidget(self.ghost_alpha)
        self.ghost_dx = QtWidgets.QSlider(Qt.Horizontal); self.ghost_dx.setRange(-100, 100); self.ghost_dx.setValue(0)
        self.ghost_dy = QtWidgets.QSlider(Qt.Horizontal); self.ghost_dy.setRange(-100, 100); self.ghost_dy.setValue(0)
        self.ghost_dx_val = QtWidgets.QSpinBox(); self.ghost_dx_val.setRange(-100, 100); self.ghost_dx_val.setSuffix(" %")
        self.ghost_dy_val = QtWidgets.QSpinBox(); self.ghost_dy_val.setRange(-100, 100); self.ghost_dy_val.setSuffix(" %")
        # keep slider and numeric box in sync (type 80 directly, or drag)
        self.ghost_dx.valueChanged.connect(self.ghost_dx_val.setValue)
        self.ghost_dx_val.valueChanged.connect(self.ghost_dx.setValue)
        self.ghost_dy.valueChanged.connect(self.ghost_dy_val.setValue)
        self.ghost_dy_val.valueChanged.connect(self.ghost_dy.setValue)
        xrow = QtWidgets.QHBoxLayout()
        xrow.addWidget(QtWidgets.QLabel("X offset")); xrow.addWidget(self.ghost_dx, 1)
        xrow.addWidget(self.ghost_dx_val)
        yrow = QtWidgets.QHBoxLayout()
        yrow.addWidget(QtWidgets.QLabel("Y offset")); yrow.addWidget(self.ghost_dy, 1)
        yrow.addWidget(self.ghost_dy_val)
        gl.addLayout(xrow); gl.addLayout(yrow)
        gl.addWidget(_hint("Offset (e.g. 80%) for a mosaic with ~20% overlap."))
        gl.addWidget(self._hline())
        gl.addWidget(QtWidgets.QLabel("Mosaic Planner"))
        mform = QtWidgets.QFormLayout()
        self.mos_w = QtWidgets.QDoubleSpinBox(); self.mos_w.setRange(0.1, 180); self.mos_w.setValue(6.0); self.mos_w.setSuffix(" °")
        self.mos_h = QtWidgets.QDoubleSpinBox(); self.mos_h.setRange(0.1, 180); self.mos_h.setValue(4.0); self.mos_h.setSuffix(" °")
        self.mos_ov = QtWidgets.QSpinBox(); self.mos_ov.setRange(0, 60); self.mos_ov.setValue(20); self.mos_ov.setSuffix(" %")
        for wdg in (self.mos_w, self.mos_h, self.mos_ov):
            wdg.valueChanged.connect(self._update_mosaic)
        mform.addRow("Target width", self.mos_w)
        mform.addRow("Target height", self.mos_h)
        mform.addRow("Overlap", self.mos_ov)
        gl.addLayout(mform)
        self.mos_out = QtWidgets.QLabel("—"); self.mos_out.setWordWrap(True)
        self.mos_out.setStyleSheet("color:#2dd4bf;")
        gl.addWidget(self.mos_out)
        gl.addStretch(1)
        self.opt_box.addItem(page_ghost, "Framing / Mosaic")

        # 7) Sequences (bursts) + integration time
        page_seq = QtWidgets.QGroupBox("Sequences (bursts)"); ql = QtWidgets.QVBoxLayout(page_seq)
        self.integ_label = QtWidgets.QLabel("0:00:00 · 0 shots")
        fI = self.integ_label.font(); fI.setBold(True); self.integ_label.setFont(fI)
        self.integ_label.setStyleSheet("color:#2dd4bf;")
        self.integ_label.setWordWrap(True)
        self.integ_label.setMinimumWidth(1)      # prevents text from forcing panel width
        ql.addWidget(self.integ_label)
        self.seq_table = QtWidgets.QTableWidget(0, 5)
        self.seq_table.setHorizontalHeaderLabels(["#", "Type", "Exposure", "Num", "Integ."])
        self.seq_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.seq_table.verticalHeader().setVisible(False)
        self.seq_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.seq_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.seq_table.setMaximumHeight(170)
        self.seq_table.setMinimumHeight(96)
        ql.addWidget(self.seq_table)
        orow = QtWidgets.QHBoxLayout()
        self.chk_integ_target = QtWidgets.QCheckBox("Target")
        self.chk_integ_target.setToolTip("Automatically stops series when "
                                         "total integration of the session reaches the target.")
        self.integ_target = QtWidgets.QDoubleSpinBox()
        self.integ_target.setRange(1, 1440); self.integ_target.setValue(60); self.integ_target.setSuffix(" min")
        self.chk_integ_target.toggled.connect(self._update_total_integ)
        self.integ_target.valueChanged.connect(self._update_total_integ)
        orow.addWidget(self.chk_integ_target); orow.addWidget(self.integ_target)
        ql.addLayout(orow)

        # --- merged page: exposure · focus · capture (interdependent) ---
        page_main = QtWidgets.QWidget(); mml = QtWidgets.QVBoxLayout(page_main)
        for gb in (page_exp, page_disp, page_cap):
            mml.addWidget(gb)
        mml.addStretch(1)
        self.opt_box.insertItem(0, page_main, "Shooting (Exposure · Focus · Capture)")
        self.opt_box.setCurrentIndex(0)

        self.lens_combo = QtWidgets.QComboBox()
        self.lens_combo.addItems(list(LENS_PROFILES.keys()))
        self.lens_combo.setToolTip("Chooses a lens: sets focal length everywhere (pose, "
                                   "solving, target framing) and max aperture.")
        self.lens_combo.currentTextChanged.connect(self._apply_lens)
        self.lens_focal = QtWidgets.QDoubleSpinBox()
        self.lens_focal.setRange(8, 2000); self.lens_focal.setSuffix(" mm"); self.lens_focal.setValue(50)
        self.lens_focal.valueChanged.connect(self._propagate_focal)

        # ---- one page, folding sections (no scrolling) ------------------------
        col = _SectionColumn()
        col.setMinimumWidth(430); col.setMaximumWidth(470)
        self.right_col = col

        # Exposure: lens + the three camera settings on one line
        sx = col.add(_Section("Exposure"))
        sx.row(self.lens_combo, self.lens_focal, stretch_last=False)
        self.lens_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Fixed)
        self.aper_fixed = QtWidgets.QLabel(""); self.aper_fixed.setObjectName("fixedVal")
        self.aper_fixed.setToolTip("Telescope: the aperture is fixed by the glass.")
        self.aper_fixed.setVisible(False)
        g = QtWidgets.QGridLayout(); g.setHorizontalSpacing(6); g.setVerticalSpacing(2)
        self._aper_cap = QtWidgets.QLabel("Aperture")
        for c, (cap, w) in enumerate((("ISO", self.cb_iso), ("Speed", self.cb_shutter),
                                      (None, self.cb_aper))):
            lb = self._aper_cap if cap is None else QtWidgets.QLabel(cap)
            lb.setObjectName("cap"); g.addWidget(lb, 0, c)
            w.setMinimumWidth(0); w.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            g.addWidget(w, 1, c)
        g.addWidget(self.aper_fixed, 1, 2)
        self.exp_calc_btn.setText("🧮"); self.exp_calc_btn.setFixedWidth(34)
        g.addWidget(self.exp_calc_btn, 1, 3)
        for c in range(3):
            g.setColumnStretch(c, 1)
        sx.lay.addLayout(g)

        # Capture
        sc = col.add(_Section("Capture")); sc.sticky = 1      # the shooting buttons: fold last
        self.dir_edit.setMinimumWidth(0)
        sc.row("Folder", self.dir_edit, browse, openf)
        self.exp_mode.setItemText(0, "Camera speed")
        self.exp_mode.setItemText(1, "Bulb")
        sc.row(self.type_combo, self.exp_mode, self.bulb_secs)
        self.chk_unlimited.setText("∞")
        sc.row("Pause", self.iv_interval, "Shots", self.iv_count, self.chk_unlimited)
        self.shot_btn.setText("Single shot")
        self.iv_start.setText("▶ Series"); self.iv_stop.setText("⏹ Stop")
        self.iv_pause = QtWidgets.QPushButton("⏸ Pause")
        self.iv_pause.setToolTip("Finishes the exposure under way, then holds the series. "
                                 "Resuming carries on with the same sequence, the same "
                                 "count and the same stack — nothing is started over.")
        self.iv_pause.setEnabled(False)
        self.iv_pause.clicked.connect(self._toggle_pause)
        sc.row(self.shot_btn, self.iv_start, self.iv_pause, self.iv_stop)
        for b in (self.shot_btn, self.iv_start, self.iv_pause, self.iv_stop):
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.chk_integ_target.setText("Stop at")
        self.integ_label.setObjectName("cap"); self.integ_label.setWordWrap(False)
        self.integ_label.setStyleSheet("")
        sc.row(self.chk_integ_target, self.integ_target, None, self.integ_label)
        self.prog_label = _ElideLabel("—"); self.cam_status_label = _ElideLabel("State: —")
        self.prog_label.setObjectName("cap"); self.cam_status_label.setObjectName("cap")
        prow = QtWidgets.QHBoxLayout(); prow.setSpacing(8)
        prow.addWidget(self.prog_label, 5); prow.addWidget(self.cam_status_label, 3)
        sc.lay.addLayout(prow)

        # Live view settings: under the picture, behind a gear
        lvw = QtWidgets.QWidget(); lvl = QtWidgets.QVBoxLayout(lvw)
        lvl.setContentsMargins(0, 0, 0, 0); lvl.setSpacing(6)
        head = QtWidgets.QLabel("Live view"); head.setObjectName("cap"); lvl.addWidget(head)
        self.src_combo.setMinimumWidth(0)
        r1 = QtWidgets.QHBoxLayout(); r1.addWidget(self.chk_live)
        r1.addWidget(self.src_combo, 1); r1.addWidget(src_ref); lvl.addLayout(r1)
        self.src_info.setObjectName("cap"); self.src_info.setStyleSheet("")
        lvl.addWidget(self.src_info)
        self.ret_label.setObjectName("cap")
        r2 = QtWidgets.QHBoxLayout(); rl_ = QtWidgets.QLabel("Reticle"); rl_.setObjectName("cap")
        r2.addWidget(rl_); r2.addWidget(self.ret_slider, 1); r2.addWidget(self.ret_label)
        lvl.addLayout(r2)
        self.lv_gear = _GearButton(lvw, "Live view settings: source, reticle, on/off")
        self._fsrow.insertWidget(self._fsrow.count() - 1, self.lv_gear)
        self.pub_btn = QtWidgets.QToolButton(); self.pub_btn.setObjectName("gear")
        self.pub_btn.setIconSize(QtCore.QSize(16, 16))
        self.pub_btn.setToolTip("Public view (V): the picture full screen for visitors — "
                                "on the second screen if one is connected. Esc to close.")
        self.pub_btn.clicked.connect(self._toggle_public_view)
        self._fsrow.insertWidget(self._fsrow.count() - 1, self.pub_btn)

        # Processing — folded until needed
        sp = col.add(_Section("Processing"))
        for w, t in ((self.chk_stack_full, "Full resolution"),
                     (self.chk_frame_reject, "Reject bad frames"),
                     (self.chk_cull, "Discard bad shots"),
                     (self.chk_objs, "Sky objects"),
                     (self.chk_autostop_pd, "Low batt./disk stop"),
                     (self.chk_track, "Tracking alert"),
                     (self.chk_kappa, "κ-σ rejection"),
                     (self.chk_dither, "Dither"),
                     (self.chk_darks, "Darks / bias"),
                     (self.chk_flats, "Flats")):
            w.setText(t)
        kbox = QtWidgets.QWidget(); kbox.setObjectName("plain")
        kh = QtWidgets.QHBoxLayout(kbox); kh.setContentsMargins(0, 0, 0, 0); kh.setSpacing(4)
        kh.addWidget(self.chk_kappa); kh.addWidget(self.kappa_val); kh.addStretch(1)
        self.kappa_val.setPrefix(""); self.kappa_val.setDecimals(1)
        pg_ = QtWidgets.QGridLayout(); pg_.setHorizontalSpacing(8); pg_.setVerticalSpacing(3)
        for i, w in enumerate((self.chk_stack, self.chk_obs, self.chk_degrad,
                               self.chk_stretch, self.chk_stack_full, self.chk_flats,
                               self.chk_darks, self.chk_cull, self.chk_frame_reject,
                               self.chk_objs, kbox)):
            pg_.addWidget(w, i // 2, i % 2)
        pg_.setColumnStretch(0, 1); pg_.setColumnStretch(1, 1)
        sp.lay.addLayout(pg_)
        sp.row("Saturation", self.sat_slider)
        self.sat_slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Fixed)
        # checkbox + its values on one line, in a 4-column grid so nothing overlaps
        self.dither_every.setPrefix("every "); self.track_thresh.setPrefix("< ")
        self.track_tol.setSuffix(" %"); self.track_tol.setDecimals(1)
        self.track_tol.setToolTip("Drift (% of the image width) at which tracking drops to 0 %.")
        self.disk_thr.setDecimals(1)
        vg = QtWidgets.QGridLayout(); vg.setHorizontalSpacing(6); vg.setVerticalSpacing(4)
        for r, (chk, *vals) in enumerate((
                (self.chk_dither, self.dither_every, self.dither_amp, self.dither_settle),
                (self.chk_track, self.track_thresh, self.track_tol),
                (self.chk_autostop_pd, self.batt_thr, self.disk_thr))):
            vg.addWidget(chk, r, 0)
            for c, v in enumerate(vals):
                v.setMinimumWidth(0); v.setMaximumWidth(92)
                v.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
                vg.addWidget(v, r, 1 + c)
        vg.setColumnStretch(0, 3)
        for c in (1, 2, 3):
            vg.setColumnStretch(c, 2)
        sp.lay.addLayout(vg)
        self.stack_reset.setText("↺ Reset stack"); self.solve_btn.setText("Identify field")
        # plate-solve settings are set once: behind a gear, next to the button
        sw = QtWidgets.QWidget(); swl = QtWidgets.QVBoxLayout(sw)
        swl.setContentsMargins(0, 0, 0, 0); swl.setSpacing(6)
        sh_ = QtWidgets.QLabel("Plate solving"); sh_.setObjectName("cap"); swl.addWidget(sh_)
        r_ = QtWidgets.QHBoxLayout(); r_.addWidget(self.solve_focal); r_.addWidget(self.solve_mode)
        swl.addLayout(r_)
        self.solve_key.setMinimumWidth(0); swl.addWidget(self.solve_key)
        sp.row(self.stack_reset, self.solve_btn,
               _GearButton(sw, "Plate-solving settings: focal, local/online, API key"))
        self.solve_label.setObjectName("cap"); self.solve_label.setStyleSheet("")
        self.solve_label.setWordWrap(True); self.solve_label.setMaximumHeight(52)
        sp.lay.addWidget(self.solve_label)

        # Automation — folded
        sa = col.add(_Section("Automation", expanded=False))
        auto_group.setTitle(""); auto_group.setObjectName("flatGroup")
        auto_group.layout().setContentsMargins(0, 0, 0, 0)
        sa.lay.addWidget(auto_group)

        # Framing / mosaic — folded
        sf = col.add(_Section("Framing / Mosaic", expanded=False))
        page_ghost.layout().setContentsMargins(0, 0, 0, 0); page_ghost.setObjectName("plain")
        sf.lay.addWidget(page_ghost)
        self._sections = {"exp": sx, "cap": sc, "proc": sp,
                          "auto": sa, "frame": sf}
        for cb in col.widget().findChildren(QtWidgets.QComboBox):
            cb.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(4)
            cb.view().setMinimumWidth(180)
        for sb in col.widget().findChildren(QtWidgets.QAbstractSpinBox):
            sb.setMinimumWidth(0)
            if sb.maximumWidth() > 120:
                sb.setMaximumWidth(120)
        self.lens_focal.setDecimals(0); self.lens_focal.setMaximumWidth(96)
        self.iv_start.setText("Series"); self.iv_stop.setText("Stop")

        # What this page no longer shows. Kept alive and hidden — other code still
        # reads and writes these widgets — but never on screen.
        self._legacy_ui = [self.opt_box, page_seq, page_main, page_exp, page_disp, page_cap,
                           self.chk_metric, self.chk_stars, self.metric_cb,
                           self.roi_label, self.roi_slider]
        self._legacy_holder = QtWidgets.QWidget(tab); self._legacy_holder.hide()
        for w in (self.chk_metric, self.chk_stars, self.metric_cb, self.roi_label,
                  self.roi_slider, page_seq):
            w.setParent(self._legacy_holder)
        self.opt_box.setParent(self._legacy_holder)
        self.chk_metric.setChecked(False); self.chk_stars.setChecked(False)

        root.addWidget(col)
        # the ⚙ popovers are children of their buttons, so this covers them too
        for w in [tab] + tab.findChildren(QtWidgets.QWidget):
            w.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)
        return tab

    def _build_gallery_tab(self):
        """A collection sheet: every Messier and the main NGC/IC objects, the ones you
        have photographed showing your own picture, the rest waiting."""
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab); lay.setSpacing(8)
        bar = QtWidgets.QHBoxLayout()
        self.gal_count = QtWidgets.QLabel(""); self.gal_count.setStyleSheet(
            "font-size:15px; font-weight:600;")
        bar.addWidget(self.gal_count)
        bar.addSpacing(16)
        f = QtWidgets.QLabel("Show"); f.setObjectName("cap"); bar.addWidget(f)
        self.gal_filter = QtWidgets.QComboBox()
        for lab, key in (("Everything", "all"), ("Captured", "done"), ("Still missing", "todo"),
                         ("Messier", "messier"), ("NGC / IC", "ngc")):
            self.gal_filter.addItem(lab, key)
        self.gal_filter.activated.connect(lambda *_: self._refresh_gallery())
        bar.addWidget(self.gal_filter)
        self.gal_search = QtWidgets.QLineEdit()
        self.gal_search.setPlaceholderText("Find an object…")
        self.gal_search.textChanged.connect(lambda *_: self._refresh_gallery())
        bar.addWidget(self.gal_search, 1)
        addb = QtWidgets.QPushButton("＋ Add a photo…")
        addb.setObjectName("primary")
        addb.clicked.connect(lambda: self._gallery_add())
        bar.addWidget(addb)
        openb = QtWidgets.QPushButton("📂 Folder")
        openb.setToolTip("Opens the folder where the gallery keeps its copies.")
        openb.clicked.connect(lambda: self._open_folder(gallery_dir()))
        bar.addWidget(openb)
        lay.addLayout(bar)

        self.gal_list = QtWidgets.QListWidget()
        self.gal_list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.gal_list.setIconSize(QtCore.QSize(180, 180))
        self.gal_list.setGridSize(QtCore.QSize(200, 214))
        self.gal_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.gal_list.setMovement(QtWidgets.QListView.Movement.Static)
        self.gal_list.setSpacing(6)
        self.gal_list.setWordWrap(True)
        self.gal_list.itemDoubleClicked.connect(self._gallery_open)
        self.gal_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.gal_list.customContextMenuRequested.connect(self._gallery_menu)
        lay.addWidget(self.gal_list, 1)
        QtCore.QTimer.singleShot(0, self._refresh_gallery)
        return tab

    def _gallery_thumb(self, path, done):
        """Thumbnail for a slot: the photo, or a dim placeholder for a missing one."""
        pm = QtGui.QPixmap(180, 180)
        if done and path and os.path.exists(path):
            img = QtGui.QPixmap(path)
            if not img.isNull():
                return QtGui.QIcon(img.scaled(180, 180, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation))
        pm.fill(QtGui.QColor("#0c0f16"))
        q = QtGui.QPainter(pm); q.setRenderHint(QtGui.QPainter.Antialiasing)
        q.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 28), 1))
        q.drawRoundedRect(QtCore.QRectF(4, 4, 172, 172), 10, 10)
        q.setPen(QtGui.QColor(255, 255, 255, 40))
        f = QtGui.QFont(); f.setPixelSize(44); q.setFont(f)
        q.drawText(QtCore.QRectF(0, 0, 180, 180), Qt.AlignCenter, "?")
        q.end()
        return QtGui.QIcon(pm)

    def _refresh_gallery(self):
        if not hasattr(self, "gal_list"):
            return
        gal = load_gallery()
        self.gal_list.clear()
        want = self.gal_filter.currentData()
        needle = self.gal_search.text().strip().lower()
        names = {}
        try:
            for o in getattr(self.skymap, "dso", []) or []:
                names[str(o.get("id", "")).upper()] = o.get("name") or ""
        except Exception:               # noqa: BLE001
            pass
        slots = gallery_slots()
        known = {k for k, _ in slots}
        slots += [(k, "Other") for k in gal if k not in known]        # anything you added
        nm_done = ngc_done = 0
        for oid, kind in slots:
            entry = gal.get(oid)
            done = bool(entry)
            if done:
                if kind == "Messier":
                    nm_done += 1
                elif kind == "NGC / IC":
                    ngc_done += 1
            if want == "done" and not done:
                continue
            if want == "todo" and done:
                continue
            if want == "messier" and kind != "Messier":
                continue
            if want == "ngc" and kind != "NGC / IC":
                continue
            label = names.get(oid.upper(), "")
            if needle and needle not in oid.lower() and needle not in label.lower():
                continue
            text = oid if not label else "{}\n{}".format(oid, label)
            it = QtWidgets.QListWidgetItem(self._gallery_thumb(
                (entry or {}).get("file"), done), text)
            it.setData(Qt.UserRole, oid)
            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            if not done:
                it.setForeground(QtGui.QBrush(QtGui.QColor("#6b7280")))
            else:
                tip = [oid]
                if entry.get("date"):
                    tip.append(entry["date"])
                if entry.get("frames"):
                    tip.append("{} frames".format(entry["frames"]))
                if entry.get("integ"):
                    tip.append(_fmt_dur(float(entry["integ"])))
                it.setToolTip("  ·  ".join(tip))
            self.gal_list.addItem(it)
        self.gal_count.setText("Messier {}/110   ·   NGC / IC {}   ·   {} in the collection"
                               .format(nm_done, ngc_done, len(gal)))

    def _gallery_add(self, oid=None, src=None, frames=None, integ=None):
        """Put a picture in the collection: pick the object, then the file."""
        if oid is None:
            goal = getattr(self, "_session_goal", None)
            default = goal[0] if goal else ""
            oid, ok = QtWidgets.QInputDialog.getText(
                self, "Add to the gallery",
                "Which object? (M31, NGC7000, IC1805…)", text=default)
            if not ok or not oid.strip():
                return
            oid = oid.strip().upper().replace(" ", "")
        if src is None:
            src, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Picture of {}".format(oid), self._effective_dir() or os.path.expanduser("~"),
                "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
            if not src:
                return
        img = QtGui.QImage(src)
        if img.isNull():
            self.statusBar().showMessage("That file is not an image NOUT can read.", 6000)
            return
        dest = os.path.join(gallery_dir(), "{}.jpg".format(oid))
        if img.width() > 2400:
            img = img.scaledToWidth(2400, Qt.SmoothTransformation)
        img.save(dest, "JPG", 92)
        gal = load_gallery()
        gal[oid] = {"file": dest, "date": datetime.now().strftime("%Y-%m-%d"),
                    "frames": frames, "integ": integ,
                    "note": (gal.get(oid) or {}).get("note", "")}
        save_gallery(gal)
        self._refresh_gallery()
        self.statusBar().showMessage("{} added to the gallery.".format(oid), 6000)

    def _gallery_open(self, item):
        oid = item.data(Qt.UserRole)
        entry = load_gallery().get(oid)
        if not entry:
            self._gallery_add(oid=oid)
            return
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle(oid)
        v = QtWidgets.QVBoxLayout(dlg)
        pic = QtWidgets.QLabel(); pic.setAlignment(Qt.AlignCenter)
        pm = QtGui.QPixmap(entry.get("file", ""))
        scr = self.screen().availableGeometry() if self.screen() else None
        side = min(1100, (scr.width() - 120) if scr else 1100)
        if not pm.isNull():
            pic.setPixmap(pm.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        v.addWidget(pic, 1)
        dist = fmt_distance(ly=dso_distance_ly(oid))
        bits = [b for b in (entry.get("date"),
                            "{} frames".format(entry["frames"]) if entry.get("frames") else "",
                            _fmt_dur(float(entry["integ"])) if entry.get("integ") else "",
                            dist) if b]
        info = QtWidgets.QLabel("   ·   ".join(bits)); info.setObjectName("cap")
        v.addWidget(info)
        row = QtWidgets.QHBoxLayout()
        rep_ = QtWidgets.QPushButton("Replace photo…")
        rep_.clicked.connect(lambda: (dlg.accept(), self._gallery_add(oid=oid)))
        rm = QtWidgets.QPushButton("Remove from gallery")
        rm.clicked.connect(lambda: (self._gallery_remove(oid), dlg.accept()))
        ok = QtWidgets.QPushButton("Close"); ok.setObjectName("primary")
        ok.clicked.connect(dlg.accept)
        row.addWidget(rep_); row.addWidget(rm); row.addStretch(1); row.addWidget(ok)
        v.addLayout(row)
        dlg.show()

    def _gallery_remove(self, oid):
        gal = load_gallery()
        if oid in gal:
            gal.pop(oid)
            save_gallery(gal)
            self._refresh_gallery()
            self.statusBar().showMessage(
                "{} removed from the gallery (the file stays in the folder).".format(oid), 6000)

    def _gallery_menu(self, pos):
        item = self.gal_list.itemAt(pos)
        if item is None:
            return
        oid = item.data(Qt.UserRole)
        m = QtWidgets.QMenu(self)
        a_add = m.addAction("Add or replace the photo…")
        a_tgt = m.addAction("Make it the session target")
        a_rm = m.addAction("Remove from the gallery")
        act = m.exec(self.gal_list.viewport().mapToGlobal(pos))
        if act is a_add:
            self._gallery_add(oid=oid)
        elif act is a_rm:
            self._gallery_remove(oid)
        elif act is a_tgt:
            self.t_search.setText(oid); self._search_target()

    def _open_folder(self, d):
        try:
            os.makedirs(d, exist_ok=True)
            if sys.platform == "darwin":
                subprocess.Popen(["open", d])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", d])
            else:
                os.startfile(d)         # noqa: B606  (Windows)
        except Exception as e:          # noqa: BLE001
            self.statusBar().showMessage("Could not open: {}".format(e), 5000)

    def _build_results_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)
        bar = QtWidgets.QHBoxLayout()
        self.combine_sel_btn = QtWidgets.QPushButton("🧩 Stack selected")
        self.combine_sel_btn.setToolTip("Aligns and averages only the bursts you selected "
                                        "(Cmd/Shift-click to select several).")
        self.combine_sel_btn.clicked.connect(self._combine_selected)
        self.combine_btn = QtWidgets.QPushButton("🧩 Stack all bursts")
        self.combine_btn.setToolTip("Aligns and averages the images of all bursts.")
        self.combine_btn.clicked.connect(self._combine_all)
        self.solve_res_btn = QtWidgets.QPushButton("🔭 Identify selected burst")
        self.solve_res_btn.clicked.connect(self._solve_result)
        self.delete_res_btn = QtWidgets.QPushButton("🗑 Delete burst")
        self.delete_res_btn.setToolTip("Removes the selected burst(s) from the Results list.")
        self.delete_res_btn.clicked.connect(self._delete_selected_results)
        bar.addWidget(self.combine_sel_btn); bar.addWidget(self.combine_btn)
        bar.addWidget(self.solve_res_btn); bar.addWidget(self.delete_res_btn)
        bar.addStretch(1)
        
        # Open Siril button + Send to Mosaic
        bar2 = QtWidgets.QHBoxLayout()
        self.open_siril_btn = QtWidgets.QPushButton("🌌 Open Siril")
        self.open_siril_btn.setToolTip("Launches the Siril application.")
        self.open_siril_btn.clicked.connect(self._open_siril_app)
        bar2.addWidget(self.open_siril_btn)
        
        self.send_mosaic_btn = QtWidgets.QPushButton("📥 Send to Active Mosaic Tile")
        self.send_mosaic_btn.setToolTip("Send selection to the active mosaic tile.")
        self.send_mosaic_btn.clicked.connect(self._send_stack_to_mosaic)
        bar2.addWidget(self.send_mosaic_btn)
        self.stack_folder_btn = QtWidgets.QPushButton("📁 Stack a folder…")
        self.stack_folder_btn.setToolTip("Stack RAW/JPEG files already on disk (a previous "
                                         "session, or to test) through the same pipeline as "
                                         "the live stacker — no camera needed.")
        self.stack_folder_btn.clicked.connect(self._stack_folder)
        bar2.addWidget(self.stack_folder_btn)
        self.solve_file_btn = QtWidgets.QPushButton("📂 Identify a file…")
        self.solve_file_btn.setToolTip("Plate-solve any image file and add it (annotated) "
                                       "to the Results list.")
        self.solve_file_btn.clicked.connect(self._solve_file)
        bar2.addWidget(self.solve_file_btn)
        self.build_flat_btn = QtWidgets.QPushButton("🔧 Build master flat…")
        self.build_flat_btn.setToolTip("Build a master flat from a folder of flat frames "
                                       "(evenly-lit shots at one aperture). Saved per f-number "
                                       "and applied automatically when “Apply flats” is on.")
        self.build_flat_btn.clicked.connect(self._build_master_flat)
        bar2.addWidget(self.build_flat_btn)
        self.build_dark_btn = QtWidgets.QPushButton("🌑 Build master dark/bias…")
        self.build_dark_btn.setToolTip("Build a master dark or bias from a folder of dark "
                                       "frames (lens capped). Saved per ISO + exposure and "
                                       "subtracted automatically when “Apply darks” is on.")
        self.build_dark_btn.clicked.connect(self._build_master_dark)
        bar2.addWidget(self.build_dark_btn)
        self.lib_btn = QtWidgets.QPushButton("📚 Manage library…")
        self.lib_btn.setToolTip("List and delete your saved master flats and darks/bias.")
        self.lib_btn.clicked.connect(lambda: LibraryDialog(self, self._flats_dir()).exec())
        bar2.addWidget(self.lib_btn)
        bar2.addStretch(1)
        
        more = QtWidgets.QToolButton(); more.setText("More  ▾")
        more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        mm = QtWidgets.QMenu(more)
        for b in (self.stack_folder_btn, self.solve_file_btn, self.send_mosaic_btn, None,
                  self.build_flat_btn, self.build_dark_btn, self.lib_btn, None,
                  self.open_siril_btn):
            if b is None:
                mm.addSeparator(); continue
            a = mm.addAction(b.text()); a.setToolTip(b.toolTip())
            a.triggered.connect(b.click)
        mm.setToolTipsVisible(True)
        more.setMenu(mm)
        bar.insertWidget(bar.count() - 1, more)
        bar.addWidget(_HelpButton("Select bursts, then Stack. Double-click to enlarge."))
        lay.addLayout(bar)
        hold = QtWidgets.QWidget(tab); hold.setLayout(bar2); hold.hide()   # kept for their handlers
        self.results_solve_label = QtWidgets.QLabel("")
        self.results_solve_label.setWordWrap(True); self.results_solve_label.setStyleSheet("color:#cbd5e1;")
        lay.addWidget(self.results_solve_label)
        self.results_list = QtWidgets.QListWidget()
        self.results_list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.results_list.setIconSize(QtCore.QSize(240, 160))
        self.results_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.results_list.setMovement(QtWidgets.QListView.Movement.Static)
        self.results_list.setSpacing(10)
        self.results_list.setWordWrap(True)
        self.results_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_list.itemDoubleClicked.connect(self._open_result)
        self._results = []           # list of (BGR image, caption)
        lay.addWidget(self.results_list)
        return tab

    def _build_mosaic_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Grid:"))
        self.mos_rows = QtWidgets.QSpinBox(); self.mos_rows.setRange(1, 10)
        self.mos_cols = QtWidgets.QSpinBox(); self.mos_cols.setRange(1, 10)
        bar.addWidget(self.mos_rows)
        bar.addWidget(QtWidgets.QLabel("×"))
        bar.addWidget(self.mos_cols)
        bar.addWidget(QtWidgets.QLabel("Overlap:"))
        self.mos_view_ov = QtWidgets.QSpinBox(); self.mos_view_ov.setRange(0, 60)
        self.mos_view_ov.setValue(20); self.mos_view_ov.setSuffix(" %")
        self.mos_view_ov.valueChanged.connect(
            lambda v: self.mos_canvas.set_overlap(v / 100.0))
        bar.addWidget(self.mos_view_ov)

        btn_apply = QtWidgets.QPushButton("Apply / Reset Grid")
        btn_apply.clicked.connect(lambda: self._reset_mosaic())
        bar.addWidget(btn_apply)

        self.mos_active_label = QtWidgets.QLabel("Active Tile: None")
        self.mos_active_label.setStyleSheet("color:#f59e0b; font-weight:bold; padding:0 10px;")
        bar.addStretch(1)
        bar.addWidget(self.mos_active_label)

        btn_load = QtWidgets.QPushButton("📂 Load Image to Tile")
        btn_load.clicked.connect(self._load_mosaic_tile)
        btn_clear = QtWidgets.QPushButton("🗑 Clear Tile")
        btn_clear.clicked.connect(self._clear_mosaic_tile)
        bar.addWidget(btn_load)
        bar.addWidget(btn_clear)
        lay.addLayout(bar)

        lay.addWidget(_hint("Click a tile to make it active. Load images, or send a stack "
                            "from the Results tab. Tiles overlap to preview the framing."))

        self.mos_canvas = MosaicCanvas()
        self.mos_canvas.set_overlap(0.20)
        lay.addWidget(self.mos_canvas, stretch=1)   # remplit la fenêtre

        self._mosaic_tiles = {}   # (r, c) -> MosaicTile
        self._mosaic_paths = {}   # (r, c) -> image path
        self._active_tile = None
        return tab

    def _reset_mosaic(self, load_paths=None):
        """Rebuilds the mosaic grid inside the canvas."""
        for t in self._mosaic_tiles.values():
            t.setParent(None)
            t.deleteLater()
        self._mosaic_tiles.clear()
        if not load_paths:
            self._mosaic_paths.clear()
        self._active_tile = None
        self._update_active_tile_label()

        rows = self.mos_rows.value()
        cols = self.mos_cols.value()
        for r in range(rows):
            for c in range(cols):
                tile = MosaicTile(r, c)
                tile.setParent(self.mos_canvas)
                tile.clicked.connect(self._on_tile_clicked)
                tile.show()
                self._mosaic_tiles[(r, c)] = tile
        self.mos_canvas.set_overlap(self.mos_view_ov.value() / 100.0)
        self.mos_canvas.set_grid(rows, cols, self._mosaic_tiles)

        if load_paths:
            try:
                paths = json.loads(load_paths)
                for k, p in paths.items():
                    if os.path.exists(p):
                        r, c = (int(x) for x in k.split(","))
                        self._set_tile_image(r, c, p)
            except Exception:         # noqa: BLE001
                pass
        self._save_settings()

    def _on_tile_clicked(self, r, c):
        if self._active_tile and self._active_tile in self._mosaic_tiles:
            self._mosaic_tiles[self._active_tile].set_active(False)
        self._active_tile = (r, c)
        self._mosaic_tiles[(r, c)].set_active(True)
        self._update_active_tile_label()

    def _update_active_tile_label(self):
        if self._active_tile:
            r, c = self._active_tile
            self.mos_active_label.setText(f"Active Tile: {r+1}×{c+1}")
        else:
            self.mos_active_label.setText("Active Tile: None")

    def _set_tile_image(self, r, c, path, bgr=None):
        """Stores the tile's source image (auto-stretched) and refits it."""
        if (r, c) not in self._mosaic_tiles:
            return
        tile = self._mosaic_tiles[(r, c)]
        if bgr is None and path and os.path.exists(path):
            bgr = cv2.imread(path)
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        if w > 660:                       # source bornée pour la mémoire
            sc = 660.0 / w
            bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)))
        bgr = auto_stretch(bgr, 0.6)      # rend les détails faibles visibles
        bgr = np.ascontiguousarray(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                            rgb.shape[1] * 3, QtGui.QImage.Format_RGB888)
        tile.set_source(QtGui.QPixmap.fromImage(qimg.copy()))
        if path:
            self._mosaic_paths[(r, c)] = path
        self._save_settings()
            
    def _load_mosaic_tile(self):
        if not self._active_tile:
            self.statusBar().showMessage("Select a tile first by clicking on it.", 5000)
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Image", self.dir_edit.text(), "Images (*.jpg *.png *.tif *.tiff)")
        if path:
            self._set_tile_image(self._active_tile[0], self._active_tile[1], path)
            
    def _clear_mosaic_tile(self):
        if not self._active_tile:
            return
        r, c = self._active_tile
        self._mosaic_paths.pop((r, c), None)
        self._mosaic_tiles[(r, c)].clear_source()
        self._save_settings()

    def _send_stack_to_mosaic(self):
        if not self._active_tile:
            self.results_solve_label.setText("Please select an Active Tile in the 'Mosaic View' tab first.")
            return
        img = self._result_img(self.results_list.currentItem())
        if img is None:
            self.results_solve_label.setText("Select a burst/stack in the list to send.")
            return
            
        # Save it to the session folder to persist between sessions
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.dir_edit.text(), f"mosaic_tile_{ts}.jpg")
        try:
            os.makedirs(self.dir_edit.text(), exist_ok=True)
            cv2.imwrite(path, img)
        except Exception:
            pass
            
        self._set_tile_image(self._active_tile[0], self._active_tile[1], path, bgr=img)
        self.results_solve_label.setText(f"Image sent successfully to Mosaic Tile {self._active_tile[0]+1}x{self._active_tile[1]+1}.")

    def _open_siril_app(self):
        import subprocess
        import sys
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-a", "Siril"])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["siril"])
            else:
                subprocess.Popen(["siril"])
            self.results_solve_label.setText("Launching Siril...")
        except Exception as e:
            self.results_solve_label.setText("Could not launch Siril: {}".format(e))

    def _sat(self):
        return self.sat_slider.value() / 100.0 if hasattr(self, "sat_slider") else 1.0

    def _disp_result(self, img, processed=False):
        """Display image for a Results entry. Stacks already flattened+stretched from
        float (`processed=True`) are shown as-is. Raw single shots get the full astro
        treatment (gradient removal + colour calibration + neutral background + MTF
        stretch + SCNR), always on."""
        if processed:
            return img
        try:
            f = remove_gradient(img.astype(np.float32), 1.0)
            f = star_white_balance(f)
            f = neutralize_background(f)
            return auto_stretch(f, 0.6, scnr=True, saturation=self._sat())
        except Exception:           # noqa: BLE001
            return auto_stretch(img, 0.6)

    def _rebuild_results(self, *args):
        self.results_list.clear()
        for idx, entry in enumerate(self._results):
            img, cap = entry[0], entry[1]
            annot = entry[2] if len(entry) > 2 else None
            processed = entry[3] if len(entry) > 3 else False
            label = cap + ("  🔭" if annot is not None else "")
            item = QtWidgets.QListWidgetItem(
                QtGui.QIcon(self._bgr_to_pixmap(self._disp_result(img, processed))), label)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setData(Qt.UserRole, idx)
            self.results_list.addItem(item)

    @Slot(int, object, str)
    def on_seq_result(self, seq_id, img, caption):
        if img is None:
            return
        if isinstance(img, np.ndarray) and img.dtype != np.uint8:
            # stacked burst: float (linear) mean -> flatten + stretch once, at full precision
            try:
                disp = auto_stretch(neutralize_background(star_white_balance(
                    remove_gradient(img.astype(np.float32), 1.0))), 0.6, scnr=True, saturation=self._sat())
            except Exception:           # noqa: BLE001
                disp = np.clip(img, 0, 255).astype(np.uint8)
            item = self._add_result(disp, caption, processed=True,
                                    raw_float=img.astype(np.float32))
            self._last_burst_idx = item.data(Qt.UserRole)
        else:
            self._last_burst_idx = self._add_result(img, caption).data(Qt.UserRole)

    def _add_result(self, img, caption, processed=False, raw_float=None):
        """Appends an image to the Results list. `processed` = already flattened+stretched
        (stacks); `raw_float` = the raw stacked mean, kept so the viewer can toggle gradient
        removal on/off."""
        idx = len(self._results)
        self._results.append((img, caption, None, processed, raw_float))
        item = QtWidgets.QListWidgetItem(
            QtGui.QIcon(self._bgr_to_pixmap(self._disp_result(img, processed))), caption)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
        item.setData(Qt.UserRole, idx)
        self.results_list.addItem(item)
        self.results_list.scrollToBottom()
        return item

    def _result_img(self, item):
        if item is None:
            return None
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self._results):
            return None
        return self._results[idx][0]

    def _result_annot(self, item):
        if item is None:
            return None
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self._results):
            return None
        entry = self._results[idx]
        return entry[2] if len(entry) > 2 else None

    def _result_processed(self, item):
        if item is None:
            return False
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self._results):
            return False
        entry = self._results[idx]
        return entry[3] if len(entry) > 3 else False

    def _result_raw(self, item):
        if item is None:
            return None
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self._results):
            return None
        entry = self._results[idx]
        return entry[4] if len(entry) > 4 else None

    def _item_for_idx(self, idx):
        for i in range(self.results_list.count()):
            it = self.results_list.item(i)
            if it.data(Qt.UserRole) == idx:
                return it
        return None

    def _delete_selected_results(self):
        items = self.results_list.selectedItems()
        if not items:
            self.results_solve_label.setText("Select one or more bursts to delete "
                                             "(Cmd/Shift-click).")
            return
        idxs = sorted({it.data(Qt.UserRole) for it in items if it.data(Qt.UserRole) is not None},
                      reverse=True)
        if not idxs:
            return
        resp = QtWidgets.QMessageBox.question(
            self, "Delete bursts",
            "Delete {} selected burst(s)? This cannot be undone.".format(len(idxs)),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for i in idxs:
            if 0 <= i < len(self._results):
                del self._results[i]
        self._rebuild_results()
        self.results_solve_label.setText("Deleted {} burst(s).".format(len(idxs)))

    def _stack_image_list(self, imgs):
        """Aligns (translation) and averages a list of BGR images. Returns (combined, n)."""
        ref = imgs[0]
        h, w = ref.shape[:2]
        rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        acc = ref.astype(np.float32); n = 1
        for im in imgs[1:]:
            im2 = cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im
            try:
                sx, sy = align_translation(rg, cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
                M = np.float32([[1, 0, -sx], [0, 1, -sy]])
                im2 = cv2.warpAffine(im2, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
            except Exception:           # noqa: BLE001
                pass
            acc += im2.astype(np.float32); n += 1
        return np.clip(acc / n, 0, 255).astype(np.uint8), n

    def _combine_all(self):
        imgs = [entry[0] for entry in self._results]
        if len(imgs) < 1:
            self.results_solve_label.setText("No bursts to stack at the moment.")
            return
        combined, n = self._stack_image_list(imgs)
        self._add_result(combined, "Stack of all {} bursts".format(n))
        self._open_image_dialog(combined, "Session Stack — {} bursts".format(n))
        self.results_solve_label.setText(
            "Session stack: {} bursts combined (added to the list).".format(n))

    def _combine_selected(self):
        items = self.results_list.selectedItems()
        imgs = [self._result_img(it) for it in items]
        imgs = [im for im in imgs if im is not None]
        if len(imgs) < 1:
            self.results_solve_label.setText(
                "Select one or more bursts first (Cmd/Shift-click).")
            return
        if len(imgs) == 1:
            self.results_solve_label.setText(
                "Only one burst selected — select at least two to stack, or use it as is.")
            return
        combined, n = self._stack_image_list(imgs)
        self._add_result(combined, "Stack of {} selected bursts".format(n))
        self._open_image_dialog(combined, "Stack of {} selected bursts".format(n))
        self.results_solve_label.setText(
            "Selected stack: {} bursts combined (added to the list).".format(n))

    def _solve_result(self):
        item = self.results_list.currentItem()
        img = self._result_img(item)
        if img is None:
            self.results_solve_label.setText("Select a burst in the list first.")
            return
        self._solving_idx = item.data(Qt.UserRole) if item else None
        self.solve_res_btn.setEnabled(False)
        self.results_solve_label.setText("Plate-solving in progress…")
        self._solver_res = SolveWorker(self._focal(), image=img,
                                       online=self._solve_online(), api_key=self.solve_key.text())
        self._solver_res.done.connect(self._solve_result_done)
        self._solver_res.start()

    @Slot(str, object, float, float, float, float)
    def _solve_result_done(self, msg, annot, ra, dec, rot, fov):
        self.solve_res_btn.setEnabled(True)
        self.results_solve_label.setText(msg)
        if annot is not None:
            # attach the annotated image to the burst that was solved, so the burst
            # preview gets a "show annotated" toggle (instead of a duplicate entry).
            idx = getattr(self, "_solving_idx", None)
            if idx is not None and 0 <= idx < len(self._results):
                entry = self._results[idx]
                img, cap = entry[0], entry[1]
                proc = entry[3] if len(entry) > 3 else False
                raw = entry[4] if len(entry) > 4 else None
                self._results[idx] = (img, cap, annot, proc, raw)
                it = self._item_for_idx(idx)
                if it is not None and "🔭" not in it.text():
                    it.setText(it.text() + "  🔭")
                self._save_annotated(annot, "annot_burst")
                self._open_image_dialog(img, cap, annot=annot, processed=proc, raw=raw)
            else:
                label = "🔭 Solved"
                if math.isfinite(ra) and math.isfinite(dec):
                    label = "🔭 Solved RA {:.2f} Dec {:+.2f}".format(ra, dec)
                self._add_result(annot, label)
                self._save_annotated(annot, "annot_burst")
        if math.isfinite(ra) and math.isfinite(dec):
            self._solved_to_skymap(ra, dec, "Solved field", rot)
            # Stay where you are: during a session you watch the picture, not the tabs.
            # The Sky Map is updated silently; the end-of-capture card offers to open it.
            self._solved_ready = True
            self.statusBar().showMessage(
                "🔭 Field identified — the Sky Map now shows where the camera points.", 9000)

    def _solved_to_skymap(self, ra, dec, label, rot=float("nan")):
        """Centre the Sky Map on the solved field (current pointing). If a session goal is
        set, show both (goal + current pointing) so you can see which way to move.
        Also applies the solved field rotation to the camera frame, if available."""
        if not self.skymap or not (math.isfinite(ra) and math.isfinite(dec)):
            return
        img = getattr(self, "_last_capture_bgr", None)
        if img is not None and hasattr(img, "shape") and img.ndim >= 2:
            self.skymap.set_portrait(img.shape[0] > img.shape[1])   # 3:4 vs 4:3
        self.skymap.set_target(label, ra % 360.0, dec)   # current pointing = solved centre
        if math.isfinite(rot):                            # apply measured field rotation
            self.skymap.set_cam_angle(rot)
            if hasattr(self, "sky_rot"):
                self.sky_rot.blockSignals(True)
                self.sky_rot.setValue(int(round(((rot + 180) % 360) - 180)))
                self.sky_rot.blockSignals(False)
        has_goal = bool(getattr(self, "_session_goal", None))
        if has_goal:
            self.skymap.show_goal = self.sky_show_goal.isChecked() if hasattr(self, "sky_show_goal") else True
            self.skymap.set_frame_allsky(True)           # show current field in all-sky
            if hasattr(self, "sky_show_frame"):
                self.sky_show_frame.blockSignals(True); self.sky_show_frame.setChecked(True)
                self.sky_show_frame.blockSignals(False)
        else:
            self.skymap.set_frame_allsky(True)
        self.skymap.set_mode("allsky")                   # the default, most readable view
        if hasattr(self, "sky_mode"):
            self.sky_mode.setCurrentText("All-sky")
        self.skymap.update()
        if self._current_hips():
            self._schedule_bg()

    def _bgr_to_pixmap(self, bgr):
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg.copy())

    def _open_image_dialog(self, bgr, title, annot=None, processed=False, raw=None):
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle(title)
        dlg.resize(900, 650)
        v = QtWidgets.QVBoxLayout(dlg)
        view = ZoomableView()
        state = {"annot": False, "grad": True}

        def _base():
            if state["grad"] or raw is None:
                return self._disp_result(bgr, processed)      # stored (gradient removed)
            try:                                              # re-render WITHOUT gradient
                f = star_white_balance(raw.astype(np.float32))
                f = neutralize_background(f)
                return auto_stretch(f, 0.6, scnr=True, saturation=self._sat())
            except Exception:           # noqa: BLE001
                return self._disp_result(bgr, processed)

        def _render():
            if state["annot"] and annot is not None:
                view.setPixmap(self._bgr_to_pixmap(annot))
            else:
                view.setPixmap(self._bgr_to_pixmap(_base()))

        row = QtWidgets.QHBoxLayout()
        if raw is not None:
            grad_chk = QtWidgets.QCheckBox("Remove gradient")
            grad_chk.setChecked(True)
            grad_chk.setToolTip("Show the image with or without background/gradient removal.")

            def _g(on):
                state["grad"] = bool(on); _render()
            grad_chk.toggled.connect(_g)
            row.addWidget(grad_chk)
        if annot is not None:
            toggle = QtWidgets.QPushButton("🔭 Show annotated")
            toggle.setToolTip("Switch between your image and the plate-solved / annotated one.")

            def _toggle():
                state["annot"] = not state["annot"]
                toggle.setText("🖼 Show original" if state["annot"] else "🔭 Show annotated")
                _render()
            toggle.clicked.connect(_toggle)
            row.addWidget(toggle)
        hist_btn = QtWidgets.QPushButton("📊 Histogram")
        hist_btn.setToolTip("Interactive histogram stretch (black/white point + midtones) to "
                            "boost the faintest details, like Siril.")
        hist_btn.clicked.connect(lambda: self._open_hist_stretch(raw, bgr, processed, title))
        row.addWidget(hist_btn)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(view)
        _render()
        dlg.show()
        self._img_dialog = dlg

    def _open_hist_stretch(self, raw, bgr, processed, title):
        # linear source in [0,1] — from the raw stacked mean if available (most headroom)
        if raw is not None:
            try:
                f = neutralize_background(star_white_balance(raw.astype(np.float32)))
            except Exception:           # noqa: BLE001
                f = raw.astype(np.float32)
            ref = float(np.percentile(f, 99.95)) + 1e-6
            src = np.clip(f / ref, 0.0, 1.0)
        else:
            src = self._disp_result(bgr, processed).astype(np.float32) / 255.0
        lum = src.mean(axis=2) if src.ndim == 3 else src
        counts, edges = np.histogram(lum, bins=160, range=(0.0, 1.0))
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Histogram stretch · " + title)
        dlg.resize(940, 760)
        v = QtWidgets.QVBoxLayout(dlg)
        view = ZoomableView(); v.addWidget(view, 1)
        hist = pg.PlotWidget(); hist.setMaximumHeight(150); hist.setBackground("#0d0d0d")
        hist.setLogMode(y=True); hist.setMouseEnabled(x=False, y=False)
        hist.plot(edges[:-1], np.maximum(counts, 1), pen=pg.mkPen("#8a95a4"))
        l_black = pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen("#3b82f6", width=2))
        l_white = pg.InfiniteLine(1.0, angle=90, pen=pg.mkPen("#ef4444", width=2))
        hist.addItem(l_black); hist.addItem(l_white)
        v.addWidget(hist)
        form = QtWidgets.QHBoxLayout()
        black = QtWidgets.QSlider(Qt.Horizontal); black.setRange(0, 400); black.setValue(0)
        mid = QtWidgets.QSlider(Qt.Horizontal); mid.setRange(5, 95); mid.setValue(30)
        white = QtWidgets.QSlider(Qt.Horizontal); white.setRange(300, 1000); white.setValue(1000)
        sat = QtWidgets.QSlider(Qt.Horizontal); sat.setRange(100, 300); sat.setValue(140)
        for lab, wdg in (("Black", black), ("Midtones", mid), ("White", white), ("Sat", sat)):
            box = QtWidgets.QVBoxLayout(); box.addWidget(QtWidgets.QLabel(lab)); box.addWidget(wdg)
            form.addLayout(box)
        v.addLayout(form)

        def _render():
            b = black.value() / 1000.0; m = mid.value() / 100.0; wv = white.value() / 1000.0
            wv = max(wv, b + 0.02)
            y = np.clip((src - b) / (wv - b), 0.0, 1.0)
            out = _mtf(y, m)
            res = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
            s = sat.value() / 100.0
            if s != 1.0 and res.ndim == 3:
                hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
                res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            view.setPixmap(self._bgr_to_pixmap(res))
            l_black.setValue(b); l_white.setValue(wv)
        for wdg in (black, mid, white, sat):
            wdg.valueChanged.connect(_render)
        _render()
        dlg.show()
        self._hist_dialog = dlg

    def _open_result(self, item):
        img = self._result_img(item)
        if img is not None:
            self._open_image_dialog(img, item.text(), annot=self._result_annot(item),
                                    processed=self._result_processed(item),
                                    raw=self._result_raw(item))

    def _update_tfov(self, *args):
        if hasattr(self, "t_fov"):
            self.t_fov.setText("Field {:.0f} mm: {}".format(
                self.t_focal.value(), fov_str(self.t_focal.value())))

    def _build_targets_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab); lay.setSpacing(8)
        form = QtWidgets.QHBoxLayout()
        self.t_lat = QtWidgets.QDoubleSpinBox(); self.t_lat.setRange(-90, 90)
        self.t_lat.setDecimals(4); self.t_lat.setValue(48.8566)
        self.t_lon = QtWidgets.QDoubleSpinBox(); self.t_lon.setRange(-180, 180)
        self.t_lon.setDecimals(4); self.t_lon.setValue(2.3522)
        self.t_lat.valueChanged.connect(self._push_site)
        self.t_lon.valueChanged.connect(self._push_site)
        self.t_date = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        self.t_date.setCalendarPopup(True)
        self.t_focal = QtWidgets.QDoubleSpinBox(); self.t_focal.setRange(8, 2000)
        self.t_focal.setValue(135); self.t_focal.setSuffix(" mm")
        self.t_minalt = QtWidgets.QDoubleSpinBox(); self.t_minalt.setRange(0, 80)
        self.t_minalt.setValue(30); self.t_minalt.setSuffix(" °")
        self.t_maxmag = QtWidgets.QDoubleSpinBox(); self.t_maxmag.setRange(2, 20)
        self.t_maxmag.setValue(13); self.t_maxmag.setSingleStep(0.5)
        self.t_maxmag.setSuffix(" mag")
        self.t_maxmag.setToolTip("Faintest objects to rank (full Messier+NGC+IC catalogue "
                                 "is ~13 000 objects). 20 = no limit (slow).")
        self.t_count = QtWidgets.QSpinBox(); self.t_count.setRange(10, 1000)
        self.t_count.setValue(150); self.t_count.setSingleStep(10)
        self.t_count.setToolTip("How many targets to list.")
        for lbl, wdg in [("Lat", self.t_lat), ("Lon", self.t_lon), ("Date", self.t_date),
                         ("Focal", self.t_focal), ("h min", self.t_minalt),
                         ("max mag", self.t_maxmag), ("count", self.t_count)]:
            form.addWidget(QtWidgets.QLabel(lbl)); form.addWidget(wdg)
        self.t_locate = QtWidgets.QPushButton("📍 Locate me")
        self.t_locate.setToolTip("Detect your location automatically (IP-based, needs Internet).")
        self.t_locate.clicked.connect(self._locate_me)
        form.addWidget(self.t_locate)
        self.t_btn = QtWidgets.QPushButton("Calculate Targets")
        self.t_btn.clicked.connect(self._run_astro)
        form.addWidget(self.t_btn)
        lay.addLayout(form)
        # keep the sky map's site in sync with these coordinates
        self.t_lat.valueChanged.connect(self._sync_sky_site)
        self.t_lon.valueChanged.connect(self._sync_sky_site)
        self.t_fov = QtWidgets.QLabel("")
        self.t_fov.setStyleSheet("color:#2dd4bf; padding:2px 4px;")
        self.t_focal.valueChanged.connect(self._update_tfov)
        self.t_header = _AutoHideLabel("")                  # night window + Moon
        self.t_header.setStyleSheet("color:#cbd5e1; padding:2px 4px;")
        hrow = QtWidgets.QHBoxLayout(); hrow.setSpacing(10)
        hrow.addWidget(self.t_fov); hrow.addWidget(self.t_header); hrow.addStretch(1)
        hl = QtWidgets.QLabel("Where am I"); hl.setObjectName("cap"); hrow.addWidget(hl)
        self.hz_combo = QtWidgets.QComboBox()
        self.hz_combo.setToolTip("Pick your place (e.g. Home): its position and its local "
                                 "horizon — trees, the neighbour's house — apply at once.")
        self.hz_combo.activated.connect(lambda _i: self._on_horizon_choice())
        hrow.addWidget(self.hz_combo)
        self.hz_edit = QtWidgets.QPushButton("Edit horizon…")
        self.hz_edit.clicked.connect(self._edit_horizon)
        hrow.addWidget(self.hz_edit)
        lay.addLayout(hrow)

        vrow = QtWidgets.QHBoxLayout(); vrow.setSpacing(8)
        sl = QtWidgets.QLabel("Sort"); sl.setObjectName("cap"); vrow.addWidget(sl)
        self.t_sort = QtWidgets.QComboBox()
        self.t_sort.addItem("Largest for my focal", "size")
        self.t_sort.addItem("Best tonight", "score")
        self.t_sort.addItem("Highest", "alt")
        self.t_sort.addItem("Longest visible", "clear")
        self.t_sort.setToolTip("Largest: the objects that fill the most of your frame at "
                               "the focal length set above — the ones worth this rig.")
        self.t_sort.activated.connect(lambda *_: self._apply_target_view())
        vrow.addWidget(self.t_sort)
        cl_ = QtWidgets.QLabel("Collection"); cl_.setObjectName("cap"); vrow.addWidget(cl_)
        self.t_season = QtWidgets.QComboBox()
        self.t_season.addItem("Tonight (all)", "all")
        self.t_season.addItem("Autumn sky", "autumn")
        self.t_season.addItem("Winter sky", "winter")
        self.t_season.addItem("Spring sky", "spring")
        self.t_season.addItem("Summer sky", "summer")
        self.t_season.addItem("Messier only", "messier")
        self.t_season.addItem("Fits my frame", "fits")
        self.t_season.setToolTip("Seasons group objects by right ascension — the part of "
                                 "the sky that culminates in the evening at that time of "
                                 "year. They are shown only if they are up tonight.")
        self.t_season.activated.connect(lambda *_: self._apply_target_view())
        vrow.addWidget(self.t_season)
        self.t_count_lbl = QtWidgets.QLabel(""); self.t_count_lbl.setObjectName("cap")
        vrow.addWidget(self.t_count_lbl); vrow.addStretch(1)
        lay.addLayout(vrow)
        QtCore.QTimer.singleShot(0, self._refresh_horizon_combo)
        QtCore.QTimer.singleShot(0, self._update_tfov)


        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("🔍 Search"))
        self.t_search = QtWidgets.QLineEdit()
        self.t_search.setPlaceholderText("Any object: M31, NGC 7000, IC1805, Andromeda, Orion Nebula…")
        self.t_search.returnPressed.connect(self._search_target)
        self.t_search.textChanged.connect(self._filter_targets)
        srow.addWidget(self.t_search, 1)
        self.t_search_btn = QtWidgets.QPushButton("Open in Sky Map")
        self.t_search_btn.clicked.connect(self._search_target)
        srow.addWidget(self.t_search_btn)
        lay.addLayout(srow)
        self.t_search_msg = _AutoHideLabel("")
        self.t_search_msg.setStyleSheet("color:#94a3b8;")
        lay.addWidget(self.t_search_msg)

        self.t_table = QtWidgets.QTableWidget(0, 9)
        self.t_table.setHorizontalHeaderLabels(
            ["Object", "Name", "Size", "Max Alt", "Clear", "Transit", "ΔMoon · avoid",
             "Framing", "rank"])
        self.t_table.horizontalHeaderItem(2).setToolTip(
            "Apparent size, and how much of the frame width it fills at your focal length.")
        self.t_table.horizontalHeaderItem(4).setToolTip(
            "Hours tonight above h min — and above your local horizon when one is active.")
        self.t_table.horizontalHeader().setStretchLastSection(True)
        self.t_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.t_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.t_table.itemDoubleClicked.connect(self._target_row_to_skymap)
        self.t_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.t_table.customContextMenuRequested.connect(self._target_context_menu)
        self.t_table.setSortingEnabled(True)
        self.t_table.setColumnHidden(8, True)      # holds the computed order
        self.t_table.setMinimumHeight(420)
        self.t_table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                   QtWidgets.QSizePolicy.Policy.Expanding)
        self.t_table.verticalHeader().setDefaultSectionSize(22)
        lay.addWidget(self.t_table, 1)
        grow = QtWidgets.QHBoxLayout()
        self.t_goto_btn = QtWidgets.QPushButton("🧭 GoTo the selected target")
        self.t_goto_btn.setToolTip("Makes the selected row the session target and opens "
                                   "Manual GoTo, ready to slew.")
        self.t_goto_btn.setMinimumHeight(32)
        self.t_goto_btn.clicked.connect(self._target_row_goto)
        grow.addWidget(self.t_goto_btn)
        grow.addStretch(1)
        lay.addLayout(grow)
        grow.insertWidget(grow.count() - 1, _HelpButton(
            "Set location + date, then Calculate Targets. "
            "Double-click a target to open it in the Sky Map."))
        return tab

    def _build_skymap_tab(self):
        self._active_sky_target = None
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)
        try:
            import sky_map
            here = os.path.dirname(os.path.abspath(__file__))
            self.skymap = sky_map.SkyMap(data_dir=here)
            self._sky_mod = sky_map
        except Exception as e:        # noqa: BLE001
            self.skymap = None
            lay.addWidget(QtWidgets.QLabel(
                "Sky Map unavailable: {}\nPlace sky_map.py, sky_stars.csv, "
                "sky_constlines.json and catalog_ngc.csv next to this script.".format(e)))
            return tab

        # search bar: jump to any object (DSO, star, constellation, planet)
        sbar = QtWidgets.QHBoxLayout()
        sbar.addWidget(QtWidgets.QLabel("🔍"))
        self.sky_search = QtWidgets.QLineEdit()
        self.sky_search.setPlaceholderText("Go to… (M31, NGC7000, Vega, Orion, Mars, Andromeda…)")
        self.sky_search.returnPressed.connect(self._sky_search_go)
        sbar.addWidget(self.sky_search, 1)
        sbtn = QtWidgets.QPushButton("Go"); sbtn.clicked.connect(self._sky_search_go)
        sbar.addWidget(sbtn)
        self.sky_search_msg = QtWidgets.QLabel(""); self.sky_search_msg.setStyleSheet("color:#94a3b8;")
        sbar.addWidget(self.sky_search_msg)
        lay.addLayout(sbar)

        bar = QtWidgets.QHBoxLayout()
        self.sky_mode = QtWidgets.QComboBox(); self.sky_mode.addItems(["All-sky", "Framing"])
        self.sky_mode.currentTextChanged.connect(
            lambda t: self.skymap.set_mode("framing" if t == "Framing" else "allsky"))
        bar.addWidget(QtWidgets.QLabel("Mode")); bar.addWidget(self.sky_mode)
        bar.addWidget(QtWidgets.QLabel("Focal"))
        self.sky_focal = QtWidgets.QDoubleSpinBox(); self.sky_focal.setRange(8, 2000)
        self.sky_focal.setValue(self._focal()); self.sky_focal.setSuffix(" mm")
        self.sky_focal.valueChanged.connect(lambda v: self.skymap.set_focal(v))
        bar.addWidget(self.sky_focal)
        bar.addWidget(QtWidgets.QLabel("Rotation"))
        self.sky_rot = QtWidgets.QSpinBox(); self.sky_rot.setRange(-180, 180); self.sky_rot.setValue(0)
        self.sky_rot.setSuffix("°"); self.sky_rot.setWrapping(True)
        self.sky_rot.setToolTip("Camera/frame rotation (equatorial). 0 = aligned to RA/Dec.")
        self.sky_rot.valueChanged.connect(lambda v: self.skymap.set_cam_angle(v))
        bar.addWidget(self.sky_rot)
        bar.addWidget(QtWidgets.QLabel("Goal rot"))
        self.sky_goal_rot = QtWidgets.QSpinBox(); self.sky_goal_rot.setRange(-180, 180)
        self.sky_goal_rot.setValue(0); self.sky_goal_rot.setSuffix("°")
        self.sky_goal_rot.setWrapping(True)
        self.sky_goal_rot.setToolTip("Rotation of the GOAL frame only — independent of the "
                                     "solved/current field rotation. Set the framing you want "
                                     "to reach.")
        self.sky_goal_rot.valueChanged.connect(lambda v: self.skymap.set_goal_angle(v))
        bar.addWidget(self.sky_goal_rot)
        # ---- display options, grouped in a popup panel to declutter the bar ----
        self.sky_show_dso = QtWidgets.QCheckBox("Deep-sky objects (DSO)"); self.sky_show_dso.setChecked(True)
        self.sky_show_con = QtWidgets.QCheckBox("Constellation lines"); self.sky_show_con.setChecked(True)
        self.sky_show_cn = QtWidgets.QCheckBox("Constellation names"); self.sky_show_cn.setChecked(True)
        self.sky_show_nm = QtWidgets.QCheckBox("DSO names"); self.sky_show_nm.setChecked(True)
        self.sky_show_sn = QtWidgets.QCheckBox("Star names"); self.sky_show_sn.setChecked(True)
        self.sky_show_sn.setToolTip("Label the bright named stars (Vega, Betelgeuse…); "
                                    "more appear as you zoom in.")
        self.sky_show_traj = QtWidgets.QCheckBox("Target trajectory"); self.sky_show_traj.setChecked(True)
        self.sky_show_traj.setToolTip("Dashed path of the target across the sky tonight "
                                      "(rise → transit → set), shown in All-sky.")
        for cb in (self.sky_show_dso, self.sky_show_con, self.sky_show_cn, self.sky_show_nm,
                   self.sky_show_sn, self.sky_show_traj):
            cb.toggled.connect(self._update_sky_flags)
        self.sky_show_frame = QtWidgets.QCheckBox("Camera frame in all-sky")
        self.sky_show_frame.setToolTip("Overlay your camera field on the all-sky horizon view.")
        self.sky_show_frame.toggled.connect(
            lambda v: self.skymap.set_frame_allsky(v) if self.skymap else None)
        self.sky_show_iss = QtWidgets.QCheckBox("ISS position")
        self.sky_show_iss.setToolTip("Show the current ISS position on the map "
                                     "(uses the TLE from the ISS passes window).")
        self.sky_show_iss.toggled.connect(self._toggle_iss)
        self.sky_show_comets = QtWidgets.QCheckBox("Comets"); self.sky_show_comets.setChecked(True)
        self.sky_show_comets.toggled.connect(
            lambda v: self.skymap.set_show_comets(v) if self.skymap else None)
        self.sky_show_goal = QtWidgets.QCheckBox("🎯 Session goal + arrow")
        self.sky_show_goal.setToolTip("Show the session target and the direction to move to "
                                      "reach it, in both Sky Map views.")
        self.sky_show_goal.setChecked(True)
        self.sky_show_goal.toggled.connect(self._on_toggle_goal)

        self.sky_comet_maglim = QtWidgets.QDoubleSpinBox()
        self.sky_comet_maglim.setRange(6.0, 16.0); self.sky_comet_maglim.setSingleStep(0.5)
        self.sky_comet_maglim.setValue(QtCore.QSettings("NOUT", "NOUT").value(
            "comet_maglim", 11.0, type=float))
        self.sky_comet_maglim.setPrefix("≤ m")
        self.sky_comet_maglim.setToolTip("Only keep comets brighter than this magnitude "
                                         "(capturable with your setup).")
        self.sky_comet_maglim.valueChanged.connect(
            lambda v: self.skymap and self.skymap.set_comet_maglim(v))
        if self.skymap:
            self.skymap.set_comet_maglim(self.sky_comet_maglim.value())

        # popup panel
        disp_panel = QtWidgets.QWidget()
        dp = QtWidgets.QVBoxLayout(disp_panel)
        dp.setContentsMargins(12, 10, 12, 10); dp.setSpacing(5)
        dp.addWidget(QtWidgets.QLabel("<b>Show on the map</b>"))
        for cb in (self.sky_show_dso, self.sky_show_con, self.sky_show_cn, self.sky_show_nm,
                   self.sky_show_sn, self.sky_show_traj, self.sky_show_frame,
                   self.sky_show_comets, self.sky_show_iss, self.sky_show_goal):
            dp.addWidget(cb)
        dp.addSpacing(6)
        mlrow = QtWidgets.QHBoxLayout()
        mlrow.addWidget(QtWidgets.QLabel("Comet magnitude limit:"))
        mlrow.addWidget(self.sky_comet_maglim); mlrow.addStretch(1)
        dp.addLayout(mlrow)
        self.sky_display_menu = QtWidgets.QMenu(self)
        wa = QtWidgets.QWidgetAction(self.sky_display_menu); wa.setDefaultWidget(disp_panel)
        self.sky_display_menu.addAction(wa)
        self.sky_display_btn = QtWidgets.QToolButton()
        self.sky_display_btn.setText("Display ▾")
        self.sky_display_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.sky_display_btn.setMenu(self.sky_display_menu)
        bar.addWidget(self.sky_display_btn)

        # action buttons stay visible on the bar
        self.sky_iss_btn = QtWidgets.QPushButton("🛰 ISS passes")
        self.sky_iss_btn.setToolTip("Predict upcoming visible passes of the ISS (or any "
                                    "satellite TLE) from your site.")
        self.sky_iss_btn.clicked.connect(self._open_iss_passes)
        bar.addWidget(self.sky_iss_btn)
        self.sky_comet_btn = QtWidgets.QPushButton("⟳ Update comets")
        self.sky_comet_btn.setToolTip("Download current comets from the Minor Planet Center "
                                      "and keep only the bright (capturable) ones.")
        self.sky_comet_btn.clicked.connect(self._update_comets)
        bar.addWidget(self.sky_comet_btn)
        self.sky_setgoal_btn = QtWidgets.QPushButton("🎯 Set goal here")
        self.sky_setgoal_btn.setToolTip("Make the current view centre the session goal — "
                                        "use it to target an empty region (no object needed): "
                                        "pan/zoom to frame the area, then click.")
        self.sky_setgoal_btn.clicked.connect(self._set_goal_here)
        bar.addWidget(self.sky_setgoal_btn)
        self.sky_mgoto_btn = QtWidgets.QPushButton("🧭 Manual GoTo")
        self.sky_mgoto_btn.setToolTip("Angles to dial by hand on the two axes (polar "
                                      "circle + graduated L-bracket) to land on the "
                                      "target right now — no motorised GoTo needed.")
        self.sky_mgoto_btn.clicked.connect(lambda: self._open_manual_goto("point"))
        bar.addWidget(self.sky_mgoto_btn)
        bar.addStretch(1)
        # single, unified target readout (goal is shown on the map itself)
        self.sky_goal_label = QtWidgets.QLabel("")      # kept for internal updates (hidden)
        self.sky_goal_label.setVisible(False)
        self.sky_target_label = QtWidgets.QLabel("Target: —")
        self.sky_target_label.setStyleSheet("color:#f59e0b; font-weight:bold;")
        bar.addWidget(self.sky_target_label)
        lay.addLayout(bar)

        # mosaic planning row
        mbar = QtWidgets.QHBoxLayout()
        self.sky_mos_show = QtWidgets.QCheckBox("Mosaic grid"); self.sky_mos_show.setChecked(False)
        self.sky_mos_rows = QtWidgets.QSpinBox(); self.sky_mos_rows.setRange(1, 10); self.sky_mos_rows.setValue(2)
        self.sky_mos_cols = QtWidgets.QSpinBox(); self.sky_mos_cols.setRange(1, 10); self.sky_mos_cols.setValue(2)
        self.sky_mos_ov = QtWidgets.QSpinBox(); self.sky_mos_ov.setRange(0, 60); self.sky_mos_ov.setValue(20); self.sky_mos_ov.setSuffix(" %")
        for wdg in (self.sky_mos_show, self.sky_mos_rows, self.sky_mos_cols, self.sky_mos_ov):
            (wdg.toggled if isinstance(wdg, QtWidgets.QCheckBox) else wdg.valueChanged).connect(self._update_sky_mosaic)
        mbar.addWidget(self.sky_mos_show)
        mbar.addWidget(QtWidgets.QLabel("rows")); mbar.addWidget(self.sky_mos_rows)
        mbar.addWidget(QtWidgets.QLabel("cols")); mbar.addWidget(self.sky_mos_cols)
        mbar.addWidget(QtWidgets.QLabel("overlap")); mbar.addWidget(self.sky_mos_ov)
        self.sky_export_btn = QtWidgets.QPushButton("📍 Export panel centers (CSV)")
        self.sky_export_btn.clicked.connect(self._export_panel_centers)
        mbar.addWidget(self.sky_export_btn)
        self.sky_send_mos_btn = QtWidgets.QPushButton("➡ Set grid in Mosaic View")
        self.sky_send_mos_btn.clicked.connect(self._sky_to_mosaic_view)
        mbar.addWidget(self.sky_send_mos_btn)
        self.sky_aladin_btn = QtWidgets.QPushButton("🌐 Open in Aladin Lite")
        self.sky_aladin_btn.clicked.connect(self._open_aladin)
        mbar.addWidget(self.sky_aladin_btn)
        mbar.addStretch(1)
        mbar.addWidget(QtWidgets.QLabel("Background"))
        self.sky_survey = QtWidgets.QComboBox()
        for label, _hips in self._SURVEYS:
            self.sky_survey.addItem(label)
        self.sky_survey.setToolTip("Real sky survey shown behind the framing map "
                                   "(like Aladin). Needs Internet.")
        self.sky_survey.currentIndexChanged.connect(self._on_survey_change)
        mbar.addWidget(self.sky_survey)
        self.sky_preload_btn = QtWidgets.QPushButton("⤓ Pre-load target (HD)")
        self.sky_preload_btn.setToolTip("Cache the target zone in high resolution (all useful "
                                        "zooms) so it stays sharp offline. Needs Internet now.")
        self.sky_preload_btn.clicked.connect(self._preload_zone)
        mbar.addWidget(self.sky_preload_btn)
        lay.addLayout(mbar)

        # date + time row (drives the real-time sky and the framing altitude)
        self._sky_time_row = QtWidgets.QWidget()
        trow = QtWidgets.QHBoxLayout(self._sky_time_row); trow.setContentsMargins(0, 0, 0, 0)
        trow.addWidget(QtWidgets.QLabel("Date"))
        self.sky_date = QtWidgets.QDateEdit(self.t_date.date()); self.sky_date.setCalendarPopup(True)
        self.sky_date.dateChanged.connect(self._update_sky_time)
        trow.addWidget(self.sky_date)
        trow.addWidget(QtWidgets.QLabel("Local time"))
        self.sky_time = QtWidgets.QSlider(Qt.Horizontal); self.sky_time.setRange(0, 1439)
        self.sky_time.setValue(22 * 60)
        self.sky_time.valueChanged.connect(self._update_sky_time)
        trow.addWidget(self.sky_time, 1)
        self.sky_time_lbl = QtWidgets.QLabel("22:00")
        trow.addWidget(self.sky_time_lbl)
        self.sky_now_btn = QtWidgets.QPushButton("Now"); self.sky_now_btn.clicked.connect(self._sky_now)
        trow.addWidget(self.sky_now_btn)
        self.sky_live = QtWidgets.QCheckBox("Live")
        self.sky_live.setToolTip("Follow the real sky in real time (updates every 30 s).")
        self.sky_live.toggled.connect(self._toggle_sky_live)
        trow.addWidget(self.sky_live)
        lay.addWidget(self._sky_time_row)
        self._sky_live_timer = QtCore.QTimer(self)
        self._sky_live_timer.setInterval(30000)
        self._sky_live_timer.timeout.connect(self._sky_now)

        # map + side image panel
        midrow = QtWidgets.QHBoxLayout()
        midrow.addWidget(self.skymap, stretch=1)
        side = QtWidgets.QVBoxLayout()
        self.sky_img_chk = QtWidgets.QCheckBox("Image (online)")
        self.sky_img_chk.setToolTip("Fetch a real sky image of the selected target "
                                    "(needs Internet).")
        self.sky_img_chk.setChecked(False)
        side.addWidget(self.sky_img_chk)
        self.sky_image = ClickableLabel("Select an object\nto load its image.\n\n(click to enlarge)")
        self.sky_image.setFixedSize(240, 240)
        self.sky_image.setAlignment(Qt.AlignCenter)
        self.sky_image.setCursor(Qt.PointingHandCursor)
        self.sky_image.setToolTip("Click to view full screen")
        self.sky_image.setStyleSheet("border:1px solid #333; background:#0a0a12; color:#888;")
        self.sky_image.clicked.connect(self._open_image_fullscreen)
        side.addWidget(self.sky_image)
        self.sky_img_btn = QtWidgets.QPushButton("⟳ Load image")
        self.sky_img_btn.clicked.connect(lambda: self._load_dso_image(force=True))
        side.addWidget(self.sky_img_btn)
        self.sky_img_full_btn = QtWidgets.QPushButton("⛶ Full screen")
        self.sky_img_full_btn.clicked.connect(self._open_image_fullscreen)
        side.addWidget(self.sky_img_full_btn)
        side.addStretch(1)
        midrow.addLayout(side)
        lay.addLayout(midrow, stretch=1)

        self.sky_status = QtWidgets.QLabel("")
        self.sky_status.setStyleSheet("color:#cbd5e1;"); self.sky_status.setWordWrap(True)
        lay.addWidget(self.sky_status)

        # Night altitude curve (collapsible)
        self.sky_curve_chk = QtWidgets.QCheckBox("📈 Night altitude curve")
        self.sky_curve_chk.setToolTip("Altitude of the current target through the night, with "
                                      "the astronomical-night band, your min altitude and transit.")
        self.sky_curve_chk.toggled.connect(self._on_curve_toggle)
        lay.addWidget(self.sky_curve_chk)
        self.sky_curve = pg.PlotWidget()
        self.sky_curve.setFixedHeight(190)
        self.sky_curve.setBackground("#0b0e16")
        self.sky_curve.setLabel("left", "Altitude", units="°")
        self.sky_curve.showGrid(x=True, y=True, alpha=0.2)
        self.sky_curve.setVisible(False)
        lay.addWidget(self.sky_curve)

        self.skymap.targetPicked.connect(self._on_sky_pick)
        self.skymap.targetPicked.connect(lambda *_: self._update_night_curve())
        self.skymap.viewChanged.connect(self._schedule_bg)
        self._bg_timer = QtCore.QTimer(self); self._bg_timer.setSingleShot(True)
        self._bg_timer.setInterval(350); self._bg_timer.timeout.connect(self._do_bg_fetch)
        self._bg_req = None
        self.skymap.set_focal(self._focal())
        self.skymap.set_site(self.t_lat.value(), self.t_lon.value())
        self._update_sky_time()
        return tab

    def _on_curve_toggle(self, on):
        self.sky_curve.setVisible(on)
        if on:
            self._update_night_curve()

    def _sun_radec(self, dt_utc):
        return self._body_radec("sun", dt_utc)

    def _body_radec(self, key, dt_utc):
        try:
            import astropy.units as u
            from astropy.coordinates import EarthLocation, get_body
            from astropy.time import Time
            loc = EarthLocation(lat=self.skymap.lat * u.deg, lon=self.skymap.lon * u.deg,
                                height=300 * u.m)
            b = get_body(key, Time(dt_utc), loc)          # GCRS direct (geocentric)
            return float(b.ra.deg), float(b.dec.deg)
        except Exception:             # noqa: BLE001
            return None

    def _update_night_curve(self):
        if not getattr(self, "sky_curve_chk", None) or not self.sky_curve_chk.isChecked():
            return
        if not self.skymap or self.skymap.dt_utc is None:
            return
        try:
            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo
            sm = self._sky_mod
            lat, lon = self.skymap.lat, self.skymap.lon
            ra, dec = self.skymap.center_ra, self.skymap.center_dec
            tz = ZoneInfo("Europe/Paris")
            qd = self.sky_date.date()
            start = datetime(qd.year(), qd.month(), qd.day(), 16, 0, tzinfo=tz)  # 16:00 local
            hrs = np.arange(0.0, 16.001, 0.25)            # → 08:00 next day, 15-min steps
            mid_utc = (start + timedelta(hours=8)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            sun_rd = self._sun_radec(mid_utc)
            moon_rd = self._body_radec("moon", mid_utc)
            alts = np.empty_like(hrs); sun = np.empty_like(hrs); moon = np.empty_like(hrs)
            for i, h in enumerate(hrs):
                ut = (start + timedelta(hours=float(h))).astimezone(
                    ZoneInfo("UTC")).replace(tzinfo=None)
                lst = sm.lst_deg(ut, lon)
                a, _az = sm.radec_to_altaz(np.array([ra]), np.array([dec]), lst, lat)
                alts[i] = a[0]
                if sun_rd:
                    sa, _ = sm.radec_to_altaz(np.array([sun_rd[0]]), np.array([sun_rd[1]]), lst, lat)
                    sun[i] = sa[0]
                if moon_rd:
                    ma, _ = sm.radec_to_altaz(np.array([moon_rd[0]]), np.array([moon_rd[1]]), lst, lat)
                    moon[i] = ma[0]
            self.sky_curve.clear()
            # astronomical-night band (sun < -18°)
            if sun_rd is not None:
                dark = np.where(sun < -18.0)[0]
                if len(dark):
                    reg = pg.LinearRegionItem(values=(hrs[dark[0]], hrs[dark[-1]]),
                                              movable=False, brush=(30, 60, 120, 60))
                    reg.setZValue(-10); self.sky_curve.addItem(reg)
            # horizon + min altitude
            self.sky_curve.addLine(y=0, pen=pg.mkPen("#475569", width=1))
            minalt = self.t_minalt.value() if hasattr(self, "t_minalt") else 30.0
            self.sky_curve.addLine(y=minalt, pen=pg.mkPen("#f59e0b", width=1,
                                                          style=Qt.PenStyle.DashLine))
            # target altitude
            self.sky_curve.plot(hrs, alts, pen=pg.mkPen("#2dd4bf", width=2))
            # Moon altitude (when above horizon) — proximity/brightness awareness
            if moon_rd is not None:
                self.sky_curve.plot(hrs, moon, pen=pg.mkPen("#fbbf24", width=1,
                                                            style=Qt.PenStyle.DashLine))
            # transit marker (max alt)
            imax = int(np.argmax(alts))
            self.sky_curve.addLine(x=hrs[imax], pen=pg.mkPen("#e5e7eb", width=1,
                                                            style=Qt.PenStyle.DotLine))
            # x ticks as local time
            ticks = [(h, "{:02d}h".format(int((16 + h) % 24))) for h in range(0, 17, 2)]
            self.sky_curve.getAxis("bottom").setTicks([ticks])
            self.sky_curve.setYRange(min(-5, float(alts.min())), 90)
            self.sky_curve.setXRange(0, 16)
            tmax = (start + timedelta(hours=float(hrs[imax]))).strftime("%Hh%M")
            self.sky_curve.setTitle("Max {:.0f}° at {}  ·  cyan = target, yellow = Moon, "
                                    "band = astro-night".format(float(alts[imax]), tmax),
                                    color="#94a3b8", size="9pt")
        except Exception as e:        # noqa: BLE001
            self.sky_curve.clear()
            self.sky_curve.setTitle("curve unavailable: {}".format(e), color="#ef4444")

    def _toggle_sky_live(self, on):
        if on:
            self._sky_now()
            self._sky_live_timer.start()
        else:
            self._sky_live_timer.stop()
        # lock the date/time controls while Live is following the real sky
        for wdg in (self.sky_time, self.sky_date, self.sky_now_btn):
            wdg.setEnabled(not on)

    def _current_hips(self):
        return self._SURVEYS[self.sky_survey.currentIndex()][1] if hasattr(self, "sky_survey") else ""

    def _on_survey_change(self, *args):
        if self._current_hips():
            self._schedule_bg()
        elif self.skymap:
            self.skymap.clear_background()
            self.sky_status.setText("Background: none (plain map).")

    def _schedule_bg(self, *args):
        if not self.skymap or getattr(self, "_shutting_down", False):
            return
        if not self._current_hips() or self.skymap.mode != "framing":
            self.skymap.clear_background()
            return
        self._bg_timer.start()

    def _preload_zone(self):
        hips = self._current_hips()
        if not self.skymap or not hips:
            self.sky_status.setText("Pick a Background survey first, then pre-load.")
            return
        if getattr(self, "_preload", None) is not None and self._preload.isRunning():
            return
        ra, dec = self.skymap.center_ra, self.skymap.center_dec
        base = self.skymap.view_fov
        # HD target zone: cache from the finest ladder level up to a wide context level.
        # Fine levels naturally cover only the small central area (small step), so the
        # target stays sharp; coarse levels give surrounding context for panning.
        snapped = _snap_fov(base)
        top = min(max(snapped * 3.0, 8.0), 40.0)
        levels = sorted({f for f in _FOV_LADDER if f <= top})   # includes 0.3, 0.5, 0.8…
        jobs = {}
        for fov in levels:
            step = max(_snap_fov(fov) / 3.0, 0.02)  # same quantisation step as the live view
            cd = max(math.cos(math.radians(dec)), 0.2)
            span = 3                                # -3..+3 tiles each axis around the target
            for dr in range(-span, span + 1):
                for dd in range(-span, span + 1):
                    tra, tdec, tfov, tsize = _bg_tile_params(
                        (ra + dr * step / cd) % 360.0,
                        max(-89.0, min(89.0, dec + dd * step)), fov)
                    jobs[(tra, tdec, tfov, tsize)] = (hips, tra, tdec, tfov, tsize)
        jobs = list(jobs.values())
        self.sky_preload_btn.setEnabled(False)
        self.sky_status.setText("Pre-loading {} tiles…".format(len(jobs)))
        self._preload = PreloadWorker(jobs)
        self._preload.progress.connect(self._on_preload_progress)
        self._track_thread(self._preload)
        self._preload.start()

    @Slot(int, int, int)
    def _on_preload_progress(self, done, total, ok):
        self.sky_status.setText("Pre-loading target… {}/{} ({} cached)".format(done, total, ok))
        if done >= total:
            self.sky_preload_btn.setEnabled(True)
            self.sky_status.setText("Target cached: {}/{} tiles available offline (HD).".format(ok, total))

    def _download_light_sky(self):
        if getattr(self, "_lightsky", None) is not None and self._lightsky.isRunning():
            return
        hips = self._LIGHT_HIPS
        jobs = {}
        for fov in self._LIGHT_LEVELS:                # coarse→fine, upscaled when zoomed past
            step = max(fov, 0.05)
            dec = -88.0
            while dec <= 88.0:
                cd = max(math.cos(math.radians(dec)), 0.04)
                ra = 0.0
                while ra < 360.0:
                    t = _light_tile_params(ra % 360.0, dec, fov)
                    jobs[(t[0], t[1], t[2], t[3])] = (hips,) + t
                    ra += step / cd
                dec += step
        jobs = list(jobs.values())
        self.sky_lightsky_btn.setEnabled(False)
        self.sky_status.setText("Downloading light all-sky backdrop ({} tiles)…".format(len(jobs)))
        self._lightsky = PreloadWorker(jobs)
        self._lightsky.progress.connect(self._on_lightsky_progress)
        self._track_thread(self._lightsky)
        self._lightsky.start()

    @Slot(int, int, int)
    def _on_lightsky_progress(self, done, total, ok):
        self.sky_status.setText("Light all-sky… {}/{} tiles ({} cached)".format(done, total, ok))
        if done >= total:
            self.sky_lightsky_btn.setEnabled(True)
            self.sky_status.setText("Light all-sky ready: {}/{} tiles. Offline backdrop active "
                                    "anywhere.".format(ok, total))

    def _offline_fallback_tile(self, ra, dec, fov):
        """Light-sky (Mellinger) backdrop is disabled — it was slow/unreliable to download
        and a partial cache caused display glitches. Pre-load target (HD) is unaffected."""
        return None

    def _track_thread(self, th):
        """Keep a reference so the QThread isn't garbage-collected while running."""
        if not hasattr(self, "_threads"):
            self._threads = []
        self._threads.append(th)
        th.finished.connect(lambda: self._threads.remove(th) if th in self._threads else None)
        return th

    def _do_bg_fetch(self):
        if getattr(self, "_shutting_down", False):
            return
        if not self.skymap or not self._current_hips() or self.skymap.mode != "framing":
            return
        # avoid piling up requests (a previous one still running)
        if getattr(self, "_bg_loader", None) is not None and self._bg_loader.isRunning():
            self._bg_timer.start()    # retry shortly
            return
        ra, dec, fov = self.skymap.center_ra, self.skymap.center_dec, self.skymap.view_fov
        if fov > 40.0:                # wide view: survey imagery too heavy/unreliable
            self.skymap.clear_background()
            self.sky_status.setText("Background: zoom in (< 40° field) to load survey imagery.")
            return
        # snap to a stable tile so the SAME area always maps to the SAME cache entry
        tra, tdec, tfov, tsize = _bg_tile_params(ra, dec, fov)
        self._bg_req = (tra, tdec, tfov)
        self.sky_status.setText("Loading {} background…".format(self.sky_survey.currentText()))
        self._bg_loader = DSOImageLoader(tra, tdec, tfov, size=tsize, hips=self._current_hips())
        self._bg_loader.done.connect(self._on_bg_done)
        self._track_thread(self._bg_loader)
        self._bg_loader.start()

    @Slot(object, str)
    def _on_bg_done(self, data, err):
        if data is None:
            # offline / not cached for this survey → fall back to the light all-sky backdrop
            if self._bg_req:
                tra, tdec, _tfov = self._bg_req
                fb = self._offline_fallback_tile(tra, tdec, self.skymap.view_fov)
                if fb:
                    fdata, fra, fdec, ffov = fb
                    fimg = QtGui.QImage.fromData(fdata)
                    if not fimg.isNull():
                        self.skymap.set_background(QtGui.QPixmap.fromImage(fimg), fra, fdec, ffov)
                        self.sky_status.setText("Background: light all-sky (offline fallback).")
                        return
            self.sky_status.setText("Background unavailable (offline?). {}".format(err)[:120])
            return
        img = QtGui.QImage.fromData(data)
        if img.isNull():
            self.sky_status.setText("Background: no image returned.")
            return
        ra, dec, fov = self._bg_req
        self.skymap.set_background(QtGui.QPixmap.fromImage(img), ra, dec, fov)
        self.sky_status.setText("Background: {} @ {:.2f}° field.".format(
            self.sky_survey.currentText(), fov))

    def _open_image_fullscreen(self):
        if not self.skymap:
            return
        ra, dec = self.skymap.center_ra, self.skymap.center_dec
        fw, _fh = self._sky_mod.fov_deg(self.sky_focal.value())
        hips = self._current_hips() or "CDS/P/DSS2/color"
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Sky image — {}".format(self.skymap.target_name))
        dlg.resize(900, 900)
        v = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel("Loading full-resolution image…")
        lbl.setAlignment(Qt.AlignCenter); lbl.setMinimumSize(820, 820)
        lbl.setStyleSheet("background:#05060d; color:#888;")
        v.addWidget(lbl, 1)
        info = QtWidgets.QLabel("{}  ·  RA {:.3f}°  Dec {:.3f}°  ·  {:.2f}° field".format(
            self.skymap.target_name, ra, dec, max(fw, 0.1)))
        info.setStyleSheet("color:#cbd5e1;"); v.addWidget(info)
        loader = DSOImageLoader(ra, dec, max(fw, 0.1), size=850, hips=hips)

        def _shown(data, err):
            if data is None:
                lbl.setText("Image unavailable (offline?).\n{}".format(err)); return
            im = QtGui.QImage.fromData(data)
            if im.isNull():
                lbl.setText("No image returned."); return
            lbl.setPixmap(QtGui.QPixmap.fromImage(im).scaled(
                lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        loader.done.connect(_shown)
        self._fs_loader = loader
        self._track_thread(loader)
        loader.start()
        dlg.exec()

    def _update_sky_flags(self, *args):
        if self.skymap:
            self.skymap.set_flags(dso=self.sky_show_dso.isChecked(),
                                  constlines=self.sky_show_con.isChecked(),
                                  names=self.sky_show_nm.isChecked(),
                                  constnames=self.sky_show_cn.isChecked())
            self.skymap.set_starnames(self.sky_show_sn.isChecked())
            self.skymap.set_trajectory(self.sky_show_traj.isChecked())

    def _update_sky_mosaic(self, *args):
        if self.skymap:
            self.skymap.set_mosaic(self.sky_mos_rows.value(), self.sky_mos_cols.value(),
                                   self.sky_mos_ov.value() / 100.0, self.sky_mos_show.isChecked())

    def _build_skymap_dt_utc(self):
        from datetime import datetime, timedelta
        qd = self.sky_date.date()
        mins = self.sky_time.value()
        local = datetime(qd.year(), qd.month(), qd.day()) + timedelta(minutes=mins)
        try:
            from zoneinfo import ZoneInfo
            aware = local.replace(tzinfo=ZoneInfo("Europe/Paris"))
            return aware.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        except Exception:             # noqa: BLE001
            return local - timedelta(hours=2)     # repli été (UTC+2)

    def _update_sky_time(self, *args):
        h, m = divmod(self.sky_time.value(), 60)
        self.sky_time_lbl.setText("{:02d}:{:02d}".format(h, m))
        if self.skymap:
            self.skymap.set_site(self.t_lat.value(), self.t_lon.value())
            self.skymap.set_time(self._build_skymap_dt_utc())
            self._update_night_curve()

    def _sky_now(self):
        from datetime import datetime
        now = datetime.now()
        self.sky_date.blockSignals(True)
        self.sky_date.setDate(QtCore.QDate(now.year, now.month, now.day))
        self.sky_date.blockSignals(False)
        self.sky_time.blockSignals(True)
        self.sky_time.setValue(now.hour * 60 + now.minute)
        self.sky_time.blockSignals(False)
        self._update_sky_time()

    def _target_row_goto(self):
        """Selected row in the Targets tab -> session goal -> Manual GoTo, in one click."""
        rows = self.t_table.selectionModel().selectedRows() if self.t_table else []
        if not rows:
            self.statusBar().showMessage("Select a target row first.", 5000)
            return
        self._target_row_to_skymap(self.t_table.item(rows[0].row(), 0))
        self._open_manual_goto()

    def _on_sky_pick(self, oid, ra, dec):
        self._active_sky_target = (oid, ra, dec)
        self.sky_target_label.setText("Target: {}".format(oid))
        self.sky_status.setText("Picked {} — RA {:.3f}°  Dec {:.3f}°.".format(oid, ra, dec))
        self._set_session_goal(oid, ra, dec)
        self._load_dso_image()
        self._schedule_bg()
        self._update_night_curve()

    def _on_toggle_goal(self, on):
        if getattr(self, "skymap", None):
            self.skymap.show_goal = bool(on) and bool(getattr(self, "_session_goal", None))
            self.skymap.update()

    def _radec_field_name(self, ra, dec):
        h = (ra % 360.0) / 15.0
        hh = int(h); mm = int(round((h - hh) * 60))
        if mm == 60:
            hh = (hh + 1) % 24; mm = 0
        return "Field {:02d}h{:02d} {:+03.0f}°".format(hh, mm, dec)

    def _set_goal_here(self):
        """Set the session goal to the current view centre — lets you target an empty
        region of sky (no specific object). Pan/zoom to frame it, then click."""
        if not self.skymap:
            return
        ra, dec = self.skymap.center_ra, self.skymap.center_dec
        name = self._radec_field_name(ra, dec)
        self._active_sky_target = (name, ra, dec)
        self.sky_target_label.setText("Target: {}".format(name))
        self._set_session_goal(name, ra, dec)
        if hasattr(self, "sky_goal_rot") and hasattr(self, "sky_rot"):
            self.sky_goal_rot.setValue(self.sky_rot.value())   # goal takes current rotation
            if self.skymap:
                self.skymap.set_goal_angle(self.sky_rot.value())
        if hasattr(self, "sky_show_goal") and not self.sky_show_goal.isChecked():
            self.sky_show_goal.setChecked(True)           # make sure it's visible
        if hasattr(self, "sky_status"):
            self.sky_status.setText(
                "Goal set at view centre — RA {:.3f}°  Dec {:+.3f}°.".format(ra, dec))
        self._load_dso_image()
        self._schedule_bg()
        self._update_night_curve()

    def _target_full_name(self, name, ra, dec):
        """Return 'ID — Common Name' when we can match the target in the Sky Map catalog,
        else the name as given."""
        try:
            objs = getattr(self.skymap, "dso", []) if getattr(self, "skymap", None) else []
            best, bd = None, 0.3     # match within ~0.3° of the target
            for o in objs:
                d = ((float(o["ra"]) - ra) ** 2 + (float(o["dec"]) - dec) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, o
            if best:
                oid, onm = best.get("id", ""), best.get("name", "")
                if onm and onm != oid:
                    return "{} — {}".format(oid, onm)
                return oid or name
        except Exception:           # noqa: BLE001
            pass
        return name

    def _set_session_goal(self, name, ra, dec):
        """The object to find this session; shown on the Sky Map alongside the live pointing."""
        self._session_goal = (name, ra, dec)
        if getattr(self, "skymap", None):
            self.skymap.set_goal(name, ra, dec)
            if hasattr(self, "sky_show_goal"):
                self.skymap.show_goal = self.sky_show_goal.isChecked()
                self.skymap.update()
        disp = self._target_full_name(name, ra, dec)
        try:
            self._update_hud_identity()
        except Exception:               # noqa: BLE001
            pass
        QtCore.QTimer.singleShot(0, self._update_sky_window)
        if hasattr(self, "sky_goal_label"):
            self.sky_goal_label.setText("🎯 {}".format(disp))
        if hasattr(self, "top_target_label"):
            self.top_target_label.setText("🎯  Target: {}".format(disp))
            self.top_target_sub.setText("RA {:.3f}°   Dec {:+.3f}°".format(ra, dec))
        # HEQ-5 Pro (2-axis) + connected: GoTo the target automatically, then track
        two = bool(QtCore.QSettings("NOUT", "NOUT").value("mount_type", 0, type=int))
        if (two and getattr(self, "_mount_connected", False)
                and not getattr(self, "_iv_running", False)
                and not getattr(self, "_ac_active", False)):
            self.statusBar().showMessage(
                "🔭 HEQ-5: GoTo {} — plate-solving then slewing RA+Dec…".format(disp), 8000)
            self._start_autocenter(self._mip())

    def _load_dso_image(self, force=False):
        if not self.skymap:
            return
        if not force and not self.sky_img_chk.isChecked():
            return
        # What you want to see is the zone YOUR camera covers. Use the last plate-solve
        # (where it really points, and how wide the field really is); the session goal is
        # only the fallback, when nothing has been solved yet.
        wcs = getattr(self, "_solve_wcs", None)
        if wcs:
            ra, dec, _rot, fw = wcs
            src = "solved field"
        else:
            ra, dec = self.skymap.center_ra, self.skymap.center_dec
            fw, _fh = self._sky_mod.fov_deg(self.sky_focal.value())
            src = "session target"
        self.sky_image.setText("Loading… ({})".format(src))
        hips = self._current_hips() or "CDS/P/DSS2/color"
        self._dso_loader = DSOImageLoader(ra, dec, max(fw, 0.1),
                                          size=self.sky_image.width(), hips=hips)
        self._dso_loader.done.connect(self._on_dso_image_done)
        self._track_thread(self._dso_loader)
        self._dso_loader.start()

    @Slot(object, str)
    def _on_dso_image_done(self, data, err):
        if data is None:
            self.sky_image.setText("Image unavailable\n(offline?)")
            self.sky_status.setText("Image fetch failed: {}".format(err)[:120])
            return
        img = QtGui.QImage.fromData(data)
        if img.isNull():
            self.sky_image.setText("No image returned")
            return
        self.sky_image.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.sky_image.width(), self.sky_image.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _export_panel_centers(self):
        if not self.skymap:
            return
        pc = self.skymap.panel_centers()
        d = self.dir_edit.text() or os.path.expanduser("~")
        try:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "mosaic_panels_{}.csv".format(
                self.skymap.target_name.replace(" ", "_")))
            with open(path, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["row", "col", "ra_deg", "dec_deg", "ra_hms", "dec_dms"])
                for r, c, ra, dec in pc:
                    wr.writerow([r, c, "{:.4f}".format(ra), "{:.4f}".format(dec),
                                 _ra_to_hms(ra), _dec_to_dms(dec)])
            self.sky_status.setText("Panel centers written: {} ({} panels).".format(path, len(pc)))
        except Exception as e:        # noqa: BLE001
            self.sky_status.setText("Export failed: {}".format(e))

    def _sky_to_mosaic_view(self):
        self.mos_rows.setValue(self.sky_mos_rows.value())
        self.mos_cols.setValue(self.sky_mos_cols.value())
        self.mos_view_ov.setValue(self.sky_mos_ov.value())
        self._reset_mosaic()
        self.sky_status.setText("Grid {}×{} sent to the Mosaic View tab.".format(
            self.sky_mos_rows.value(), self.sky_mos_cols.value()))

    def _open_aladin(self):
        import webbrowser
        if not self.skymap:
            return
        ra, dec = self.skymap.center_ra, self.skymap.center_dec
        fw, _ = self._sky_mod.fov_deg(self.sky_focal.value())
        url = ("https://aladin.cds.unistra.fr/AladinLite/?target={:.5f}%20{:+.5f}"
               "&fov={:.3f}&survey=P%2FDSS2%2Fcolor").format(ra, dec, max(fw * 1.5, 0.2))
        try:
            webbrowser.open(url)
            self.sky_status.setText("Opening Aladin Lite at RA {:.3f}° Dec {:.3f}°…".format(ra, dec))
        except Exception as e:        # noqa: BLE001
            self.sky_status.setText("Could not open browser: {}".format(e))

    def _sync_sky_site(self, *args):
        if getattr(self, "skymap", None):
            self.skymap.set_site(self.t_lat.value(), self.t_lon.value())

    def _locate_me(self):
        self.t_locate.setEnabled(False); self.t_locate.setText("📍 Locating…")
        self._geo = GeolocateWorker()
        self._geo.done.connect(self._on_located)
        self._track_thread(self._geo)
        self._geo.start()

    @Slot(object, str)
    def _on_located(self, d, err):
        self.t_locate.setEnabled(True); self.t_locate.setText("📍 Locate me")
        if not d:
            if hasattr(self, "t_search_msg"):
                self.t_search_msg.setText("Location failed: {}".format(err)[:90])
            return
        try:
            self.t_lat.setValue(float(d["lat"])); self.t_lon.setValue(float(d["lon"]))
        except Exception:             # noqa: BLE001
            return
        self._sync_sky_site()
        if d.get("source") == "ip":
            self.t_search_msg.setText(
                "Approx location via IP (often your ISP's city, e.g. Paris): "
                "{}, {} ({:.3f}, {:.3f}). Adjust Lat/Lon by hand if needed.".format(
                    d.get("city", "?"), d.get("country", "?"), d["lat"], d["lon"]))
        elif hasattr(self, "t_search_msg"):
            self.t_search_msg.setText("GPS location: {:.4f}, {:.4f}.".format(d["lat"], d["lon"]))

    def _ensure_search_index(self):
        if getattr(self, "_search_index", None) is not None:
            return
        self._search_index = []
        try:
            import astro_targets
            for t in astro_targets.load_catalog():
                self._search_index.append((t.id, t.name or "",
                                           float(t.coord.ra.deg), float(t.coord.dec.deg)))
        except Exception:             # noqa: BLE001
            if getattr(self, "skymap", None):
                for d in self.skymap.dso:
                    self._search_index.append((d["id"], d.get("name", ""), d["ra"], d["dec"]))

    @staticmethod
    def _norm_id(s):
        import re
        s = s.strip().upper().replace(" ", "")
        m = re.match(r"^(NGC|IC|M|MESSIER)0*(\d+)$", s)
        if m:
            pre = "M" if m.group(1) in ("M", "MESSIER") else m.group(1)
            return pre + str(int(m.group(2)))
        return s

    def _search_target(self):
        q = self.t_search.text().strip()
        if not q:
            return
        self._ensure_search_index()
        nq = self._norm_id(q); ql = q.lower(); hit = None
        for oid, name, ra, dec in self._search_index:       # exact id
            if self._norm_id(oid) == nq:
                hit = (oid, ra, dec); break
        if not hit:                                         # exact name
            for oid, name, ra, dec in self._search_index:
                if name and name.lower() == ql:
                    hit = (oid, ra, dec); break
        if not hit:                                         # substring
            for oid, name, ra, dec in self._search_index:
                if ql in (name or "").lower() or ql in oid.lower():
                    hit = (oid, ra, dec); break
        if not hit:
            self.t_search_msg.setText("No match for \u201c{}\u201d.".format(q)); return
        oid, ra, dec = hit
        if getattr(self, "skymap", None):
            self.skymap.set_target(oid, ra, dec)
            self._show_on_skymap()
            self._active_sky_target = (oid, ra, dec)
            self._set_session_goal(oid, ra, dec)
            self.sky_target_label.setText("Target: {}".format(oid))
            self.tabs.setCurrentIndex(self._skymap_index)
            self._load_dso_image()
            if hasattr(self, "_schedule_bg"):
                self._schedule_bg()
        self.t_search_msg.setText("Found {} — RA {:.3f}° Dec {:.3f}°.".format(oid, ra, dec))

    def _sky_search_go(self):
        if not getattr(self, "skymap", None):
            return
        q = self.sky_search.text()
        res = self.skymap.find_object(q)
        if not res:
            self.sky_search_msg.setText("not found")
            return
        name, ra, dec = res
        self.sky_search_msg.setText("→ {}".format(name))
        self.skymap.set_target(name, ra % 360.0, dec)
        self._active_sky_target = (name, ra % 360.0, dec)
        if hasattr(self, "sky_target_label"):
            self.sky_target_label.setText("Target: {}".format(name))
        self._load_dso_image(); self._schedule_bg(); self._update_night_curve()

    def _target_context_menu(self, pos):
        """Everything you do with a target, from the list, without changing tab."""
        item = self.t_table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        self.t_table.selectRow(row)
        first = self.t_table.item(row, 0)
        data = first.data(Qt.UserRole) if first else None
        if not data:
            return
        oid, ra, dec = data
        name = self.t_table.item(row, 1).text() if self.t_table.item(row, 1) else oid
        m = QtWidgets.QMenu(self)
        head = m.addAction("{} — {}".format(oid, name) if name and name != oid else oid)
        head.setEnabled(False)
        m.addSeparator()
        a_goal = m.addAction("Set as session target")
        a_goto = m.addAction("GoTo…")
        a_map = m.addAction("Frame in Sky Map")
        m.addSeparator()
        a_run = m.addAction("Start a series on it (already pointed)")
        a_run.setToolTip("Makes it the target, switches to Shooting and starts the series "
                         "with the current settings.")
        m.addSeparator()
        a_copy = m.addAction("Copy coordinates")
        m.setToolTipsVisible(True)
        act = m.exec(self.t_table.viewport().mapToGlobal(pos))
        if act is a_goal:
            self._set_session_goal(oid, ra, dec)
            self.statusBar().showMessage("🎯 Session target: {}".format(oid), 5000)
        elif act is a_goto:
            self._target_row_goto()
        elif act is a_map:
            self._target_row_to_skymap(first, framing=True)
        elif act is a_run:
            self._set_session_goal(oid, ra, dec)
            self.tabs.setCurrentIndex(self._shoot_index)
            self._start_interval()
        elif act is a_copy:
            h = ra / 15.0
            ad = abs(dec)
            txt = "{}  RA {:02d}h{:02d}m{:04.1f}s  Dec {}{:02d}°{:02d}′{:02d}″  (J2000)".format(
                oid, int(h), int(h * 60) % 60, (h * 3600) % 60,
                "-" if dec < 0 else "+", int(ad), int(ad * 60) % 60, int(ad * 3600) % 60)
            QtWidgets.QApplication.clipboard().setText(txt)
            self.statusBar().showMessage("Copied: " + txt, 5000)

    def _show_on_skymap(self, framing=False):
        """All-sky, turned toward the target with the camera frame on it — or the
        Framing view when that is what was asked for."""
        if framing:
            self.sky_mode.setCurrentText("Framing"); self.skymap.set_mode("framing")
        else:
            self.skymap.set_frame_allsky(True)
            if hasattr(self, "sky_show_frame"):
                self.sky_show_frame.blockSignals(True); self.sky_show_frame.setChecked(True)
                self.sky_show_frame.blockSignals(False)
            self.sky_mode.setCurrentText("All-sky"); self.skymap.set_mode("allsky")
            self.skymap.aim_at_target(); self.skymap.update()

    def _target_row_to_skymap(self, item, framing=False):
        row = item.row()
        first = self.t_table.item(row, 0)
        data = first.data(Qt.UserRole) if first else None
        if not data:
            return
        oid, ra_deg, dec_deg = data
        if not hasattr(self, "skymap"):
            return
        self.skymap.set_target(oid, ra_deg, dec_deg)
        self._show_on_skymap(framing)
        self._active_sky_target = (oid, ra_deg, dec_deg)
        self._set_session_goal(oid, ra_deg, dec_deg)
        self.sky_target_label.setText("Target: {}".format(oid))
        self.tabs.setCurrentIndex(self._skymap_index)
        self._load_dso_image()
        self._schedule_bg()
        self._update_night_curve()

    # -- UI callbacks -------------------------------------------------------
    def _set(self, name, combo):
        self.worker.post("set_config", name=name, value=combo.currentText())

    def _on_roi(self, v):
        self.roi_frac = v / 100.0
        self.roi_label.setText("Analysis zone: {}%".format(v))
        self.worker.post("set_roi", frac=self.roi_frac)
        self._sync_hdmi()

    # -- HDMI live source ---------------------------------------------------
    def _refresh_video_sources(self, restore=None):
        """Fill the source list with the Mac's video inputs, capture cards first."""
        cur = self.src_combo.currentData()
        want = restore if restore is not None else (cur[1] if cur else None)
        devs = list_video_devices()
        self.src_combo.blockSignals(True)
        while self.src_combo.count() > 1:
            self.src_combo.removeItem(1)
        ext = [d for d in devs if not _is_builtin_camera(d[1])]
        builtin = [d for d in devs if _is_builtin_camera(d[1])]
        for idx, name in ext + builtin:
            tag = "HDMI — " if (idx, name) in ext else "Mac camera — "
            self.src_combo.addItem("{}{} [{}]".format(tag, name, idx), (idx, name))
        sel = 0
        for i in range(1, self.src_combo.count()):
            if want and self.src_combo.itemData(i)[1] == want:
                sel = i
        self.src_combo.setCurrentIndex(sel)
        self.src_combo.blockSignals(False)
        if restore is None:
            self.src_info.setText(
                "{} capture card(s) found.".format(len(ext)) if ext else
                "No capture card found — plug it in (camera on, HDMI connected), then ↻.")
        if restore:                       # startup: reopen the card used last time
            if sel:
                self._on_live_source()
        elif restore is None and self.src_combo.currentData() != cur:
            self._on_live_source()        # the card in use disappeared: back to USB

    def _sync_hdmi(self):
        h = getattr(self, "_hdmi", None)
        if h is None:
            return
        h.metric = self.metric_cb.currentText()
        h.roi_frac = self.roi_frac
        h.metric_on = self.chk_metric.isChecked()
        h.stars_on = self.chk_stars.isChecked()
        h.paused = not self.chk_live.isChecked()

    def _stop_hdmi(self):
        h, self._hdmi = getattr(self, "_hdmi", None), None
        if h is None:
            return
        for sig in (h.frame_ready, h.status, h.opened, h.rec_progress, h.rec_done):
            try:
                sig.disconnect()
            except Exception:           # noqa: BLE001
                pass
        h.stop()
        _retire_thread(h, 4000)

    def _on_live_source(self):
        data = self.src_combo.currentData()
        self._stop_hdmi()
        if data is None:                                  # back to the USB live view
            self.worker.post("set_live", on=self.chk_live.isChecked())
            self.src_info.setText("USB live view (1024×680).")
            return
        idx, name = data
        self.worker.post("set_live", on=False)            # USB stays for photos only
        h = HdmiSource(idx)
        h.frame_ready.connect(self.on_frame)
        h.status.connect(lambda m: (self.statusBar().showMessage(m, 8000),
                                    self.src_info.setText(m)))
        h.opened.connect(lambda w, hh, fps: self.src_info.setText(
            "HDMI: {}×{} at {:.0f} fps — {}".format(w, hh, fps, name)
            + ("" if w >= 1900 else "  ⚠ the card is not delivering 1080p")))
        h.rec_progress.connect(self._on_hdmi_rec_progress)
        h.rec_done.connect(self._on_hdmi_rec_done)
        self._hdmi = h
        self._sync_hdmi()
        self.view.fit()
        self.src_info.setText("Opening {}…".format(name))
        h.start()

    def _hdmi_record(self, seconds, mono):
        h = getattr(self, "_hdmi", None)
        lbl = getattr(self, "_plan_readout_lbl", None)
        if h is None:
            if lbl is not None:
                lbl.setText("Choose the HDMI source first (Display & Focus → Live source).")
            return
        if h.recording:
            h.stop_recording(); return
        path = os.path.join(self._effective_dir() or os.path.expanduser("~"),
                            "nout_hdmi_{}.ser".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
        self._plan_active = False
        h.start_recording(path, seconds, mono)
        if lbl is not None:
            lbl.setText("🎥 Recording HDMI ({} s, {})…".format(
                seconds, "mono" if mono else "colour"))

    def _on_hdmi_rec_progress(self, n, secs):
        lbl = getattr(self, "_plan_readout_lbl", None)
        if lbl is not None:
            try:
                lbl.setText("🎥 HDMI: {} frames · {:.1f} s · {:.0f} fps".format(
                    n, secs, n / max(secs, 1e-3)))
            except RuntimeError:
                self._plan_readout_lbl = None

    def _on_hdmi_rec_done(self, path, n, fps):
        mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0
        self.statusBar().showMessage("🎥 {} frames at {:.0f} fps → {} ({:.0f} MB)".format(
            n, fps, os.path.basename(path), mb), 15000)
        if n < 10:
            lbl = getattr(self, "_plan_readout_lbl", None)
            if lbl is not None:
                lbl.setText("Recording too short ({} frames) — no HDMI signal?".format(n))
            return
        if not self._plan_dlg:
            self._open_planetary_dialog()
        self._process_video(path)

    def _on_reticle_alpha(self, v):
        self.ret_label.setText("{}%".format(v) if v else "off")
        if hasattr(self, "view") and hasattr(self.view, "set_reticle_opacity"):
            self.view.set_reticle_opacity(v / 100.0)

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save folder",
                                                       self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def _on_autostart_toggled(self, on):
        """Enabling auto-start switches the sleep mode to 'during capture' so the Mac stays
        awake while armed/capturing and sleeps once the session ends."""
        if on and hasattr(self, "awake_mode"):
            i = self.awake_mode.findData("capture")
            if i >= 0 and self.awake_mode.currentData() != "capture":
                self.awake_mode.setCurrentIndex(i)
                self.statusBar().showMessage(
                    "Sleep mode → 'during capture': Mac stays awake until the session ends.",
                    6000)
        if not on:
            self._auto_started = False
        self._update_awake()

    def _goal_radec(self):
        g = getattr(self, "_session_goal", None)
        if g and g[1] is not None and g[2] is not None:
            return float(g[1]), float(g[2])
        return None, None

    def _push_autostop(self, *a):
        if not hasattr(self, "chk_autostop"):
            return
        ra, dec = self._goal_radec()
        sun_alt = fixed_min = target_alt = None
        if self.chk_autostop.isChecked():
            d = self.autostop_when.currentData()
            if d is None:
                t = self.autostop_time.time(); fixed_min = t.hour() * 60 + t.minute()
            else:
                sun_alt = float(d)
        if self.chk_stop_target.isChecked() and ra is not None:
            target_alt = float(self.autostop_alt.value())
        self.worker.post("set_autostop", sun_alt=sun_alt, fixed_min=fixed_min,
                         target_alt=target_alt, ra=ra, dec=dec)

    def _automation_tick(self):
        if not hasattr(self, "chk_autostart"):
            return
        from datetime import datetime
        lat, lon = self.t_lat.value(), self.t_lon.value()
        now = datetime.utcnow()
        try:
            sun = _sun_altitude_deg(lat, lon, now)
        except Exception:               # noqa: BLE001
            return
        ra, dec = self._goal_radec()
        tgt = _object_altitude_deg(lat, lon, ra, dec, now) if ra is not None else None
        # Polaris indicator: naked-eye visible for alignment once the Sun is below ~-6°
        if lat > 0 and sun < -6:
            self.polaris_lbl.setText("🧭 Polaris visible — polar alignment OK")
            self.polaris_lbl.setStyleSheet("color:#7dd3a0;")
        else:
            self.polaris_lbl.setText("🧭 Polaris — waiting for dusk (Sun {:+.0f}°)".format(sun))
            self.polaris_lbl.setStyleSheet("color:#94a3b8;")
        self._push_autostop()           # keep the worker's stop config current
        self._update_awake()            # armed auto-start keeps the Mac awake; release when done
        if not self.chk_autostart.isChecked():
            self._auto_started = False
            return
        d = self.autostart_when.currentData()
        if d is None:
            t = self.autostart_time.time(); fm = t.hour() * 60 + t.minute()
            loc = datetime.now(); cur = loc.hour * 60 + loc.minute
            dark_ok = fm <= cur < fm + 180
        else:
            dark_ok = sun < float(d)
        if not dark_ok:
            self._auto_started = False   # re-arm for the next night
        alt_ok = (tgt is None) or (tgt >= self.autostart_alt.value())
        if (dark_ok and alt_ok and not getattr(self, "_iv_running", False)
                and not getattr(self, "_auto_started", False)):
            self._auto_started = True
            self.statusBar().showMessage(
                "🌙 Auto-start: sky dark enough and target high — starting capture.", 9000)
            self._start_interval()

    def _effective_dir(self):
        return os.path.join(self.dir_edit.text(), self.type_combo.currentText().strip())

    def _single_capture(self):
        # Une photo unique se comporte comme une séquence d'UNE vue : elle passe
        # par la même machinerie (salve), apparaît dans Results et est empilable.
        self._single_shot = True          # one frame: plate-solve it as well
        bulb = self.exp_mode.currentIndex() == 1
        self.worker.post("set_exposure_mode", bulb=bulb, seconds=self.bulb_secs.value())
        self.worker.post("start_interval", interval=0, count=1, bulb=bulb,
                         shutter=self._ui_shutter(),
                         seconds=self.bulb_secs.value(), save_dir=self._effective_dir())
        # le passage en mode revue + le bouton « Done » sont gérés par
        # on_interval_state(True/False), exactement comme pour une séquence.

    def _toggle_interval(self):
        """The S key: pause or resume a running series, or start one. Stopping for
        good stays a deliberate click on Stop."""
        if getattr(self, "_iv_paused", False) or (self.review_mode and not self._await_done):
            self._toggle_pause()
        else:
            self._start_interval()

    def _toggle_pause(self):
        if getattr(self, "_iv_paused", False):
            self.worker.post("resume_interval")
        elif getattr(self, "_iv_running", False):
            self.worker.post("pause_interval")

    @Slot(bool)
    def on_interval_paused(self, paused):
        self._iv_paused = bool(paused)
        self.iv_pause.setText("▶ Resume" if paused else "⏸ Pause")
        self.iv_pause.setEnabled(bool(paused) or bool(getattr(self, "_iv_running", False)))
        self.iv_start.setEnabled(not paused and not getattr(self, "_iv_running", False))
        self.rec_label.setText("⏸ PAUSED — the series is held, press Resume"
                               if paused else "● REC — sequence in progress")
        self.rec_label.setStyleSheet(
            "color:#fff;background:{};font-weight:bold;padding:3px;".format(
                "#92400e" if paused else "#b91c1c"))
        if paused:
            self.prog_label.setText("Paused — Resume to carry on")

    @Slot(str)
    def on_cam_status(self, s):
        self.cam_status_label.setText("State: " + s)
        m = re.match(r"Battery\s+(.*?)\s+·\s+(.*?)\s+free", s)
        self._sb["batt"] = ("🔋 {}   💾 {}".format(m.group(1), m.group(2)) if m else s)
        self._refresh_statusbar()

    def _on_count_changed(self, v):
        # allow extending/shortening the burst live (finite series in progress)
        if getattr(self, "_iv_running", False) and not self.chk_unlimited.isChecked():
            self.worker.post("set_count", count=int(v))

    def _start_interval(self):
        bulb = self.exp_mode.currentIndex() == 1
        count = 0 if self.chk_unlimited.isChecked() else self.iv_count.value()
        if self.chk_dither.isChecked():                # mount in use: make sure it's tracking
            self.worker.post("mount_track_start", ip=self._mip())
            self.statusBar().showMessage("🟢 Tracking: auto-started for the session.", 6000)
        self.worker.post("set_exposure_mode", bulb=bulb, seconds=self.bulb_secs.value())
        self.worker.post("start_interval",
                         interval=self.iv_interval.value(),
                         count=count,
                         bulb=bulb,
                         seconds=self.bulb_secs.value(),
                         shutter=self._ui_shutter(),   # what 'camera speed' means here
                         save_dir=self._effective_dir())

    def _on_live_toggle(self, on):
        hdmi = getattr(self, "_hdmi", None) is not None
        self.worker.post("set_live", on=on and not hdmi)
        self._sync_hdmi()
        if not on and not self.review_mode:
            self.view.setPixmap(QtGui.QPixmap())
            self.view.setText("Live view disabled (battery saving).")

    def _on_metric_toggle(self, on):
        self.worker.post("set_metric_enabled", on=on)
        self.plot.setVisible(on and not self.review_mode)
        if not on:
            self.readout.setText("")

    # -- local horizon profiles ------------------------------------------------
    def _refresh_horizon_combo(self, select=None):
        profs = horizon_profiles()
        cur = select if select is not None else QtCore.QSettings("NOUT", "NOUT").value(
            "horizon_active", "", type=str)
        self.hz_combo.blockSignals(True); self.hz_combo.clear()
        self.hz_combo.addItem("Elsewhere — open horizon", "")
        for nm in sorted(profs):
            self.hz_combo.addItem("🏠 " + nm, nm)
        i = self.hz_combo.findData(cur)
        self.hz_combo.setCurrentIndex(max(i, 0))
        self.hz_combo.blockSignals(False)
        self._apply_horizon(push_site=False)

    def _active_horizon(self):
        nm = self.hz_combo.currentData() if hasattr(self, "hz_combo") else ""
        p = horizon_profiles().get(nm) if nm else None
        return p.get("alt") if p else None

    def _apply_horizon(self, push_site=True):
        nm = self.hz_combo.currentData() or ""
        QtCore.QSettings("NOUT", "NOUT").setValue("horizon_active", nm)
        p = horizon_profiles().get(nm) if nm else None
        if p and push_site:                         # "I am at home": home's coordinates too
            self.t_lat.setValue(float(p.get("lat", self.t_lat.value())))
            self.t_lon.setValue(float(p.get("lon", self.t_lon.value())))
        if getattr(self, "skymap", None) is not None:
            self.skymap.set_local_horizon(p.get("alt") if p else None)
        QtCore.QTimer.singleShot(0, self._update_sky_window)

    def _on_horizon_choice(self):
        self._apply_horizon()
        nm = self.hz_combo.currentData()
        self.statusBar().showMessage(
            "🏠 {}: position and local horizon applied — Calculate Targets to update the "
            "list.".format(nm) if nm else "Open horizon.", 7000)

    def _edit_horizon(self):
        nm = self.hz_combo.currentData() or ""
        profs = horizon_profiles()
        p = profs.get(nm, {})
        dlg = HorizonEditor(self, nm or "Home", p.get("alt"),
                            float(p.get("lat", self.t_lat.value())),
                            float(p.get("lon", self.t_lon.value())), self.t_minalt.value())
        dlg.delete_btn.setEnabled(bool(nm))
        r = dlg.exec()
        if r == 2 and nm:                                       # delete
            profs.pop(nm, None); save_horizon_profiles(profs)
            self._refresh_horizon_combo(select="")
            return
        if r != QtWidgets.QDialog.Accepted:
            return
        new = dlg.name.text().strip() or "Home"
        if nm and new != nm:
            profs.pop(nm, None)
        profs[new] = {"lat": float(p.get("lat", self.t_lat.value())),
                      "lon": float(p.get("lon", self.t_lon.value())),
                      "alt": [round(float(a), 1) for a in dlg.canvas.alt]}
        save_horizon_profiles(profs)
        self._refresh_horizon_combo(select=new)
        self.statusBar().showMessage("🏠 Horizon « {} » saved.".format(new), 6000)

    def _run_astro(self):
        from datetime import date as _date
        qd = self.t_date.date()
        when = _date(qd.year(), qd.month(), qd.day())
        self.t_btn.setEnabled(False)
        self.t_header.setText("Calculation in progress…")
        mm = self.t_maxmag.value()
        max_mag = None if mm >= 20 else float(mm)
        self._astro = AstroWorker(self.t_lat.value(), self.t_lon.value(), 300.0,
                                  "Site", "Europe/Paris", when,
                                  self.t_focal.value(), self.t_minalt.value(),
                                  self.t_count.value(), max_mag=max_mag,
                                  horizon=self._active_horizon())
        self._astro.done.connect(self._astro_done)
        self._astro.start()

    @Slot(object, str)
    def _astro_done(self, payload, err):
        self.t_btn.setEnabled(True)
        if payload is None:
            self.t_header.setText("Error: {}\n(install astropy: pip install astropy, "
                                  "and place astro_targets.py next to this script).".format(err))
            self.t_table.setRowCount(0)
            return
        hdr, rows = payload
        self._last_plan = (hdr, rows)          # keep for restoring after a search
        if not self.t_search.text().strip():   # don't clobber an active search
            self._populate_target_table(hdr, rows)

    #: right-ascension windows of the four seasonal skies (hours), evening culmination
    _SEASON_RA = {"winter": (4, 9), "spring": (9, 15), "summer": (15, 21), "autumn": (21, 4)}

    def _apply_target_view(self):
        """Re-show the last plan with the current sort order and collection."""
        plan = getattr(self, "_last_plan", None)
        if plan:
            self._populate_target_table(plan[0], plan[1])

    @staticmethod
    def _frame_fit(size_arcmin, fov_w_deg):
        """How well an object suits the frame: 1.0 fills it, less is smaller, and
        anything wider than the frame is knocked back."""
        frac = (float(size_arcmin) / 60.0) / max(float(fov_w_deg), 1e-6)
        return min(frac, 1.0) - 0.35 * max(frac - 1.0, 0.0)

    def _filter_sort_targets(self, rows):
        mode = self.t_sort.currentData() if hasattr(self, "t_sort") else "size"
        coll = self.t_season.currentData() if hasattr(self, "t_season") else "all"
        fw, _fh = fov_deg(self.t_focal.value())
        out = []
        for r in rows:
            oid = r[0]
            ra = float(r[6]) if len(r) >= 8 and r[6] is not None else None
            size_am = float(r[11]) if len(r) >= 12 and r[11] else 0.0
            if coll in self._SEASON_RA and ra is not None:
                lo, hi = self._SEASON_RA[coll]
                h = (ra / 15.0) % 24.0
                inside = (lo <= h < hi) if lo < hi else (h >= lo or h < hi)
                if not inside:
                    continue
            elif coll == "messier" and not re.match(r"(?i)^m\s*\d+$", oid):
                continue
            elif coll == "fits" and not (0 < size_am / 60.0 <= fw * 0.9):
                continue
            out.append(r)
        if mode == "size":
            # Largest first, but an object wider than the frame is not a better target
            # than one that fills it: rank by how much of the frame it covers, capped.
            out.sort(key=lambda r: self._frame_fit(
                float(r[11]) if len(r) >= 12 and r[11] else 0.0, fw), reverse=True)
        elif mode == "alt":
            out.sort(key=lambda r: float(r[2]), reverse=True)
        elif mode == "clear":
            out.sort(key=lambda r: float(r[10] or 0), reverse=True)
        return out

    def _populate_target_table(self, hdr, rows):
        self.t_header.setText(hdr)
        total = len(rows)
        rows = self._filter_sort_targets(rows)
        if hasattr(self, "t_count_lbl"):
            self.t_count_lbl.setText("{} of {} targets".format(len(rows), total))
        self.t_table.setSortingEnabled(False)  # avoid reordering mid-fill
        self.t_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            oid, name, hmax, transit, sep, fit = row[:6]
            ra_deg, dec_deg = (row[6], row[7]) if len(row) >= 8 else (None, None)
            otype = row[8] if len(row) >= 9 else ""
            moon_av = row[9] if len(row) >= 10 else None
            clear_h = row[10] if len(row) >= 11 else None
            size_am = float(row[11]) if len(row) >= 12 and row[11] else 0.0
            if name and name != oid:
                disp_name = _translate_target(name)   # FR planner names -> EN; EN names unchanged
            elif otype and otype not in ("?", ""):
                disp_name = otype
            else:
                disp_name = oid
            if sep != sep:                                    # NaN
                sep_txt = "—"
            elif moon_av is not None:
                sep_txt = "{:.0f}° · {}%".format(sep, moon_av)   # separation + avoidance score
            else:
                sep_txt = "{:.0f}°".format(sep)
            mo = re.match(r"([A-Za-z]+)\s*0*(\d+)", oid)
            id_key = (mo.group(1).upper(), int(mo.group(2))) if mo else (oid.upper(), 0)
            if clear_h is None:
                clear_txt, clear_key = "—", -1.0
            else:
                mins = int(round(clear_h * 60))
                clear_txt = "{}h{:02d}".format(mins // 60, mins % 60) if mins else "0"
                clear_key = clear_h
            if size_am <= 0:
                size_txt, size_key = "—", -1.0
            else:
                fw, _fh = fov_deg(self.t_focal.value())
                frac = (size_am / 60.0) / max(fw, 1e-6)
                size_txt = ("{:.1f}°".format(size_am / 60.0) if size_am >= 60
                            else "{:.0f}′".format(size_am))
                size_txt += "  ({:.0f}%)".format(frac * 100) if frac >= 0.01 else "  (<1%)"
                # Sorting on how much of the frame it fills, not on raw arcminutes:
                # an object twice as wide as the frame is not a better target than one
                # that fills it, so anything past 100 % is pushed back down.
                size_key = self._frame_fit(size_am, fw)
            cells = [(oid, id_key), (disp_name, disp_name.lower()), (size_txt, size_key),
                     ("{:.0f}°".format(hmax), hmax), (clear_txt, clear_key), (transit, transit),
                     (sep_txt, 999 if sep != sep else sep), (str(fit), str(fit))]
            for j, (txt, key) in enumerate(cells):
                it = _SortItem(txt, key)
                if j == 0 and ra_deg is not None:
                    it.setData(Qt.UserRole, (oid, ra_deg, dec_deg))
                self.t_table.setItem(i, j, it)
            self.t_table.setItem(i, 8, _SortItem("", float(i)))   # the order we computed
        self.t_table.setSortingEnabled(True)
        # Enabling sorting re-sorts by whatever column carries the indicator — by
        # default the first one, which threw away the ranking every time. Sort
        # explicitly on the column the chosen mode is about.
        mode = self.t_sort.currentData() if hasattr(self, "t_sort") else "size"
        col = {"size": 2, "alt": 3, "clear": 4}.get(mode)
        if col is not None:
            self.t_table.sortItems(col, Qt.DescendingOrder)
        else:
            self.t_table.sortItems(8, Qt.AscendingOrder)
        self.t_table.resizeColumnsToContents()

    # ---- live search / filter over the full catalogue --------------------
    def _full_catalog(self):
        """Lazy-loaded full DSO catalogue for search: list of dicts. Enriched with a
        curated list of common names (English + French) so objects like the Heart /
        Soul / Shark nebulae are searchable even when the base catalogue lacks names."""
        if getattr(self, "_catalog_cache", None) is not None:
            return self._catalog_cache
        out = []
        by_id = {}
        here = os.path.dirname(os.path.abspath(__file__))
        import csv as _csv
        path = os.path.join(here, "catalog_ngc.csv")
        try:
            with open(path, newline="") as f:
                for r in _csv.DictReader(f):
                    try:
                        ra = float(r["ra_hours"]) * 15.0; dec = float(r["dec_deg"])
                    except (ValueError, KeyError):
                        continue
                    sz = float(r["size_arcmin"]) if r.get("size_arcmin") else 0.0
                    o = {"id": r["id"], "name": r.get("name") or r["id"], "name_fr": "",
                         "type": r.get("type") or "", "ra": ra, "dec": dec, "size": sz}
                    out.append(o); by_id[r["id"].upper().replace(" ", "")] = o
        except Exception:             # noqa: BLE001
            pass
        # merge curated common names
        npath = os.path.join(here, "dso_names.csv")
        try:
            with open(npath, newline="") as f:
                for r in _csv.DictReader(f):
                    key = (r.get("id") or "").upper().replace(" ", "")
                    name = (r.get("name") or "").strip()
                    name_fr = (r.get("name_fr") or "").strip()
                    exist = by_id.get(key)
                    if exist is not None:                 # enrich existing catalogue entry
                        if name and (not exist["name"] or exist["name"] == exist["id"]):
                            exist["name"] = name
                        exist["name_fr"] = name_fr
                        if exist["name"] not in (name, "") and name:
                            # keep both english common name and the catalogue name
                            exist["name"] = name
                    else:                                  # new object not in base catalogue
                        try:
                            ra = float(r["ra_deg"]); dec = float(r["dec_deg"])
                        except (ValueError, KeyError):
                            continue
                        sz = float(r["size_arcmin"]) if r.get("size_arcmin") else 0.0
                        o = {"id": r.get("id") or name, "name": name or r.get("id"),
                             "name_fr": name_fr, "type": r.get("type") or "Nebula",
                             "ra": ra, "dec": dec, "size": sz}
                        out.append(o); by_id[key] = o
        except Exception:             # noqa: BLE001
            pass
        self._catalog_cache = out
        return out

    def _local_transit(self, ra_deg):
        """Local transit time (meridian crossing) for the chosen date — closed form."""
        try:
            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo
            import sky_map
            qd = self.t_date.date()
            base = datetime(qd.year(), qd.month(), qd.day())           # 0h UT
            g0 = sky_map.lst_deg(base, 0.0) % 360.0                    # GMST at 0h
            target = (ra_deg - self.t_lon.value()) % 360.0            # need GMST=RA-lon
            hours_ut = ((target - g0) % 360.0) / 15.0410686           # sidereal deg/h
            t_ut = base + timedelta(hours=hours_ut)
            loc = t_ut.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Paris"))
            return loc.strftime("%Hh%M")
        except Exception:             # noqa: BLE001
            return "—"

    def _fits_size(self, size_arcmin):
        fw, fh = self._sky_mod.fov_deg(self.t_focal.value()) if getattr(self, "_sky_mod", None) \
            else (10.0, 7.0)
        if not size_arcmin:
            return "—"
        return "✓ fits" if (size_arcmin / 60.0) <= min(fw, fh) else "✗ wide"

    def _filter_targets(self, text):
        q = (text or "").strip()
        if not q:
            if getattr(self, "_last_plan", None):
                self._populate_target_table(*self._last_plan)
            return
        import unicodedata

        def norm(s):
            s = (s or "").replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae")
            s = unicodedata.normalize("NFKD", s)
            s = "".join(c for c in s if not unicodedata.combining(c))
            return "".join(ch for ch in s.upper() if ch.isalnum())

        qn = norm(q)
        short = len(qn) <= 2          # avoid name-substring flooding for "M", "IC"…
        lat = self.t_lat.value()
        t1, t2, t3 = [], [], []
        for o in self._full_catalog():
            idn = norm(o["id"]); nmn = norm(o["name"]); frn = norm(o.get("name_fr", ""))
            if idn.startswith(qn):
                t1.append(o)
            elif nmn.startswith(qn) or (frn and frn.startswith(qn)):
                t2.append(o)
            elif (not short) and (qn in idn or qn in nmn or (frn and qn in frn)):
                t3.append(o)
            if len(t1) + len(t2) + len(t3) >= 800:
                break
        matches = t1 + t2 + t3
        rows = []
        for o in matches:
            mx = 90.0 - abs(lat - o["dec"])
            rows.append((o["id"], o["name"], mx, self._local_transit(o["ra"]),
                         float("nan"), self._fits_size(o["size"]),
                         o["ra"], o["dec"], o["type"]))
        self._populate_target_table(
            "🔍 “{}” — {} objects (Max Alt = altitude at transit; click a header to sort)".format(
                q, len(rows)), rows)
        self.t_table.sortItems(0, Qt.SortOrder.AscendingOrder)   # M1, M2…, then NGC, IC

    # -- slots from worker --------------------------------------------
    @Slot(object, float, object)
    def on_frame(self, frame, val, stars):
        if self.review_mode:          # during intervalometer: no live view
            return
        if getattr(self, "_plan_recording", False):
            self._plan_rec_write(frame)
            return
        if getattr(self, "_plan_active", False):
            self._planetary_process(frame)
            return
        if getattr(self, "_drift_active", False):
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            self._update_drift(g)
        h, w = frame.shape[:2]
        disp = frame.copy()
        # ROI only if sharpness measurement is active
        if self.chk_metric.isChecked():
            rw, rh = int(w * self.roi_frac), int(h * self.roi_frac)
            x0, y0 = (w - rw) // 2, (h - rh) // 2
            cv2.rectangle(disp, (x0, y0), (x0 + rw, y0 + rh), (45, 212, 191), 2)
        # star markers
        if stars is not None:
            for (sx, sy, r) in stars:
                cv2.circle(disp, (int(sx), int(sy)), max(4, int(r) + 3), (80, 200, 255), 1)
            self.star_label.setText("★ {} stars detected".format(len(stars)))
        elif not self.chk_stars.isChecked():
            self.star_label.setText("")

        # reference ghost (framing / mosaic): movable overlay
        if self.chk_ghost.isChecked() and self._ghost_img is not None:
            gh = cv2.resize(self._ghost_img, (w, h))
            tx = int(self.ghost_dx.value() / 100.0 * w)
            ty = int(self.ghost_dy.value() / 100.0 * h)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            gh = cv2.warpAffine(gh, M, (w, h))
            alpha = self.ghost_alpha.value() / 100.0
            disp = cv2.addWeighted(disp, 1.0, gh, alpha, 0)
        rgb = np.ascontiguousarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))
        pv = getattr(self, "_public", None)
        if pv is not None and pv.isVisible():
            now = time.monotonic()
            if now - getattr(self, "_pub_live_t", 0.0) > 0.25:
                self._pub_live_t = now
                pv._t = 1.0; pv._prev = None                # live: no cross-fade
                self._publish_public(frame)

        # curve (only if metric active: val non-NaN)
        if val != val:        # NaN -> metric disabled
            return
        self.hist.append(val)
        self.session_peak = max(self.session_peak, val)
        ys = list(self.hist)
        self.curve.setData(ys)
        if ys:
            self.peak_line.setValue(max(ys))
        pct = 100.0 * val / self.session_peak if self.session_peak > 0 else 0.0
        color = "#22c55e" if pct > 92 else "#f59e0b" if pct > 70 else "#ef4444"
        self.readout.setText("sharpness: {:.0f}   ({:.0f}% of peak)".format(val, pct))
        self.readout.setStyleSheet("color:{};".format(color))

    def _set_review_mode(self, on):
        """Switches between live view (adjustment) and review of the last photo (intervalometer)."""
        self.review_mode = on
        self.rec_label.setVisible(on)
        self.plot.setVisible((not on) and self.chk_metric.isChecked())
        self.hist_plot.setVisible(on)        # histogram visible in review
        self.track_plot.setVisible(on)       # HFR tracking visible in review
        self.iv_start.setEnabled(not on)
        self.star_label.setText("")
        if not on:
            self.clip_label.setText("")
        if on:
            self._track_x.clear(); self._track_y.clear()
            self.track_curve.setData([], [])
        if on:
            self._elapsed_start = time.time()
            self._last_progress = (0, 0)
            self._prog_timer.start()
            self.view.setPixmap(QtGui.QPixmap())
            self.view.setText("Intervalometer started — waiting for first photo…")
            self.readout.setStyleSheet("color:#cbd5e1;")
            self.readout.setText("last photo")
        else:
            self._prog_timer.stop()
            self.prog_label.setText("—")
            self.readout.setStyleSheet("color:#cbd5e1;")
            self.readout.setText("")
            self.view.setText("Live view (adjustment)…" if self.chk_live.isChecked()
                              else "Live view disabled (battery saving).")

    @Slot(int, int, float)
    def on_interval_progress(self, done, total, avg):
        self._last_progress = (done, total)
        self._eta_deadline = (time.time() + (total - done) * avg) if (total and avg > 0) else None
        self._tick_progress()

    def _tick_progress(self):
        done, total = self._last_progress
        if total:
            remain = int(self._eta_deadline - time.time()) if self._eta_deadline else 0
            self.prog_label.setText("Shot {}/{}  ·  remaining ~{}".format(
                done, total, _fmt_dur(remain)))
        else:
            elapsed = int(time.time() - self._elapsed_start) if self._elapsed_start else 0
            self.prog_label.setText("Shot #{}  ·  {} elapsed  ·  manual stop".format(
                done, _fmt_dur(elapsed)))

    @Slot(bool)
    def on_interval_state(self, running):
        self._iv_running = running
        if not running and getattr(self, "_plan_burst_dir", None):
            folder, self._plan_burst_dir = self._plan_burst_dir, None
            QtCore.QTimer.singleShot(1200, lambda: self._plan_after_burst(folder))
        self._update_awake()                              # awake follows capture/armed state
        if running:
            self._gx_flat = None; self._gx_done_id = None    # fresh GraXpert per burst
            self._await_done = False
            self._track_ref_gray = None
            self._track_ref_stars = None
            self._track_warned = False
            self._star_hist = []
            self._cloud_warned = False
            self.track_match.setText("")
            self.done_btn.setVisible(False)
            self._burst_autosolved = False        # auto plate-solve the 1st frame of a big burst
            self._sess_hfr = []; self._sess_t0 = time.monotonic()
            self._series_frames = 0
            self._iv_paused = False
            self.iv_pause.setEnabled(True); self.iv_pause.setText("⏸ Pause")
            self._set_review_mode(True)
        else:
            self._single_shot = False
            # sequence finished: we STAY on the final preview until "Done" clicked
            self._await_done = True
            self.rec_label.setVisible(False)
            self._prog_timer.stop()
            self.prog_label.setText("Finished — press Done")
            self.iv_start.setEnabled(True)        # possibility to relaunch a burst
            self._iv_paused = False
            self.iv_pause.setEnabled(False); self.iv_pause.setText("⏸ Pause")
            if not getattr(self, "_series_frames", 0):
                # Nothing was taken (an auto-stop rule fired at once, or the camera
                # refused): staying in review would show a stale frame with no way out
                # but the Done button. Go back to the live view and say so.
                self.prog_label.setText("Series ended before any frame")
                self._return_to_live()
                self.statusBar().showMessage(
                    "The series ended before taking a single frame — see the message "
                    "above; check the Automation stop rules.", 12000)
                return
            self.done_btn.setVisible(True)
            self._notify("NOUT", "Burst finished.")
            QtCore.QTimer.singleShot(600, self._show_session_summary)
            # review_mode stays True: live view doesn't overwrite final preview

    def _show_session_summary(self):
        """A short card at the end of a series: what was gathered, and how well."""
        w = self.worker
        kept = int(getattr(w, "_stack_count", 0) or 0)
        rej = int(getattr(w, "_stack_rejected", 0) or 0)
        n, integ = getattr(self, "_pub_stats", (0, 0.0))
        taken = int(getattr(w, "_done", 0) or 0)
        hf = getattr(self, "_sess_hfr", [])
        dur = time.monotonic() - getattr(self, "_sess_t0", time.monotonic())
        rows = [("Frames taken", str(taken)), ("Session length", _fmt_dur(dur))]
        if getattr(self, "_solve_wcs", None) and getattr(self, "_solved_ready", False):
            ra0, dec0, _rot, fov_w = self._solve_wcs
            rows.append(("Field identified", "RA {:.2f}°  Dec {:+.2f}°  ·  {:.1f}° wide".format(
                ra0, dec0, fov_w)))
        if self.chk_stack.isChecked():
            rows += [("In the stack", "{}  ·  {} rejected".format(kept, rej)),
                     ("Light gathered", _fmt_dur(integ))]
        if hf:
            vals = sorted(h for _i, h in hf)
            best = min(hf, key=lambda t: t[1]); worst = max(hf, key=lambda t: t[1])
            rows += [("Median HFR", "{:.2f} px".format(vals[len(vals) // 2])),
                     ("Sharpest / softest", "#{} ({:.2f})  ·  #{} ({:.2f})".format(
                         best[0], best[1], worst[0], worst[1]))]
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Session summary")
        v = QtWidgets.QVBoxLayout(dlg); v.setContentsMargins(18, 16, 18, 14); v.setSpacing(10)
        head = QtWidgets.QLabel("Series finished"); head.setStyleSheet("font-size:15px; font-weight:600;")
        v.addWidget(head)
        g = QtWidgets.QGridLayout(); g.setHorizontalSpacing(18); g.setVerticalSpacing(6)
        for r, (k, val) in enumerate(rows):
            a = QtWidgets.QLabel(k); a.setObjectName("cap")
            b = QtWidgets.QLabel(val); b.setStyleSheet("font-weight:600;")
            g.addWidget(a, r, 0); g.addWidget(b, r, 1)
        v.addLayout(g)
        br = QtWidgets.QHBoxLayout()
        sky = QtWidgets.QPushButton("Show frame in Sky Map")
        sky.setToolTip("Opens the Sky Map on the field just photographed — the camera "
                       "frame is drawn where the plate-solve put it.")
        sky.clicked.connect(lambda: (self.tabs.setCurrentIndex(self._skymap_index),
                                     dlg.accept()))
        res = QtWidgets.QPushButton("Open Results")
        res.clicked.connect(lambda: (self.tabs.setCurrentIndex(self._results_index),
                                     dlg.accept()))
        br.addWidget(sky); br.addWidget(res); br.addStretch(1)
        pb = QtWidgets.QPushButton("Public view"); pb.clicked.connect(self._toggle_public_view)
        ok = QtWidgets.QPushButton("OK"); ok.setObjectName("primary"); ok.clicked.connect(dlg.accept)
        br.addWidget(pb); br.addWidget(ok); v.addLayout(br)
        dlg.setModal(False); dlg.show()
        self._summary_dlg = dlg

    def _return_to_live(self):
        self._await_done = False
        self.done_btn.setVisible(False)
        self._set_review_mode(False)

    def _exposure_hint(self, img):
        """Sky level of the last frame, read like the camera's own histogram: the
        background peak between ~1/7 and ~1/2 is where a 30 s sub is well used."""
        try:
            g = cv2.cvtColor(img[::4, ::4], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img[::4, ::4]
            pct = 100.0 * float(np.median(g)) / 255.0
        except Exception:               # noqa: BLE001
            return
        if pct < 8:
            txt = "Sky {:.0f}%  ·  under-exposed: raise ISO".format(pct)
        elif pct < 14:
            txt = "Sky {:.0f}%  ·  a little low".format(pct)
        elif pct <= 45:
            txt = "Sky {:.0f}%  ·  well exposed".format(pct)
        else:
            txt = "Sky {:.0f}%  ·  bright: lower ISO".format(pct)
        self._hud_info["br"] = txt
        self.view.set_hud_info(dict(self._hud_info))

    @Slot(object, str)
    def on_capture_image(self, img, caption):
        self._series_frames = getattr(self, "_series_frames", 0) + 1
        if img is not None:
            self._exposure_hint(img)
        if img is not None and self.chk_track.isChecked() \
                and (self.review_mode or getattr(self, "_iv_running", False)):
            self._update_tracking(img)
        # auto-identify the field: a single shot, or the first frame of a long burst
        if (not getattr(self, "_burst_autosolved", True)
                and (getattr(self, "_single_shot", False) or self.chk_unlimited.isChecked()
                     or self.iv_count.value() > 10)
                and getattr(self, "_last_capture_path", None)
                and os.path.exists(self._last_capture_path)):
            self._burst_autosolved = True
            self.solve_label.setText("🔭 Identifying the field…")
            self._run_solve()
        # auto-center: once its 5 s frame is saved, plate-solve it (drives _ac_on_solved)
        if (getattr(self, "_ac_active", False) and self._ac_phase == "capture"
                and getattr(self, "_last_capture_path", None)
                and os.path.exists(self._last_capture_path)):
            self._ac_phase = "solving"
            self._run_solve()
        if self.chk_stack.isChecked():
            return                       # display + readout come from on_stack_ready
        self.readout.setText(caption)
        self._last_capture_bgr = img
        if img is None:
            self.view.setText("Photo saved — RAW preview unavailable.\n"
                              "To view .ARW : pip install rawpy")
            for c in (self.hist_curve_r, self.hist_curve_g, self.hist_curve_b):
                c.setData([0])
            return
        self._render_review_image()

    def _render_review_image(self):
        img = self._last_capture_bgr
        if img is None:
            return
        # clipping alert (on UNPROCESSED image)
        frac = _clipping_fraction(img) * 100.0
        if frac > 1.0:
            self.clip_label.setText("⚠ clipping: {:.1f}% saturated pixels".format(frac))
            self.clip_label.setStyleSheet("color:#ef4444;")
        elif frac > 0.05:
            self.clip_label.setText("light clipping: {:.1f}%".format(frac))
            self.clip_label.setStyleSheet("color:#f59e0b;")
        else:
            self.clip_label.setText("no clipping")
            self.clip_label.setStyleSheet("color:#22c55e;")
        disp = img
        src = self._stack_float if (self.chk_stack.isChecked()
                                    or self._folder_stacking) else None
        if src is not None and self.chk_stretch.isChecked() and self.chk_obs.isChecked():
            flat = None
            if self.chk_degrad.isChecked() and self._use_graxpert_live():
                flat = self._graxpert_live(src)         # None until GraXpert answers
                if flat is not None and flat.shape != src.shape:
                    flat = None
            self._obs_request(src, flat)                # shown when the thread answers
            return
        if src is not None and (self.chk_degrad.isChecked() or self.chk_stretch.isChecked()):
            # full-precision path: gradient removal + Siril MTF stretch on the float mean
            work = src
            if self.chk_degrad.isChecked():
                if self._use_graxpert_live():
                    gxf = self._graxpert_live(src)      # async; None until first result
                    work = gxf if gxf is not None else remove_gradient(src, 1.0)
                else:
                    work = remove_gradient(work, 1.0)   # float in -> float out (polynomial)
                work = star_white_balance(work)        # neutralise star colour (PCC-like)
                work = neutralize_background(work)      # neutral gray sky (kills colour cast)
            if self.chk_stretch.isChecked():
                disp = auto_stretch(work, 0.5, scnr=True, saturation=self._sat())
            else:
                disp = np.clip(work / (float(work.max()) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        else:
            if self.chk_degrad.isChecked():                     # remove gradient (vignetting/LP)
                disp = remove_gradient(disp, 1.0)
            if self.chk_stretch.isChecked():
                disp = auto_stretch(disp, 0.5, scnr=True, saturation=self._sat())
        disp = np.ascontiguousarray(disp)
        if getattr(self, "chk_objs", None) is not None and self.chk_objs.isChecked() \
                and self._solve_wcs is not None:
            disp = self._sky_objects_overlay(
                disp, getattr(self.worker, "_stack_view_rect", None) if src is not None else None)
        self._show_rendered(disp)

    def _show_rendered(self, disp):
        """Put a finished render on screen (and on the public view, if open)."""
        h, w = disp.shape[:2]
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))
        self._publish_public(disp)
        hb, hg, hr = rgb_hist(disp)
        self.hist_curve_r.setData(hr)
        self.hist_curve_g.setData(hg)
        self.hist_curve_b.setData(hb)

    def _obs_request(self, src, flat):
        """Latest-wins: one render at a time, and only the newest stack waits."""
        # Full resolution: keep every pixel in the render too, so zooming in shows the
        # detail the 24 MP stack holds (~5 s per render instead of ~1 s, off-thread).
        # Otherwise the screen size is all that is ever shown.
        full = self.chk_stack_full.isChecked()
        self._obs_start((src, flat, self._sat(), 100000 if full else 2400))

    def _obs_start(self, job):
        if getattr(self, "_obs_worker", None) is not None and self._obs_worker.isRunning():
            self._obs_pending = job
            return
        self._obs_pending = None
        self._obs_worker = ObsRenderWorker(*job)
        self._obs_worker.done.connect(self._obs_done)
        self._obs_worker.start()

    def _obs_done(self, img):
        old = self.sender()
        if old is not None:
            _retire_thread(old, 2000)
        if img is not None and self.review_mode:
            disp = np.ascontiguousarray(img)
            if getattr(self, "chk_objs", None) is not None and self.chk_objs.isChecked() \
                    and self._solve_wcs is not None:
                disp = self._sky_objects_overlay(
                    disp, getattr(self.worker, "_stack_view_rect", None))
            self._show_rendered(disp)
        job, self._obs_pending = getattr(self, "_obs_pending", None), None
        if job is not None:
            self._obs_start(job)

    # -- public view -----------------------------------------------------------
    def _toggle_public_view(self):
        pv = getattr(self, "_public", None)
        if pv is not None and pv.isVisible():
            pv.close(); return
        pv = self._public = PublicView()
        screens = QtGui.QGuiApplication.screens()
        mine = self.screen()
        other = [sc for sc in screens if sc is not mine]
        target = other[0] if other else mine
        pv.setGeometry(target.geometry())
        pv.showFullScreen()
        if pv.windowHandle() is not None:
            pv.windowHandle().setScreen(target)
        self._update_public_caption()
        last = getattr(self, "_last_public_img", None)
        if last is not None:
            pv.set_image(last)
        self.statusBar().showMessage(
            "Public view on {} — Esc or V to close.".format(
                "the second screen" if other else "this screen"), 6000)

    def _publish_public(self, bgr):
        self._last_public_img = bgr
        pv = getattr(self, "_public", None)
        if pv is not None and pv.isVisible():
            pv.set_image(bgr)
            self._update_public_caption()

    def _target_identity(self):
        """(title, distance text) for the session target: its common name with the
        catalogue id, distance in words. Moon and planets: their real distance now."""
        goal = getattr(self, "_session_goal", None)
        if not goal:
            return "", ""
        name, ra, dec = goal
        key = (name or "").strip().lower()
        if key in _BODY_EN:                                  # Moon, planets, Sun
            body = _BODY_EN[key]
            try:
                import astropy.units as u
                from astropy.coordinates import get_body, EarthLocation
                from astropy.time import Time
                loc = EarthLocation(lat=self.t_lat.value() * u.deg, lon=self.t_lon.value() * u.deg)
                b = get_body(body.lower(), Time.now(), loc)
                return body, fmt_distance(km=float(b.distance.to(u.km).value))
            except Exception:           # noqa: BLE001
                return body, ""
        oid, cat_name, typ = name, "", ""
        try:
            objs = list(getattr(self.skymap, "dso", [])) if getattr(self, "skymap", None) else []
            best, bd = None, 0.5
            for o in objs:
                d = ((float(o["ra"]) - ra) ** 2 + (float(o["dec"]) - dec) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, o
            if best:
                oid, cat_name, typ = best.get("id", name), best.get("name", ""), best.get("type", "")
        except Exception:               # noqa: BLE001
            pass
        nm = cat_name if (cat_name and cat_name != oid
                          and not re.match(r"(?i)^messier\s*\d+$", cat_name)) else ""
        if nm:
            same = oid and oid.upper().replace(" ", "") in nm.upper().replace(" ", "")
            title = nm if same or not oid else "{}  ({})".format(nm, oid)
        else:                                                 # unnamed: say what it is
            t = (typ or "").lower()
            kind = next((v for k, v in _TYPE_EN if k in t), "")
            title = "{} {}".format(kind, oid).strip() if kind else oid
        return title, fmt_distance(ly=dso_distance_ly(oid))

    def _update_hud_identity(self):
        title, dist = self._target_identity()
        self._hud_info["tl"] = "{}  ·  {}".format(title, dist) if dist else title
        self.view.set_hud_info(dict(self._hud_info))

    def _update_public_caption(self):
        pv = getattr(self, "_public", None)
        if pv is None:
            return
        title, dist = self._target_identity()
        n, integ = getattr(self, "_pub_stats", (0, 0.0))
        # for the visitors: in words rather than 00:09:30
        m = int(round(integ / 60.0))
        dur = ("{} s".format(int(integ)) if integ < 60 else
               "{} min".format(m) if m < 60 else "{} h {:02d}".format(m // 60, m % 60))
        sub = ("{} exposure{} · {} of light gathered".format(n, "s" if n != 1 else "", dur)
               if n else "")
        pv.set_caption(title, sub, dist)

    def _toggle_stretch_shortcut(self):
        self.chk_stretch.setChecked(not self.chk_stretch.isChecked())

    def _toggle_fullscreen_preview(self):
        if getattr(self, "_fs_preview", False):
            self._exit_fullscreen_preview()
            return
        self._fs_win = QtWidgets.QWidget()
        self._fs_win.setStyleSheet("background:#000;color:#ddd;")
        fl = QtWidgets.QVBoxLayout(self._fs_win)
        fl.setContentsMargins(0, 0, 0, 0); fl.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        btn = QtWidgets.QPushButton("✕ Exit fullscreen (Esc)")
        btn.clicked.connect(self._exit_fullscreen_preview)
        bar.addStretch(1); bar.addWidget(btn)
        fl.addLayout(bar)
        self.view.setParent(None)
        fl.addWidget(self.view, 1)
        self._fs_win.showFullScreen()
        self._fs_preview = True

    def _exit_fullscreen_preview(self):
        if not getattr(self, "_fs_preview", False):
            return
        self._fs_win.layout().removeWidget(self.view)
        self.view_box.set_child(self.view)
        self.view.fit()
        self._fs_win.close()
        self._fs_win = None
        self._fs_preview = False

    def _restretch(self, *args):
        if self._last_capture_bgr is not None and self.review_mode:
            self._render_review_image()

    def _update_tracking(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if self._track_ref_gray is None:        # first exposure = reference
            self._track_ref_gray = gray
            _stars, n = detect_stars(gray)
            self.track_match.setText("Tracking: reference acquired ({} stars)".format(n))
            self.track_match.setStyleSheet("color:#94a3b8;")
            return
        pct, drift = tracking_match(self._track_ref_gray, gray,
                                    self.track_tol.value() / 100.0)
        col = "#22c55e" if pct >= 85 else ("#f59e0b" if pct >= self.track_thresh.value() else "#ef4444")
        self.track_match.setStyleSheet("color:{};".format(col))
        self.track_match.setText("Tracking: {:.0f}%  ·  drift {:.0f} px".format(pct, drift))
        self._sb["track"] = "Tracking {:.0f}%".format(pct)
        self._refresh_statusbar()
        if pct < self.track_thresh.value() and not self._track_warned:
            self._track_warned = True
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setWindowTitle("Degraded tracking")
            box.setText("Tracking dropped to {:.0f}% (threshold {} %).\n\n"
                        "The field has significantly drifted: check balancing "
                        "and redo polar alignment.".format(pct, self.track_thresh.value()))
            box.setWindowModality(Qt.NonModal)
            box.show()
            self._track_popup = box

    def _focal(self):
        return self.solve_focal.value() if hasattr(self, "solve_focal") else 135.0

    def _solve_online(self):
        return self.solve_mode.currentText() == "Online"

    def _run_solve(self):
        path = self._last_capture_path
        if not path or not os.path.exists(path):
            self.solve_label.setText("Take a capture first, then launch identification.")
            return
        self.solve_btn.setEnabled(False)
        self.solve_label.setText("Plate-solving in progress…")
        self._solver = SolveWorker(self._focal(), image_path=path,
                                   online=self._solve_online(), api_key=self.solve_key.text())
        self._solver.done.connect(self._solve_done)
        self._solver.start()

    def _solve_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Photo to identify", self.dir_edit.text(),
            "Images (*.arw *.jpg *.jpeg *.png *.tif *.tiff *.fits *.fit)")
        if not path:
            return
        f = _read_focal_exif(path)
        if f:
            self.solve_focal.setValue(f)         # auto focal from EXIF
        self.solve_file_btn.setEnabled(False)
        self._solve_file_path = path
        self.solve_label.setText("Plate-solving {}…".format(os.path.basename(path)))
        self._solver_file = SolveWorker(self._focal(), image_path=path,
                                        online=self._solve_online(), api_key=self.solve_key.text())
        self._solver_file.done.connect(self._solve_file_done)
        self._solver_file.start()

    @Slot(str, object)
    @Slot(str, object, float, float, float, float)
    def _solve_done(self, msg, annot, ra, dec, rot, fov):
        self.solve_btn.setEnabled(True)
        self.solve_label.setText(msg)
        if math.isfinite(ra) and math.isfinite(dec):
            self._solved_to_skymap(ra, dec, "Solved field", rot)
            self._solve_wcs = (ra % 360.0, dec, rot if math.isfinite(rot) else 0.0,
                               fov if math.isfinite(fov) and fov > 0 else 10.0)
        # auto-center routine intercepts the solve (no burst attach / dialog pop-up)
        if getattr(self, "_ac_active", False):
            if math.isfinite(ra) and math.isfinite(dec):
                self._ac_on_solved(ra % 360.0, dec)
            else:
                self._ac_finish("Auto-center: plate-solve failed — stopped.")
            return
        # attach the annotation to the burst already in Results (same salvo), not a duplicate
        idx = getattr(self, "_last_burst_idx", None)
        if annot is not None and idx is not None and 0 <= idx < len(self._results):
            e = self._results[idx]
            self._results[idx] = (e[0], e[1], annot, e[3] if len(e) > 3 else False,
                                  e[4] if len(e) > 4 else None)
            it = self._item_for_idx(idx)
            if it is not None and "🔭" not in it.text():
                it.setText(it.text() + "  🔭")
            self._save_annotated(annot, "annot")
            self.tabs.setCurrentIndex(2)
            self._open_image_dialog(e[0], e[1], annot=annot,
                                    processed=e[3] if len(e) > 3 else False,
                                    raw=e[4] if len(e) > 4 else None)
        else:
            self._add_solved_result(getattr(self, "_last_capture_bgr", None), annot,
                                    ra, dec, "🔭 Identify (last photo)")
        if math.isfinite(ra) and math.isfinite(dec):
            self._solved_to_skymap(ra, dec, "Solved field", rot)
            self._solve_wcs = (ra % 360.0, dec, rot if math.isfinite(rot) else 0.0,
                               fov if math.isfinite(fov) and fov > 0 else 10.0)
            if hasattr(self, "chk_objs") and self.chk_objs.isChecked():
                self._render_review_image()
        if getattr(self, "_autocenter", {}).get("active"):
            self._autocenter_after_solve()

    @Slot(str, object, float, float, float, float)
    def _solve_file_done(self, msg, annot, ra, dec, rot, fov):
        self.solve_file_btn.setEnabled(True)
        self.solve_label.setText(msg)
        base = None
        p = getattr(self, "_solve_file_path", None)
        if p:
            try:
                base = _decode_preview_image(p, full_demosaic=False)
            except Exception:           # noqa: BLE001
                base = None
        self._add_solved_result(base, annot, ra, dec,
                                "🔭 " + (os.path.basename(p) if p else "Identified file"))
        if math.isfinite(ra) and math.isfinite(dec):
            self._solved_to_skymap(ra, dec, "Solved field", rot)

    def _add_solved_result(self, base, annot, ra, dec, prefix):
        """Add a plate-solve result to the Results list with the annotated image attached
        (toggle in the viewer). Used for 'Identify a file' and as a fallback."""
        if annot is None and base is None:
            self._show_annotated(annot, "annot")
            return
        if base is None:
            base = annot
        label = prefix
        if math.isfinite(ra) and math.isfinite(dec):
            label = "{} · RA {:.2f} Dec {:+.2f}".format(prefix, ra, dec)
        item = self._add_result(base, label)
        idx = item.data(Qt.UserRole)
        entry = self._results[idx]
        self._results[idx] = (entry[0], entry[1], annot, entry[3], entry[4])
        if annot is not None:
            item.setText(item.text() + "  🔭")
            self._save_annotated(annot, "annot")
        self.tabs.setCurrentIndex(2)
        self._open_image_dialog(base, label, annot=annot)

    def _save_annotated(self, annot, name):
        if annot is None:
            return
        try:
            d = self.dir_edit.text(); os.makedirs(d, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(d, "{}_{}.png".format(name, ts))
            cv2.imwrite(path, annot)
            self.statusBar().showMessage("Annotated image saved: " + path, 9000)
        except Exception:           # noqa: BLE001
            pass

    def _show_annotated(self, annot, name):
        if annot is None:
            return
        self._save_annotated(annot, name)
        # zoomable view
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Annotated field (plate-solving)")
        dlg.resize(900, 650)
        v = QtWidgets.QVBoxLayout(dlg)
        view = ZoomableView(); view.setPixmap(self._bgr_to_pixmap(annot))
        v.addWidget(view)
        dlg.show()
        self._annot_dialog = dlg

    def _push_autostop_pd(self, *args):
        self.worker.post("set_autostop", on=self.chk_autostop_pd.isChecked(),
                         batt=self.batt_thr.value(), disk=self.disk_thr.value())

    def _load_ghost(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Reference image", self.dir_edit.text(),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.arw)")
        if not path:
            return
        img = _decode_preview_image(path)
        if img is None:
            self.statusBar().showMessage("Reference unreadable.", 5000)
            return
        self._ghost_img = img
        self.chk_ghost.setChecked(True)
        self.statusBar().showMessage("Reference loaded: {}".format(os.path.basename(path)), 5000)

    @Slot(object, int, float)
    def on_stack_ready(self, stacked, count, integ):
        if stacked is None:
            return
        if isinstance(stacked, np.ndarray) and stacked.dtype != np.uint8:
            self._stack_float = stacked                              # linear/float mean
            self._last_capture_bgr = np.clip(stacked, 0, 255).astype(np.uint8)
        else:
            self._stack_float = None
            self._last_capture_bgr = stacked
        self.readout.setText("Stack: {} shots  ·  integration {}".format(
            count, _fmt_dur(integ)))
        self._pub_stats = (int(count), float(integ))
        self._render_review_image()

    def _sky_objects_overlay(self, disp, rect=None):
        """Project catalog objects (incl. dark nebulae like the Shark) onto the stack using
        the last plate-solve (gnomonic + field rotation), so you can see where the first
        faint details should appear. Uses the SAME catalog as the Sky Map. BETA: if markers
        land mirrored/rotated, tell me and I'll flip the convention."""
        try:
            import math
            ra0, dec0, rot, fov_w = self._solve_wcs
            H, W = disp.shape[:2]
            ra0r, dec0r = math.radians(ra0), math.radians(dec0)
            # The live stack is cropped to what every frame covered, then scaled for the
            # screen: measure the field on the FULL frame and put the solved centre where
            # the frame's centre now sits, or markers slide off their objects.
            cx0, cy0, full_w = W / 2.0, H / 2.0, float(W)
            if rect is not None and rect[2] > rect[0]:
                x0r, y0r, x1r, y1r, fw, fh = rect
                k = W / float(x1r - x0r)                  # display px per stack px
                full_w = fw * k
                cx0, cy0 = (fw / 2.0 - x0r) * k, (fh / 2.0 - y0r) * k
            scale = full_w / math.radians(max(fov_w, 0.1))  # pixels per radian
            # Measured on a real solved frame: objects landed on the same diagonal but
            # on the far side of centre — a 180° error. The solver reports the
            # orientation of the frame it was handed, which is upside down with respect
            # to the array we draw on. Settings key 'overlay_rot_offset' if it ever
            # needs to change again.
            rot += float(QtCore.QSettings("NOUT", "NOUT").value(
                "overlay_rot_offset", 180.0, type=float))
            th = math.radians(rot); cth, sth = math.cos(th), math.sin(th)
            objs = list(getattr(self.skymap, "dso", [])) if getattr(self, "skymap", None) else []
            out = disp.copy()
            col = (255, 170, 80)                          # BGR ~ EVA orange
            for o in objs:
                ra, dec = float(o["ra"]), float(o["dec"])
                rar, decr = math.radians(ra), math.radians(dec)
                dra = rar - ra0r
                denom = (math.sin(dec0r) * math.sin(decr)
                         + math.cos(dec0r) * math.cos(decr) * math.cos(dra))
                if denom <= 0:
                    continue
                X = math.cos(decr) * math.sin(dra) / denom
                Y = (math.cos(dec0r) * math.sin(decr)
                     - math.sin(dec0r) * math.cos(decr) * math.cos(dra)) / denom
                xr = X * cth - Y * sth; yr = X * sth + Y * cth
                px = int(round(cx0 - xr * scale))         # RA increases eastward (image left)
                py = int(round(cy0 - yr * scale))
                if not (0 <= px < W and 0 <= py < H):
                    continue
                r = int(max(10, min(0.5 * float(o.get("size", 0)) / 60.0
                                    * math.radians(1) * scale, 120)))  # object size -> px
                cv2.circle(out, (px, py), r, col, 1, cv2.LINE_AA)
                label = o.get("name") or o.get("id") or ""
                cv2.putText(out, str(label), (px + r + 4, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            return out
        except Exception:               # noqa: BLE001
            return disp

    #: frames held back at the start of a recording, to time the live view before
    #: the file is opened — a VideoWriter needs its rate up front and cannot be told later
    _REC_PROBE_FRAMES = 8

    def _plan_rec_write(self, frame):
        """Append one live-view frame to the movie being recorded.

        The first few frames are buffered rather than written, to measure how fast
        the live view actually runs. A VideoWriter fixes its frame rate when it is
        created, so a guessed 15 fps would leave the file claiming a length it does
        not have — 381 frames of a 4 s recording would read as 25 s, and every tool
        that opens it afterwards would inherit the lie."""
        if self._plan_rec_writer is None:
            buf = getattr(self, "_plan_rec_buf", None)
            if buf is None:
                buf = self._plan_rec_buf = []
            buf.append((time.monotonic(), frame.copy()))
            if len(buf) < self._REC_PROBE_FRAMES:
                if self._plan_readout_lbl is not None:
                    try:
                        self._plan_readout_lbl.setText("🎥 Timing the live view…")
                    except RuntimeError:
                        pass
                return
            gaps = sorted(buf[i + 1][0] - buf[i][0] for i in range(len(buf) - 1))
            dt = gaps[len(gaps) // 2] or 0.066            # median, robust to one stall
            fps = max(1.0, min(120.0, 1.0 / dt))
            h, w = frame.shape[:2]
            self._plan_rec_path = os.path.join(
                self._effective_dir() or os.path.expanduser("~"),
                "nout_movie_{}.mp4".format(int(time.time())))
            try:
                os.makedirs(os.path.dirname(self._plan_rec_path), exist_ok=True)
                self._plan_rec_writer = cv2.VideoWriter(
                    self._plan_rec_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            except Exception:           # noqa: BLE001
                self._plan_recording = False
                self._plan_rec_buf = None
                return
            self._plan_rec_fps = fps
            self._plan_rec_n = 0
            self._plan_rec_t0 = buf[0][0]
            for _t, f in buf:                              # nothing is thrown away
                self._plan_rec_writer.write(f)
                self._plan_rec_n += 1
            self._plan_rec_buf = None
        else:
            self._plan_rec_writer.write(frame)
            self._plan_rec_n += 1
        if self._plan_readout_lbl is not None:
            try:
                self._plan_readout_lbl.setText(
                    "🎥 Recording… {} frames at {:.0f} fps".format(
                        self._plan_rec_n, self._plan_rec_fps))
            except RuntimeError:        # noqa: BLE001  (dialog closed)
                pass

    def _ui_shutter(self):
        """The shutter speed showing in the toolbar, or None if it reads Bulb."""
        cb = getattr(self, "cb_shutter", None)
        try:
            v = cb.currentText().strip() if cb is not None else ""
        except RuntimeError:
            return None
        return None if (not v or "bulb" in v.lower()) else v

    def _plan_shoot_burst(self, n):
        """Fire a rapid burst of FULL-resolution stills, then lucky-stack it.

        This is the sharpest path on this camera by a wide margin. The live view is
        fixed at 1024×680 and a 1080p movie downsamples the sensor from 6000 to 1920
        pixels — at 357 mm that is 173 pixels across the Moon against 541 for a full
        frame. Same lucky-imaging maths, three times the detail."""
        if getattr(self, "_iv_running", False):
            self._plan_readout_lbl.setText("A capture is already running — stop it first.")
            return
        base = self._effective_dir() or os.path.expanduser("~")
        folder = os.path.join(base, "lucky_{}".format(int(time.time())))
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:          # noqa: BLE001
            self._plan_readout_lbl.setText("Cannot create {}: {}".format(folder, e))
            return
        self._plan_active = False               # the live-view stacker would fight this
        self._plan_burst_dir = folder
        self._plan_burst_total = int(n)
        self._plan_readout_lbl.setText(
            "🌙 Shooting {} full-resolution frames — a 48 MB RAW takes about 5 s each, "
            "so allow ~{:.0f} min. Progress below.".format(n, n * 5.5 / 60.0))
        self.worker.post("start_interval", interval=0, count=int(n), bulb=False,
                         save_dir=folder, shutter=self._ui_shutter())

    def _plan_burst_progress(self, done, total, avg):
        """Say how the burst is going. Without this the readout sat unchanged for
        minutes and the whole thing looked like it had stopped after one frame."""
        if not getattr(self, "_plan_burst_dir", None):
            return
        lbl = getattr(self, "_plan_readout_lbl", None)
        if lbl is None:
            return
        tot = total or getattr(self, "_plan_burst_total", 0)
        left = (tot - done) * avg if (tot and avg) else 0
        try:
            lbl.setText("🌙 {}/{} frames · {:.1f} s each · about {} left".format(
                done, tot or "?", avg, CameraWorker._eta_str(left)))
        except RuntimeError:            # dialog closed mid-burst
            self._plan_readout_lbl = None

    def _plan_after_burst(self, folder):
        """The burst finished: stack it, and say so if the camera wrote nothing."""
        try:
            shots = [f for f in os.listdir(folder)
                     if not f.startswith(".") and os.path.isfile(os.path.join(folder, f))]
        except Exception:               # noqa: BLE001
            shots = []
        if self._plan_readout_lbl is not None:
            if not shots:
                self._plan_readout_lbl.setText(
                    "The burst produced no files in {} — check the camera is in PC "
                    "Remote and saving to the computer.".format(folder))
                return
            self._plan_readout_lbl.setText(
                "🌙 {} frames captured — lucky-stacking…".format(len(shots)))
            QtWidgets.QApplication.processEvents()
        self._process_image_folder(folder)

    def _plan_record_start(self, seconds):
        if not self.chk_live.isChecked():
            self.chk_live.setChecked(True)             # recording taps the live-view stream
        self._plan_active = False
        self._plan_burst_dir = None
        self._plan_rec_writer = None; self._plan_rec_path = None
        self._plan_rec_buf = None; self._plan_rec_n = 0
        self._plan_rec_fps = 0.0; self._plan_rec_t0 = None
        self._plan_recording = True
        if self._plan_readout_lbl is not None:
            self._plan_readout_lbl.setText("🎥 Recording {} s…".format(seconds))
        QtCore.QTimer.singleShot(int(seconds * 1000), self._plan_record_stop)

    def _plan_record_stop(self):
        if not self._plan_recording:
            return
        self._plan_recording = False
        path = self._plan_rec_path
        if self._plan_rec_writer is not None:
            try:
                self._plan_rec_writer.release()
            except Exception:           # noqa: BLE001
                pass
            self._plan_rec_writer = None
        if path and os.path.exists(path):
            if self._plan_readout_lbl is not None:
                took = (time.monotonic() - (self._plan_rec_t0 or time.monotonic())) or 1.0
                self._plan_readout_lbl.setText(
                    "🎥 {} frames in {:.1f} s ({:.0f} fps) — processing…".format(
                        getattr(self, "_plan_rec_n", 0), took,
                        getattr(self, "_plan_rec_n", 0) / took))
            QtWidgets.QApplication.processEvents()
            self._process_video(path)                  # auto-process the recorded movie
        elif self._plan_readout_lbl is not None:
            self._plan_readout_lbl.setText("Recording produced no file.")

    def _process_image_folder(self, folder):
        """Highest quality on Sony: lucky-stack a burst of FULL-RESOLUTION stills (which DO
        download over gphoto2, unlike movies). Two passes: score, then sub-pixel align+stack
        the best X%, then sharpen."""
        import glob
        exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".arw", ".cr2", ".cr3",
                ".nef", ".raf", ".rw2", ".orf", ".dng")
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if f.lower().endswith(exts))
        if not files:
            self._plan_readout_lbl.setText("No images found in that folder."); return
        self._plan_reset()
        scores = [None] * len(files)
        for i, fp in enumerate(files):                 # pass 1 — score
            img = _decode_preview_image(fp, full_demosaic=True)
            if img is None:
                scores[i] = -1.0; continue
            _c, g = self._plan_crop(img)
            scores[i] = float(cv2.Laplacian(g, cv2.CV_32F).var())
            if i % 3 == 0:
                self._plan_readout_lbl.setText("Scoring {}/{}…".format(i + 1, len(files)))
                QtWidgets.QApplication.processEvents()
        valid = [s for s in scores if s >= 0]
        if not valid:
            self._plan_readout_lbl.setText("Could not decode those images."); return
        thr = float(np.percentile(valid, (1.0 - self._plan_keep_pct / 100.0) * 100.0))
        ref = None; hann = None; mean = None; count = 0
        for i, fp in enumerate(files):                 # pass 2 — align + stack the best
            if scores[i] < thr:
                continue
            img = _decode_preview_image(fp, full_demosaic=True)
            if img is None:
                continue
            crop, g = self._plan_crop(img)
            if ref is None:
                ref = g.copy(); hann = cv2.createHanningWindow((g.shape[1], g.shape[0]),
                                                               cv2.CV_32F)
                mean = crop.copy(); count = 1
            elif g.shape == ref.shape:
                try:
                    (dx, dy), _ = cv2.phaseCorrelate(ref * hann, g * hann)
                except Exception:       # noqa: BLE001
                    dx = dy = 0.0
                M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                aligned = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                count += 1; mean += (aligned - mean) / count
            self._plan_readout_lbl.setText("Stacking {}…".format(count))
            QtWidgets.QApplication.processEvents()
        if mean is None:
            self._plan_readout_lbl.setText("Nothing stacked."); return
        self._plan_mean = mean; self._plan_count = count; self._plan_total = len(files)
        self._plan_finalize()
        self._plan_readout_lbl.setText(
            "Done: stacked {} best of {} full-res photos (sub-pixel + sharpened). Save it.".format(
                count, len(files)))

    def _on_movie_ready(self, path):
        """A camera movie was recorded and downloaded — lucky-stack it automatically."""
        if not self._plan_dlg:
            self._open_planetary_dialog()
        if os.path.exists(path):
            self._process_video(path)

    def _plan_crop(self, frame):
        """Return (crop_float_bgr, gray_float) for the current ROI setting."""
        H, W = frame.shape[:2]
        r = int(self._plan_roi)
        if r <= 0 or r >= min(H, W):
            crop = frame.astype(np.float32)
        else:
            g0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            thr = max(30, int(g0.max() * 0.4)); ys, xs = np.where(g0 >= thr)
            cx, cy = (int(xs.mean()), int(ys.mean())) if len(xs) > 20 else (W // 2, H // 2)
            x0 = int(np.clip(cx - r // 2, 0, W - r)); y0 = int(np.clip(cy - r // 2, 0, H - r))
            crop = frame[y0:y0 + r, x0:x0 + r].astype(np.float32)
        g = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) \
            if crop.ndim == 3 else crop.astype(np.float32)
        return crop, g

    def _process_video(self, path):
        """Highest-quality path: lucky-stack a full-resolution movie recorded in-camera
        (1080p/4K). Two passes: score every frame, then sub-pixel align + stack the best X%."""
        n, frames = open_video_frames(path)
        if frames is None:
            self._plan_readout_lbl.setText("Could not open the video."); return
        self._plan_reset()
        # pass 1 — score every frame
        scores = []
        for f in frames():
            _crop, g = self._plan_crop(f)
            scores.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
            if len(scores) % 25 == 0:
                self._plan_readout_lbl.setText("Scoring frames… {}".format(len(scores)))
                QtWidgets.QApplication.processEvents()
        if not scores:
            self._plan_readout_lbl.setText("No frames decoded."); return
        thr = float(np.percentile(scores, (1.0 - self._plan_keep_pct / 100.0) * 100.0))
        # pass 2 — align + stack the best frames
        ref = None; hann = None; mean = None; count = 0; idx = 0
        for f in frames():
            if idx >= len(scores):
                break
            if scores[idx] >= thr:
                crop, g = self._plan_crop(f)
                if ref is None:
                    ref = g.copy(); hann = cv2.createHanningWindow((g.shape[1], g.shape[0]),
                                                                   cv2.CV_32F)
                    mean = crop.copy(); count = 1
                elif g.shape == ref.shape:
                    try:
                        (dx, dy), _ = cv2.phaseCorrelate(ref * hann, g * hann)
                    except Exception:       # noqa: BLE001
                        dx = dy = 0.0
                    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                    aligned = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                    count += 1; mean += (aligned - mean) / count
            idx += 1
            if idx % 15 == 0:
                self._plan_readout_lbl.setText("Stacking best frames… {}/{}".format(count, n or idx))
                QtWidgets.QApplication.processEvents()
        if mean is None:
            self._plan_readout_lbl.setText("Nothing stacked."); return
        self._plan_mean = mean; self._plan_count = count; self._plan_total = len(scores)
        out8 = self._plan_finalize()
        self._plan_readout_lbl.setText(
            "Done: stacked {} best of {} frames (sub-pixel + sharpened). Save to keep it.".format(
                count, len(scores)))
        return out8

    def _planetary_process(self, frame):
        """Lucky imaging for the Moon/Sun (extended objects): score sharpness, keep the best
        X%, align each kept frame to a reference with SUB-PIXEL phase correlation, and stack
        (running mean). Sharpening is applied at the end (Stop/Finalize)."""
        H, W = frame.shape[:2]
        r = int(self._plan_roi)
        if r <= 0 or r >= min(H, W):                   # full frame (Moon/Sun fill the field)
            crop = frame.astype(np.float32)
        else:                                          # small object: crop around its centroid
            gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            thr = max(30, int(gray0.max() * 0.4))
            ys, xs = np.where(gray0 >= thr)
            cx, cy = (int(xs.mean()), int(ys.mean())) if len(xs) > 20 else (W // 2, H // 2)
            x0 = int(np.clip(cx - r // 2, 0, W - r)); y0 = int(np.clip(cy - r // 2, 0, H - r))
            crop = frame[y0:y0 + r, x0:x0 + r].astype(np.float32)
        gcrop = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) \
            if crop.ndim == 3 else crop.astype(np.float32)
        score = float(cv2.Laplacian(gcrop, cv2.CV_32F).var())
        self._plan_total += 1
        self._plan_scores.append(score)
        keep = self._plan_keep_pct / 100.0
        thr_score = (np.percentile(self._plan_scores, (1.0 - keep) * 100.0)
                     if len(self._plan_scores) >= 8 else -1.0)
        if score >= thr_score:
            if self._plan_ref is None or self._plan_ref.shape != gcrop.shape:
                self._plan_ref = gcrop.copy()          # reference for sub-pixel alignment
                self._plan_hann = cv2.createHanningWindow((gcrop.shape[1], gcrop.shape[0]),
                                                          cv2.CV_32F)
                self._plan_mean = crop.copy(); self._plan_count = 1
            else:
                try:
                    (dx, dy), _resp = cv2.phaseCorrelate(self._plan_ref * self._plan_hann,
                                                         gcrop * self._plan_hann)
                except Exception:       # noqa: BLE001
                    dx = dy = 0.0
                M = np.float32([[1, 0, -dx], [0, 1, -dy]])   # sub-pixel shift to the reference
                aligned = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                self._plan_count += 1
                self._plan_mean += (aligned - self._plan_mean) / self._plan_count
            kept = True
        else:
            kept = False
        base = self._plan_mean if self._plan_mean is not None else crop
        prev = self._plan_sharpen(base, self._plan_sharpen_amt * 0.5)   # light live preview
        self._plan_show(self._plan_stretch(prev))
        if getattr(self, "_plan_readout_lbl", None) is not None:
            try:
                self._plan_readout_lbl.setText(
                    "kept {} / {}  ({:.0f}%)   ·   sharpness {:.0f}   ·   {}".format(
                        self._plan_count, self._plan_total,
                        100.0 * self._plan_count / max(self._plan_total, 1),
                        score, "✓ kept" if kept else "✗ dropped"))
            except RuntimeError:
                self._plan_readout_lbl = None

    def _plan_show(self, bgr8):
        h, w = bgr8.shape[:2]
        big = 640
        scale = big / max(h, w)
        show = cv2.resize(bgr8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        hh, ww = show.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
        qimg = QtGui.QImage(rgb.data, ww, hh, 3 * ww, QtGui.QImage.Format_RGB888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))

    def _plan_sharpen(self, img, amount):
        """Multi-scale unsharp mask (wavelet-like) — approximates RegiStax wavelets so the
        module is self-contained. amount 0 = none."""
        if amount <= 0:
            return img.astype(np.float32)
        f = img.astype(np.float32)
        out = f.copy()
        for sigma in (1.0, 2.0, 4.0, 8.0):
            out += amount * (f - cv2.GaussianBlur(f, (0, 0), sigma))
        return out

    def _plan_stretch(self, img):
        f = img.astype(np.float32)
        lo = float(np.percentile(f, 2)); hi = float(np.percentile(f, 99.9))
        if hi <= lo:
            hi = lo + 1.0
        return (np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    def _plan_reset(self):
        self._plan_mean = None; self._plan_count = 0; self._plan_total = 0
        self._plan_ref = None; self._plan_scores.clear()

    def _plan_finalize(self):
        """Produce the final sharpened image from the stack and show it."""
        if self._plan_mean is None:
            return None
        final = self._plan_sharpen(self._plan_mean, self._plan_sharpen_amt)
        out8 = self._plan_stretch(final)
        self._plan_show(out8)
        return out8

    def _open_planetary_dialog(self):
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("🌙 Moon / ☀ Sun — lucky imaging")
        dlg.resize(440, 360); v = QtWidgets.QVBoxLayout(dlg)
        wrow = QtWidgets.QHBoxLayout()
        warn = QtWidgets.QLabel("☀ Sun: full-aperture solar filter only")
        warn.setStyleSheet("color:#ff7a29; font-weight:600;")
        wrow.addWidget(warn); wrow.addStretch(1)
        wrow.addWidget(_HelpButton(
            "<b>Solar safety:</b> never point at the Sun without a full-aperture solar "
            "filter — it destroys the sensor and your eyes. Cap the finder and polar scope.<br><br>"
            "Live-view lucky imaging keeps the sharpest frames, aligns them sub-pixel "
            "(phase correlation) and stacks them. Stop applies multi-scale sharpening for a "
            "finished image — no RegiStax needed."))
        v.addLayout(wrow)
        krow = QtWidgets.QHBoxLayout()
        krow.addWidget(QtWidgets.QLabel("Keep best"))
        keep = QtWidgets.QSpinBox(); keep.setRange(5, 90); keep.setValue(self._plan_keep_pct)
        keep.setSuffix(" %"); keep.valueChanged.connect(
            lambda x: setattr(self, "_plan_keep_pct", x))
        krow.addWidget(keep)
        krow.addWidget(QtWidgets.QLabel("ROI"))
        roi = QtWidgets.QSpinBox(); roi.setRange(0, 2000); roi.setSingleStep(20)
        roi.setValue(self._plan_roi); roi.setSuffix(" px (0=full)")
        roi.valueChanged.connect(lambda x: setattr(self, "_plan_roi", x))
        krow.addWidget(roi); v.addLayout(krow)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Sharpening"))
        sharp = QtWidgets.QSlider(Qt.Horizontal); sharp.setRange(0, 200)
        sharp.setValue(int(self._plan_sharpen_amt * 100))
        sharp.valueChanged.connect(lambda x: (setattr(self, "_plan_sharpen_amt", x / 100.0),
                                              self._plan_finalize() if not self._plan_active
                                              and self._plan_mean is not None else None))
        srow.addWidget(sharp); v.addLayout(srow)
        self._plan_readout_lbl = QtWidgets.QLabel("Not started.")
        self._plan_readout_lbl.setStyleSheet("color:#94a3b8;"); v.addWidget(self._plan_readout_lbl)
        brow = QtWidgets.QHBoxLayout()
        startb = QtWidgets.QPushButton("▶ Start"); stopb = QtWidgets.QPushButton("⏹ Stop && sharpen")
        resetb = QtWidgets.QPushButton("↺ Reset"); saveb = QtWidgets.QPushButton("💾 Save")
        stopb.setEnabled(False)

        def _start():
            if not self.chk_live.isChecked():
                self.chk_live.setChecked(True)
            self._plan_reset(); self._plan_active = True
            startb.setEnabled(False); stopb.setEnabled(True)

        def _stop():
            self._plan_active = False
            startb.setEnabled(True); stopb.setEnabled(False)
            self._plan_finalize()
            if self._plan_readout_lbl is not None and self._plan_count:
                self._plan_readout_lbl.setText(
                    "Final: stacked {} best frames, sub-pixel aligned + sharpened.".format(
                        self._plan_count))

        def _save():
            out8 = self._plan_finalize()
            if out8 is None:
                self._plan_readout_lbl.setText("Nothing to save yet."); return
            out = os.path.join(self._effective_dir() or os.path.expanduser("~"),
                               "lucky_{}.png".format(int(time.time())))
            try:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                cv2.imwrite(out, out8)
                self._plan_readout_lbl.setText("Saved: " + out)
            except Exception as e:      # noqa: BLE001
                self._plan_readout_lbl.setText("Save failed: {}".format(e))

        startb.clicked.connect(_start); stopb.clicked.connect(_stop)
        resetb.clicked.connect(self._plan_reset); saveb.clicked.connect(_save)
        for b in (startb, stopb, resetb, saveb):
            brow.addWidget(b)
        v.addLayout(brow)
        vidbtn = QtWidgets.QPushButton("📁 Load a movie (1080p/4K from the camera) — best quality")
        vidbtn.setToolTip("Record a movie of the Moon/Sun in-camera, then load it here to "
                          "lucky-stack the full-resolution frames.")

        def _loadvid():
            fn, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg, "Choose a movie", os.path.expanduser("~"),
                "Movies (*.ser *.mp4 *.mov *.avi *.m4v *.mts)")
            if fn:
                self._plan_active = False
                self._process_video(fn)

        vidbtn.clicked.connect(_loadvid)
        v.addWidget(vidbtn)
        rrow = QtWidgets.QHBoxLayout()
        recdur = QtWidgets.QSpinBox(); recdur.setRange(3, 120); recdur.setValue(20)
        recdur.setSuffix(" s")
        recbtn = QtWidgets.QPushButton("🎥 Record REAL movie (camera) && auto-process")
        recbtn.setToolTip("Triggers the camera's own movie recording (full resolution, to "
                          "card), downloads it, switches the mount to SOLAR rate, and "
                          "lucky-stacks it automatically.")
        recbtn.clicked.connect(lambda: (
            self._plan_readout_lbl.setText("🎥 Recording on the camera…"),
            self.worker.post("record_movie", seconds=recdur.value(),
                             save_dir=self._effective_dir())))
        rrow.addWidget(recbtn); rrow.addWidget(recdur)
        v.addLayout(rrow)
        hrow = QtWidgets.QHBoxLayout()
        hdmibtn = QtWidgets.QPushButton("🎥 Record HDMI → SER (uncompressed) && auto-process")
        hdmibtn.setMinimumHeight(32)
        hdmibtn.setToolTip(
            "Records EVERY frame from the capture card, uncompressed, to a .ser file "
            "(AutoStakkert!, Siril, PIPP read it too), then lucky-stacks it here.\n"
            "Needs the HDMI live source (Display & Focus). Press again to stop early.\n"
            "Disk: 1080p60 ≈ 125 MB/s in mono, ≈ 375 MB/s in colour.")
        hmono = QtWidgets.QCheckBox("Mono (Moon/Sun, 3× smaller)"); hmono.setChecked(True)
        hdmibtn.clicked.connect(lambda: self._hdmi_record(recdur.value(), hmono.isChecked()))
        hrow.addWidget(hdmibtn, 1); hrow.addWidget(hmono)
        v.addLayout(hrow)
        lvbtn = QtWidgets.QPushButton("🎥 Record in NOUT (live view) && auto-process — reliable")
        lvbtn.setToolTip("Records the live-view stream inside NOUT (lower resolution than the "
                         "camera movie, but works on every camera, fully automatic).")
        lvbtn.clicked.connect(lambda: self._plan_record_start(recdur.value()))
        v.addWidget(lvbtn)
        brow2 = QtWidgets.QHBoxLayout()
        nburst = QtWidgets.QSpinBox(); nburst.setRange(10, 500); nburst.setValue(60)
        nburst.setSuffix(" frames")
        shootb = QtWidgets.QPushButton("🌙 Shoot a burst NOW && lucky-stack (full resolution)")
        shootb.setMinimumHeight(34)
        shootb.setToolTip(
            "The sharpest path on this camera. Live view is fixed at 1024×680 and a "
            "1080p movie downsamples the sensor from 6000 to 1920 px — at 357 mm that "
            "is 173 pixels across the Moon against 541 for a full frame.\n"
            "Set a short exposure first (1/125 s or so on the Moon), then press this: "
            "NOUT shoots the burst into its own folder and stacks it when done.")
        shootb.clicked.connect(lambda: self._plan_shoot_burst(nburst.value()))
        brow2.addWidget(shootb, 1); brow2.addWidget(nburst)
        v.addLayout(brow2)
        photobtn = QtWidgets.QPushButton("📁 Lucky-stack a FOLDER of photos shot earlier")
        photobtn.setToolTip("Point at the Moon/Sun, shoot a burst of short-exposure stills with "
                            "the intervalometer, then pick that folder here to lucky-stack the "
                            "full 24 MP frames (far sharper than any video).")

        def _loadfolder():
            d = QtWidgets.QFileDialog.getExistingDirectory(
                dlg, "Folder of Moon/Sun photos", self._effective_dir() or os.path.expanduser("~"))
            if d:
                self._plan_active = False
                self._process_image_folder(d)

        photobtn.clicked.connect(_loadfolder)
        v.addWidget(photobtn)
        dlg.finished.connect(lambda *_: (setattr(self, "_plan_active", False),
                                         setattr(self, "_plan_readout_lbl", None)))
        dlg.show(); self._plan_dlg = dlg

    def _flats_dir(self):
        d = QtCore.QSettings("NOUT", "NOUT").value(
            "flats_dir", os.path.expanduser("~/NOUT_flats"), type=str)
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
        return d

    def _push_flats(self, on):
        QtCore.QSettings("NOUT", "NOUT").setValue("apply_flats", bool(on))
        self.worker.post("set_flats", on=bool(on), flats_dir=self._flats_dir())

    def _start_caffeinate(self):
        """Keep the machine awake while NOUT runs.

        macOS: 'caffeinate'. Linux: 'systemd-inhibit'. Windows: the thread execution
        state flags, which is how an application says 'I am busy' there."""
        if os.name == "nt":
            try:
                import ctypes
                # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
                self._caffeinate = "win"
            except Exception:           # noqa: BLE001
                self._caffeinate = None
            return
        if sys.platform.startswith("linux"):
            if self._caffeinate is not None and getattr(self._caffeinate, "poll", lambda: 0)() is None:
                return
            try:
                self._caffeinate = subprocess.Popen(
                    ["systemd-inhibit", "--what=idle:sleep:handle-lid-switch",
                     "--why=NOUT is capturing", "sleep", "infinity"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:           # noqa: BLE001
                self._caffeinate = None
            return
        if sys.platform != "darwin":
            return
        if self._caffeinate is not None and self._caffeinate.poll() is None:
            return                                          # already running
        try:
            import subprocess
            # -d display, -i idle, -m disk, -s system, -u user-active (blocks the lock)
            self._caffeinate = subprocess.Popen(
                ["caffeinate", "-dimsu"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:               # noqa: BLE001
            self._caffeinate = None

    def _stop_caffeinate(self):
        if self._caffeinate == "win":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)   # ES_CONTINUOUS
            except Exception:           # noqa: BLE001
                pass
            self._caffeinate = None
            return
        if self._caffeinate is not None:
            try:
                self._caffeinate.terminate()
            except Exception:           # noqa: BLE001
                pass
            self._caffeinate = None

    def _apply_awake_mode(self, *args):
        """Start/stop 'caffeinate' according to the chosen mode."""
        mode = self.awake_mode.currentData()
        QtCore.QSettings("SonyTether", "SonyTether").setValue("keep_awake_mode", mode)
        if mode == "always":
            self._start_caffeinate()
            msg = "Keeping the Mac awake (no sleep/lock) while NOUT is open."
        elif mode == "off":
            self._stop_caffeinate()
            msg = "Mac may sleep/lock normally."
        else:                                               # during capture
            if self._awake_capture_active():
                self._start_caffeinate()
            else:
                self._stop_caffeinate()
            msg = "Mac kept awake while capturing (auto-start armed counts too)."
        if sys.platform == "darwin" and mode != "off" and self._caffeinate is None \
                and (mode == "always" or self._awake_capture_active()):
            msg = "Could not start 'caffeinate' — the Mac may sleep."
        self.statusBar().showMessage(msg)

    def _awake_capture_active(self):
        """True while we must keep the Mac awake in 'during capture' mode: either a burst is
        running, or auto-start is armed and still waiting for its trigger."""
        if getattr(self, "_iv_running", False):
            return True
        if (hasattr(self, "chk_autostart") and self.chk_autostart.isChecked()
                and not getattr(self, "_auto_started", False)):
            return True
        return False

    def _update_awake(self):
        """Re-evaluate caffeinate for the current mode/state (called on capture + automation
        state changes)."""
        if not hasattr(self, "awake_mode"):
            return
        if self.awake_mode.currentData() == "capture":
            if self._awake_capture_active():
                self._start_caffeinate()
            else:
                self._stop_caffeinate()

    def _push_darks(self, on):
        QtCore.QSettings("NOUT", "NOUT").setValue("apply_darks", bool(on))
        self.worker.post("set_darks", on=bool(on), flats_dir=self._flats_dir())

    def _ac_set_buttons(self, running):
        for name, en in (("autocenter_btn", not running), ("autocenter_stop_btn", running)):
            b = getattr(self, name, None)
            if b is not None:
                try:
                    b.setEnabled(en)
                except RuntimeError:            # dialog was closed
                    setattr(self, name, None)

    def _start_autocenter(self, ip=None):
        """Automated centering loop: capture → plate-solve → slew RA toward goal → repeat
        until RA converges (or max iterations / user Stop). Dec residual is reported."""
        if getattr(self, "_ac_active", False):
            return
        goal = getattr(self, "_session_goal", None)
        if not goal or goal[1] is None or goal[2] is None:
            self._set_goal_slew_msg("Set a goal first (Set goal here, or pick a target).")
            return
        if getattr(self, "_iv_running", False):
            self._set_goal_slew_msg("Stop the current capture before auto-centering.")
            return
        self._ac_ip = ip or self._mip()
        self._ac_two_axis = bool(QtCore.QSettings("NOUT", "NOUT").value("mount_type", 0, type=int))
        self._ac_active = True; self._ac_iter = 0
        self._ac_set_buttons(True)
        self._set_goal_slew_msg("🎯 Auto-center started — capturing…")
        self._ac_capture()

    def _stop_autocenter(self):
        if getattr(self, "_ac_active", False):
            self._ac_active = False; self._ac_phase = None
            self._ac_set_buttons(False)
            self._set_goal_slew_msg("⏹ Auto-center stopped.")

    def _ac_capture(self):
        if not getattr(self, "_ac_active", False):
            return
        self._ac_phase = "capture"
        self._set_goal_slew_msg("🎯 Auto-center iter {}: 5 s capture…".format(self._ac_iter + 1))
        # fixed 5 s exposure for centering — independent of the Shooting-tab exposure
        self.worker.post("set_exposure_mode", bulb=True, seconds=5.0)
        self.worker.post("start_interval", interval=0, count=1, bulb=True,
                         seconds=5.0, save_dir=self._effective_dir())

    def _ac_on_solved(self, ra, dec):
        goal = getattr(self, "_session_goal", None)
        if not self._ac_active or not goal:
            return
        ra_goal, dec_goal = float(goal[1]), float(goal[2])
        dra = ((ra_goal - ra + 180.0) % 360.0) - 180.0
        ddec = dec_goal - dec
        two = getattr(self, "_ac_two_axis", False)
        self._ac_iter += 1
        ra_ok = abs(dra) <= self._ac_tol_deg
        dec_ok = (not two) or abs(ddec) <= self._ac_tol_deg
        if ra_ok and dec_ok:
            if two:
                self._ac_finish("✅ Centered on goal (RA+Dec within {:.0f}′) — tracking."
                                .format(self._ac_tol_deg * 60))
            else:
                self._ac_finish("✅ Centered — RA within {:.0f}′. Dec off {:+.2f}° "
                                "(adjust the mount head).".format(self._ac_tol_deg * 60, ddec))
            return
        if self._ac_iter >= self._ac_max:
            self._ac_finish("Stopped after {} tries — RA {:+.2f}°, Dec {:+.2f}°."
                            .format(self._ac_max, dra, ddec))
            return
        self._set_goal_slew_msg("🎯 Iter {}: RA {:+.2f}°{} → slewing…".format(
            self._ac_iter, dra, (", Dec {:+.2f}°".format(ddec)) if two else ""))
        self._ac_phase = "slew"
        kw = dict(ip=self._ac_ip, arcsec=dra * 3600.0)
        if two:
            kw["dec_arcsec"] = ddec * 3600.0            # HEQ-5 Pro: correct Dec too
        self.worker.post("mount_goto_ra", **kw)

    def _ac_after_slew(self, moved):
        if not getattr(self, "_ac_active", False) or self._ac_phase != "slew":
            return
        QtCore.QTimer.singleShot(2500, self._ac_capture)  # brief settle, then next frame

    def _ac_finish(self, msg):
        self._ac_active = False; self._ac_phase = None
        self._ac_set_buttons(False)
        self._set_goal_slew_msg(msg)

    def _mip(self):
        return QtCore.QSettings("NOUT", "NOUT").value("mount_ip", "192.168.4.1", type=str)

    def _slew_ra_to_goal(self, ip=None):
        """Rotate the RA axis to bring the goal to centre, using the last plate-solve as the
        current pointing. RA-only: reports the residual Dec (the mount can't correct it)."""
        ip = ip or self._mip()
        if not self._solve_wcs:
            self._set_goal_slew_msg("Plate-solve a photo first (identify the field), then retry.")
            return
        goal = getattr(self, "_session_goal", None)
        if not goal or goal[1] is None or goal[2] is None:
            self._set_goal_slew_msg("Set a goal first (Set goal here, or pick a target).")
            return
        ra_now, dec_now = float(self._solve_wcs[0]), float(self._solve_wcs[1])
        ra_goal, dec_goal = float(goal[1]), float(goal[2])
        dra = ((ra_goal - ra_now + 180.0) % 360.0) - 180.0     # shortest RA delta (deg)
        ddec = dec_goal - dec_now
        arcsec = dra * 3600.0
        self.worker.post("mount_goto_ra", ip=ip, arcsec=arcsec)
        self._set_goal_slew_msg(
            "Rotating RA by {:+.2f}°  ({:+.0f}\u2033).  Dec off {:+.2f}° — nudge the mount head "
            "in Dec (RA-only mount can't). Re-solve to refine.".format(dra, arcsec, ddec))

    # --- automatic centring: capture -> solve -> slew -> repeat -----------------
    def _autocenter_start(self):
        goal = getattr(self, "_session_goal", None)
        if not goal or goal[1] is None or goal[2] is None:
            self._set_goal_slew_msg("Set a goal first, then start auto-centre."); return
        if getattr(self, "_iv_running", False):
            self._set_goal_slew_msg("Stop the running capture before auto-centring."); return
        self._autocenter = {"active": True, "iter": 0, "phase": "capture"}
        if getattr(self, "autocenter_stop_btn", None) is not None:
            self.autocenter_stop_btn.setEnabled(True)
        if getattr(self, "autocenter_btn", None) is not None:
            self.autocenter_btn.setEnabled(False)
        self._set_goal_slew_msg("🎯 Auto-centre: taking a photo…")
        self._autocenter_capture()

    def _autocenter_capture(self):
        if not getattr(self, "_autocenter", {}).get("active"):
            return
        self._autocenter["phase"] = "capture"
        self._single_capture()          # 1 shot; its image+path arrive via on_capture_image

    def _autocenter_after_capture(self):
        ac = getattr(self, "_autocenter", {})
        if not ac.get("active") or ac.get("phase") != "capture":
            return
        if not (getattr(self, "_last_capture_path", None)
                and os.path.exists(self._last_capture_path)):
            return
        ac["phase"] = "solve"
        self._solve_wcs = None          # so a failed solve is detected (stays None)
        self._set_goal_slew_msg("🎯 Auto-centre: plate-solving…")
        self._run_solve()

    def _autocenter_after_solve(self):
        ac = getattr(self, "_autocenter", {})
        if not ac.get("active") or ac.get("phase") != "solve":
            return
        if not self._solve_wcs:
            self._autocenter_finish("Solve failed — stopping auto-centre. Try a longer exposure.")
            return
        goal = self._session_goal
        ra_now, dec_now = float(self._solve_wcs[0]), float(self._solve_wcs[1])
        dra = ((float(goal[1]) - ra_now + 180.0) % 360.0) - 180.0
        ddec = float(goal[2]) - dec_now
        ac["iter"] += 1
        if abs(dra) <= 0.05:            # centred in RA (~3′)
            self._autocenter_finish(
                "✅ Centred in RA. Dec off {:+.2f}° — nudge the mount head if needed.".format(ddec))
            return
        if ac["iter"] >= 6:
            self._autocenter_finish(
                "Stopped after 6 tries (RA still {:+.2f}°). Check tracking / polar align.".format(dra))
            return
        self.worker.post("mount_goto_ra", ip=self._mip(), arcsec=dra * 3600.0)
        self._set_goal_slew_msg("🎯 Auto-centre #{}: RA {:+.2f}° → slewing, settling…".format(
            ac["iter"], dra))
        ac["phase"] = "slew"
        QtCore.QTimer.singleShot(5000, self._autocenter_capture)   # settle then re-shoot

    def _autocenter_finish(self, msg):
        self._autocenter = {"active": False, "iter": 0, "phase": None}
        if getattr(self, "autocenter_stop_btn", None) is not None:
            self.autocenter_stop_btn.setEnabled(False)
        if getattr(self, "autocenter_btn", None) is not None:
            self.autocenter_btn.setEnabled(True)
        self._set_goal_slew_msg(msg)

    def _autocenter_stop(self):
        if getattr(self, "_autocenter", {}).get("active"):
            self._autocenter_finish("⏹ Auto-centre stopped.")

    def _set_goal_slew_msg(self, msg):
        if getattr(self, "_goal_slew_lbl", None) is not None:
            self._goal_slew_lbl.setText(msg)
        self.statusBar().showMessage(msg, 9000)

    def _push_dither(self, *args):
        if not hasattr(self, "chk_dither"):
            return
        self.worker.post("set_dither", on=self.chk_dither.isChecked(), ip=self._mip(),
                         every=self.dither_every.value(), amp=float(self.dither_amp.value()),
                         settle=float(self.dither_settle.value()))

    # ------------------------------------------------------------------
    #  Manual GoTo — dial readings for a mount with graduated circles
    # ------------------------------------------------------------------
    def _mg_engine(self):
        """The ManualGoto solver, created once and kept in sync with the site."""
        if getattr(self, "_manual_goto", None) is None:
            import manual_goto
            self._manual_goto = manual_goto.ManualGoto()
            try:
                raw = QtCore.QSettings("NOUT", "NOUT").value("manual_goto_cal", "", type=str)
                if raw:
                    self._manual_goto.load_dict(json.loads(raw))
            except Exception:           # noqa: BLE001  (corrupt/absent calibration)
                pass
        self._manual_goto.set_site(self.t_lat.value(), self.t_lon.value())
        return self._manual_goto

    def _mg_save_cal(self):
        try:
            QtCore.QSettings("NOUT", "NOUT").setValue(
                "manual_goto_cal", json.dumps(self._mg_engine().to_dict()))
        except Exception:               # noqa: BLE001
            pass

    def _mg_when_utc(self):
        """UTC instant the readings are computed for: the real now, or the Sky Map
        time when you are planning rather than shooting."""
        if getattr(self, "_mg_live", None) is not None and not self._mg_live.isChecked():
            return self._build_skymap_dt_utc()
        from datetime import datetime as _dt
        return _dt.utcnow()

    def _open_manual_goto(self, tab="point"):
        """Dial readings that put the session goal in the centre of the frame.

        The two readings are the whole point, so they stay pinned at the top; the
        setup that produces them — calibration, polar alignment, two-star solve —
        lives behind tabs, because it is done once a session and then never again."""
        try:
            import manual_goto                                # noqa: F401
        except Exception as e:          # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Manual GoTo",
                                          "manual_goto.py is missing: {}".format(e))
            return
        if getattr(self, "_mg_dlg", None) is not None:        # already open: raise it
            try:
                self._mg_show_tab(tab)
                self._mg_dlg.raise_(); self._mg_dlg.activateWindow(); return
            except RuntimeError:
                self._mg_dlg = None
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Mount — pointing, setup and control")
        dlg.setMinimumWidth(720)
        self._mg_dlg = dlg
        v = QtWidgets.QVBoxLayout(dlg)
        v.setSpacing(6)

        # ---- the answer, always visible -------------------------------------
        self._mg_target_lbl = QtWidgets.QLabel("—")
        self._mg_target_lbl.setStyleSheet("color:#f59e0b; font-weight:bold; font-size:16px;")
        self._mg_target_lbl.setWordWrap(True)
        v.addWidget(self._mg_target_lbl)
        self._mg_coord_lbl = QtWidgets.QLabel("")
        self._mg_coord_lbl.setStyleSheet("color:#94a3b8;")
        self._mg_coord_lbl.setWordWrap(True)
        v.addWidget(self._mg_coord_lbl)

        dials = QtWidgets.QHBoxLayout()
        self._mg_dial_lbl, self._mg_sub_lbl, self._mg_turn_lbl = {}, {}, {}
        for key, title, hint in (
                ("a1", "AXIS 1 — polar dial",
                 "The graduated circle behind the polar scope (hour angle). The motor "
                 "drives this one."),
                ("a2", "AXIS 2 — L-bracket",
                 "Your hand-graduated L-plate. Graduated 0 at the pole, the reading is "
                 "the angle between the camera and the polar axis. You turn this one.")):
            box = QtWidgets.QGroupBox(title)
            box.setToolTip(hint)
            bl = QtWidgets.QVBoxLayout(box); bl.setSpacing(2)
            big = QtWidgets.QLabel("—")
            big.setAlignment(Qt.AlignCenter)
            big.setStyleSheet("font-size:30px; font-weight:bold; color:#e2e8f0;")
            bl.addWidget(big)
            sub = QtWidgets.QLabel(""); sub.setAlignment(Qt.AlignCenter)
            sub.setStyleSheet("color:#94a3b8;")
            bl.addWidget(sub)
            turn = QtWidgets.QLabel(""); turn.setAlignment(Qt.AlignCenter)
            turn.setStyleSheet("color:#38bdf8; font-size:13px; font-weight:bold;")
            bl.addWidget(turn)
            self._mg_dial_lbl[key], self._mg_sub_lbl[key], self._mg_turn_lbl[key] = big, sub, turn
            dials.addWidget(box, 1)
        v.addLayout(dials)

        self._mg_info_lbl = QtWidgets.QLabel("")
        self._mg_info_lbl.setStyleSheet("color:#cbd5e1;"); self._mg_info_lbl.setWordWrap(True)
        v.addWidget(self._mg_info_lbl)
        self._mg_warn_lbl = QtWidgets.QLabel("")
        self._mg_warn_lbl.setStyleSheet("color:#fbbf24;"); self._mg_warn_lbl.setWordWrap(True)
        v.addWidget(self._mg_warn_lbl)
        self._mg_alt_lbl = QtWidgets.QLabel("")
        self._mg_alt_lbl.setStyleSheet("color:#94a3b8;"); self._mg_alt_lbl.setWordWrap(True)
        v.addWidget(self._mg_alt_lbl)

        tabs = QtWidgets.QTabWidget()
        v.addWidget(tabs, 1)

        def _page():
            """A scrollable tab page, so nothing is ever clipped off the bottom."""
            inner = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(inner)
            lay.setContentsMargins(4, 6, 4, 6); lay.setSpacing(7)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True); scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            # never scroll sideways: that is what lets the long labels wrap instead
            # of running off the edge and taking the buttons with them
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(inner)
            return scroll, lay

        # =====================================================================
        #  TAB 1 — Point: what you touch on every target
        # =====================================================================
        page1, p1 = _page()
        tabs.addTab(page1, "▶  Point")

        mo = QtWidgets.QGroupBox("Motorised axis 1 (Wi-Fi) — you keep the L-plate")
        mol = QtWidgets.QVBoxLayout(mo); mol.setSpacing(5)
        mol.addWidget(_hint(
            "<b>For every target:</b> click it on the Sky Map → <b>GoTo axis 1</b> → "
            "turn the L-plate to the angle shown above. That is the whole loop.<br><br>"
            "Axis 1 has to be synced once per session first — that lives in Set up, "
            "step 4, with the rest of the calibration."))

        mrow = QtWidgets.QHBoxLayout()
        self._mg_a1_goto_btn = QtWidgets.QPushButton("▶  GoTo axis 1")
        self._mg_a1_goto_btn.setToolTip(
            "Slews axis 1 onto the target and keeps correcting until it is there. "
            "This mount does about 0.30°/s, so allow ~3 s + 3.3 s per degree.")
        self._mg_a1_goto_btn.clicked.connect(self._mg_axis1_goto)
        self._mg_a1_stop_btn = QtWidgets.QPushButton("⏹  Stop")
        self._mg_a1_stop_btn.clicked.connect(lambda: self.worker.post("mount_axis1_stop"))
        self._mg_a1_tosetup_btn = QtWidgets.QPushButton("⚙  Sync it (Set up)")
        self._mg_a1_tosetup_btn.setToolTip("Jumps to the calibration step that ties the "
                                           "mount's encoder to the sky.")
        self._mg_a1_tosetup_btn.clicked.connect(lambda: tabs.setCurrentIndex(1))
        for b, wgt in ((self._mg_a1_goto_btn, 3), (self._mg_a1_stop_btn, 1),
                       (self._mg_a1_tosetup_btn, 2)):
            b.setMinimumHeight(34); mrow.addWidget(b, wgt)
        mol.addLayout(mrow)

        lim = QtWidgets.QHBoxLayout()
        lim.addWidget(QtWidgets.QLabel("Max travel from the sync point"))
        self._mg_a1_travel = QtWidgets.QDoubleSpinBox()
        self._mg_a1_travel.setRange(5.0, 350.0); self._mg_a1_travel.setValue(120.0)
        self._mg_a1_travel.setSuffix(" °")
        self._mg_a1_travel.setToolTip("NOUT refuses to take axis 1 further than this from "
                                      "where you synced it — that is what runs the camera "
                                      "into a tripod leg. Keep it tight at first.")
        lim.addWidget(self._mg_a1_travel)
        lim.addSpacing(12)
        lim.addWidget(QtWidgets.QLabel("Stop within"))
        self._mg_a1_tol = QtWidgets.QDoubleSpinBox()
        self._mg_a1_tol.setRange(0.02, 2.0); self._mg_a1_tol.setValue(0.15)
        self._mg_a1_tol.setSuffix(" °"); self._mg_a1_tol.setDecimals(2)
        lim.addWidget(self._mg_a1_tol)
        lim.addStretch(1)
        mol.addLayout(lim)

        self._mg_flip = QtWidgets.QCheckBox(
            "Put the camera on the other side of the pole (mirrored position)")
        self._mg_flip.setToolTip(
            "Same point of sky, L-plate on the other side, axis 1 exactly 12 h away. "
            "It does NOT extend how far south you can reach — both positions use the "
            "same |angle| — but it decides which side the camera physically sits on, "
            "which is what clears the tripod. Only you can see that.")
        self._mg_flip.toggled.connect(lambda *_: self._mg_refresh())
        mol.addWidget(self._mg_flip)
        self._mg_a1_bar = QtWidgets.QProgressBar()
        self._mg_a1_bar.setRange(0, 1000); self._mg_a1_bar.setValue(0)
        self._mg_a1_bar.setTextVisible(False); self._mg_a1_bar.setFixedHeight(14)
        self._mg_a1_bar.setVisible(False)
        mol.addWidget(self._mg_a1_bar)
        self._mg_a1_eta = QtWidgets.QLabel("")
        self._mg_a1_eta.setStyleSheet("color:#38bdf8; font-weight:bold;")
        self._mg_a1_eta.setVisible(False)
        mol.addWidget(self._mg_a1_eta)
        self._mg_a1_msg = QtWidgets.QLabel("Pick a target and press GoTo.")
        self._mg_a1_msg.setWordWrap(True); self._mg_a1_msg.setStyleSheet("color:#94a3b8;")
        mol.addWidget(self._mg_a1_msg)
        self._mg_a1_plate = QtWidgets.QLabel("")
        self._mg_a1_plate.setStyleSheet("color:#fbbf24; font-size:17px; font-weight:bold;")
        self._mg_a1_plate.setWordWrap(True)
        mol.addWidget(self._mg_a1_plate)
        p1.addWidget(mo)

        rg = QtWidgets.QGroupBox("Tracking rate")
        rgl = QtWidgets.QHBoxLayout(rg)
        self._mg_rate = self._make_rate_combo()
        rgl.addWidget(self._mg_rate, 1)
        rb = QtWidgets.QPushButton("Apply rate"); rb.clicked.connect(self._mg_apply_rate)
        rgl.addWidget(rb)
        p1.addWidget(rg)
        self._mg_track_lbl = QtWidgets.QLabel("⚪  Tracking: measuring…")
        self._mg_track_lbl.setWordWrap(True)
        self._mg_track_lbl.setStyleSheet("font-size:14px; font-weight:bold;")
        rgl2 = QtWidgets.QVBoxLayout()
        rgl2.addWidget(self._mg_track_lbl)
        self._mg_track_sub = QtWidgets.QLabel("")
        self._mg_track_sub.setWordWrap(True); self._mg_track_sub.setStyleSheet("color:#94a3b8;")
        rgl2.addWidget(self._mg_track_sub)
        rg.layout().addLayout(rgl2) if False else None
        p1.addLayout(rgl2)
        self._mg_rate_lbl = QtWidgets.QLabel("")
        self._mg_rate_lbl.setStyleSheet("color:#94a3b8;"); self._mg_rate_lbl.setWordWrap(True)
        p1.addWidget(self._mg_rate_lbl)
        self._mg_rate_tick = 0
        # the probe runs from the main window; this panel only displays it

        p1.addWidget(_hint(
            "No Wi-Fi? Set both dials by hand instead: stop tracking, set the L-plate "
            "first (it never drifts), then axis 1 last — its reading above keeps running "
            "at 0.25°/min — then start tracking again at once."))
        p1.addStretch(1)

        # =====================================================================
        #  TAB 2 — Set up: once a session
        # =====================================================================
        page2, p2 = _page()
        tabs.addTab(page2, "⚙  Set up")
        self._mg_tabs = tabs
        eng = self._mg_engine()

        pa = QtWidgets.QGroupBox("1 · Polar alignment — where Polaris sits right now")
        pal = QtWidgets.QVBoxLayout(pa)
        prow = QtWidgets.QHBoxLayout()
        self._mg_reticle = PolarReticle()
        prow.addWidget(self._mg_reticle)
        pside = QtWidgets.QVBoxLayout()
        self._mg_polaris_lbl = QtWidgets.QLabel("")
        self._mg_polaris_lbl.setWordWrap(True); self._mg_polaris_lbl.setStyleSheet("color:#cbd5e1;")
        pside.addWidget(self._mg_polaris_lbl)
        pside.addWidget(_hint(
            "<b>The reticle turns with the RA axis</b>, so it is rarely upright. Tell "
            "NOUT how far it has turned and the drawing matches your eyepiece: the "
            "engraved scale rotates, Polaris stays where the sky puts it.<br><br>"
            "If your polar scope shows the view turned round or flipped, set the "
            "options below once so the picture matches the eyepiece."))
        rrow = QtWidgets.QHBoxLayout()
        rrow.addWidget(QtWidgets.QLabel("RA turned"))
        self._mg_ret_h = QtWidgets.QSpinBox(); self._mg_ret_h.setRange(-23, 23)
        self._mg_ret_h.setSuffix(" h"); self._mg_ret_h.setWrapping(True)
        self._mg_ret_m = QtWidgets.QSpinBox(); self._mg_ret_m.setRange(0, 59)
        self._mg_ret_m.setSuffix(" m"); self._mg_ret_m.setWrapping(True)
        for sp in (self._mg_ret_h, self._mg_ret_m):
            sp.valueChanged.connect(self._mg_ret_manual)
            rrow.addWidget(sp)
        rrow.addStretch(1)
        pside.addLayout(rrow)
        trow2 = QtWidgets.QHBoxLayout()
        self._mg_ret_track = QtWidgets.QPushButton("▶ Start tracking")
        self._mg_ret_track.setCheckable(True)
        self._mg_ret_track.setToolTip(
            "From now on NOUT adds the sidereal rotation itself. Polaris should then "
            "stay put on the engraved scale — if it drifts, the polar alignment is off, "
            "and that drift is the measurement.")
        self._mg_ret_track.toggled.connect(self._mg_ret_start)
        rst = QtWidgets.QPushButton("↺ Reset")
        rst.setToolTip("Back to zero: the reticle is upright, 12 at the top.")
        rst.clicked.connect(self._mg_ret_reset)
        trow2.addWidget(self._mg_ret_track); trow2.addWidget(rst)
        pside.addLayout(trow2)
        self._mg_ret_msg = QtWidgets.QLabel("")
        self._mg_ret_msg.setWordWrap(True); self._mg_ret_msg.setStyleSheet("color:#94a3b8;")
        pside.addWidget(self._mg_ret_msg)
        self._mg_ret_t0 = None
        self._mg_ret_flip = QtWidgets.QCheckBox("My reticle is rotated 180°")
        self._mg_ret_mirror = QtWidgets.QCheckBox("My reticle is mirrored")
        self._mg_ret_ccw = QtWidgets.QCheckBox("Its numbers run anticlockwise")
        self._mg_ret_ccw.setToolTip(
            "Look through and follow 12 → 1 → 2 → 3. Clockwise, like a watch: leave "
            "this off. Anticlockwise: tick it. Five seconds of looking, and the "
            "reading below is then right for your glass — I cannot tell from here how "
            "yours is engraved.")
        for cbx in (self._mg_ret_flip, self._mg_ret_mirror, self._mg_ret_ccw):
            cbx.toggled.connect(lambda *_: self._mg_refresh())
            pside.addWidget(cbx)
        pside.addStretch(1)
        prow.addLayout(pside, 1)
        pal.addLayout(prow)
        pcheck = QtWidgets.QPushButton("✔  Check the polar axis from the last plate-solve")
        pcheck.setMinimumHeight(30)
        pcheck.setToolTip("Put the L-plate on 0 (camera parallel to the polar axis), take "
                          "a photo, plate-solve it, then click here. The solved centre IS "
                          "where your polar axis points — the one measurement that owes "
                          "nothing to any dial or reticle convention.")
        pcheck.clicked.connect(self._mg_check_polar)
        pal.addWidget(pcheck)
        self._mg_polar_msg = QtWidgets.QLabel("")
        self._mg_polar_msg.setWordWrap(True); self._mg_polar_msg.setStyleSheet("color:#94a3b8;")
        pal.addWidget(self._mg_polar_msg)
        p2.addWidget(pa)

        lp = QtWidgets.QGroupBox("2 · L-plate zero")
        lpl = QtWidgets.QVBoxLayout(lp)
        poleb = QtWidgets.QPushButton("⊙  My L-plate reads 0 at the pole")
        poleb.setMinimumHeight(32)
        poleb.setToolTip("Your setup: camera parallel to the polar axis, plate on 0. The "
                         "reading is then the angle from the celestial pole (90° − "
                         "declination). One click, no star needed.")
        poleb.clicked.connect(self._mg_pole_zero)
        lpl.addWidget(poleb)
        grd = QtWidgets.QHBoxLayout()
        grd.addWidget(QtWidgets.QLabel("Graduation"))
        self._mg_wrap = QtWidgets.QComboBox()
        self._mg_wrap.addItem("± either side of the pole", False)
        self._mg_wrap.addItem("0–360° continuous", True)
        self._mg_wrap.setCurrentIndex(1 if eng.axis2.wrap else 0)
        self._mg_wrap.setToolTip(
            "How your plate is engraved. On a 0–360 scale the mirrored position reads "
            "311.4° where a ± scale would read −48.6° — the same place, no minus signs "
            "to misread in the dark.")
        grd.addWidget(self._mg_wrap, 1)
        lpl.addLayout(grd)
        rng = QtWidgets.QHBoxLayout()
        rng.addWidget(QtWidgets.QLabel("The plate can swing up to"))
        self._mg_travel2 = QtWidgets.QDoubleSpinBox()
        self._mg_travel2.setRange(10.0, 180.0); self._mg_travel2.setValue(eng.axis2_travel)
        self._mg_travel2.setSuffix(" °")
        self._mg_travel2.setToolTip("Measured from the pole position, either way. Beyond "
                                    "it NOUT offers the mirrored position instead — and "
                                    "warns when neither side can reach.")
        rng.addWidget(self._mg_travel2)
        rng.addWidget(QtWidgets.QLabel("from the pole, either way"))
        rng.addStretch(1)
        lpl.addLayout(rng)
        p2.addWidget(lp)

        cal = QtWidgets.QGroupBox("3 · Polar-dial zero — sync on a star")
        cg = QtWidgets.QVBoxLayout(cal)
        cg.addWidget(_hint(
            "Centre a bright star far from the pole, click it on the Sky Map, type what "
            "the dials read, then Sync — within 30 s, the polar dial keeps running. "
            "Keep the L-plate on its positive side, the one NOUT shows."))
        srow = QtWidgets.QHBoxLayout()
        self._mg_lbl_r1 = QtWidgets.QLabel("")
        srow.addWidget(self._mg_lbl_r1)
        self._mg_r1 = DialSpin(); self._mg_r1.setMinimumWidth(110)
        srow.addWidget(self._mg_r1)
        self._mg_lbl_r2 = QtWidgets.QLabel("")
        srow.addWidget(self._mg_lbl_r2)
        self._mg_r2 = DialSpin(); self._mg_r2.setMinimumWidth(110)
        self._mg_r2.setToolTip("Not used to calibrate anything — your plate's zero is "
                               "already exact from the pole. NOUT only reports how far "
                               "this star says you are from it, as a health check.")
        srow.addWidget(self._mg_r2)
        srow.addWidget(QtWidgets.QLabel("<i>(check only)</i>"))
        srow.addStretch(1)
        cg.addLayout(srow)
        srow3 = QtWidgets.QHBoxLayout()
        syncb = QtWidgets.QPushButton("⊕  Sync the polar dial on this star")
        syncb.setMinimumHeight(32)
        syncb.clicked.connect(self._mg_sync)
        markb = QtWidgets.QPushButton("⊙  Mark dials as current"); markb.setMinimumHeight(32)
        markb.setToolTip("Remember where the dials are, so the next target is shown as "
                         "'still to turn' instead of an absolute reading.")
        markb.clicked.connect(self._mg_mark)
        srow3.addWidget(syncb, 1); srow3.addWidget(markb, 1)
        cg.addLayout(srow3)
        self._mg_cal_msg = QtWidgets.QLabel("")
        self._mg_cal_msg.setStyleSheet("color:#94a3b8;"); self._mg_cal_msg.setWordWrap(True)
        cg.addWidget(self._mg_cal_msg)
        p2.addWidget(cal)

        mot = QtWidgets.QGroupBox("4 · Motorised sync — tie the encoder to the sky")
        mtl = QtWidgets.QVBoxLayout(mot)
        mtl.addWidget(_hint(
            "Needed only if you drive axis 1 over Wi-Fi. Centre the same bright star, "
            "click it on the Sky Map, then sync — nothing to type, NOUT reads the "
            "mount's step counter. Keep the L-plate on its positive side. The first "
            "sync of a session takes ~20 s and nudges the axis 1° and back while it "
            "works out the directions; the reference is taken before that, so your "
            "centring still counts."))
        self._mg_a1_sync_btn = QtWidgets.QPushButton("⌖  Sync axis 1 here")
        self._mg_a1_sync_btn.setMinimumHeight(32)
        self._mg_a1_sync_btn.setToolTip(
            "The very first sync of a session nudges the axis 1° and brings it back, "
            "to find out which way the direction bit really drives. That is normal, "
            "and the reference is taken before it — your centring is not lost.")
        self._mg_a1_sync_btn.clicked.connect(self._mg_axis1_sync)
        mtl.addWidget(self._mg_a1_sync_btn)
        self._mg_a1_sync_msg = QtWidgets.QLabel(
            "Not synced — axis 1 does not know where it is.")
        self._mg_a1_sync_msg.setWordWrap(True)
        self._mg_a1_sync_msg.setStyleSheet("color:#94a3b8;")
        mtl.addWidget(self._mg_a1_sync_msg)
        p2.addWidget(mot)

        adv = QtWidgets.QGroupBox("Dial conventions — already set for this mount")
        ag = QtWidgets.QGridLayout(adv)

        def _axis_row(row, cal_obj, name):
            ag.addWidget(QtWidgets.QLabel(name), row, 0)
            units = QtWidgets.QComboBox(); units.addItems(["hours", "degrees"])
            units.setCurrentText(cal_obj.units)
            ag.addWidget(units, row, 1)
            ag.addWidget(QtWidgets.QLabel("zero"), row, 2)
            zero = QtWidgets.QDoubleSpinBox()
            zero.setRange(-360.0, 360.0); zero.setDecimals(3); zero.setValue(cal_obj.zero)
            ag.addWidget(zero, row, 3)
            direc = QtWidgets.QComboBox()
            direc.addItem("normal (+)", 1.0); direc.addItem("reversed (−)", -1.0)
            direc.setCurrentIndex(0 if cal_obj.sign >= 0 else 1)
            direc.setToolTip("Reverse it if the target ends up twice as far off as it "
                             "started — the graduations run the other way.")
            ag.addWidget(direc, row, 4)
            return units, zero, direc

        self._mg_u1, self._mg_z1, self._mg_d1 = _axis_row(0, eng.axis1, "Axis 1 (polar)")
        self._mg_u2, self._mg_z2, self._mg_d2 = _axis_row(1, eng.axis2, "Axis 2 (L-plate)")
        self._mg_s2 = QtWidgets.QCheckBox(
            "Sync also re-solves the L-plate zero (only for a plate not graduated "
            "0 at the pole)")
        self._mg_s2.setChecked(False)
        self._mg_s2.setToolTip(
            "Leave this OFF. A plate graduated 0 at the pole is already calibrated "
            "exactly, by mechanics, with no star involved. Solving it from a star "
            "instead would bake that star's polar-alignment, squareness and centring "
            "errors into the plate zero — trading an exact calibration for a worse one. "
            "It is here only for a bracket whose zero mark means nothing in particular.")
        ag.addWidget(self._mg_s2, 2, 0, 1, 5)
        for wdg in (self._mg_u1, self._mg_u2, self._mg_d1, self._mg_d2, self._mg_wrap):
            wdg.currentIndexChanged.connect(self._mg_apply_cal)
        for wdg in (self._mg_z1, self._mg_z2, self._mg_travel2):
            wdg.valueChanged.connect(self._mg_apply_cal)
        self._mg_sync_units()          # needs the unit combos, so only once they exist
        p2.addWidget(adv)
        p2.addStretch(1)

        # The two-star alignment tab is gone: this rig is polar-aligned with the
        # polar scope, and a pointing model that does not fix tracking was never used.
        # The solver itself stays in manual_goto.py for anyone who wants it back.

        tabs.addTab(self._build_mount_page(), "🔭  Mount")

        # ---- footer ----------------------------------------------------------
        foot = QtWidgets.QHBoxLayout()
        self._mg_live = QtWidgets.QCheckBox("Live (now)")
        self._mg_live.setChecked(True)
        self._mg_live.setToolTip("On: readings for the current instant, refreshed every "
                                 "second. Off: readings for the date/time set in the Sky "
                                 "Map tab, to prepare a session in advance.")
        self._mg_live.toggled.connect(lambda *_: self._mg_refresh())
        foot.addWidget(self._mg_live)
        foot.addStretch(1)
        cpy = QtWidgets.QPushButton("⧉  Copy readings"); cpy.clicked.connect(self._mg_copy)
        foot.addWidget(cpy)
        closeb = QtWidgets.QPushButton("Close"); closeb.clicked.connect(dlg.close)
        foot.addWidget(closeb)
        v.addLayout(foot)

        timer = QtCore.QTimer(dlg); timer.setInterval(1000)
        timer.timeout.connect(self._mg_refresh)
        timer.start()
        dlg.finished.connect(lambda *_: (timer.stop(),
                                         setattr(self, "_mg_track_lbl", None),
                                         setattr(self, "_mg_dlg", None),
                                         setattr(self, "_mg_tabs", None),
                                         setattr(self, "_mount_dlg_lbl", None),
                                         setattr(self, "_goal_slew_lbl", None),
                                         setattr(self, "autocenter_btn", None),
                                         setattr(self, "autocenter_stop_btn", None),
                                         setattr(self, "_dither_plot", None),
                                         setattr(self, "_mg_dial_lbl", None),
                                         setattr(self, "_mg_live", None)))
        self._mg_refresh()
        # A long label that cannot wrap sets a minimum width for the whole tab and
        # pushes everything else off the right edge — so make sure none of them can.
        for _lb in dlg.findChildren(QtWidgets.QLabel):
            if len(_lb.text()) > 60:
                _lb.setWordWrap(True)
        # fit the screen: tabs keep it short, a scroll area catches the rest
        scr = self.screen().availableGeometry() if self.screen() else None
        dlg.resize(780, min(800, scr.height() - 80) if scr else 740)
        self._mg_show_tab(tab)
        dlg.show()

    def _mg_show_tab(self, tab):
        """'point', 'setup' or 'mount'."""
        tabs = getattr(self, "_mg_tabs", None)
        if tabs is None:
            return
        tabs.setCurrentIndex({"point": 0, "setup": 1, "mount": 2}.get(tab, 0))

    def _mg_record_star(self, idx):
        """Freeze the current target + instant + dial readings as one align star."""
        import manual_goto
        goal = getattr(self, "_session_goal", None)
        if not goal or goal[1] is None:
            self._mg_align_msg.setText("Pick the star on the Sky Map first.")
            return
        stars = getattr(self, "_mg_align_stars", None) or [None, None]
        stars[idx] = manual_goto.AlignStar(
            goal[0], float(goal[1]), float(goal[2]), self._mg_when_utc(),
            float(self._mg_r1.value()), float(self._mg_r2.value()))
        self._mg_align_stars = stars
        self._mg_align_refresh()

    def _mg_align_refresh(self):
        if not getattr(self, "_mg_star_lbl", None):     # tab removed: nothing to show
            return
        """Show what is recorded, and judge the pair before anything is solved."""
        import manual_goto
        stars = getattr(self, "_mg_align_stars", None) or [None, None]
        for i, lb in enumerate(self._mg_star_lbl):
            st = stars[i]
            lb.setText("star {} — empty".format(i + 1) if st is None else
                       "star {} — {}   dial {}   plate {:+.2f}°   at {}".format(
                           i + 1, st.name, self._mg_ha_str(st.reading1), st.reading2,
                           self._mg_local_str(st.when_utc)))
        if not (stars[0] and stars[1]):
            self._mg_pair_lbl.setText(""); return
        eng = self._mg_engine()
        v = [manual_goto.star_vec(*manual_goto.precess_from_j2000(
                st.ra_j2000, st.dec_j2000, st.when_utc), st.when_utc, eng.lat, eng.lon)
             for st in stars]
        grade, why = manual_goto.pair_quality(manual_goto.sep_deg(*v),
                                              stars[0].reading2, stars[1].reading2)
        col = {"good": "#4ade80", "ok": "#fbbf24"}.get(grade, "#f87171")
        self._mg_pair_lbl.setStyleSheet("color:{};".format(col))
        self._mg_pair_lbl.setText(why)

    def _mg_solve_align(self):
        import manual_goto
        stars = getattr(self, "_mg_align_stars", None) or [None, None]
        if not (stars[0] and stars[1]):
            self._mg_align_msg.setText("Record two stars first.")
            return
        eng = self._mg_engine()
        res = manual_goto.two_star_align(stars, eng.lat, eng.lon, eng.axis1,
                                         precess=eng.precess)
        if not res.ok:
            self._mg_align_msg.setStyleSheet("color:#f87171;")
            self._mg_align_msg.setText(res.message)
            return
        eng.apply_align(res)
        self._mg_z1.blockSignals(True); self._mg_z1.setValue(res.zero1)
        self._mg_z1.blockSignals(False)
        self._mg_save_cal()
        self._mg_align_msg.setStyleSheet(
            "color:{};".format("#fbbf24" if res.ambiguous else "#4ade80"))
        d_alt = eng.lat - res.axis_alt
        d_az = -manual_goto.wrap180(res.axis_az)
        self._mg_align_msg.setText(
            res.message + "  Residuals {:.2f}° / {:.2f}°.\n\nTo polar-align from here, "
            "touch only the tripod knobs: {} the altitude by {:.2f}°, and turn the mount "
            "{:.2f}° towards the {}. Then re-run this align to confirm — the two numbers "
            "should collapse towards zero.".format(
                *(res.residuals + [0.0, 0.0])[:2],
                "raise" if d_alt > 0 else "lower", abs(d_alt),
                abs(d_az), "east" if d_az > 0 else "west"))
        self._mg_refresh()

    def _mg_clear_align(self):
        eng = self._mg_engine()
        eng.clear_align()
        self._mg_save_cal()
        self._mg_align_msg.setStyleSheet("color:#cbd5e1;")
        self._mg_align_msg.setText("Back to assuming the axis is on the celestial pole.")
        self._mg_refresh()

    _RATE_BODIES = ("sun", "moon", "mercury", "venus", "mars", "jupiter",
                    "saturn", "uranus", "neptune")

    def _make_rate_combo(self):
        """A tracking-rate selector. Several of them exist (pointing tab, mount tab);
        they share one setting, so picking a rate anywhere updates them all — and it
        works with no target chosen, which is exactly when you need to say what to
        follow."""
        cb = QtWidgets.QComboBox()
        cb.addItem("Auto (from target)", "auto")
        cb.addItem("Sidereal (stars, deep sky)", "sidereal")
        cb.addItem("Solar", "sun")
        cb.addItem("Lunar", "moon")
        for b in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"):
            cb.addItem(b, b.lower())
        cb.setToolTip("The Sun, Moon and planets drift against the stars, so they need "
                      "their own rate. NOUT takes it from the ephemeris, which also gets "
                      "the sign right: a planet in retrograde needs tracking slightly "
                      "FASTER than sidereal.\n"
                      "With no target chosen, 'Auto' means sidereal.")
        cb.setCurrentIndex(max(0, cb.findData(getattr(self, "_track_rate_choice", "auto"))))
        combos = getattr(self, "_rate_combos", None)
        if combos is None:
            combos = self._rate_combos = []
        combos.append(cb)
        cb.currentIndexChanged.connect(lambda *_: self._on_rate_choice(cb.currentData()))
        return cb

    def _on_rate_choice(self, choice):
        self._track_rate_choice = choice
        for cb in list(getattr(self, "_rate_combos", [])):
            try:
                if cb.currentData() != choice:
                    cb.blockSignals(True)
                    cb.setCurrentIndex(max(0, cb.findData(choice)))
                    cb.blockSignals(False)
            except RuntimeError:                       # its window was closed
                self._rate_combos.remove(cb)

    def _mg_rate_body(self):
        """Which body's rate to use — explicit, or worked out from the target.

        Reads a plain attribute rather than the combo box: the top-bar indicator
        needs this after the GoTo window has been closed, and by then that widget
        may no longer exist."""
        sel = getattr(self, "_track_rate_choice", "auto")
        if sel != "auto":
            return sel
        goal = getattr(self, "_session_goal", None)
        name = (goal[0] if goal else "").strip().lower()
        return name if name in self._RATE_BODIES else "sidereal"

    def _mount_stop_now(self):
        """Everything stops: manual slew, GoTo, tracking. Reachable from anywhere,
        because the moment you need it is never the moment to go looking for it."""
        for cmd in ("mount_slew_stop", "mount_axis1_stop", "mount_track_stop"):
            self.worker.post(cmd)
        self.statusBar().showMessage("⛔ Mount stop requested — slew, GoTo and tracking.",
                                     8000)

    def _start_tracking(self, ip=None):
        """Start the axis at the chosen rate: plain sidereal, or the rate of whatever
        body is selected (or of the target, when the choice is 'Auto')."""
        ip = ip or self._mip()
        body = self._mg_rate_body()
        if body == "sidereal":
            self.worker.post("mount_track_start", ip=ip)
        else:
            self.worker.post("mount_set_track_rate", ip=ip, body=body,
                             lat=self.t_lat.value(), lon=self.t_lon.value())
        self.statusBar().showMessage(
            "Tracking at the {} rate…".format(self._rate_mode_name(body)), 6000)

    def _mg_apply_rate(self, *args):
        import manual_goto
        from datetime import datetime
        body = self._mg_rate_body()
        self.worker.post("mount_set_track_rate", ip=self._mip(), body=body,
                         lat=self.t_lat.value(), lon=self.t_lon.value())
        try:
            self._mg_rate_lbl.setText(manual_goto.rate_summary(
                body, datetime.utcnow(), self.t_lat.value(), self.t_lon.value()))
        except Exception:               # noqa: BLE001
            pass
        self._mg_rate_tick = 0

    def _mg_axis1_target(self):
        goal = getattr(self, "_session_goal", None)
        if not goal or goal[1] is None:
            self._mg_a1_msg.setText("Pick the target on the Sky Map first.")
            return None
        return dict(ip=self._mip(), ra=float(goal[1]), dec=float(goal[2]),
                    name=goal[0], lat=self.t_lat.value(), lon=self.t_lon.value(),
                    plate=float(self._mg_r2.value()) if getattr(self, "_mg_r2", None)
                    else 0.0,
                    flip=bool(getattr(self, "_mg_flip", None)
                              and self._mg_flip.isChecked()))

    def _mg_axis1_sync(self):
        kw = self._mg_axis1_target()
        if kw is None:
            return
        self._mg_last_a1_op = "sync"
        self._mg_a1_sync_msg.setText("Syncing… (measuring the axis directions on the "
                                     "first call — about 10 s)")
        self.worker.post("mount_axis1_sync", **kw)

    def _mg_axis1_goto(self):
        kw = self._mg_axis1_target()
        if kw is None:
            return
        kw["tol_deg"] = float(self._mg_a1_tol.value())
        kw["max_from_sync"] = float(self._mg_a1_travel.value())
        kw["max_iter"] = 6
        kw["rate"] = 256
        self._mg_last_a1_op = "goto"
        self._mg_a1_msg.setText("Slewing axis 1 — this mount does about 0.30°/s, so a "
                                "large move takes minutes. Stop is live.")
        self._mg_a1_plate.setText("")
        self._mg_a1_bar.setValue(0); self._mg_a1_bar.setVisible(True)
        self._mg_a1_eta.setText("starting…"); self._mg_a1_eta.setVisible(True)
        self.worker.post("mount_axis1_goto", **kw)

    @staticmethod
    def _rate_mode_name(body):
        """The four tracking modes, named the same way everywhere.

        A planet is named as well as categorised: its rate is barely a twentieth
        of a percent off sidereal, so 'planet · Saturn' says which one is being
        followed where a bare 'planet' would leave you guessing."""
        if body in ("sidereal", "", None):
            return "sidereal"
        if body == "sun":
            return "solar"
        if body == "moon":
            return "lunar"
        return "planet · {}".format(body.capitalize())

    def _track_pill_menu(self, pos):
        m = QtWidgets.QMenu(self)
        stop = m.addAction("⛔  Stop the mount now")
        m.addSeparator()
        openw = m.addAction("Open the mount window")
        act = m.exec(self.track_pill.mapToGlobal(pos))
        if act is stop:
            self._mount_stop_now()
        elif act is openw:
            self._open_manual_goto("mount")

    def _track_pill_clicked(self):
        """The indicator doubles as the connect button when there is nothing to show.

        'Connected' means two different things — the Mac being on the mount's Wi-Fi,
        and NOUT having opened the mount — and only the second one makes the axis
        answer. Saying so where the confusion happens beats a message elsewhere."""
        kind = getattr(self, "_track_state_kind", "off")
        if kind == "off" or getattr(self.worker, "_dither_mount", None) is None:
            self.statusBar().showMessage(
                "Connecting to the mount at {}…".format(self._mip()), 6000)
            self.worker.post("mount_connect", ip=self._mip(),
                             two_axis=bool(QtCore.QSettings("NOUT", "NOUT").value(
                                 "mount_type", 0, type=int)))
            return
        if kind in ("wrong", "stopped"):
            # applying a rate also starts the tracking, so one action covers both
            body = self._mg_rate_body()
            self.worker.post("mount_set_track_rate", ip=self._mip(), body=body,
                             lat=self.t_lat.value(), lon=self.t_lon.value())
            self.statusBar().showMessage(
                "Applying the {} tracking rate…".format(self._rate_mode_name(body)), 6000)
            return
        self._open_mount_dialog()

    def _track_verdict(self, measured, commanded, cpr):
        """What the axis is doing, judged against what the target needs.

        Returns (colour, short, headline, detail). The rate is measured off the
        step counter, so this catches a mount that quietly stopped and a mount
        tracking at the wrong rate for what you are pointing at — neither of which
        any 'tracking started' message would."""
        import manual_goto
        from datetime import datetime
        if not cpr or measured != measured:             # NaN: nothing to read
            self._track_state_kind = "off"
            return ("#94a3b8", "⚪  mount off",
                    "⚪  Mount not connected",
                    "Click to connect.")
        sid = cpr * manual_goto.SIDEREAL_RATE_DPS / 360.0
        ratio = measured / sid
        body = self._mg_rate_body()
        try:
            need = manual_goto.body_track_rate(
                body, datetime.utcnow(), self.t_lat.value(),
                self.t_lon.value()) / manual_goto.SIDEREAL_RATE_DPS
        except Exception:               # noqa: BLE001
            need = 1.0
        want = self._rate_mode_name(body)
        if abs(ratio) > 3.0:
            self._track_state_kind = "slew"
            return ("#38bdf8", "⏩  slewing", "⏩  Slewing — not tracking right now",
                    "{:.0f}× sidereal".format(abs(ratio)))
        if abs(ratio) < 0.25:
            self._track_state_kind = "stopped"
            return ("#f87171", "⏸  STOPPED — click to start",
                    "⏸  STOPPED — the axis is not moving",
                    "Nothing is following the sky. Click to start tracking at the "
                    "{} rate.".format(want))
        off = (ratio / need - 1.0) * 100.0 if need else 0.0
        detail = ("measured {:.4f}× sidereal, {} wants {:.4f}× — {:+.2f}% off "
                  "({:.1f} steps/s)".format(ratio, want, need, off, measured))
        if want.startswith("planet") and abs(need - 1.0) < 0.002:
            detail += (".  A planet sits so close to the sidereal rate that plain "
                       "sidereal tracking is fine for it")
        if abs(off) < 0.5:
            self._track_state_kind = "ok"
            return ("#4ade80", "🟢  {}".format(want),
                    "🟢  Tracking — {} rate, {}".format(
                        want, "as your target needs" if body != "sidereal"
                        else "as the stars need"), detail)
        self._track_state_kind = "wrong"
        return ("#fbbf24", "⚠  {} ≠ {} — click to fix".format(
                    "sidereal" if abs(ratio - 1.0) < 0.005 else "{:.3f}×".format(ratio), want),
                "⚠  Tracking, but NOT at the {} rate your target needs".format(want),
                detail + ".  Click the indicator to apply the {} rate.".format(want))

    def _on_track_state(self, measured, commanded, cpr):
        col, short, head, detail = self._track_verdict(measured, commanded, cpr)
        self._sb["mount"] = "🔭 " + re.sub(r"^[^\w]+", "", short).strip()
        self._refresh_statusbar()
        pill = getattr(self, "track_pill", None)
        if pill is not None:
            try:
                pill.setStyleSheet(
                    "color:{}; font-weight:bold; padding:0 8px;".format(col))
                pill.setCursor(Qt.PointingHandCursor)
                pill.setText(short)
                pill.setToolTip((head + "\n" + detail).strip())
            except RuntimeError:
                self.track_pill = None
        lbl = getattr(self, "_mg_track_lbl", None)
        if lbl is None:
            return
        try:
            lbl.setStyleSheet("font-size:14px; font-weight:bold; color:{};".format(col))
            lbl.setText(head)
            self._mg_track_sub.setText(detail)
        except RuntimeError:            # dialog closed
            self._mg_track_lbl = None

    def _on_axis1_progress(self, frac, left_deg, left_s):
        """Drive the GoTo bar. It lives in the dialog because that is what is in
        front of you during a slew — the main window's status bar is behind it."""
        bar = getattr(self, "_mg_a1_bar", None)
        if bar is None:
            return
        try:
            if not bar.isVisible():
                bar.setVisible(True); self._mg_a1_eta.setVisible(True)
            bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
            self._mg_a1_eta.setText("{:.0f}%  —  {:.2f}° left, about {}".format(
                frac * 100.0, left_deg, CameraWorker._eta_str(left_s)))
        except RuntimeError:            # dialog closed mid-slew
            self._mg_a1_bar = None

    def _mg_a1_hide_progress(self):
        for name in ("_mg_a1_bar", "_mg_a1_eta"):
            wdg = getattr(self, name, None)
            if wdg is not None:
                try:
                    wdg.setVisible(False)
                except RuntimeError:
                    setattr(self, name, None)

    def _on_axis1_result(self, ok, msg, plate):
        """A sync is calibration and a GoTo is pointing, so each reports where it
        was asked for rather than both landing in the same box."""
        syncing = getattr(self, "_mg_last_a1_op", "goto") == "sync"
        self._mg_a1_hide_progress()     # the slew is over, one way or another
        lbl = getattr(self, "_mg_a1_sync_msg" if syncing else "_mg_a1_msg", None)
        if lbl is None:
            self.statusBar().showMessage(msg, 9000)
            return
        try:
            lbl.setStyleSheet("color:{};".format("#4ade80" if ok else "#f87171"))
            lbl.setText(msg)
            if not syncing:
                self._mg_a1_plate.setText(
                    "→ now set the L-plate to {:+.1f}°".format(plate)
                    if ok and plate else "")
        except RuntimeError:            # dialog closed while the slew ran
            self._mg_a1_msg = self._mg_a1_sync_msg = None
            self.statusBar().showMessage(msg, 9000)

    def _mg_sync_units(self):
        """Label the two sync boxes with the unit each dial is actually graduated in."""
        for spin, lbl, combo, name in ((self._mg_r1, self._mg_lbl_r1, self._mg_u1, "axis 1"),
                                       (self._mg_r2, self._mg_lbl_r2, self._mg_u2, "axis 2")):
            hours = combo.currentText() == "hours"
            spin.set_hours(hours)
            lbl.setText("{} ({})".format(name, "h — 23h30 or 23.5" if hours else "°"))

    def _mg_apply_cal(self, *args):
        """Push the calibration widgets into the solver and persist it."""
        eng = self._mg_engine()
        eng.axis1.units = self._mg_u1.currentText()
        eng.axis1.sign = float(self._mg_d1.currentData())
        eng.axis1.zero = float(self._mg_z1.value())
        eng.axis1.wrap = True
        eng.axis2.units = self._mg_u2.currentText()
        eng.axis2.sign = float(self._mg_d2.currentData())
        eng.axis2.zero = float(self._mg_z2.value())
        eng.axis2.wrap = bool(self._mg_wrap.currentData())
        eng.axis2_travel = float(self._mg_travel2.value())
        self._mg_sync_units()
        self._mg_save_cal()
        self._mg_refresh()

    def _mg_check_polar(self):
        """Measure the polar axis itself: with the L-plate on 0 the solved frame centre
        is exactly where the axis points, so its distance to Dec +90° is the error."""
        import manual_goto
        wcs = getattr(self, "_solve_wcs", None)
        if not wcs:
            self._mg_polar_msg.setText(
                "No plate-solve yet. Set the L-plate to 0, take a photo, run the solve "
                "in the Shooting tab, then come back.")
            return
        ra, dec = float(wcs[0]), float(wcs[1])
        err, ha, direction = manual_goto.polar_axis_error(
            ra, dec, self._mg_when_utc(), self.t_lon.value())
        if err <= 0.2:
            verdict = "excellent — nothing to redo."
        elif err <= 0.5:
            verdict = "fine for wide field; worth improving past ~300 mm."
        elif err <= 1.5:
            verdict = "rough. Redo the alignment."
        else:
            verdict = ("way off — this is about the size of putting Polaris on the "
                       "wrong side of the reticle (twice its 0.63° offset).")
        self._mg_polar_msg.setText(
            "Polar axis is {:.2f}° from the pole (solved centre Dec {:+.3f}°, "
            "{} of it) — {}".format(err, dec, direction, verdict))

    def _mg_sync(self):
        import manual_goto
        goal = getattr(self, "_session_goal", None)
        if not goal or goal[1] is None:
            self._mg_cal_msg.setText("Pick a target on the Sky Map first.")
            return
        eng = self._mg_engine()
        # the polar dial is the only thing a star sync is really for; axis 2 is an
        # opt-in for a plate whose zero is not the pole
        do1, do2 = True, self._mg_s2.isChecked()
        when = self._mg_when_utc()
        # the plate check is worth having whether or not we rewrite its zero
        dec_d = eng.resolve(goal[0], float(goal[1]), float(goal[2]), when)[1]
        gap = eng.axis2_pole_error(dec_d, self._mg_r2.value())
        z1, z2 = eng.sync(float(goal[1]), float(goal[2]), self._mg_r1.value(),
                          self._mg_r2.value(), when, axis1=do1, axis2=do2,
                          name=goal[0])
        for w, val in ((self._mg_z1, z1), (self._mg_z2, z2)):
            w.blockSignals(True); w.setValue(val); w.blockSignals(False)
        self._mg_save_cal()
        msg = "Synced on {} — polar-dial zero {:.3f} h. ".format(goal[0], z1)
        _unused = do1
        if do2:
            msg += "L-plate zero rewritten to {:+.2f}.".format(z2)
        else:
            msg += ("Check: this star puts the L-plate {:+.2f}° from the pole reading "
                    "(alignment + squareness + centring; under ~1° is normal)."
                    .format(gap))
        self._mg_cal_msg.setText(msg)
        self._mg_refresh()

    def _mg_pole_zero(self):
        """Calibrate axis 2 the easy way: the plate reads 0 with the camera aimed
        along the polar axis, so a reading is just the angle from the pole."""
        eng = self._mg_engine()
        eng.set_axis2_pole_zero(sign=float(self._mg_d2.currentData()),
                                wrap=bool(self._mg_wrap.currentData()))
        self._mg_z2.blockSignals(True); self._mg_z2.setValue(eng.axis2.zero)
        self._mg_z2.blockSignals(False)
        self._mg_u2.blockSignals(True); self._mg_u2.setCurrentText("degrees")
        self._mg_u2.blockSignals(False)
        self._mg_sync_units()
        self._mg_save_cal()
        self._mg_cal_msg.setText(
            "L-plate: 0 = pole. A target at declination δ sits (90° − δ) from the pole, "
            "reachable from either side — the two are the same point of sky with axis 1 "
            "half a turn apart, so take whichever your bracket reaches. {}".format(
                "On your 0–360 scale the second one reads 360° minus the first."
                if eng.axis2.wrap else "The second one is the first with a minus sign."))
        self._mg_refresh()

    def _mg_mark(self):
        self._mg_engine().mark_current(self._mg_r1.value(), self._mg_r2.value())
        self._mg_cal_msg.setText("Current dial position noted — targets now show "
                                 "how far to turn.")
        self._mg_refresh()

    def _mg_copy(self):
        txt = getattr(self, "_mg_clip", "")
        if txt:
            QtWidgets.QApplication.clipboard().setText(txt)
            self.statusBar().showMessage("Manual GoTo readings copied.", 4000)

    def _mg_refresh(self):
        """Recompute both dial readings for the goal at the chosen instant."""
        if not getattr(self, "_mg_dial_lbl", None):
            return
        goal = getattr(self, "_session_goal", None)
        if not goal or goal[1] is None or goal[2] is None:
            self._mg_target_lbl.setText("No target — pick one on the Sky Map.")
            self._mg_coord_lbl.setText("")
            for k in ("a1", "a2"):
                self._mg_dial_lbl[k].setText("—"); self._mg_turn_lbl[k].setText("")
            for k in ("a1", "a2"):
                self._mg_sub_lbl[k].setText("")
            self._mg_update_polaris(self._mg_when_utc())   # independent of any target
            self._mg_info_lbl.setText(""); self._mg_warn_lbl.setText("")
            self._mg_alt_lbl.setText("")
            return
        eng = self._mg_engine()
        when = self._mg_when_utc()
        res = eng.solve(goal[0], float(goal[1]), float(goal[2]), when)
        flip = bool(getattr(self, "_mg_flip", None) and self._mg_flip.isChecked())
        p, other = (res.alternate, res.primary) if flip else (res.primary, res.alternate)
        if getattr(self, "_mg_name_for", None) != goal:          # refreshes once a second:
            self._mg_name_for = goal                             # don't rescan the catalog
            self._mg_name = self._target_full_name(goal[0], float(goal[1]), float(goal[2]))
        self._mg_target_lbl.setText("🎯 {}".format(self._mg_name))
        self._mg_coord_lbl.setText(
            "J2000  RA {:.4f}°  Dec {:+.4f}°     ·     of date  RA {:.4f}°  Dec {:+.4f}°"
            .format(res.ra_j2000, res.dec_j2000, res.ra_date, res.dec_date))
        for key, dial, cal in (("a1", p.axis1, eng.axis1), ("a2", p.axis2, eng.axis2)):
            self._mg_dial_lbl[key].setText(dial.text)
            aligned = eng.axis_altaz is None
            self._mg_sub_lbl[key].setText(
                ("angle from the pole" if aligned else "angle from the axis")
                if key == "a2" else
                ("hour angle {}".format(self._mg_ha_str(res.ha_hours)) if aligned
                 else "axis angle (mount {:.1f}° off the pole)".format(eng.axis_error())))
            if dial.turn_from is None:
                self._mg_turn_lbl[key].setText("(mark the dials to see what is left)")
            elif cal.units == "hours":
                self._mg_turn_lbl[key].setText("still to turn  {}  ({:+.1f}° of sky)".format(
                    self._mg_ha_str(dial.turn_from), dial.turn_sky_deg))
            else:
                self._mg_turn_lbl[key].setText(
                    "still to turn  {:+.2f}°".format(dial.turn_from))
        local = self._mg_local_str(when)
        self._mg_info_lbl.setText(
            "Valid at {} local   ·   hour angle {} ({})   ·   "
            "altitude {:+.1f}°  azimuth {:.1f}°".format(
                local, self._mg_ha_str(res.ha_hours),
                "west" if res.hour_angle >= 0 else "east", res.alt, res.az))
        self._mg_warn_lbl.setText("  ".join("⚠ " + w for w in res.warnings))
        a = other
        self._mg_alt_lbl.setText(
            "{} — same point of sky, L-plate on the other side of the pole: "
            "axis 1 {}  ·  axis 2 {}{}".format(
                "Direct position" if flip else "Mirrored position",
                a.axis1.text, a.axis2.text,
                "" if a.reachable else "  — outside the L-plate travel you declared"))
        self._mg_update_polaris(when)
        # the Moon's rate wanders as it runs along its orbit; refresh it now and then
        if getattr(self, "_mg_rate", None) is not None:
            self._mg_rate_tick = getattr(self, "_mg_rate_tick", 0) + 1
            if self._mg_rate_tick >= 300 and self._mg_rate_body() != "sidereal":
                self._mg_apply_rate()
        self._mg_clip = "{}  |  axis1 {}  |  axis2 {}  |  {} local".format(
            goal[0], p.axis1.text, p.axis2.text, local)

    def _mg_ret_hours(self):
        """How far the reticle has turned, in hours of RA."""
        if getattr(self, "_mg_ret_t0", None) is not None:
            import time
            elapsed = time.monotonic() - self._mg_ret_t0[0]
            return self._mg_ret_t0[1] + elapsed / 3600.0 * (86400.0 / 86164.0905)
        h = self._mg_ret_h.value()
        return h + (self._mg_ret_m.value() / 60.0) * (1 if h >= 0 else -1)

    def _mg_ret_manual(self, *args):
        if getattr(self, "_mg_ret_t0", None) is not None:
            return                      # the timer owns the value while it runs
        self._mg_refresh()

    def _mg_ret_start(self, on):
        import time
        if on:
            self._mg_ret_t0 = (time.monotonic(), self._mg_ret_hours())
            self._mg_ret_track.setText("⏸ Stop counting")
        else:
            held = self._mg_ret_hours()
            self._mg_ret_t0 = None
            self._mg_ret_track.setText("▶ Start tracking")
            self._mg_ret_set_spins(held)
        self._mg_refresh()

    def _mg_ret_reset(self):
        self._mg_ret_t0 = None
        if self._mg_ret_track.isChecked():
            self._mg_ret_track.setChecked(False)
        self._mg_ret_set_spins(0.0)
        self._mg_refresh()

    def _mg_ret_set_spins(self, hours):
        hours = ((hours + 12.0) % 24.0) - 12.0
        sign = -1 if hours < 0 else 1
        a = abs(hours)
        hh, mm = int(a), int(round((a - int(a)) * 60.0))
        if mm == 60:
            hh, mm = hh + 1, 0
        for sp, val in ((self._mg_ret_h, sign * hh), (self._mg_ret_m, mm)):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)

    def _mg_update_polaris(self, when):
        """Polaris' place around the pole, described in the real sky (naked eye,
        facing north) — a polar scope may rotate or mirror that view."""
        lbl = getattr(self, "_mg_polaris_lbl", None)
        if lbl is None:
            return
        import manual_goto
        ha, pdist, clock, direction = manual_goto.polaris_state(when, self.t_lon.value())
        ret = getattr(self, "_mg_reticle", None)
        if ret is not None:
            try:
                ha_app, arcmin = manual_goto.polaris_apparent(
                    when, self.t_lat.value(), self.t_lon.value())
                rot = self._mg_ret_hours()
                ret.set_state(ha_app, arcmin, rotation_h=rot,
                              flip=self._mg_ret_flip.isChecked(),
                              mirror=self._mg_ret_mirror.isChecked(),
                              numbers_ccw=self._mg_ret_ccw.isChecked(),
                              night=bool(getattr(self, "_night", False)))
                if self._mg_ret_t0 is not None:
                    self._mg_ret_set_spins(rot)
                c = ret.clock_reading()
                self._mg_ret_msg.setText(
                    "Put Polaris at <b>{}:{:02.0f}</b> on the engraved scale, "
                    "<b>{:.1f}′</b> from the centre.".format(
                        int(c) or 12, (c - int(c)) * 60.0, arcmin))
            except RuntimeError:        # dialog closed under us
                self._mg_reticle = None
        try:
            lbl.setText(
                "Polaris is <b>{:.2f}° from the true pole</b>, at hour angle {} — {} of "
                "the pole as you see it facing north. The pole is on the opposite side: "
                "{:.2f}° from Polaris, along the line running towards <b>Kochab</b> (the "
                "bright star of the Little Dipper's bowl). That line owes nothing to any "
                "reticle convention and is good to 3′.".format(
                    pdist, self._mg_ha_str(ha), direction, pdist))
        except RuntimeError:            # dialog closed under us
            self._mg_polaris_lbl = None

    @staticmethod
    def _mg_clock_str(clock_hours):
        h = int(clock_hours) or 12
        m = (clock_hours - int(clock_hours)) * 60.0
        return "{}:{:02.0f}".format(h, m)

    @staticmethod
    def _mg_ha_str(ha_hours):
        """Hour angle as ±HhMMm — signed even inside the first hour."""
        sign = "-" if ha_hours < 0 else "+"
        h, m = divmod(round(abs(ha_hours) * 60.0), 60)
        return "{}{:.0f}h{:02.0f}m".format(sign, h, m)

    @staticmethod
    def _mg_local_str(dt_utc):
        from datetime import timezone as _tz
        try:
            from zoneinfo import ZoneInfo
            return dt_utc.replace(tzinfo=_tz.utc).astimezone(
                ZoneInfo("Europe/Paris")).strftime("%H:%M:%S")
        except Exception:               # noqa: BLE001
            return dt_utc.strftime("%H:%M:%S") + " UTC"

    def _open_mount_dialog(self):
        """Kept for the tracking pill and older call sites: same window, Mount tab."""
        self._open_manual_goto(tab="mount")

    def _build_mount_page(self):
        """Connection, tracking, manual slews, auto-centring and the dither trail."""
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("Mount"))
        mtype = QtWidgets.QComboBox()
        mtype.addItem("Star Adventurer 2i (Wi-Fi)", False)
        mtype.addItem("HEQ-5 Pro (SynScan Wi-Fi, 2-axis) — beta", True)
        saved_type = QtCore.QSettings("NOUT", "NOUT").value("mount_type", 0, type=int)
        mtype.setCurrentIndex(min(saved_type, mtype.count() - 1))
        mtype.currentIndexChanged.connect(
            lambda i: QtCore.QSettings("NOUT", "NOUT").setValue("mount_type", i))
        trow.addWidget(mtype, 1)
        v.addLayout(trow)
        iprow = QtWidgets.QHBoxLayout()
        ip = QtWidgets.QLineEdit(self._mip())
        ip.editingFinished.connect(
            lambda: QtCore.QSettings("NOUT", "NOUT").setValue("mount_ip", ip.text().strip()))
        iprow.addWidget(QtWidgets.QLabel("Wi-Fi IP")); iprow.addWidget(ip)
        v.addLayout(iprow)

        def _ip():
            return ip.text().strip() or "192.168.4.1"

        def _two():
            return bool(mtype.currentData())
        row1 = QtWidgets.QHBoxLayout()
        keep = QtWidgets.QCheckBox("kept running")
        keep.setToolTip("Tick this when the mount has stayed powered since the last "
                        "session: NOUT then keeps its step counter, and the axis-1 sync "
                        "from last time is restored instead of being thrown away.")
        self._mount_keep = keep
        cbtn = QtWidgets.QPushButton("🔌 Connect")
        cbtn.clicked.connect(lambda: self.worker.post("mount_connect", ip=_ip(),
                                                      two_axis=_two(),
                                                      preserve=keep.isChecked()))
        tbtn = QtWidgets.QPushButton("▶ Start tracking")
        tbtn.setToolTip("Starts the axis at the rate chosen below.")
        tbtn.clicked.connect(lambda: self._start_tracking(_ip()))
        sbtn = QtWidgets.QPushButton("⏸ Stop tracking")
        sbtn.clicked.connect(lambda: self.worker.post("mount_track_stop"))
        row1.addWidget(cbtn); row1.addWidget(keep); row1.addWidget(tbtn); row1.addWidget(sbtn)
        v.addLayout(row1)
        stopall = QtWidgets.QPushButton("⛔  STOP THE MOUNT  (X)")
        stopall.setToolTip("Stops every motion at once: slew, GoTo and tracking. "
                           "The X key does the same from anywhere in the app.")
        stopall.setStyleSheet("background:#7f1d1d; color:#fff; font-weight:600;")
        stopall.setMinimumHeight(32)
        stopall.clicked.connect(self._mount_stop_now)
        v.addWidget(stopall)
        rrow = QtWidgets.QHBoxLayout()
        rlbl = QtWidgets.QLabel("Rate"); rlbl.setObjectName("cap")
        rrow.addWidget(rlbl)
        rrow.addWidget(self._make_rate_combo(), 1)
        rapply = QtWidgets.QPushButton("Apply")
        rapply.setToolTip("Applies the rate right away — and starts tracking if it is stopped.")
        rapply.clicked.connect(self._mg_apply_rate)
        rrow.addWidget(rapply)
        v.addLayout(rrow)

        row2 = QtWidgets.QHBoxLayout()
        neg = QtWidgets.QPushButton("⏪ Rotate RA− (hold)")
        pos = QtWidgets.QPushButton("Rotate RA+ (hold) ⏩")
        neg.setToolTip("Hold to rotate the arm (manual slew, outside tracking); release to stop.")
        pos.setToolTip(neg.toolTip())
        neg.pressed.connect(lambda: self.worker.post("mount_slew", ip=_ip(), dir=-1, rate=128))
        neg.released.connect(lambda: self.worker.post("mount_slew_stop"))
        pos.pressed.connect(lambda: self.worker.post("mount_slew", ip=_ip(), dir=1, rate=128))
        pos.released.connect(lambda: self.worker.post("mount_slew_stop"))
        row2.addWidget(neg); row2.addWidget(pos)
        v.addLayout(row2)
        row3 = QtWidgets.QHBoxLayout()
        goalbtn = QtWidgets.QPushButton("🎯 Slew RA to goal")
        goalbtn.setToolTip("One-shot: rotate RA once to bring the goal to centre (needs a "
                           "recent plate-solve).")
        goalbtn.clicked.connect(lambda: self._slew_ra_to_goal(_ip()))
        row3.addWidget(goalbtn)
        v.addLayout(row3)
        row3b = QtWidgets.QHBoxLayout()
        self.autocenter_btn = QtWidgets.QPushButton("🤖 Auto-centre on goal")
        self.autocenter_btn.setToolTip("Automatic: photo → plate-solve → slew RA → repeat "
                                       "until the goal is centred (RA only).")
        self.autocenter_btn.clicked.connect(lambda: self._start_autocenter(_ip()))
        self.autocenter_stop_btn = QtWidgets.QPushButton("⏹ Stop")
        self.autocenter_stop_btn.setEnabled(getattr(self, "_ac_active", False))
        self.autocenter_stop_btn.clicked.connect(self._stop_autocenter)
        row3b.addWidget(self.autocenter_btn); row3b.addWidget(self.autocenter_stop_btn)
        v.addLayout(row3b)
        self._goal_slew_lbl = QtWidgets.QLabel("")
        self._goal_slew_lbl.setStyleSheet("color:#94a3b8;"); self._goal_slew_lbl.setWordWrap(True)
        v.addWidget(self._goal_slew_lbl)
        lbl = QtWidgets.QLabel("Mount: " + self._mount_status_text)
        lbl.setStyleSheet("color:#94a3b8;"); lbl.setWordWrap(True)
        v.addWidget(lbl)
        self._mount_dlg_lbl = lbl
        v.addWidget(QtWidgets.QLabel("Dither trail (RA offset over shots):"))
        dplot = pg.PlotWidget(); dplot.setMaximumHeight(150); dplot.setBackground("#0d0d0d")
        dplot.setLabel("left", "RA offset (\")"); dplot.setLabel("bottom", "shot")
        dplot.showGrid(x=True, y=True, alpha=0.3)
        dplot.addLine(y=0, pen=pg.mkPen("#3b4653"))
        self._dither_curve = dplot.plot(
            pen=pg.mkPen("#ff7a29", width=2), symbol="o", symbolSize=6,
            symbolBrush="#ff7a29")
        v.addWidget(dplot)
        self._dither_plot = dplot
        self._refresh_dither_plot()
        v.addStretch(1)
        return page

    def _on_dither_point(self, offset, shot):
        self._dither_hist.append((int(shot), float(offset)))
        if len(self._dither_hist) > 300:
            self._dither_hist = self._dither_hist[-300:]
        self._refresh_dither_plot()

    def _refresh_dither_plot(self):
        plot = getattr(self, "_dither_plot", None)
        if plot is None or not hasattr(self, "_dither_curve"):
            return
        if self._dither_hist:
            xs = [p[0] for p in self._dither_hist]; ys = [p[1] for p in self._dither_hist]
            self._dither_curve.setData(xs, ys)

    def _on_mount_status(self, msg):
        self._mount_status_text = msg
        if "connected" in msg.lower():
            self._mount_connected = True
        elif "connect failed" in msg.lower() or "disconnect" in msg.lower():
            self._mount_connected = False
        if self._mount_dlg_lbl is not None:
            try:
                self._mount_dlg_lbl.setText("Mount: " + msg)
            except RuntimeError:
                self._mount_dlg_lbl = None
        self.statusBar().showMessage(msg, 8000)

    def _build_master_dark(self):
        iso = None
        if hasattr(self, "iso_combo"):
            try:
                iso = int(self.iso_combo.currentText())
            except (ValueError, AttributeError):
                iso = None
        DarkDialog(self, self._flats_dir(), current_iso=iso).exec()

    def _build_master_flat(self):
        cur = self.lens_combo.currentText() if hasattr(self, "lens_combo") else ""
        focal = self.lens_focal.value() if hasattr(self, "lens_focal") else None
        MasterFlatDialog(self, self._flats_dir(), current_lens=cur, current_focal=focal).exec()

    def _stack_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a folder of RAW/JPEG frames to stack")
        if not folder:
            return
        self._folder_stacking = True
        exe = self._graxpert_exe
        use_gx = self.chk_degrad.isChecked() and exe is not None
        self._folder_stack = FolderStackWorker(
            folder, linear=True,
            kappa=self.chk_kappa.isChecked(),
            kappa_k=self.kappa_val.value(),
            reject_weak=self.chk_frame_reject.isChecked(),
            use_graxpert=use_gx, graxpert_exe=exe,
            apply_flats=self.chk_flats.isChecked(), flats_dir=self._flats_dir(),
            apply_darks=self.chk_darks.isChecked())
        self._folder_stack.progress.connect(self._on_folder_progress)
        self._folder_stack.stack_ready.connect(self.on_stack_ready)
        self._folder_stack.done.connect(self._on_folder_done)
        self._folder_stack.failed.connect(self._on_folder_failed)
        self.chk_stretch.setChecked(True)              # reveal faint detail
        self.stack_folder_btn.setEnabled(False)
        self.tabs.setCurrentIndex(2)                   # Results tab (result lands there)
        self.statusBar().showMessage("Stacking folder…")
        self._folder_stack.start()

    def _on_folder_progress(self, done, total, name):
        self.statusBar().showMessage("Stacking {}/{} — {}".format(done, total, name))

    def _on_folder_done(self, mean, raw_mean, kept, rejected, used_gx=False):
        self.stack_folder_btn.setEnabled(True)
        self._folder_stacking = False
        if not getattr(self, "review_mode", False):
            self.iv_start.setEnabled(True)             # make sure capture is usable again
        method = "GraXpert" if used_gx else "built-in gradient removal"
        msg = "Folder stack: {} frames kept{} · background: {}.".format(
            kept, ", {} rejected".format(rejected) if rejected else "", method)
        self.statusBar().showMessage(msg)
        if self._graxpert_exe and not used_gx:
            self.statusBar().showMessage(msg + "  " + str(_GRAXPERT_LOG))
            QtWidgets.QMessageBox.warning(
                self, "GraXpert",
                "GraXpert didn't run — NOUT used its built-in gradient removal.\n\n"
                "Reason: {}\n\n"
                "Most common fix: open GraXpert once (the app) and run a Background "
                "Extraction so it downloads its AI model; then try again."
                .format(_GRAXPERT_LOG))
        try:                                           # `mean` is already background-flat
            result = auto_stretch(neutralize_background(
                star_white_balance(mean.astype(np.float32))), 0.6, scnr=True, saturation=self._sat())
            item = self._add_result(result, "📁 Folder stack · {} frames".format(kept),
                                    processed=True, raw_float=raw_mean)
            self._last_burst_idx = item.data(Qt.UserRole)
        except Exception:           # noqa: BLE001
            pass

    def _use_graxpert_live(self):
        return self._graxpert_exe is not None

    def _graxpert_live(self, src):
        """Async GraXpert on the live-stack float mean, cached between subs (GraXpert is
        slow). Returns the cached flattened float, or None until the first result."""
        if not self._gx_busy and id(src) != self._gx_done_id:
            self._gx_busy = True
            self._gx_pending_id = id(src)
            self._gx_worker = GraXpertWorker(src.copy(), self._graxpert_exe, 0.2)
            self._gx_worker.done.connect(self._on_graxpert_live)
            self._gx_worker.start()
            self.statusBar().showMessage("GraXpert: extracting background…")
        return self._gx_flat

    def _on_graxpert_live(self, flat):
        self._gx_busy = False
        self._gx_done_id = getattr(self, "_gx_pending_id", None)
        if flat is not None:
            self._gx_flat = flat
            self.statusBar().showMessage("GraXpert: background extracted.")
        else:
            self.statusBar().showMessage("GraXpert failed — using built-in gradient removal.")
        self._render_review_image()

    def _on_folder_failed(self, msg):
        self.stack_folder_btn.setEnabled(True)
        self._folder_stacking = False
        if not getattr(self, "review_mode", False):
            self.iv_start.setEnabled(True)
        self.statusBar().showMessage("Folder stacking failed.")
        QtWidgets.QMessageBox.warning(self, "Folder stacking", msg)

    @Slot(int, float, int, float)
    def on_track_point(self, idx, hfr, stars, ecc):
        if hfr > 0:
            self._sess_hfr = getattr(self, "_sess_hfr", []) + [(int(idx), float(hfr))]
        self._track_x.append(idx)
        self._track_y.append(hfr)
        self.track_curve.setData(list(self._track_x), list(self._track_y))
        txt = "HFR {:.2f} px  ·  eccentricity {:.2f}  ·  ★{}".format(hfr, ecc, stars)
        if ecc > 0.6:
            txt += "   ⚠ elongated stars (tracking?)"
            self.track_info.setStyleSheet("color:#ef4444;")
        else:
            self.track_info.setStyleSheet("color:#94a3b8;")
        self.track_info.setText(txt)
        self._sb["hfr"] = "HFR {:.2f}px".format(hfr)
        self._refresh_statusbar()
        # transparency monitor: drop in star count = clouds/haze
        if stars > 0:
            self._star_hist.append(stars)
            self._star_hist = self._star_hist[-12:]
        if len(self._star_hist) >= 5:
            med = sorted(self._star_hist)[len(self._star_hist) // 2]
            if med > 0 and stars < 0.5 * med:
                self.track_info.setText(txt + "   ☁ dropping transparency")
                self.track_info.setStyleSheet("color:#f59e0b;")
                if not self._cloud_warned:
                    self._cloud_warned = True
                    self.statusBar().showMessage(
                        "☁ Transparency drop (clouds / haze / Moon?)", 8000)
                    self._notify("NOUT", "Transparency drop detected (clouds?)")
            else:
                self._cloud_warned = False

    def _push_kappa(self, *args):
        self.worker.post("set_kappa", on=self.chk_kappa.isChecked(),
                         kappa=self.kappa_val.value())

    def _push_site(self, *args):
        self.worker.post("set_site", lat=self.t_lat.value(), lon=self.t_lon.value())

    def _open_iss_passes(self):
        dlg = SatellitePassDialog(self, lat=self.t_lat.value(), lon=self.t_lon.value())
        dlg.exec()
        if self.skymap and self.sky_show_iss.isChecked():   # refresh TLE if it changed
            self._toggle_iss(True)

    def _toggle_iss(self, on):
        if not self.skymap:
            return
        if on:
            tle = QtCore.QSettings("NOUT", "NOUT").value("iss_tle", _DEFAULT_ISS_TLE, type=str)
            self.skymap.set_iss_tle(tle)
            self.skymap.set_show_iss(True)
            self.skymap.refresh_iss()             # place it immediately
            self._iss_timer.start()               # then keep it live
        else:
            self._iss_timer.stop()
            self.skymap.set_show_iss(False)

    def _tick_iss(self):
        """Move the ISS icon in real time while it's shown and the Sky Map is visible."""
        if not self.skymap or not getattr(self, "sky_show_iss", None) \
                or not self.sky_show_iss.isChecked():
            self._iss_timer.stop()
            return
        if self.tabs.currentIndex() != getattr(self, "_skymap_index", -1):
            return                                # save CPU when not looking at the map
        self.skymap.refresh_iss()

    def _update_comets(self):
        import sky_map
        lim = self.sky_comet_maglim.value()
        QtCore.QSettings("NOUT", "NOUT").setValue("comet_maglim", lim)
        self.sky_comet_btn.setEnabled(False)
        self.sky_search_msg.setText("Fetching bright comets from the MPC…")
        self._comet_fetch = CometFetchWorker(lim, sky_map.comets_user_path())
        self._comet_fetch.done.connect(self._on_comets_fetched)
        self._comet_fetch.start()

    def _on_comets_fetched(self, rows, err):
        self.sky_comet_btn.setEnabled(True)
        if rows is None:
            self.sky_search_msg.setText("Comet update failed: {}".format((err or "")[:70]))
            return
        if self.skymap:
            self.skymap.reload_comets()
        names = ", ".join(r[0].split("(")[0].strip() for r in rows[:4])
        self.sky_search_msg.setText("{} capturable comet(s) (≤ m{:.1f}){}".format(
            len(rows), self.sky_comet_maglim.value(),
            ": " + names + ("…" if len(rows) > 4 else "") if rows else
            " — none right now; try a fainter limit."))

    def _open_exposure_calc(self):
        focal = self.lens_focal.value() if hasattr(self, "lens_focal") else 135.0
        fnum = getattr(self, "_last_fnumber", 2.8)
        dlg = ExposureCalcDialog(self, focal=focal, fnumber=fnum,
                                 lat=self.t_lat.value(), lon=self.t_lon.value())
        dlg.exec()

    def _apply_lens(self, name):
        p = LENS_PROFILES.get(name)
        if p is None:                      # Custom: free focal length
            self.lens_focal.setEnabled(True)
            self.lens_focal.setRange(8, 2000)
            self._update_aperture_widget()
            return
        self.lens_focal.setRange(p["fmin"], p["fmax"])
        if p["fmin"] == p["fmax"]:          # prime lens
            self.lens_focal.setValue(p["fmin"])
            self.lens_focal.setEnabled(False)
        else:                               # zoom: adjustable focal length
            self.lens_focal.setEnabled(True)
            if not (p["fmin"] <= self.lens_focal.value() <= p["fmax"]):
                self.lens_focal.setValue(int((p["fmin"] + p["fmax"]) / 2))
        self._refresh_aperture_menu()
        self._update_aperture_widget()
        self._propagate_focal()

    def _update_aperture_widget(self):
        """A telescope's aperture is fixed by the glass (with or without reducer or
        Barlow): show its focal ratio as plain text instead of a menu."""
        p = LENS_PROFILES.get(self.lens_combo.currentText()) if hasattr(self, "lens_combo") else None
        scope = bool(p and p.get("scope"))
        if hasattr(self, "aper_fixed"):
            self.aper_fixed.setText("f/{:g}".format(p["amax"]) if scope else "")
            self.aper_fixed.setVisible(scope)
            self.cb_aper.setVisible(not scope)

    def _lens_amax(self):
        p = LENS_PROFILES.get(self.lens_combo.currentText())
        return p["amax"] if p else 1.0

    @staticmethod
    def _fval(s):
        try:
            return float(s.replace("f/", "").replace(",", "."))
        except (ValueError, AttributeError):
            return None

    def _refresh_aperture_menu(self):
        """Adapts aperture menu to the lens: f-stops >= max aperture."""
        if not hasattr(self, "cb_aper"):
            return
        amax = self._lens_amax()
        raw = self._cam_aper_choices or [
            "f/1.4", "f/1.8", "f/2", "f/2.5", "f/2.8", "f/3.5", "f/4", "f/4.5",
            "f/5.6", "f/6.3", "f/8", "f/11", "f/13", "f/16", "f/22"]
        filt = [s for s in raw if (self._fval(s) is None or self._fval(s) + 1e-6 >= amax)]
        if not filt:
            filt = raw
        cur = self.cb_aper.currentText()
        self.cb_aper.blockSignals(True)
        self.cb_aper.clear(); self.cb_aper.addItems(filt)
        self.cb_aper.setCurrentText(cur) if cur in filt else self.cb_aper.setCurrentIndex(0)
        self.cb_aper.setEnabled(True)
        self.cb_aper.blockSignals(False)

    def _propagate_focal(self, *args):
        f = self.lens_focal.value()
        for attr in ("solve_focal", "t_focal", "sky_focal"):
            sp = getattr(self, attr, None)
            if sp is not None:
                sp.blockSignals(True); sp.setValue(f); sp.blockSignals(False)
        if getattr(self, "skymap", None):
            self.skymap.set_focal(f)

    def _hline(self):
        ln = QtWidgets.QFrame(); ln.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        ln.setStyleSheet("color:#333;")
        return ln

    def _update_mosaic(self, *args):
        f = self.lens_focal.value() if hasattr(self, "lens_focal") else 50.0
        fw, fh = fov_deg(f)
        ov = self.mos_ov.value() / 100.0
        step_w = fw * (1 - ov); step_h = fh * (1 - ov)
        cols = max(1, math.ceil((self.mos_w.value() - fw) / max(step_w, 1e-3)) + 1)
        rows = max(1, math.ceil((self.mos_h.value() - fh) / max(step_h, 1e-3)) + 1)
        pct_w = step_w / max(fw, 1e-6) * 100.0
        pct_h = step_h / max(fh, 1e-6) * 100.0
        self.mos_out.setText(
            "Field ({:.0f} mm): {} \u2192 grid {}×{} = {} panels\n"
            "Offset between centers: {:.2f}° (X, {:.0f}%) · {:.2f}° (Y, {:.0f}%)".format(
                f, fov_str(f), cols, rows, cols * rows,
                step_w, pct_w, step_h, pct_h))

    # --- Polar Alignment (Drift) -----------------------------------------
    def _toggle_drift(self):
        self._drift_active = not getattr(self, "_drift_active", False)
        if self._drift_active:
            self._drift_ref_gray = None
            self._drift_t0 = time.time()
            self.drift_btn.setText("⏸ Stop measurement")
            self.drift_out.setText("Measurement in progress… keep Live view active.")
        else:
            self.drift_btn.setText("▶ Start measurement")

    def _reset_drift_ref(self):
        self._drift_ref_gray = None
        self._drift_t0 = time.time()
        self.drift_out.setText("Reference reset.")

    def _update_drift(self, gray):
        ref = getattr(self, "_drift_ref_gray", None)
        if ref is None:
            self._drift_ref_gray = gray
            self._drift_t0 = time.time()
            return
        h, w = ref.shape[:2]
        ms = 700.0; sc = ms / max(h, w) if max(h, w) > ms else 1.0
        r = cv2.resize(ref, None, fx=sc, fy=sc) if sc != 1.0 else ref
        c = cv2.resize(gray, None, fx=sc, fy=sc) if sc != 1.0 else gray
        try:
            (dx, dy), _ = cv2.phaseCorrelate(r.astype(np.float32), c.astype(np.float32))
        except Exception:             # noqa: BLE001
            return
        dx, dy = dx / sc, dy / sc
        drift_px = (dx * dx + dy * dy) ** 0.5
        dt = max(1e-3, (time.time() - self._drift_t0) / 60.0)   # minutes
        fw, _fh = fov_deg(self._focal())
        arcmin = drift_px / w * fw * 60.0
        rate = arcmin / dt
        horiz = "right" if dx > 0 else "left"
        vert = "down" if dy > 0 else "up"
        self.drift_out.setText(
            "Drift {:.1f}′ in {:.1f} min  ·  {:.2f}′/min\n"
            "Direction: to the {} and {}.\n"
            "Aim at the relevant axis (Az/Alt) and adjust to cancel the drift.".format(
                arcmin, dt, rate, horiz, vert))

    def _notify(self, title, message):
        if not getattr(self, "chk_notify", None) or not self.chk_notify.isChecked():
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["osascript", "-e",
                                  'display notification "{}" with title "{}" sound name "Glass"'.format(
                                      message.replace('"', "'"), title.replace('"', "'"))])
                return
            if sys.platform.startswith("linux") and shutil.which("notify-send"):
                subprocess.Popen(["notify-send", title, message])
                return
        except Exception:             # noqa: BLE001
            pass
        try:                          # Windows, or anything without a notifier: use the tray
            tray = getattr(self, "_tray", None)
            if tray is None:
                tray = self._tray = QtWidgets.QSystemTrayIcon(self.windowIcon(), self)
                tray.show()
            tray.showMessage(title, message, QtWidgets.QSystemTrayIcon.MessageIcon.Information, 6000)
        except Exception:             # noqa: BLE001
            self.statusBar().showMessage("{} — {}".format(title, message), 8000)

    def _open_session_folder(self):
        import subprocess
        d = self.dir_edit.text()
        try:
            os.makedirs(d, exist_ok=True)
            if sys.platform == "darwin":
                subprocess.Popen(["open", d])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", d])
            else:
                os.startfile(d)         # noqa: B606  (Windows)
        except Exception as e:          # noqa: BLE001
            self.statusBar().showMessage("Could not open: {}".format(e), 5000)

    def _prep_calibration(self, kind):
        labels = {
            "darks": "Darks: lens covered, SAME exposure time & ISO as lights, ~20–30.",
            "flats": "Flats: uniform background (twilight sky / white screen), histogram at ~50%, ~20–30.",
            "bias": "Bias: lens covered, SHORTEST possible exposure, same ISO, ~50.",
        }
        self.type_combo.setCurrentText(kind)
        self._sections["cap"].set_expanded(True)      # the folder/type row is in Capture
        self.statusBar().showMessage(
            "Calibration — " + labels[kind] + "  (exposure settings kept from your lights.)",
            12000)

    def _refresh_statusbar(self):
        """One line, always in view: camera, mount, what has been gathered, and when
        the target goes behind the horizon (the local one, trees and all, if set)."""
        if not hasattr(self, "status_perm"):
            return
        parts = [p for p in (self._sb.get("batt"), self._sb.get("mount"),
                             self._sb.get("integ"), self._sb.get("hfr"),
                             self._sb.get("track"), self._sb.get("sky")) if p]
        self.status_perm.setText("    │    ".join(parts))

    def _update_sky_window(self):
        """When the session target disappears behind the horizon — or reappears.
        Uses the local horizon (trees, houses) when one is active."""
        goal = getattr(self, "_session_goal", None)
        if not goal or not hasattr(self, "t_lat"):
            self._sb.pop("sky", None); self._refresh_statusbar(); return
        name, ra, dec = goal
        try:
            import manual_goto as mg
            from datetime import datetime, timedelta, timezone
            key = (name or "").strip().lower()
            now = datetime.now(timezone.utc)
            prof = self._active_horizon() if hasattr(self, "hz_combo") else None
            lat, lon = self.t_lat.value(), self.t_lon.value()

            track = None
            if key in _BODY_EN:                         # Moon/planets move: hourly, then
                track = []                              # interpolated (ephemeris is slow)
                for hh in range(0, 18):
                    rd = mg.body_radec(_BODY_EN[key].lower(), now + timedelta(hours=hh), lat, lon)
                    if rd:
                        track.append((hh * 60.0, rd[0], rd[1]))

            def visible(t):
                r, d = ra, dec
                if track:
                    mins = (t - now).total_seconds() / 60.0
                    xs = [p[0] for p in track]
                    ras = np.unwrap(np.radians([p[1] for p in track]))
                    r = float(np.degrees(np.interp(mins, xs, ras))) % 360.0
                    d = float(np.interp(mins, xs, [p[2] for p in track]))
                alt, az = mg.radec_to_altaz(mg.lst_deg(t, lon) - r, d, lat)
                return alt > max(0.0, float(np.interp(az % 360.0, np.arange(0.0, 362.0, 2.0),
                                                      list(prof) + [prof[0]])) if prof else 0.0)
            now_vis = visible(now)
            when = None
            for k in range(1, 12 * 16 + 1):             # 16 h ahead, 5 min steps
                t = now + timedelta(minutes=5 * k)
                if visible(t) != now_vis:
                    when = t; break
            behind = "behind your horizon" if prof else "below the horizon"
            short = re.sub(r"\s*\(.*\)$", "", name) if name else "Target"
            if when is None:
                txt = "{} {} all night".format(short, "visible" if now_vis else behind)
            else:
                hhmm = when.astimezone().strftime("%H:%M")
                txt = ("{} goes {} at {}".format(short, behind, hhmm) if now_vis
                       else "{} rises above your horizon at {}".format(short, hhmm) if prof
                       else "{} rises at {}".format(short, hhmm))
            self._sb["sky"] = ("🌳 " if prof else "🌅 ") + txt
        except Exception:               # noqa: BLE001
            self._sb.pop("sky", None)
        self._refresh_statusbar()


    def _update_total_integ(self, *args):
        total_s = sum(self._seq_integs.values())
        total_n = sum(self._seq_counts.values())
        snr = math.sqrt(max(total_n, 1))                   # noise ↓ as √N -> SNR gain ×√N
        txt = "{} · {} shots".format(_fmt_dur(total_s), total_n)
        self.integ_label.setToolTip("Total integration {} · {} shots · SNR ×{:.1f}".format(
            _fmt_dur(total_s), total_n, snr))
        if self.chk_integ_target.isChecked():
            target_s = self.integ_target.value() * 60.0
            remain = max(target_s - total_s, 0.0)
            txt = "{} / {}".format(_fmt_dur(total_s), _fmt_dur(target_s))
            # auto stop when target reached (during a series)
            if self.review_mode and total_s >= target_s and total_s > 0:
                self.worker.post("stop_interval")
                self.statusBar().showMessage("🎯 Integration target reached.", 8000)
        self.integ_label.setText(txt)
        self._sb["integ"] = "Integ {}".format(_fmt_dur(total_s))
        self._refresh_statusbar()

    def _set_cell(self, row, col, text):
        self.seq_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(text)))

    @Slot(int, str, str)
    def on_seq_started(self, seq_id, seq_type, mode):
        # A new series starts its own count: the integration shown is the one you are
        # gathering now, not everything since the app was launched. Pausing and
        # resuming keeps it, because that does not start a new sequence.
        self._seq_counts.clear(); self._seq_integs.clear()
        self._update_total_integ()
        row = self.seq_table.rowCount()
        self.seq_table.insertRow(row)
        self._seq_rows[seq_id] = row
        self._seq_counts[seq_id] = 0
        self._seq_integs[seq_id] = 0.0
        for col, val in enumerate([seq_id, seq_type, mode, 0, "0:00"]):
            self._set_cell(row, col, val)
        self.seq_table.scrollToBottom()
        if seq_type == "lights":
            iso = self.cb_iso.currentText() or "?"
            self._cal_hint = ("Lights: exposure '{}', ISO {} — reproduce these for the "
                              "darks.".format(mode, iso))

    @Slot(int, int, float)
    def on_seq_progress(self, seq_id, count, integ):
        self._seq_counts[seq_id] = count
        self._seq_integs[seq_id] = integ
        row = self._seq_rows.get(seq_id)
        if row is not None:
            self._set_cell(row, 3, count)
            self._set_cell(row, 4, _fmt_dur(integ))
        self._update_total_integ()

    @Slot(int, int, float, float)
    def on_seq_ended(self, seq_id, count, integ, dur):
        self._seq_counts[seq_id] = count
        self._seq_integs[seq_id] = integ
        row = self._seq_rows.get(seq_id)
        if row is not None:
            self._set_cell(row, 3, count)
            self._set_cell(row, 4, _fmt_dur(integ))
        self._update_total_integ()

    @Slot(str)
    def on_saved(self, path):
        self._last_capture_path = path
        self.statusBar().showMessage("✓ {}".format(path), 8000)

    @Slot(dict)
    def on_config(self, cfg):
        for key, combo in (("iso", self.cb_iso), ("shutterspeed", self.cb_shutter)):
            choices, current = cfg.get(key, ([], ""))
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(choices)
            if current in choices:
                combo.setCurrentText(current)
            combo.setEnabled(bool(choices))
            combo.blockSignals(False)
        # aperture: camera list filtered according to selected lens
        choices, current = cfg.get("f-number", ([], ""))
        self._cam_aper_choices = choices
        self._refresh_aperture_menu()
        if current and current in [self.cb_aper.itemText(i) for i in range(self.cb_aper.count())]:
            self.cb_aper.setCurrentText(current)

    @Slot(str)
    def on_failed(self, msg):
        QtWidgets.QMessageBox.critical(self, "Camera", msg)
        self.statusBar().showMessage("Connection failed.", 0)

    def closeEvent(self, ev):
        self._shutting_down = True
        pv = getattr(self, "_public", None)
        if pv is not None:
            pv.close()
        self._stop_caffeinate()        # let the Mac sleep/lock again
        try:
            self._save_settings()
        except Exception:           # noqa: BLE001
            pass
        try:
            if getattr(self, "_bg_timer", None) is not None:
                self._bg_timer.stop()
            if getattr(self, "_iss_timer", None) is not None:
                self._iss_timer.stop()
        except Exception:           # noqa: BLE001
            pass
        # Gather EVERY background QThread we may have spawned (camera worker, survey/
        # zone downloads, plate-solving, geolocation…). A QThread still running at
        # teardown makes Qt abort the process (exit code 134 = SIGABRT) — which is
        # exactly what happened when closing during a zone download.
        threads = []
        for attr in ("worker", "_preload", "_lightsky", "_bg_loader",
                     "_astro", "_solve", "_geo", "_dso_loader", "_folder_stack", "_gx_worker",
                     "_hdmi", "_obs_worker"):
            t = getattr(self, attr, None)
            if isinstance(t, QtCore.QThread):
                threads.append(t)
        threads += [t for t in getattr(self, "_threads", [])
                    if isinstance(t, QtCore.QThread)]
        seen = set(); uniq = []
        for t in threads:
            if id(t) not in seen:
                seen.add(id(t)); uniq.append(t)
        # ask them all to stop first (non-blocking)
        for t in uniq:
            try:
                if hasattr(t, "stop"):
                    t.stop()
                t.requestInterruption()
            except Exception:       # noqa: BLE001
                pass
        # wait up to a shared budget, then force any straggler (network fetch stuck
        # on a timeout) so Qt never destroys a live thread
        import time as _t
        end = _t.monotonic() + 5.0
        for t in uniq:
            try:
                rem = max(1, int((end - _t.monotonic()) * 1000))
                t.wait(rem)
            except Exception:       # noqa: BLE001
                pass
        for t in uniq:
            try:
                if t.isRunning():
                    t.terminate(); t.wait(1500)
            except Exception:       # noqa: BLE001
                pass
        try:
            GPhoto2Backend.kill_any_hammer(); GPhoto2Backend.resume_daemon()
        except Exception:           # noqa: BLE001
            pass
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    # macOS passe "-psn_..." aux .app : on le retire pour ne pas faire échouer argparse
    cli = [a for a in sys.argv[1:] if not a.startswith("-psn_")]
    ap = argparse.ArgumentParser(description="Tether + focus for Sony A7II")
    ap.add_argument("--simulate", action="store_true",
                    help="Uses a simulated camera (no hardware required).")
    ap.add_argument("--save-dir", default="~/SonyTether",
                    help="Folder to save photos.")
    ap.add_argument("--metric", default="tenengrad", choices=list(METRICS),
                    help="Initial sharpness metric.")
    ap.add_argument("--roi", type=float, default=0.5,
                    help="Central fraction analyzed (0.1–1.0).")
    args, _unknown = ap.parse_known_args(cli)

    save_dir = os.path.expanduser(args.save_dir)
    factory = SimBackend if args.simulate else GPhoto2Backend

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("NOUT")
    app.setApplicationDisplayName("NOUT")
    app.setOrganizationName("NOUT")
    # Avoid astropy trying to download IERS earth-orientation data (network + a
    # root-owned ~/.cache/astropy from a past sudo run cause permission errors).
    # The bundled IERS-B table is plenty accurate for a planetarium.
    try:
        from astropy.utils import iers
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
    except Exception:             # noqa: BLE001
        pass
    try:
        import warnings as _w
        from astropy.utils.exceptions import AstropyWarning
        _w.simplefilter("ignore", AstropyWarning)
    except Exception:             # noqa: BLE001
        pass
    _logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nout_1024.png")
    if os.path.exists(_logo):
        app.setWindowIcon(QtGui.QIcon(_logo))
    worker = CameraWorker(factory, metric=args.metric, roi_frac=args.roi)
    win = MainWindow(worker, save_dir, args.roi)

    def _cleanup():
        try:
            GPhoto2Backend.kill_any_hammer(); GPhoto2Backend.resume_daemon()
        except Exception:             # noqa: BLE001
            pass
    app.aboutToQuit.connect(_cleanup)

    win.showMaximized()          # a session screen: use the whole display
    worker.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()