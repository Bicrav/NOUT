#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sky_map.py — Planetarium widget for NOUT.

Two modes painted natively with QPainter (offline, no web):
  * "framing"  : gnomonic (tangent-plane) map centered on a target, equatorial-
                 aligned. Shows stars, constellation lines, DSO at their true
                 angular size, the camera FIELD rectangle for the current focal,
                 and an optional MOSAIC grid (with overlap). Pan = drag, zoom = wheel.
  * "allsky"   : azimuthal (alt-az dome) for the whole sky at a given instant,
                 with horizon, cardinal points, Moon/planets and a time cursor.

Data files expected next to this module (shipped with the app):
  sky_stars.csv        ra_deg,dec_deg,mag,bv      (~5000 stars, mag<=6.5)
  sky_constlines.json  list of polylines [[ra_deg,dec_deg], ...]
  catalog_ngc.csv      id,name,type,ra_hours,dec_deg,mag,size_arcmin  (DSO)
"""
import csv
import json
import math
import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

_HERE = os.path.dirname(os.path.abspath(__file__))
D2R = math.pi / 180.0
R2D = 180.0 / math.pi


# --------------------------------------------------------------------------- data
def _load_stars(path):
    ra, dec, mag, bv = [], [], [], []
    if not os.path.exists(path):
        return (np.array([]),) * 4
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ra.append(float(row["ra_deg"])); dec.append(float(row["dec_deg"]))
                mag.append(float(row["mag"])); bv.append(float(row.get("bv", 0) or 0))
            except (ValueError, KeyError):
                continue
    return np.array(ra), np.array(dec), np.array(mag), np.array(bv)


def _load_starnames(path):
    """Named bright stars: list of dicts {name, ra, dec, mag}."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append({"name": row["name"], "ra": float(row["ra_deg"]),
                            "dec": float(row["dec_deg"]), "mag": float(row.get("mag", 3) or 3)})
            except (ValueError, KeyError):
                continue
    return out


def _load_constlines(path):
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path))
    except Exception:           # noqa: BLE001
        return []


def _load_constnames(path):
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path))      # [[ra_deg, dec_deg, name], ...]
    except Exception:           # noqa: BLE001
        return []


# Famous targets absent from NGC/IC (dark/molecular nebulae, etc.)
_EXTRA_DSO = [
    {"id": "LDN1235", "name": "Shark Nebula", "type": "Dark Nebula",
     "ra": 333.25, "dec": 73.33, "mag": float("nan"), "size": 90.0},
    {"id": "Ou4", "name": "Squid Nebula", "type": "Emission Nebula",
     "ra": 317.88, "dec": 59.95, "mag": float("nan"), "size": 60.0},
    {"id": "vdB152", "name": "Wolf's Cave", "type": "Reflection Nebula",
     "ra": 327.30, "dec": 69.95, "mag": float("nan"), "size": 30.0},
    {"id": "B33", "name": "Horsehead Nebula", "type": "Dark Nebula",
     "ra": 85.246, "dec": -2.458, "mag": float("nan"), "size": 8.0},
    {"id": "LBN552", "name": "Ghost of Cepheus (vdB141)", "type": "Reflection Nebula",
     "ra": 328.90, "dec": 68.17, "mag": float("nan"), "size": 20.0},
]


def _load_ldn(path):
    """Optional Lynds Dark Nebulae (or any) catalogue: CSV with columns
    id, name, ra_deg, dec_deg, size_arcmin. Drop the full Lynds catalog here to get them
    all on the map and in the preview overlay."""
    out = []
    if not os.path.exists(path):
        return out
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ra = float(row["ra_deg"]); dec = float(row["dec_deg"])
                except (ValueError, KeyError):
                    continue
                out.append({"id": row.get("id") or "LDN", "type": row.get("type") or "Dark Nebula",
                            "name": row.get("name") or row.get("id") or "LDN",
                            "ra": ra, "dec": dec, "mag": float("nan"),
                            "size": float(row["size_arcmin"]) if row.get("size_arcmin") else 20.0})
    except Exception:               # noqa: BLE001
        pass
    return out


def _load_search_catalog(path):
    """Full DSO list (id, name, name_fr, ra, dec) for the search box. Merges a
    curated common-name list (English + French) from dso_names.csv beside it."""
    out = []
    by_id = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ra = float(row["ra_hours"]) * 15.0; dec = float(row["dec_deg"])
                except (ValueError, KeyError):
                    continue
                o = {"id": row["id"], "name": row.get("name") or row["id"],
                     "name_fr": "", "ra": ra, "dec": dec}
                out.append(o); by_id[row["id"].upper().replace(" ", "")] = o
    npath = os.path.join(os.path.dirname(path), "dso_names.csv")
    if os.path.exists(npath):
        with open(npath, newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("id") or "").upper().replace(" ", "")
                name = (row.get("name") or "").strip()
                name_fr = (row.get("name_fr") or "").strip()
                exist = by_id.get(key)
                if exist is not None:
                    if name:
                        exist["name"] = name
                    exist["name_fr"] = name_fr
                else:
                    try:
                        ra = float(row["ra_deg"]); dec = float(row["dec_deg"])
                    except (ValueError, KeyError):
                        continue
                    o = {"id": row.get("id") or name, "name": name or row.get("id"),
                         "name_fr": name_fr, "ra": ra, "dec": dec}
                    out.append(o); by_id[key] = o
    return out


def _load_dso(path, max_mag=10.5, min_size=6.0):
    """DSO for display: keep bright OR large objects (caps the count for speed)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ra = float(row["ra_hours"]) * 15.0
                dec = float(row["dec_deg"])
            except (ValueError, KeyError):
                continue
            mag = float(row["mag"]) if row.get("mag") else float("nan")
            size = float(row["size_arcmin"]) if row.get("size_arcmin") else 0.0
            keep = (not math.isnan(mag) and mag <= max_mag) or size >= min_size
            if not keep:
                continue
            out.append({"id": row["id"], "name": row.get("name") or row["id"],
                        "type": row.get("type") or "", "ra": ra, "dec": dec,
                        "mag": mag, "size": size})
    return out


# ----------------------------------------------------------------- astro helpers
def _az_name(az):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((az % 360) / 22.5 + 0.5) % 16]


def fov_deg(focal_mm, sensor_w=35.8, sensor_h=23.9):
    fw = 2 * math.degrees(math.atan(sensor_w / (2 * max(focal_mm, 1))))
    fh = 2 * math.degrees(math.atan(sensor_h / (2 * max(focal_mm, 1))))
    return fw, fh


def _jd_utc(dt):
    y, m, d = dt.year, dt.month, dt.day
    frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    if m <= 2:
        y -= 1; m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5 + frac


def lst_deg(dt_utc, lon_east_deg):
    """Local apparent sidereal time (deg) — good enough for a planetarium."""
    t = _jd_utc(dt_utc) - 2451545.0
    gmst = 280.46061837 + 360.98564736629 * t
    return (gmst + lon_east_deg) % 360.0


_GAUSS_K = 0.01720209895          # Gaussian gravitational constant (rad/day)


def _earth_helio_ecl(jd):
    """Earth heliocentric ecliptic rectangular position (AU), low precision."""
    n = jd - 2451545.0
    g = math.radians((357.529 + 0.98560028 * n) % 360.0)
    L = (280.459 + 0.98564736 * n) % 360.0
    lam = math.radians((L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360.0)
    r = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)   # AU (Sun-Earth)
    # Earth is opposite the Sun as seen from the Sun
    return (-r * math.cos(lam), -r * math.sin(lam), 0.0)


def comet_state(el, jd):
    """Two-body state of a comet at JD: returns (ra_deg, dec_deg, r_au, delta_au)
    where r = heliocentric distance and delta = geocentric distance."""
    q = el["q"]; e = el["e"]
    i = math.radians(el["i"]); node = math.radians(el["node"]); w = math.radians(el["peri"])
    dt = jd - el["tp"]
    if e < 1.0:                                   # elliptical
        a = q / (1.0 - e)
        nmean = _GAUSS_K / (a ** 1.5)
        M = nmean * dt
        E = M if e < 0.8 else math.pi
        for _ in range(60):
            dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
            E -= dE
            if abs(dE) < 1e-10:
                break
        xv = a * (math.cos(E) - e)
        yv = a * (math.sqrt(1.0 - e * e) * math.sin(E))
    else:                                         # near-parabolic (Barker's equation)
        W = 3.0 * _GAUSS_K * dt / math.sqrt(2.0 * q ** 3)
        s = W / 3.0
        for _ in range(60):
            s2 = s * s
            f = s * (s2 + 3.0) - W
            df = 3.0 * (s2 + 1.0)
            ds = f / df
            s -= ds
            if abs(ds) < 1e-10:
                break
        nu = 2.0 * math.atan(s)
        r = q * (1.0 + s * s)
        xv = r * math.cos(nu); yv = r * math.sin(nu)
    cw, sw = math.cos(w), math.sin(w)
    cn, sn = math.cos(node), math.sin(node)
    ci, si = math.cos(i), math.sin(i)
    Px = cw * cn - sw * sn * ci; Qx = -sw * cn - cw * sn * ci
    Py = cw * sn + sw * cn * ci; Qy = -sw * sn + cw * cn * ci
    Pz = sw * si;                Qz = cw * si
    xh = Px * xv + Qx * yv; yh = Py * xv + Qy * yv; zh = Pz * xv + Qz * yv
    r_au = math.sqrt(xh * xh + yh * yh + zh * zh)
    ex, ey, ez = _earth_helio_ecl(jd)
    xg, yg, zg = xh - ex, yh - ey, zh - ez            # geocentric ecliptic
    delta = math.sqrt(xg * xg + yg * yg + zg * zg)
    eps = math.radians(23.43928)
    xe = xg
    ye = yg * math.cos(eps) - zg * math.sin(eps)
    ze = yg * math.sin(eps) + zg * math.cos(eps)
    ra = math.degrees(math.atan2(ye, xe)) % 360.0
    dec = math.degrees(math.atan2(ze, math.sqrt(xe * xe + ye * ye)))
    return ra, dec, r_au, delta


def comet_apparent_mag(H, G, r_au, delta_au):
    """MPC total magnitude: m = H + 5 log10(delta) + 2.5*G*log10(r)."""
    if r_au <= 0 or delta_au <= 0:
        return None
    return H + 5.0 * math.log10(delta_au) + 2.5 * G * math.log10(r_au)


def comet_radec(el, jd):
    """Geocentric RA/Dec (deg) of a comet (two-body model)."""
    ra, dec, _r, _d = comet_state(el, jd)
    return ra, dec


def comets_user_path():
    """User-writable comets file (takes precedence over the bundled example)."""
    base = os.path.expanduser("~/Library/Application Support/NOUT")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:                 # noqa: BLE001
        pass
    return os.path.join(base, "comets.csv")


def _comets_path():
    up = comets_user_path()
    if os.path.exists(up):
        return up
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "comets.csv")


def _load_comets(path):
    """Read comet orbital elements from a CSV the user can edit/update from the MPC.
    Columns: name,q_au,e,i_deg,node_deg,peri_deg,tp (ISO date or JD)[,epoch]."""
    out = []
    if not os.path.exists(path):
        return out
    import datetime as _dt
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                tp_raw = (row.get("tp") or "").strip()
                if not tp_raw:
                    continue
                if any(c in tp_raw for c in "-/:") and len(tp_raw) >= 8:
                    s = tp_raw.replace("/", "-").replace("T", " ")
                    parts = s.split(" ")
                    ymd = parts[0].split("-")
                    yy, mm = int(ymd[0]), int(ymd[1])
                    dd = float(ymd[2]) if len(ymd) > 2 else 1.0
                    hh = 0.0
                    if len(parts) > 1 and ":" in parts[1]:
                        t = parts[1].split(":")
                        hh = int(t[0]) + int(t[1]) / 60.0 + (float(t[2]) if len(t) > 2 else 0) / 3600.0
                    tp = _jd_utc(_dt.datetime(yy, mm, int(dd))) + (dd - int(dd)) + hh / 24.0
                else:
                    tp = float(tp_raw)
                out.append({"name": row["name"].strip(),
                            "q": float(row["q_au"]), "e": float(row["e"]),
                            "i": float(row["i_deg"]), "node": float(row["node_deg"]),
                            "peri": float(row["peri_deg"]), "tp": tp,
                            "mag": (float(row["mag"]) if (row.get("mag") or "").strip()
                                    else None)})
            except (ValueError, KeyError):
                continue
    return out


def radec_to_altaz(ra_deg, dec_deg, lst_d, lat_deg):
    ha = np.radians((lst_d - ra_deg) % 360.0)
    dec = np.radians(dec_deg); lat = math.radians(lat_deg)
    sinalt = np.sin(dec) * math.sin(lat) + np.cos(dec) * math.cos(lat) * np.cos(ha)
    alt = np.degrees(np.arcsin(np.clip(sinalt, -1, 1)))
    cosA = (np.sin(dec) - np.sin(np.radians(alt)) * math.sin(lat)) / (
        np.cos(np.radians(alt)) * math.cos(lat) + 1e-9)
    A = np.degrees(np.arccos(np.clip(cosA, -1, 1)))
    az = np.where(np.sin(ha) > 0, 360.0 - A, A)
    return alt, az


def gnomonic(ra_deg, dec_deg, ra0, dec0):
    """Tangent-plane projection. Returns (x east, y north) in radians + front mask."""
    ra = np.radians(ra_deg); dec = np.radians(dec_deg)
    ra0r = math.radians(ra0); dec0r = math.radians(dec0)
    cosc = (math.sin(dec0r) * np.sin(dec) +
            math.cos(dec0r) * np.cos(dec) * np.cos(ra - ra0r))
    cosc = np.where(np.abs(cosc) < 1e-6, 1e-6, cosc)
    x = np.cos(dec) * np.sin(ra - ra0r) / cosc
    y = (math.cos(dec0r) * np.sin(dec) -
         math.sin(dec0r) * np.cos(dec) * np.cos(ra - ra0r)) / cosc
    return x, y, cosc > 0


def inv_gnomonic(x, y, ra0, dec0):
    rho = math.hypot(x, y)
    if rho < 1e-12:
        return ra0 % 360.0, dec0
    c = math.atan(rho)
    dec0r = math.radians(dec0); ra0r = math.radians(ra0)
    dec = math.asin(math.cos(c) * math.sin(dec0r) + y * math.sin(c) * math.cos(dec0r) / rho)
    ra = ra0r + math.atan2(x * math.sin(c),
                           rho * math.cos(dec0r) * math.cos(c) - y * math.sin(dec0r) * math.sin(c))
    return (math.degrees(ra)) % 360.0, math.degrees(dec)


def _bv_color(bv):
    """Approximate star colour from B-V index."""
    if bv < 0.0:
        return QtGui.QColor(170, 190, 255)
    if bv < 0.3:
        return QtGui.QColor(210, 220, 255)
    if bv < 0.6:
        return QtGui.QColor(255, 255, 245)
    if bv < 1.0:
        return QtGui.QColor(255, 240, 200)
    return QtGui.QColor(255, 210, 170)


_TYPE_COLOR = {
    "Galaxy": QtGui.QColor(255, 130, 130), "Galaxy Pair": QtGui.QColor(255, 130, 130),
    "Galaxy Triplet": QtGui.QColor(255, 130, 130), "Galaxy Group": QtGui.QColor(255, 130, 130),
    "Globular Cluster": QtGui.QColor(255, 220, 120), "Open Cluster": QtGui.QColor(255, 235, 150),
    "Planetary Nebula": QtGui.QColor(120, 230, 230), "Nebula": QtGui.QColor(140, 235, 160),
    "HII Region": QtGui.QColor(140, 235, 160), "Emission Nebula": QtGui.QColor(140, 235, 160),
    "Reflection Nebula": QtGui.QColor(150, 200, 255), "Supernova Remnant": QtGui.QColor(200, 160, 255),
    "Cluster + Nebula": QtGui.QColor(180, 235, 170),
}


def _dso_color(typ):
    return _TYPE_COLOR.get(typ, QtGui.QColor(200, 200, 210))


# ------------------------------------------------------------------- the widget
class SkyMap(QtWidgets.QWidget):
    targetPicked = Signal(str, float, float)      # name, ra_deg, dec_deg
    viewChanged = Signal()                        # framing center/zoom changed (for bg refetch)

    def __init__(self, data_dir=None):
        super().__init__()
        d = data_dir or _HERE
        self.star_ra, self.star_dec, self.star_mag, self.star_bv = _load_stars(
            os.path.join(d, "sky_stars.csv"))
        self.constlines = _load_constlines(os.path.join(d, "sky_constlines.json"))
        self.constnames = _load_constnames(os.path.join(d, "sky_constnames.json"))
        self.starnames = _load_starnames(os.path.join(d, "sky_starnames.csv"))
        self.show_starnames = True
        self.show_trajectory = True   # dashed diurnal path of the target (all-sky)
        self.comets = _load_comets(_comets_path())
        self._comet_pos = []          # [(name, ra, dec)] for the current time
        self._iss_tle = None          # ISS/satellite TLE text (set by the app)
        self._iss_pos = None          # (ra, dec, alt) topocentric at current time
        self.show_iss = False
        self.show_comets = True
        self.comet_mag_limit = None   # if set, only show comets brighter than this
        self.dso = _load_dso(os.path.join(d, "catalog_ngc.csv")) + list(_EXTRA_DSO) \
            + _load_ldn(os.path.join(d, "catalog_ldn.csv"))
        self._search_dso = _load_search_catalog(os.path.join(d, "catalog_ngc.csv")) \
            + [{"id": x["id"], "name": x["name"], "ra": x["ra"], "dec": x["dec"]}
               for x in _EXTRA_DSO]
        # state
        self.mode = "framing"
        self.center_ra = 83.8         # Orion by default (M42 area)
        self.center_dec = -5.4
        self.target_name = "M42"
        self.view_fov = 25.0          # degrees across the widget (framing)
        self.focal_mm = 135.0
        self.cam_angle = 0.0          # camera rotation (deg), 0 = aligned to RA/Dec
        self.goal_angle = 0.0         # goal-frame rotation (deg), independent of cam_angle
        self.portrait = False         # frame orientation: True = taller than wide (3:4)
        self.show_mosaic = False
        self.mos_rows = 2
        self.mos_cols = 2
        self.mos_overlap = 0.20
        self.show_dso = True
        self.show_constlines = True
        self.show_constnames = True
        self.show_names = True
        # all-sky / horizon view (Stellarium-like)
        self.lat = 43.694
        self.lon = 5.737
        self.dt_utc = None            # datetime in UTC
        self._planets = []            # [(name, ra, dec, color)]
        self.view_az = 180.0          # looking azimuth (deg from N, S=180)
        self.view_alt = 20.0          # looking elevation (deg)
        self.h_fov = 100.0            # horizontal field of view (deg)
        # survey background (framing)
        self.bg_pixmap = None
        self.bg_ra = 0.0; self.bg_dec = 0.0; self.bg_fov = 1.0
        self.show_frame_allsky = False
        # session goal (the object to find) shown alongside the current pointing
        self.goal_ra = 0.0; self.goal_dec = 0.0
        self.goal_name = ""; self.show_goal = False
        # interaction
        self._pan_offset = QtCore.QPointF(0, 0)
        self._drag = None
        self.setMinimumSize(360, 300)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#05060d;")
        self.setFocusPolicy(Qt.StrongFocus)

    # -- configuration -----------------------------------------------------
    def set_target(self, name, ra_deg, dec_deg):
        self.target_name = name
        self.center_ra = ra_deg % 360.0
        self.center_dec = dec_deg
        self.update()

    def set_focal(self, mm):
        self.focal_mm = float(mm); self.update()

    def set_mode(self, mode):
        self.mode = mode
        if mode == "allsky" and self.dt_utc is not None:
            self.aim_at_target()
        self.update()

    def set_mosaic(self, rows, cols, overlap, show):
        self.mos_rows = int(rows); self.mos_cols = int(cols)
        self.mos_overlap = float(overlap); self.show_mosaic = bool(show)
        self.update()

    def set_site(self, lat, lon):
        self.lat = float(lat); self.lon = float(lon); self._recompute_bodies(); self.update()

    def set_time(self, dt_utc):
        self.dt_utc = dt_utc; self._recompute_bodies(); self.update()

    def set_cam_angle(self, deg):
        self.cam_angle = float(deg); self.update()

    def set_portrait(self, on):
        on = bool(on)
        if on != self.portrait:
            self.portrait = on; self.update()

    def set_goal_angle(self, deg):
        self.goal_angle = float(deg); self.update()

    def find_object(self, query):
        """Resolve a name (DSO, named star, constellation, planet/Sun/Moon) to
        (label, ra_deg, dec_deg). Returns None if nothing matches."""
        q = (query or "").strip()
        if not q:
            return None
        import unicodedata

        def norm(s):
            s = (s or "").replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae")
            s = unicodedata.normalize("NFKD", s)
            s = "".join(c for c in s if not unicodedata.combining(c))
            return "".join(ch for ch in s.upper() if ch.isalnum())

        qn = norm(q)
        # planets / Sun / Moon (dynamic positions)
        for nm, ra, dec, _col in self._planets:
            if norm(nm) == qn or norm(nm).startswith(qn):
                return (nm, ra, dec)
        # comets (dynamic positions)
        for nm, ra, dec, _mag in self._comet_pos:
            nn = norm(nm)
            if qn in nn or nn.startswith(qn):
                return (nm, ra, dec)
        # ISS
        if self._iss_pos and qn in ("ISS", "ISSZARYA", "STATION"):
            return ("ISS", self._iss_pos[0], self._iss_pos[1])
        cands = []  # (id, display, ra, dec, name_en, name_fr)
        for d in self._search_dso:
            cands.append((d["id"], d["name"], d["ra"], d["dec"], d["name"], d.get("name_fr", "")))
        for s in self.starnames:
            cands.append((s["name"], s["name"], s["ra"], s["dec"], s["name"], ""))
        for ra, dec, nm in self.constnames:
            cands.append((nm, nm, ra, dec, nm, ""))
        # 1) exact match on id / English name / French name
        for cid, disp, ra, dec, en, fr in cands:
            if norm(cid) == qn or norm(en) == qn or (fr and norm(fr) == qn):
                return (disp or cid, ra, dec)
        # 2) starts with the query
        for cid, disp, ra, dec, en, fr in cands:
            if norm(en).startswith(qn) or norm(cid).startswith(qn) \
                    or (fr and norm(fr).startswith(qn)):
                return (disp or cid, ra, dec)
        # 3) substring on the name (English or French)
        for cid, disp, ra, dec, en, fr in cands:
            if qn in norm(en) or (fr and qn in norm(fr)):
                return (disp or cid, ra, dec)
        return None

    def set_background(self, pixmap, ra, dec, fov):
        self.bg_pixmap = pixmap; self.bg_ra = ra; self.bg_dec = dec; self.bg_fov = fov
        self.update()

    def clear_background(self):
        if self.bg_pixmap is not None:
            self.bg_pixmap = None; self.update()

    def set_frame_allsky(self, on):
        self.show_frame_allsky = bool(on); self.update()

    def set_starnames(self, on):
        self.show_starnames = bool(on); self.update()

    def set_trajectory(self, on):
        self.show_trajectory = bool(on); self.update()

    def set_iss_tle(self, tle_text):
        self._iss_tle = (tle_text or "").strip() or None
        self._recompute_bodies(); self.update()

    def reload_comets(self):
        d = os.path.dirname(os.path.abspath(__file__))
        self.comets = _load_comets(_comets_path())
        self._recompute_bodies(); self.update()

    def set_comet_maglim(self, lim):
        self.comet_mag_limit = lim
        self._recompute_bodies(); self.update()

    def set_show_iss(self, on):
        self.show_iss = bool(on)
        self._recompute_bodies(); self.update()

    def set_show_comets(self, on):
        self.show_comets = bool(on); self.update()

    def refresh_iss(self, dt_now=None):
        """Recompute ONLY the ISS position at real time (for live tracking) and repaint."""
        if not (self.show_iss and self._iss_tle) or self.lat is None:
            return
        import datetime as _dt
        if dt_now is None:
            dt_now = _dt.datetime.utcnow()
        try:
            import satellites
            res = satellites.topocentric_radec(self._iss_tle, self.lat, self.lon, dt_now)
            self._iss_pos = (res[0], res[1], res[2]) if res else None
        except Exception:             # noqa: BLE001
            return
        self.update()

    def _paint_trajectory_allsky(self, p):
        """Dashed diurnal path of the target across the sky (rise → transit → set)."""
        if not self.show_trajectory or self.lat is None:
            return
        lat = math.radians(self.lat); dec = math.radians(self.center_dec)
        ha = np.radians(np.arange(-180.0, 180.5, 2.0))
        sin_alt = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * np.cos(ha)
        sin_alt = np.clip(sin_alt, -1.0, 1.0)
        alt = np.degrees(np.arcsin(sin_alt))
        cos_alt = np.cos(np.arcsin(sin_alt))
        cosA = (math.sin(dec) - sin_alt * math.sin(lat)) / (cos_alt * math.cos(lat) + 1e-9)
        cosA = np.clip(cosA, -1.0, 1.0)
        A = np.degrees(np.arccos(cosA))
        az = np.where(np.sin(ha) > 0, 360.0 - A, A)
        sx, sy, infront = self._project_horizon(alt, az)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 175, 80, 210), 1.7, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        path = QtGui.QPainterPath(); started = False
        for i in range(len(alt)):
            if alt[i] > 0 and infront[i] and np.isfinite(sx[i]) and np.isfinite(sy[i]):
                if started:
                    path.lineTo(float(sx[i]), float(sy[i]))
                else:
                    path.moveTo(float(sx[i]), float(sy[i])); started = True
            else:
                started = False
        p.drawPath(path)
        # transit marker (highest point)
        up = alt > 0
        if np.any(up):
            it = int(np.argmax(np.where(up, alt, -90)))
            if infront[it]:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 205, 120), 1.5))
                p.drawEllipse(QtCore.QPointF(float(sx[it]), float(sy[it])), 3, 3)

    def _starname_mag_limit(self):
        if self.view_fov > 30: return 1.6
        if self.view_fov > 15: return 2.2
        if self.view_fov > 6: return 3.1
        return 6.5

    def _paint_starnames_framing(self, p):
        if not self.starnames:
            return
        lim = self._starname_mag_limit()
        ras = np.array([s["ra"] for s in self.starnames])
        decs = np.array([s["dec"] for s in self.starnames])
        x, y, front = self._project_framing(ras, decs)
        f = QtGui.QFont(); f.setPointSize(8); p.setFont(f)
        p.setPen(QtGui.QColor(170, 205, 255))
        for i, s in enumerate(self.starnames):
            if s["mag"] <= lim and front[i] and np.isfinite(x[i]) and np.isfinite(y[i]) \
                    and 0 <= x[i] <= self.width() and 0 <= y[i] <= self.height():
                p.drawText(QtCore.QPointF(float(x[i]) + 5, float(y[i]) - 4), s["name"])

    def _paint_starnames_allsky(self, p, lst):
        if not self.starnames:
            return
        lim = 2.6 if self.h_fov > 90 else (3.4 if self.h_fov > 45 else 6.5)
        f = QtGui.QFont(); f.setPointSize(8); p.setFont(f)
        p.setPen(QtGui.QColor(170, 205, 255))
        for s in self.starnames:
            if s["mag"] > lim:
                continue
            sx, sy, vis, alt = self._radec_h(s["ra"], s["dec"], lst)
            if vis[0] and alt[0] > 0 and 0 <= sx[0] <= self.width() and 0 <= sy[0] <= self.height():
                p.drawText(QtCore.QPointF(float(sx[0]) + 5, float(sy[0]) - 4), s["name"])

    def set_goal(self, name, ra_deg, dec_deg):
        self.goal_name = name; self.goal_ra = ra_deg % 360.0; self.goal_dec = dec_deg
        self.show_goal = True; self.update()

    def clear_goal(self):
        self.show_goal = False; self.update()

    def _sep_bearing(self, ra1, dec1, ra2, dec2):
        """Angular separation (deg) and compass bearing from point 1 to point 2."""
        r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
        dra = r2 - r1
        sep = math.degrees(math.acos(max(-1.0, min(1.0,
              math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(dra)))))
        y = math.sin(dra) * math.cos(d2)
        x = math.cos(d1) * math.sin(d2) - math.sin(d1) * math.cos(d2) * math.cos(dra)
        pa = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0   # 0=N, 90=E
        names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        comp = names[int((pa + 22.5) // 45) % 8]
        return sep, pa, comp

    def _field_polys_radec(self, cra=None, cdec=None, mosaic=None, angle=None):
        """Camera field (or mosaic panels) as closed RA/Dec polygons, centred on
        (cra, cdec) — defaults to the current pointing — rotated by `angle`
        (defaults to the camera rotation)."""
        cra = self.center_ra if cra is None else cra
        cdec = self.center_dec if cdec is None else cdec
        mosaic = self.show_mosaic if mosaic is None else mosaic
        ang = self.cam_angle if angle is None else angle
        fw, fh = fov_deg(self.focal_mm)
        if self.portrait:
            fw, fh = fh, fw          # 3:4 frame: field is taller than wide
        hw = math.radians(fw / 2.0); hh = math.radians(fh / 2.0)
        th = math.radians(ang); cth, sth = math.cos(th), math.sin(th)

        def corner(ox, oy):
            ex = ox * cth - oy * sth; ey = ox * sth + oy * cth
            return inv_gnomonic(-ex, -ey, cra, cdec)

        polys = []
        if mosaic:
            ov = self.mos_overlap
            stepx = 2 * hw * (1 - ov); stepy = 2 * hh * (1 - ov)
            cols, rows = self.mos_cols, self.mos_rows
            for r in range(rows):
                for c in range(cols):
                    ax = (c - (cols - 1) / 2.0) * stepx
                    ay = (r - (rows - 1) / 2.0) * stepy
                    polys.append([corner(ax - hw, ay - hh), corner(ax + hw, ay - hh),
                                  corner(ax + hw, ay + hh), corner(ax - hw, ay + hh)])
        else:
            polys.append([corner(-hw, -hh), corner(hw, -hh), corner(hw, hh), corner(-hw, hh)])
        return polys

    def _draw_goal_frame_framing(self, p):
        """Draw the camera field placed on the goal, to compare with the current field."""
        if not self.show_goal:
            return
        p.setPen(QtGui.QPen(QtGui.QColor(255, 205, 70, 230), 2, Qt.DashLine))
        p.setBrush(QtGui.QColor(255, 205, 70, 22))
        for poly in self._field_polys_radec(self.goal_ra, self.goal_dec, mosaic=False,
                                            angle=self.goal_angle):
            ras = np.array([q[0] for q in poly]); decs = np.array([q[1] for q in poly])
            xs, ys, front = self._project_framing(ras, decs)
            if np.all(front) and np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
                path = QtGui.QPainterPath(); path.moveTo(float(xs[0]), float(ys[0]))
                for i in range(1, len(poly)):
                    path.lineTo(float(xs[i]), float(ys[i]))
                path.closeSubpath(); p.drawPath(path)

    def set_flags(self, dso=None, constlines=None, names=None, constnames=None):
        if dso is not None:
            self.show_dso = dso
        if constlines is not None:
            self.show_constlines = constlines
        if names is not None:
            self.show_names = names
        if constnames is not None:
            self.show_constnames = constnames
        self.update()

    def _recompute_bodies(self):
        """Sun/Moon/planets at the current instant (optional, needs astropy)."""
        self._planets = []
        if self.dt_utc is None:
            return
        try:
            import astropy.units as u
            from astropy.coordinates import EarthLocation, get_body
            from astropy.time import Time
            loc = EarthLocation(lat=self.lat * u.deg, lon=self.lon * u.deg, height=300 * u.m)
            t = Time(self.dt_utc)
            spec = [("Sun", "sun", QtGui.QColor(255, 215, 90)),
                    ("Moon", "moon", QtGui.QColor(225, 225, 235)),
                    ("Mercury", "mercury", QtGui.QColor(200, 200, 200)),
                    ("Venus", "venus", QtGui.QColor(255, 255, 230)),
                    ("Mars", "mars", QtGui.QColor(255, 140, 110)),
                    ("Jupiter", "jupiter", QtGui.QColor(255, 225, 180)),
                    ("Saturn", "saturn", QtGui.QColor(240, 225, 175)),
                    ("Uranus", "uranus", QtGui.QColor(170, 230, 240)),
                    ("Neptune", "neptune", QtGui.QColor(150, 180, 255))]
            for nm, key, col in spec:
                # geocentric apparent direction (GCRS). Do NOT transform to ICRS:
                # that shifts the origin to the barycentre and ruins nearby bodies (Sun/Moon).
                b = get_body(key, t, loc)
                self._planets.append((nm, float(b.ra.deg), float(b.dec.deg), col))
        except Exception:           # noqa: BLE001
            self._planets = []
        # comets (offline two-body propagation of orbital elements)
        self._comet_pos = []
        if self.comets and self.dt_utc is not None:
            jd = _jd_utc(self.dt_utc)
            for c in self.comets:
                mag = c.get("mag")
                if self.comet_mag_limit is not None and (mag is None
                                                         or mag > self.comet_mag_limit):
                    continue            # show only capturable comets
                try:
                    ra, dec = comet_radec(c, jd)
                    self._comet_pos.append((c["name"], ra, dec, mag))
                except Exception:   # noqa: BLE001
                    continue
        # ISS / satellite (topocentric, current time)
        self._iss_pos = None
        if self.show_iss and self._iss_tle and self.dt_utc is not None:
            try:
                import satellites
                res = satellites.topocentric_radec(self._iss_tle, self.lat, self.lon,
                                                   self.dt_utc)
                if res is not None:
                    ra, dec, alt, _az = res
                    self._iss_pos = (ra, dec, alt)
            except Exception:       # noqa: BLE001
                self._iss_pos = None

    # -- projection helpers ------------------------------------------------
    def _scale_framing(self):
        # px per radian so that view_fov spans the smaller widget dimension
        half = math.radians(self.view_fov / 2.0)
        return (min(self.width(), self.height()) / 2.0) / max(math.tan(half), 1e-6)

    def _project_framing(self, ra, dec):
        # Le CIEL reste nord-en-haut ; c'est le CADRE (rectangle/mosaïque) qui tourne.
        x, y, front = gnomonic(ra, dec, self.center_ra, self.center_dec)
        s = self._scale_framing()
        cx, cy = self.width() / 2.0, self.height() / 2.0
        sx = cx - x * s          # east to the left
        sy = cy - y * s          # north up
        return sx, sy, front

    def _allsky_radius(self):
        return min(self.width(), self.height()) / 2.0 - 32

    def _project_allsky(self, ra, dec):
        if self.dt_utc is None:
            return None, None, np.array([])
        lst = lst_deg(self.dt_utc, self.lon)
        alt, az = radec_to_altaz(np.atleast_1d(ra), np.atleast_1d(dec), lst, self.lat)
        R = self._allsky_radius()
        r = (90.0 - alt) / 90.0 * R
        th = np.radians(az)
        cx = self.width() / 2.0 + self._pan_offset.x()
        cy = self.height() / 2.0 + self._pan_offset.y()
        sx = cx - r * np.sin(th)        # east (az 90) to the left
        sy = cy - r * np.cos(th)        # north (az 0) up
        return sx, sy, alt > 0

    def _horizon_scale(self):
        return (self.width() / 2.0) / max(math.tan(math.radians(self.h_fov) / 4.0), 1e-3)

    def _project_horizon(self, alt_deg, az_deg):
        """Stereographic projection centered on the viewing direction (Stellarium-like).
        Returns sx, sy, infront(bool array)."""
        alt = np.radians(np.atleast_1d(np.asarray(alt_deg, float)))
        az = np.radians(np.atleast_1d(np.asarray(az_deg, float)))
        vx = np.cos(alt) * np.sin(az)      # east
        vy = np.cos(alt) * np.cos(az)      # north
        vz = np.sin(alt)                   # up
        a0 = math.radians(self.view_az); h0 = math.radians(self.view_alt)
        fwd = np.array([math.cos(h0) * math.sin(a0), math.cos(h0) * math.cos(a0), math.sin(h0)])
        right = np.array([fwd[1], -fwd[0], 0.0])       # cross(fwd, world_up=(0,0,1))
        rn = np.hypot(right[0], right[1])
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
        upc = np.array([right[1] * fwd[2],              # cross(right, fwd)
                        -right[0] * fwd[2],
                        right[0] * fwd[1] - right[1] * fwd[0]])
        f = vx * fwd[0] + vy * fwd[1] + vz * fwd[2]
        xs = vx * right[0] + vy * right[1] + vz * right[2]
        ys = vx * upc[0] + vy * upc[1] + vz * upc[2]
        denom = 1.0 + f
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        s = self._horizon_scale()
        sx = self.width() / 2.0 + (xs / denom) * s
        sy = self.height() / 2.0 - (ys / denom) * s
        return sx, sy, f > -0.1

    def aim_at_target(self):
        """Point the horizon view toward the current target (if above horizon)."""
        if self.dt_utc is None:
            return
        lst = lst_deg(self.dt_utc, self.lon)
        alt, az = radec_to_altaz(np.array([self.center_ra]), np.array([self.center_dec]),
                                 lst, self.lat)
        if alt[0] > 0:
            self.view_az = float(az[0])
            self.view_alt = float(np.clip(alt[0], 5, 80))
        else:
            self.view_az = float(az[0]); self.view_alt = 12.0

    # -- painting ----------------------------------------------------------
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), QtGui.QColor(5, 6, 13))
            if self.mode == "framing":
                self._paint_framing(p)
            else:
                self._paint_allsky(p)
        except Exception as exc:        # noqa: BLE001 — never let a draw error crash the app
            try:
                p.fillRect(self.rect(), QtGui.QColor(5, 6, 13))
                p.setPen(QtGui.QColor(200, 120, 120))
                p.drawText(QtCore.QRectF(10, 10, max(self.width() - 20, 50), 60),
                           Qt.TextWordWrap, "Sky render error (try zooming back in):\n{}".format(exc))
            except Exception:           # noqa: BLE001
                pass
        finally:
            p.end()

    def _star_radius(self, mag):
        return max(0.6, (6.8 - mag) * 0.62)

    def _paint_constlines_framing(self, p):
        p.setPen(QtGui.QPen(QtGui.QColor(70, 90, 140, 150), 1))
        diag = math.hypot(self.width(), self.height())
        for poly in self.constlines:
            ra = np.array([q[0] for q in poly]); dec = np.array([q[1] for q in poly])
            sx, sy, front = self._project_framing(ra, dec)
            for i in range(len(poly) - 1):
                if not (front[i] and front[i + 1]):
                    continue
                a = QtCore.QPointF(sx[i], sy[i]); b = QtCore.QPointF(sx[i + 1], sy[i + 1])
                if (a - b).manhattanLength() > diag:
                    continue
                p.drawLine(a, b)

    def _paint_constnames_framing(self, p):
        if not self.constnames:
            return
        p.setPen(QtGui.QColor(120, 150, 200))
        f = QtGui.QFont(); f.setPointSize(9); f.setItalic(True); p.setFont(f)
        ra = np.array([c[0] for c in self.constnames])
        dec = np.array([c[1] for c in self.constnames])
        sx, sy, front = self._project_framing(ra, dec)
        for i in np.where(front)[0]:
            if 0 <= sx[i] <= self.width() and 0 <= sy[i] <= self.height():
                p.drawText(QtCore.QPointF(sx[i], sy[i]), self.constnames[i][2])

    def _paint_cardinals_framing(self, p):
        # En framing équatorial : N haut, S bas, E gauche, O droite (l'AD croît vers la gauche).
        p.setPen(QtGui.QColor(150, 170, 200))
        f = QtGui.QFont(); f.setPointSize(10); f.setBold(True); p.setFont(f)
        w, h = self.width(), self.height()
        for label, x, y, al in [("N", w / 2, 16, Qt.AlignCenter), ("S", w / 2, h - 8, Qt.AlignCenter),
                                ("E", 12, h / 2, Qt.AlignLeft), ("W", w - 20, h / 2, Qt.AlignRight)]:
            p.drawText(QtCore.QRectF(x - 14, y - 10, 28, 20), al, label)

    def _draw_bg_framing(self, p):
        if self.bg_pixmap is None or self.bg_pixmap.isNull() or self.bg_pixmap.width() < 1:
            return
        sx, sy, front = self._project_framing(np.array([self.bg_ra]), np.array([self.bg_dec]))
        if not front[0] or not (np.isfinite(sx[0]) and np.isfinite(sy[0])):
            return
        s = self._scale_framing()
        bw = self.bg_pixmap.width()
        bg_s = (bw / 2.0) / max(math.tan(math.radians(self.bg_fov) / 2.0), 1e-6)
        scale = s / bg_s if bg_s > 1e-9 else 0.0
        w = self.bg_pixmap.width() * scale; h = self.bg_pixmap.height() * scale
        if not (math.isfinite(w) and math.isfinite(h)) or w < 1 or h < 1 or w > 20000 or h > 20000:
            return
        p.save(); p.translate(float(sx[0]), float(sy[0]))
        p.drawPixmap(QtCore.QRectF(-w / 2, -h / 2, w, h), self.bg_pixmap,
                     QtCore.QRectF(self.bg_pixmap.rect()))
        p.restore()

    def _paint_framing(self, p):
        # real-sky survey background (Aladin-like), if loaded
        self._draw_bg_framing(p)
        # constellation lines
        if self.show_constlines and len(self.constlines):
            self._paint_constlines_framing(p)
        # stars
        if len(self.star_ra):
            x, y, front = self._project_framing(self.star_ra, self.star_dec)
            for i in np.where(front)[0]:
                if -20 <= x[i] <= self.width() + 20 and -20 <= y[i] <= self.height() + 20:
                    r = self._star_radius(self.star_mag[i])
                    p.setBrush(_bv_color(self.star_bv[i])); p.setPen(Qt.NoPen)
                    p.drawEllipse(QtCore.QPointF(x[i], y[i]), r, r)
        # DSO
        if self.show_dso and self.dso:
            self._paint_dso_framing(p)
        # constellation names
        if self.show_constnames:
            self._paint_constnames_framing(p)
        # named bright stars
        if self.show_starnames:
            self._paint_starnames_framing(p)
        # planets / Moon / Sun (those falling inside the framed region)
        self._paint_planets_framing(p)
        # field rectangle + mosaic
        self._paint_field(p)
        # session goal (navigation): where to move to find the target
        self._paint_goal_framing(p)
        # cardinals (equatorial: N up, E left)
        self._paint_cardinals_framing(p)
        # crosshair on target center
        p.setPen(QtGui.QPen(QtGui.QColor(255, 90, 90, 180), 1))
        cx, cy = self.width() / 2.0, self.height() / 2.0
        p.drawLine(QtCore.QPointF(cx - 9, cy), QtCore.QPointF(cx + 9, cy))
        p.drawLine(QtCore.QPointF(cx, cy - 9), QtCore.QPointF(cx, cy + 9))
        alt_txt = ""
        if self.dt_utc is not None:
            lst = lst_deg(self.dt_utc, self.lon)
            al, _az = radec_to_altaz(np.array([self.center_ra]), np.array([self.center_dec]),
                                     lst, self.lat)
            alt_txt = "  ·  alt {:+.0f}°{}".format(float(al[0]),
                                                   " (below horizon)" if al[0] < 0 else "")
        _fw, _fh = fov_deg(self.focal_mm)
        if self.portrait:
            _fw, _fh = _fh, _fw
        self._paint_hud(p, "Framing · {}  ·  field {:.1f}°×{:.1f}° @ {:.0f} mm  ·  rot {:.0f}°{}".format(
            self.target_name, _fw, _fh, self.focal_mm, self.cam_angle, alt_txt))

    def _paint_planets_framing(self, p):
        """Draw Sun/Moon/planets that fall within the framed sky region."""
        if not self._planets:
            return
        ras = np.array([b[1] for b in self._planets])
        decs = np.array([b[2] for b in self._planets])
        xs, ys, front = self._project_framing(ras, decs)
        for i, (nm, _ra, _dec, col) in enumerate(self._planets):
            if not front[i]:
                continue
            if -30 <= xs[i] <= self.width() + 30 and -30 <= ys[i] <= self.height() + 30:
                rr = 8 if nm in ("Sun", "Moon") else 5
                p.setBrush(col); p.setPen(Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(float(xs[i]), float(ys[i])), rr, rr)
                p.setPen(QtGui.QColor(230, 235, 250))
                p.drawText(QtCore.QPointF(float(xs[i]) + rr + 3, float(ys[i]) + 4), nm)
        self._paint_comets_framing(p)
        self._paint_iss_framing(p)

    def _paint_iss_framing(self, p):
        if not self._iss_pos:
            return
        ra, dec, _alt = self._iss_pos
        xs, ys, front = self._project_framing(np.array([ra]), np.array([dec]))
        if front[0] and -30 <= xs[0] <= self.width() + 30 and -30 <= ys[0] <= self.height() + 30:
            self._draw_iss_marker(p, float(xs[0]), float(ys[0]))

    def _draw_iss_marker(self, p, x, y):
        col = QtGui.QColor(255, 120, 200)
        p.setPen(QtGui.QPen(col, 2)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(QtCore.QPointF(x, y), 5, 5)
        p.drawLine(QtCore.QPointF(x - 9, y), QtCore.QPointF(x - 5, y))   # solar-panel hint
        p.drawLine(QtCore.QPointF(x + 5, y), QtCore.QPointF(x + 9, y))
        p.setPen(col)
        p.drawText(QtCore.QPointF(x + 8, y - 6), "🛰 ISS")

    def _paint_comets_framing(self, p):
        if not self._comet_pos or not self.show_comets:
            return
        ras = np.array([c[1] for c in self._comet_pos])
        decs = np.array([c[2] for c in self._comet_pos])
        xs, ys, front = self._project_framing(ras, decs)
        col = QtGui.QColor(120, 240, 220)
        for i, (nm, _ra, _dec, mag) in enumerate(self._comet_pos):
            if not front[i]:
                continue
            if -30 <= xs[i] <= self.width() + 30 and -30 <= ys[i] <= self.height() + 30:
                self._draw_comet_marker(p, float(xs[i]), float(ys[i]), nm, col, mag)
        self._paint_comet_path_framing(p)

    def _draw_comet_marker(self, p, x, y, name, col, mag=None):
        p.setPen(QtGui.QPen(col, 1.5)); p.setBrush(col)
        p.drawEllipse(QtCore.QPointF(x, y), 3.5, 3.5)
        p.setPen(QtGui.QPen(col, 1.2))                  # little coma/tail flick
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + 10, y - 7))
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + 13, y - 4))
        p.setPen(col)
        label = "☄ " + name + ("  m{:.0f}".format(mag) if mag is not None else "")
        p.drawText(QtCore.QPointF(x + 6, y + 12), label)

    def _comet_elements(self, name):
        for c in self.comets:
            if c["name"] == name:
                return c
        return None

    def _paint_comet_path_framing(self, p):
        """Dashed path of the *selected* comet across the stars over ~6 weeks."""
        el = self._comet_elements(self.target_name)
        if el is None or self.dt_utc is None:
            return
        import datetime as _dt
        jd0 = _jd_utc(self.dt_utc)
        pts = []
        for k in range(-7, 36, 2):
            try:
                ra, dec = comet_radec(el, jd0 + k)
            except Exception:           # noqa: BLE001
                continue
            x, y, front = self._project_framing(np.array([ra]), np.array([dec]))
            if front[0]:
                pts.append((float(x[0]), float(y[0]), k))
        if len(pts) < 2:
            return
        p.setPen(QtGui.QPen(QtGui.QColor(120, 240, 220, 170), 1.6, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        path = QtGui.QPainterPath(QtCore.QPointF(pts[0][0], pts[0][1]))
        for (x, y, _k) in pts[1:]:
            path.lineTo(x, y)
        p.drawPath(path)
        p.setPen(QtGui.QColor(150, 245, 225)); p.setBrush(QtGui.QColor(150, 245, 225))
        for (x, y, k) in pts:                       # weekly date ticks
            if k % 7 == 0:
                p.drawEllipse(QtCore.QPointF(x, y), 2.2, 2.2)
                d = self.dt_utc + _dt.timedelta(days=k)
                tag = "today" if k == 0 else d.strftime("%d %b")
                p.drawText(QtCore.QPointF(x + 4, y - 4), tag)

    def _paint_goal_framing(self, p):
        if not self.show_goal:
            return
        self._draw_goal_frame_framing(p)      # camera field placed on the goal
        # offset of the goal from the frame centre, as % of the field (map: right=+X, up=+Y)
        xs, ys, _fr = self._project_framing(np.array([self.goal_ra]), np.array([self.goal_dec]))
        s = self._scale_framing(); fw, fh = fov_deg(self.focal_mm)
        halfw = max(math.radians(fw / 2.0) * s, 1e-6)
        halfh = max(math.radians(fh / 2.0) * s, 1e-6)
        offx = (float(xs[0]) - self.width() / 2.0) / (2 * halfw) * 100.0
        offy = -(float(ys[0]) - self.height() / 2.0) / (2 * halfh) * 100.0
        p.setPen(QtGui.QColor(255, 205, 70))
        fb = QtGui.QFont(); fb.setPointSize(9); fb.setBold(True); p.setFont(fb)
        p.drawText(QtCore.QPointF(10, self.height() - 12),
                   "Goal offset   X {:+.0f}%   Y {:+.0f}%   (of frame)".format(offx, offy))
        gx, gy, gfront = self._project_framing(np.array([self.goal_ra]), np.array([self.goal_dec]))
        cx, cy = self.width() / 2.0, self.height() / 2.0
        sep, _pa, comp = self._sep_bearing(self.center_ra, self.center_dec,
                                           self.goal_ra, self.goal_dec)
        gold = QtGui.QColor(255, 205, 70)
        f = QtGui.QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        onscreen = (bool(gfront[0]) and np.isfinite(gx[0]) and np.isfinite(gy[0])
                    and 0 <= gx[0] <= self.width() and 0 <= gy[0] <= self.height())
        if onscreen:
            x, y = float(gx[0]), float(gy[0])
            p.setPen(QtGui.QPen(gold, 2)); p.setBrush(Qt.NoBrush)
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y - 9), QtCore.QPointF(x + 9, y),
                                           QtCore.QPointF(x, y + 9), QtCore.QPointF(x - 9, y)]))
            p.setPen(gold)
            p.drawText(QtCore.QPointF(x + 12, y + 4),
                       "{}  {:.2f}°".format(self.goal_name, sep))
        else:
            ddec = self.goal_dec - self.center_dec
            dra = ((self.goal_ra - self.center_ra + 540.0) % 360.0) - 180.0
            east = dra * math.cos(math.radians(self.center_dec))
            vx, vy = -east, -ddec                    # E->left, N->up
            n = math.hypot(vx, vy) or 1.0
            vx, vy = vx / n, vy / n
            R = min(self.width(), self.height()) * 0.40
            ex, ey = cx + vx * R, cy + vy * R
            p.setPen(QtGui.QPen(gold, 2.5))
            p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(ex, ey))
            ang = math.atan2(vy, vx)
            for da in (2.62, -2.62):                 # +/-150 deg arrow head
                p.drawLine(QtCore.QPointF(ex, ey),
                           QtCore.QPointF(ex + 13 * math.cos(ang + da),
                                          ey + 13 * math.sin(ang + da)))
            p.setPen(gold)
            tx = max(8.0, min(ex + 8.0, self.width() - 160.0))
            p.drawText(QtCore.QPointF(tx, max(16.0, min(ey, self.height() - 8.0))),
                       "-> {}  {:.1f}° {}".format(self.goal_name, sep, comp))

    def _paint_dso_framing(self, p):
        s = self._scale_framing()
        f = QtGui.QFont(); f.setPointSize(8); p.setFont(f)
        ras = np.array([d["ra"] for d in self.dso])
        decs = np.array([d["dec"] for d in self.dso])
        x, y, front = self._project_framing(ras, decs)
        for i in np.where(front)[0]:
            if not (-30 <= x[i] <= self.width() + 30 and -30 <= y[i] <= self.height() + 30):
                continue
            d = self.dso[i]
            col = _dso_color(d["type"])
            rad = max(2.0, math.radians((d["size"] or 2.0) / 60.0) * s / 2.0)
            p.setPen(QtGui.QPen(col, 1.3)); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QtCore.QPointF(x[i], y[i]), rad, rad * 0.7)
            if self.show_names and (d["size"] >= 10 or (not math.isnan(d["mag"]) and d["mag"] <= 8)):
                p.setPen(QtGui.QColor(180, 200, 210))
                p.drawText(QtCore.QPointF(x[i] + rad + 2, y[i] + 3), d["id"])

    def _paint_field(self, p):
        s = self._scale_framing()
        cx, cy = self.width() / 2.0, self.height() / 2.0
        fw, fh = fov_deg(self.focal_mm)
        hw = math.radians(fw / 2.0) * s
        hh = math.radians(fh / 2.0) * s
        p.save()
        p.translate(cx, cy)
        if self.cam_angle:
            p.rotate(self.cam_angle)      # le cadre tourne, le ciel reste nord-en-haut
        if not self.show_mosaic:
            p.setPen(QtGui.QPen(QtGui.QColor(110, 240, 220), 2))
            p.setBrush(QtGui.QColor(110, 240, 220, 28))
            p.drawRect(QtCore.QRectF(-hw, -hh, 2 * hw, 2 * hh))
            p.restore()
            return
        # mosaic grid of panels with overlap, centered on target
        ov = self.mos_overlap
        stepx = 2 * hw * (1 - ov); stepy = 2 * hh * (1 - ov)
        cols, rows = self.mos_cols, self.mos_rows
        total_w = 2 * hw + (cols - 1) * stepx
        total_h = 2 * hh + (rows - 1) * stepy
        x0 = -total_w / 2.0; y0 = -total_h / 2.0
        p.setPen(QtGui.QPen(QtGui.QColor(110, 240, 220, 220), 1.5))
        p.setBrush(QtGui.QColor(110, 240, 220, 22))
        for r in range(rows):
            for c in range(cols):
                p.drawRect(QtCore.QRectF(x0 + c * stepx, y0 + r * stepy, 2 * hw, 2 * hh))
        p.restore()

    def panel_centers(self):
        """RA/Dec of each mosaic panel centre (for pointing each tile)."""
        fw, fh = fov_deg(self.focal_mm)
        hw = math.radians(fw / 2.0); hh = math.radians(fh / 2.0)
        ov = self.mos_overlap
        stepx = 2 * hw * (1 - ov); stepy = 2 * hh * (1 - ov)
        cols, rows = self.mos_cols, self.mos_rows
        th = math.radians(self.cam_angle)
        cth, sth = math.cos(th), math.sin(th)
        out = []
        for r in range(rows):
            for c in range(cols):
                lx = (c - (cols - 1) / 2.0) * stepx       # repère écran (x droite, y bas)
                ly = (r - (rows - 1) / 2.0) * stepy
                ex = lx * cth - ly * sth                  # rotation horaire (peintre)
                ey = lx * sth + ly * cth
                ra, dec = inv_gnomonic(-ex, -ey, self.center_ra, self.center_dec)  # est=gauche, nord=haut
                out.append((r + 1, c + 1, ra, dec))
        return out

    def _radec_h(self, ra, dec, lst):
        alt, az = radec_to_altaz(np.atleast_1d(np.asarray(ra, float)),
                                 np.atleast_1d(np.asarray(dec, float)), lst, self.lat)
        sx, sy, infront = self._project_horizon(alt, az)
        return sx, sy, infront, alt

    def _paint_allsky(self, p):
        W, H = self.width(), self.height()
        sky = QtGui.QLinearGradient(0, 0, 0, H)
        sky.setColorAt(0.0, QtGui.QColor(6, 9, 22))
        sky.setColorAt(0.72, QtGui.QColor(14, 22, 46))
        sky.setColorAt(1.0, QtGui.QColor(28, 42, 76))
        p.fillRect(self.rect(), sky)
        if self.dt_utc is None:
            self._paint_hud(p, "Horizon view · set a date/time")
            return
        lst = lst_deg(self.dt_utc, self.lon)
        self._horizon_grid(p)
        self._horizon_content(p, lst)
        self._horizon_ground(p)
        self._horizon_cardinals(p)
        self._paint_hud(p, "Horizon view · {:%Y-%m-%d %H:%M UTC}  ·  looking {} alt {:+.0f}°  ·  "
                           "drag to look around, wheel to zoom".format(
                               self.dt_utc, _az_name(self.view_az), self.view_alt))

    def _horizon_grid(self, p):
        # almucantars (constant altitude) + azimuth verticals
        p.setPen(QtGui.QPen(QtGui.QColor(80, 100, 135, 70), 1))
        azs = np.linspace(self.view_az - 175, self.view_az + 175, 120)
        for alt in (15, 30, 45, 60, 75):
            sx, sy, vis = self._project_horizon(np.full_like(azs, alt), azs)
            self._polyline(p, sx, sy, vis)
        alts = np.linspace(0, 85, 40)
        for az in range(0, 360, 30):
            sx, sy, vis = self._project_horizon(alts, np.full_like(alts, az))
            self._polyline(p, sx, sy, vis)

    def _polyline(self, p, sx, sy, vis, maxseg=400):
        for i in range(len(sx) - 1):
            if vis[i] and vis[i + 1]:
                a = QtCore.QPointF(sx[i], sy[i]); b = QtCore.QPointF(sx[i + 1], sy[i + 1])
                if (a - b).manhattanLength() < maxseg:
                    p.drawLine(a, b)

    def _horizon_content(self, p, lst):
        # constellation lines
        if self.show_constlines:
            p.setPen(QtGui.QPen(QtGui.QColor(95, 115, 160, 120), 1))
            for poly in self.constlines:
                ra = np.array([q[0] for q in poly]); dec = np.array([q[1] for q in poly])
                sx, sy, vis, _alt = self._radec_h(ra, dec, lst)
                self._polyline(p, sx, sy, vis)
        # stars (magnitude-limited)
        if len(self.star_ra):
            sx, sy, vis, alt = self._radec_h(self.star_ra, self.star_dec, lst)
            for i in np.where(vis)[0]:
                m = self.star_mag[i]
                if m > 5.4 or alt[i] < -1:
                    continue
                r = max(0.7, (5.7 - m) * 1.05)
                p.setBrush(_bv_color(self.star_bv[i])); p.setPen(Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(sx[i], sy[i]), r, r)
        # DSO (showy ones)
        if self.show_dso and self.dso:
            ras = np.array([d["ra"] for d in self.dso]); decs = np.array([d["dec"] for d in self.dso])
            sx, sy, vis, alt = self._radec_h(ras, decs, lst)
            for i in np.where(vis)[0]:
                d = self.dso[i]
                if alt[i] < 0:
                    continue
                if d["size"] < 20 and not (not math.isnan(d["mag"]) and d["mag"] <= 6):
                    continue
                p.setPen(QtGui.QPen(_dso_color(d["type"]), 1)); p.setBrush(Qt.NoBrush)
                p.drawEllipse(QtCore.QPointF(sx[i], sy[i]), 3, 3)
        # constellation names
        if self.show_constnames and self.constnames:
            f = QtGui.QFont(); f.setPointSize(8); f.setItalic(True); p.setFont(f)
            ra = np.array([c[0] for c in self.constnames]); dec = np.array([c[1] for c in self.constnames])
            sx, sy, vis, alt = self._radec_h(ra, dec, lst)
            for i in np.where(vis)[0]:
                if alt[i] < 6:
                    continue
                p.setPen(QtGui.QColor(10, 14, 26))
                p.drawText(QtCore.QPointF(sx[i] + 1, sy[i] + 1), self.constnames[i][2])
                p.setPen(QtGui.QColor(155, 185, 230))
                p.drawText(QtCore.QPointF(sx[i], sy[i]), self.constnames[i][2])
        # named bright stars
        if self.show_starnames:
            self._paint_starnames_allsky(p, lst)
        # planets / Moon / Sun
        for nm, ra, dec, col in self._planets:
            sx, sy, vis, alt = self._radec_h(ra, dec, lst)
            if vis[0] and alt[0] > 0:
                rr = 7 if nm in ("Sun", "Moon") else 4
                p.setBrush(col); p.setPen(Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(float(sx[0]), float(sy[0])), rr, rr)
                p.setPen(QtGui.QColor(220, 230, 245))
                p.drawText(QtCore.QPointF(float(sx[0]) + rr + 2, float(sy[0]) + 3), nm)
        # comets
        if self.show_comets:
            for nm, ra, dec, mag in self._comet_pos:
                sx, sy, vis, alt = self._radec_h(ra, dec, lst)
                if vis[0] and alt[0] > 0:
                    self._draw_comet_marker(p, float(sx[0]), float(sy[0]), nm,
                                            QtGui.QColor(120, 240, 220), mag)
        # ISS / satellite
        if self._iss_pos and self._iss_pos[2] > 0:
            sx, sy, vis, alt = self._radec_h(self._iss_pos[0], self._iss_pos[1], lst)
            if vis[0] and alt[0] > 0:
                self._draw_iss_marker(p, float(sx[0]), float(sy[0]))
        # target trajectory (diurnal arc) + marker
        self._paint_trajectory_allsky(p)
        sx, sy, vis, alt = self._radec_h(self.center_ra, self.center_dec, lst)
        if vis[0] and alt[0] > 0:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 90, 90), 2)); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QtCore.QPointF(float(sx[0]), float(sy[0])), 9, 9)
            p.setPen(QtGui.QColor(255, 150, 150))
            p.drawText(QtCore.QPointF(float(sx[0]) + 11, float(sy[0]) + 3), self.target_name)
        # camera frame overlay (optional)
        if self.show_frame_allsky:
            p.setPen(QtGui.QPen(QtGui.QColor(110, 240, 220, 235), 2))
            p.setBrush(QtGui.QColor(110, 240, 220, 30))
            for poly in self._field_polys_radec():
                ra = np.array([q[0] for q in poly]); dec = np.array([q[1] for q in poly])
                fsx, fsy, fvis, falt = self._radec_h(ra, dec, lst)
                if np.all(fvis) and np.all(falt > -2):
                    path = QtGui.QPainterPath(); path.moveTo(float(fsx[0]), float(fsy[0]))
                    for i in range(1, len(poly)):
                        path.lineTo(float(fsx[i]), float(fsy[i]))
                    path.closeSubpath(); p.drawPath(path)
        # session goal: marker + dashed line from current pointing + bearing
        if self.show_goal:
            gsx, gsy, gvis, galt = self._radec_h(self.goal_ra, self.goal_dec, lst)
            csx, csy, cvis, _ca = self._radec_h(self.center_ra, self.center_dec, lst)
            sep, _pa, comp = self._sep_bearing(self.center_ra, self.center_dec,
                                               self.goal_ra, self.goal_dec)
            gold = QtGui.QColor(255, 205, 70)
            f = QtGui.QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
            if gvis[0] and cvis[0]:
                p.setPen(QtGui.QPen(gold, 1.6, Qt.DashLine)); p.setBrush(Qt.NoBrush)
                p.drawLine(QtCore.QPointF(float(csx[0]), float(csy[0])),
                           QtCore.QPointF(float(gsx[0]), float(gsy[0])))
            # goal camera field (to compare with the current pointing)
            p.setPen(QtGui.QPen(gold, 1.8, Qt.DashLine)); p.setBrush(QtGui.QColor(255, 205, 70, 22))
            for poly in self._field_polys_radec(self.goal_ra, self.goal_dec, mosaic=False,
                                                angle=self.goal_angle):
                ra = np.array([q[0] for q in poly]); dec = np.array([q[1] for q in poly])
                fsx, fsy, fv, fa = self._radec_h(ra, dec, lst)
                if np.all(fv) and np.all(fa > -2):
                    pth = QtGui.QPainterPath(); pth.moveTo(float(fsx[0]), float(fsy[0]))
                    for i in range(1, len(poly)):
                        pth.lineTo(float(fsx[i]), float(fsy[i]))
                    pth.closeSubpath(); p.drawPath(pth)
            if gvis[0]:
                x, y = float(gsx[0]), float(gsy[0])
                p.setPen(QtGui.QPen(gold, 2)); p.setBrush(Qt.NoBrush)
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y - 9), QtCore.QPointF(x + 9, y),
                                               QtCore.QPointF(x, y + 9), QtCore.QPointF(x - 9, y)]))
                p.setPen(gold)
                tag = "{}  {:.1f}° {}".format(self.goal_name, sep, comp)
                if galt[0] < 0:
                    tag += " (below horizon)"
                p.drawText(QtCore.QPointF(x + 12, y + 4), tag)
            else:
                p.setPen(gold)
                p.drawText(QtCore.QPointF(12, self.height() - 30),
                           "Goal {}: {:.1f}° {} — not in view".format(self.goal_name, sep, comp))

    def _horizon_ground(self, p):
        W, H = self.width(), self.height()
        azs = np.linspace(self.view_az - 175, self.view_az + 175, 160)
        sx, sy, vis = self._project_horizon(np.zeros_like(azs), azs)
        pts = [(float(sx[i]), float(sy[i])) for i in range(len(azs)) if vis[i]]
        if len(pts) < 2:
            return
        path = QtGui.QPainterPath()
        path.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        path.lineTo(W + 60, pts[-1][1]); path.lineTo(W + 60, H + 60)
        path.lineTo(-60, H + 60); path.lineTo(-60, pts[0][1])
        path.closeSubpath()
        top = min(y for _x, y in pts)
        gg = QtGui.QLinearGradient(0, top, 0, H)
        gg.setColorAt(0.0, QtGui.QColor(46, 50, 44))
        gg.setColorAt(1.0, QtGui.QColor(10, 12, 10))
        p.fillPath(path, gg)
        # marked horizon line
        p.setPen(QtGui.QPen(QtGui.QColor(150, 200, 235), 2))
        for i in range(len(pts) - 1):
            a = QtCore.QPointF(*pts[i]); b = QtCore.QPointF(*pts[i + 1])
            if (a - b).manhattanLength() < 400:
                p.drawLine(a, b)

    def _horizon_cardinals(self, p):
        fb = QtGui.QFont(); fb.setPointSize(11); fb.setBold(True); p.setFont(fb)
        for label, az, major in [("N", 0, 1), ("NE", 45, 0), ("E", 90, 1), ("SE", 135, 0),
                                 ("S", 180, 1), ("SW", 225, 0), ("W", 270, 1), ("NW", 315, 0)]:
            sx, sy, vis = self._project_horizon(np.array([0.0]), np.array([float(az)]))
            if not vis[0]:
                continue
            x, y = float(sx[0]), float(sy[0])
            if -10 <= x <= self.width() + 10:
                p.setPen(QtGui.QColor(220, 235, 255) if major else QtGui.QColor(130, 150, 180))
                p.drawText(QtCore.QRectF(x - 16, y - 22, 32, 18), Qt.AlignCenter, label)

    def _paint_hud(self, p, text):
        p.setPen(QtGui.QColor(150, 200, 210))
        f = QtGui.QFont(); f.setPointSize(9); p.setFont(f)
        p.drawText(QtCore.QRectF(8, 6, self.width() - 16, 20), Qt.AlignLeft, text)

    # -- interaction -------------------------------------------------------
    def wheelEvent(self, ev):
        d = ev.angleDelta().y()
        if self.mode == "framing":
            self.view_fov = float(np.clip(self.view_fov * (0.86 if d > 0 else 1.16), 0.3, 100))
            self.viewChanged.emit()
        else:
            self.h_fov = float(np.clip(self.h_fov * (0.86 if d > 0 else 1.16), 15, 150))
        self.update()

    def mousePressEvent(self, ev):
        self._drag = ev.position()
        self._drag_moved = False

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        delta = ev.position() - self._drag
        self._drag = ev.position()
        if delta.manhattanLength() > 2:
            self._drag_moved = True
        if self.mode == "framing":
            s = self._scale_framing()
            dra = -delta.x() / s          # east-left -> +x moves center west
            ddec = delta.y() / s
            self.center_ra = (self.center_ra - math.degrees(dra) / math.cos(math.radians(self.center_dec) or 1)) % 360.0
            self.center_dec = float(np.clip(self.center_dec + math.degrees(ddec), -89.9, 89.9))
        else:
            # horizon view: look around (azimuth/altitude)
            degpx = self.h_fov / self.width()
            self.view_az = (self.view_az - delta.x() * degpx) % 360.0
            self.view_alt = float(np.clip(self.view_alt + delta.y() * degpx, -10, 89))
        self.update()

    def mouseReleaseEvent(self, ev):
        moved = getattr(self, "_drag_moved", False)
        if self._drag is not None and not moved:
            self._pick(ev.position())
        self._drag = None
        if moved and self.mode == "framing":
            self.viewChanged.emit()       # refetch survey background for new center

    def _pick(self, pos):
        """Pick the nearest object to the click — DSO, planet, Sun, Moon or named
        star — and emit it."""
        cands = []  # (label, ra, dec)
        for d in self.dso:
            cands.append((d["id"], d["ra"], d["dec"]))
        for nm, ra, dec, _c in self._planets:     # planets / Sun / Moon (clickable)
            cands.append((nm, ra, dec))
        for nm, ra, dec, _m in self._comet_pos:   # comets (clickable -> shows path)
            cands.append((nm, ra, dec))
        if self._iss_pos:
            cands.append(("ISS", self._iss_pos[0], self._iss_pos[1]))
        if self.show_starnames:
            for s in self.starnames:
                cands.append((s["name"], s["ra"], s["dec"]))
        if not cands or (self.mode != "framing" and self.dt_utc is None):
            return
        ras = np.array([c[1] for c in cands]); decs = np.array([c[2] for c in cands])
        if self.mode == "framing":
            x, y, front = self._project_framing(ras, decs)
        else:
            x, y, front, _alt = self._radec_h(ras, decs, lst_deg(self.dt_utc, self.lon))
        best = None; bestd = 22.0 ** 2
        for i in np.where(front)[0]:
            dd = (x[i] - pos.x()) ** 2 + (y[i] - pos.y()) ** 2
            if dd < bestd:
                bestd = dd; best = cands[i]
        if best:
            self.set_target(best[0], best[1], best[2])
            self.targetPicked.emit(*best)
