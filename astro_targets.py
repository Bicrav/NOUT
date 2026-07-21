#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
astro_targets.py
================

Moteur de planification pour l'astrophotographie, à coupler avec le contrôle
tethered de l'A7II (sony_tether_focus.py).

Donne, pour un LIEU (lat/lon) et une DATE :
  * la fenêtre de nuit astronomique (Soleil < -18°) -> quand il fait vraiment noir
  * l'état de la Lune (fraction éclairée + hauteur) -> impact sur le fond de ciel
  * un classement des cibles du ciel profond observables cette nuit-là :
        - hauteur maximale atteinte pendant la nuit
        - heure de passage au méridien (transit), en heure locale
        - séparation angulaire à la Lune
        - si la cible "tient" dans le champ de ton A7II à une focale donnée

Dépendances :
    pip install astropy numpy
    # (zoneinfo est dans la stdlib >= 3.9)

Exemples :
    # Saint-Paul-lez-Durance, cette nuit, objectif 200 mm :
    python astro_targets.py --lat 43.694 --lon 5.737 --focal 200

    # une date précise, top 30 cibles, hauteur mini 25° :
    python astro_targets.py --lat 43.694 --lon 5.737 --date 2026-08-15 \
                            --focal 135 --min-alt 25 --limit 30

API (pour l'intégrer dans l'appli) :
    site = Site(lat=43.694, lon=5.737, height=300, name="Cadarache")
    night = night_window(site, when=date.today())
    recos = plan(site, when=date.today(), focal_mm=200, min_alt=30, limit=20)
"""

import argparse
import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta

import numpy as np

# astropy peut tenter de télécharger des tables IERS récentes : on tolère des
# tables un peu anciennes pour rester utilisable hors-ligne (site sombre isolé).
from astropy.utils import iers
from astropy.utils.exceptions import AstropyWarning
iers.conf.auto_max_age = None
# avertissements attendus hors-ligne (IERS indisponible, séparation Lune) : on les tait.
warnings.simplefilter("ignore", category=AstropyWarning)

import astropy.units as u
from astropy.coordinates import (AltAz, EarthLocation, SkyCoord, get_body,
                                 get_sun)
from astropy.time import Time

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


# ===========================================================================
#  Capteur / champ couvert (A7II par défaut, plein format)
# ===========================================================================
@dataclass
class Sensor:
    name: str = "Sony A7II (plein format)"
    width_mm: float = 35.8
    height_mm: float = 23.9
    px_x: int = 6000
    px_y: int = 4000


def field_of_view(focal_mm, sensor=Sensor()):
    """Champ couvert (degrés) et échelle (arcsec/pixel) pour une focale donnée."""
    fov_w = math.degrees(2 * math.atan(sensor.width_mm / (2 * focal_mm)))
    fov_h = math.degrees(2 * math.atan(sensor.height_mm / (2 * focal_mm)))
    pix_um = sensor.width_mm / sensor.px_x * 1000.0
    scale = 206.265 * pix_um / focal_mm  # arcsec / pixel
    return {"fov_w_deg": fov_w, "fov_h_deg": fov_h,
            "fov_w_arcmin": fov_w * 60, "fov_h_arcmin": fov_h * 60,
            "scale_arcsec_px": scale}


def fits_in_frame(size_arcmin, focal_mm, sensor=Sensor(), margin=1.3):
    """La cible (taille en arcmin) tient-elle confortablement dans le cadre ?"""
    if not size_arcmin or size_arcmin <= 0:
        return None
    fov = field_of_view(focal_mm, sensor)
    return size_arcmin * margin <= min(fov["fov_w_arcmin"], fov["fov_h_arcmin"])


# ===========================================================================
#  Catalog. The full Messier + NGC + IC catalogue (J2000, English names/types)
#  is loaded from "catalog_ngc.csv" sitting next to this file (OpenNGC-derived).
#  A small curated list is embedded as a fallback if that file is missing.
# ===========================================================================
import csv as _csv
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CATALOG_CSV = _os.path.join(_HERE, "catalog_ngc.csv")

# (id, name, type, RA "HH MM.m", Dec "+DD MM", magnitude, size_arcmin) — fallback only
_CATALOG_RAW = [
    ("M31", "Andromeda Galaxy", "Galaxy", "00 42.7", "+41 16", 3.4, 178),
    ("M33", "Triangulum Galaxy", "Galaxy", "01 33.9", "+30 39", 5.7, 70),
    ("M42", "Orion Nebula", "Nebula", "05 35.4", "-05 27", 4.0, 85),
    ("M45", "Pleiades", "Open Cluster", "03 47.0", "+24 07", 1.6, 110),
    ("M44", "Beehive Cluster", "Open Cluster", "08 40.1", "+19 59", 3.7, 95),
    ("M1", "Crab Nebula", "Supernova Remnant", "05 34.5", "+22 01", 8.4, 6),
    ("M51", "Whirlpool Galaxy", "Galaxy", "13 29.9", "+47 12", 8.4, 11),
    ("M81", "Bode's Galaxy", "Galaxy", "09 55.6", "+69 04", 6.9, 27),
    ("M82", "Cigar Galaxy", "Galaxy", "09 55.8", "+69 41", 8.4, 11),
    ("M101", "Pinwheel Galaxy", "Galaxy", "14 03.2", "+54 21", 7.9, 29),
    ("M104", "Sombrero Galaxy", "Galaxy", "12 40.0", "-11 37", 8.0, 9),
    ("M27", "Dumbbell Nebula", "Planetary Nebula", "19 59.6", "+22 43", 7.4, 8),
    ("M57", "Ring Nebula", "Planetary Nebula", "18 53.6", "+33 02", 8.8, 3),
    ("M8", "Lagoon Nebula", "Nebula", "18 03.8", "-24 23", 5.0, 90),
    ("M20", "Trifid Nebula", "Nebula", "18 02.6", "-23 02", 6.3, 28),
    ("M16", "Eagle Nebula", "Nebula", "18 18.8", "-13 47", 6.0, 35),
    ("M17", "Omega Nebula", "Nebula", "18 20.8", "-16 11", 6.0, 46),
    ("M13", "Hercules Cluster", "Globular Cluster", "16 41.7", "+36 28", 5.8, 20),
    ("NGC7000", "North America Nebula", "Nebula", "20 58.8", "+44 20", 4.0, 120),
    ("IC1396", "Elephant's Trunk Nebula", "Nebula", "21 39.1", "+57 30", 3.5, 170),
    ("NGC7293", "Helix Nebula", "Planetary Nebula", "22 29.6", "-20 48", 7.6, 16),
    ("NGC869", "Double Cluster", "Open Cluster", "02 19.0", "+57 09", 4.3, 60),
    ("NGC2237", "Rosette Nebula", "Nebula", "06 31.7", "+05 03", 5.5, 80),
    ("IC434", "Horsehead Nebula", "Nebula", "05 41.0", "-02 27", 6.8, 30),
    ("NGC6960", "Veil Nebula", "Supernova Remnant", "20 45.7", "+30 43", 7.0, 70),
    ("NGC1499", "California Nebula", "Nebula", "04 03.3", "+36 25", 5.0, 145),
]

# Planets computed dynamically (true position at the requested instant)
_PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn"]
_PLANET_EN = {"mercury": "Mercury", "venus": "Venus", "mars": "Mars",
              "jupiter": "Jupiter", "saturn": "Saturn"}


def _parse_ra_hours(s):
    """'00 42.7' -> decimal hours."""
    parts = s.split()
    h = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    return h + m / 60.0


def _parse_dec_deg(s):
    """'+41 16' / '-05 27' -> decimal degrees (handles -00)."""
    s = s.strip()
    sign = -1.0 if s[0] == "-" else 1.0
    parts = s.lstrip("+-").split()
    d = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    return sign * (d + m / 60.0)


@dataclass
class Target:
    id: str
    name: str
    type: str
    coord: SkyCoord
    mag: float
    size_arcmin: float


def _load_embedded():
    out = []
    for tid, name, typ, ra, dec, mag, size in _CATALOG_RAW:
        coord = SkyCoord(ra=_parse_ra_hours(ra) * u.hourangle,
                         dec=_parse_dec_deg(dec) * u.deg, frame="icrs")
        out.append(Target(tid, name, typ, coord, float(mag), float(size)))
    return out


def load_catalog(path=None):
    """Full Messier + NGC + IC catalogue from catalog_ngc.csv (English).

    Builds ONE SkyCoord array then slices it per target, so loading ~13 000
    objects stays fast (no per-target SkyCoord construction)."""
    path = path or _CATALOG_CSV
    if not _os.path.exists(path):
        return _load_embedded()
    ids, names, types, ras, decs, mags, sizes = [], [], [], [], [], [], []
    try:
        with open(path, newline="") as f:
            for row in _csv.DictReader(f):
                try:
                    ra = float(row["ra_hours"]); dec = float(row["dec_deg"])
                except (ValueError, KeyError):
                    continue
                ids.append(row["id"])
                names.append(row.get("name") or row["id"])
                types.append(row.get("type") or "?")
                ras.append(ra); decs.append(dec)
                mags.append(float(row["mag"]) if row.get("mag") else float("nan"))
                sizes.append(float(row["size_arcmin"]) if row.get("size_arcmin") else 0.0)
    except Exception:
        return _load_embedded()
    if not ids:
        return _load_embedded()
    coords = SkyCoord(ra=np.array(ras) * u.hourangle,
                      dec=np.array(decs) * u.deg, frame="icrs")
    return [Target(ids[i], names[i], types[i], coords[i], mags[i], sizes[i])
            for i in range(len(ids))]


def resolve_name(name):
    """Resolve an object by name via SIMBAD/SESAME (needs internet).
    Handy to point at a target absent from the embedded catalogue."""
    coord = SkyCoord.from_name(name)
    return Target(name, name, "?", coord, float("nan"), 0.0)


# ===========================================================================
#  Lieu et temps
# ===========================================================================
@dataclass
class Site:
    lat: float
    lon: float
    height: float = 0.0
    name: str = "Site"
    tz: str = "Europe/Paris"

    @property
    def location(self):
        return EarthLocation(lat=self.lat * u.deg, lon=self.lon * u.deg,
                             height=self.height * u.m)


def _local_tz(site):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(site.tz)
        except Exception:  # noqa: BLE001
            pass
    return None


def _to_local(t, site):
    """astropy.Time (UTC) -> datetime local lisible."""
    dt = t.to_datetime()  # naïf, UTC
    tz = _local_tz(site)
    if tz is not None:
        from datetime import timezone
        return dt.replace(tzinfo=timezone.utc).astimezone(tz)
    return dt


def _night_grid(site, when, step_min=5):
    """Grille de temps de midi (local) à midi+24h, en UTC."""
    tz = _local_tz(site)
    noon_local = datetime.combine(when, dtime(12, 0))
    if tz is not None:
        noon_local = noon_local.replace(tzinfo=tz)
    t0 = Time(noon_local)  # astropy convertit en UTC en interne
    n = int(24 * 60 / step_min) + 1
    return t0 + np.arange(n) * step_min * u.min


# ===========================================================================
#  Nuit astronomique & Lune
# ===========================================================================
@dataclass
class NightWindow:
    start: Time
    end: Time
    kind: str  # "astronomique" / "nautique" / "aucune"


def night_window(site, when):
    """Fenêtre de nuit la plus sombre autour du minuit local.
    Renvoie une NightWindow (Soleil < -18°), ou un repli nautique (-12°),
    ou kind='aucune' si le Soleil ne descend jamais assez bas."""
    grid = _night_grid(site, when)
    altaz = AltAz(obstime=grid, location=site.location)
    sun_alt = get_sun(grid).transform_to(altaz).alt.deg

    for thr, kind in ((-18, "astronomical"), (-12, "nautical")):
        mask = sun_alt < thr
        if not mask.any():
            continue
        # on garde le segment continu contenant le minimum (cœur de nuit)
        imin = int(np.argmin(sun_alt))
        i = imin
        while i > 0 and mask[i - 1]:
            i -= 1
        j = imin
        while j < len(mask) - 1 and mask[j + 1]:
            j += 1
        return NightWindow(grid[i], grid[j], kind)
    return NightWindow(None, None, "none")


def moon_state(site, t):
    """Fraction éclairée (0..1), hauteur (deg) et coordonnée de la Lune à l'instant t."""
    altaz = AltAz(obstime=t, location=site.location)
    moon_topo = get_body("moon", t, site.location)        # topocentrique -> hauteur
    moon_geo = get_body("moon", t)                         # géocentrique -> phase
    sun = get_sun(t)
    elong = sun.separation(moon_geo)
    # angle de phase (formule standard) -> fraction éclairée
    phase = np.arctan2(sun.distance * np.sin(elong),
                       moon_geo.distance - sun.distance * np.cos(elong))
    illum = float((1 + np.cos(phase)) / 2)
    alt = float(moon_topo.transform_to(altaz).alt.deg)
    # geocentric apparent direction (do NOT transform_to icrs: barycentric shift ruins it)
    return {"illum": illum, "alt": alt,
            "coord": SkyCoord(moon_geo.ra, moon_geo.dec, frame="icrs")}


# ===========================================================================
#  Visibilité d'une cible & classement
# ===========================================================================
@dataclass
class Reco:
    target: Target
    max_alt: float
    transit_local: datetime
    alt_now_window: float
    moon_sep: float
    fits: object
    score: float
    moon_av: int = 100        # Moon-avoidance score 0..100 (100 = no interference)


def _target_track(target, site, window, step_min=2):
    n = int((window.end - window.start).to(u.min).value / step_min) + 1
    times = window.start + np.arange(n) * step_min * u.min
    altaz = AltAz(obstime=times, location=site.location)
    alt = target.coord.transform_to(altaz).alt.deg
    k = int(np.argmax(alt))
    return float(alt[k]), times[k], alt


def plan(site, when, focal_mm=200, min_alt=30.0, limit=20,
         catalog=None, include_planets=True, max_mag=None):
    """Renvoie les meilleures cibles pour la nuit, triées par score décroissant.
    Transformation alt-az vectorisée (toutes les cibles × tous les instants en
    un seul appel) -> ~1 s au lieu de plusieurs dizaines."""
    night = night_window(site, when)
    if night.kind == "none" or night.start is None:
        return night, []

    mid = night.start + (night.end - night.start) / 2
    moon = moon_state(site, mid)
    moon_icrs = moon["coord"]

    targets = list(catalog if catalog is not None else load_catalog())
    # pré-filtre éclat : avec un boîtier + objectif photo, les objets très faibles
    # ne sont pas des cibles réalistes ; on les écarte pour garder le calcul rapide.
    # (max_mag=None => on garde tout)
    if max_mag is not None:
        targets = [t for t in targets
                   if (math.isnan(t.mag) and t.size_arcmin >= 8.0) or
                   (not math.isnan(t.mag) and t.mag <= max_mag)]
    if include_planets:
        for p in _PLANETS:
            body = get_body(p, mid, site.location)   # GCRS apparent direction
            targets.append(Target(p.upper(), _PLANET_EN[p], "Planet",
                                  SkyCoord(body.ra, body.dec, frame="icrs"),
                                  float("nan"), 0.0))
    if not targets:
        return night, []

    # grille de temps de la nuit (~6 min) -> une seule transformation broadcastée
    span = (night.end - night.start).to(u.min).value
    n = max(2, int(span / 6) + 1)
    times = night.start + np.linspace(0, span, n) * u.min
    frame = AltAz(obstime=times[np.newaxis, :], location=site.location)  # (1, N)
    ra1d = u.Quantity([t.coord.ra for t in targets])                    # (M,)
    dec1d = u.Quantity([t.coord.dec for t in targets])
    coords1d = SkyCoord(ra=ra1d, dec=dec1d, frame="icrs")
    coords = coords1d[:, np.newaxis]                                    # (M, 1)
    alt = coords.transform_to(frame).alt.deg                            # (M, N)
    seps = coords1d.separation(moon_icrs).deg                           # (M,) vectorisé

    recos = []
    for i, tgt in enumerate(targets):
        k = int(np.argmax(alt[i]))
        max_alt = float(alt[i, k])
        if max_alt < min_alt:
            continue
        sep = float(seps[i])

        # score : hauteur + bonus éclat + bonus taille (grand champ) - pénalité Lune.
        # La Lune ne gêne que lorsqu'elle est levée ; son rayon d'influence grandit
        # avec la phase (~30° en croissant, ~100° en pleine Lune) et son éclat dépend
        # aussi de sa hauteur (une Lune basse gêne moins).
        moon_factor = 0.0
        if moon["alt"] > 0:
            illum = float(moon["illum"])
            radius = 30.0 + 70.0 * illum
            if sep < radius:
                closeness = (radius - sep) / radius
                altw = min(1.0, moon["alt"] / 40.0)
                moon_factor = closeness * illum * altw
        moon_pen = moon_factor * 60.0
        moon_av = int(round(100.0 * (1.0 - moon_factor)))
        mag = tgt.mag if not math.isnan(tgt.mag) else 12.0
        size_bonus = min(tgt.size_arcmin, 60.0) / 60.0 * 12.0
        score = max_alt + size_bonus - moon_pen - 1.2 * max(mag, 0)

        recos.append(Reco(
            target=tgt, max_alt=max_alt,
            transit_local=_to_local(times[k], site),
            alt_now_window=max_alt, moon_sep=sep,
            fits=fits_in_frame(tgt.size_arcmin, focal_mm) if tgt.size_arcmin else None,
            score=score, moon_av=moon_av))

    recos.sort(key=lambda r: r.score, reverse=True)
    return night, recos[:limit]


# ===========================================================================
#  CLI
# ===========================================================================
def _fmt_t(dt):
    return dt.strftime("%Hh%M") if dt else "—"


def main():
    ap = argparse.ArgumentParser(
        description="Cibles astrophoto selon lieu et date (moteur astropy).")
    ap.add_argument("--lat", type=float, required=True, help="Latitude (°, N>0)")
    ap.add_argument("--lon", type=float, required=True, help="Longitude (°, E>0)")
    ap.add_argument("--height", type=float, default=300.0, help="Altitude (m)")
    ap.add_argument("--name", default="Site", help="Nom du lieu")
    ap.add_argument("--tz", default="Europe/Paris", help="Fuseau (ex: Europe/Paris)")
    ap.add_argument("--date", default=None, help="AAAA-MM-JJ (défaut : aujourd'hui)")
    ap.add_argument("--focal", type=float, default=200.0, help="Focale (mm)")
    ap.add_argument("--min-alt", type=float, default=30.0, help="Hauteur mini (°)")
    ap.add_argument("--limit", type=int, default=20, help="Nb de cibles")
    args = ap.parse_args()

    when = date.fromisoformat(args.date) if args.date else date.today()
    site = Site(args.lat, args.lon, args.height, args.name, args.tz)

    fov = field_of_view(args.focal)
    night, recos = plan(site, when, focal_mm=args.focal,
                        min_alt=args.min_alt, limit=args.limit)

    print("\n" + "=" * 78)
    print(" {}  —  nuit du {}".format(site.name, when.isoformat()))
    print(" lat {:.3f}°  lon {:.3f}°  alt {:.0f} m  ({})".format(
        site.lat, site.lon, site.height, site.tz))
    print(" A7II @ {:.0f} mm : champ {:.2f}° × {:.2f}°  |  {:.2f}\"/px".format(
        args.focal, fov["fov_w_deg"], fov["fov_h_deg"], fov["scale_arcsec_px"]))
    print("=" * 78)

    if night.kind == "none":
        print(" ⚠ Pas de nuit astronomique à cette date (Soleil jamais sous -18°).")
        return
    print(" Nuit {} : {} → {} (heure locale)".format(
        night.kind, _fmt_t(_to_local(night.start, site)),
        _fmt_t(_to_local(night.end, site))))
    mid = night.start + (night.end - night.start) / 2
    moon = moon_state(site, mid)
    print(" Lune : {:.0f}% éclairée, hauteur {:+.0f}° au cœur de nuit{}".format(
        moon["illum"] * 100, moon["alt"],
        "  (sous l'horizon → ciel sombre 🌑)" if moon["alt"] < 0 else ""))
    print("-" * 78)

    if not recos:
        print(" Aucune cible au-dessus de {:.0f}° cette nuit.".format(args.min_alt))
        return

    print(" {:<9}{:<30}{:>6}{:>9}{:>8}{:>9}".format(
        "Objet", "Nom", "h.max", "au mieux", "ΔLune", "cadre"))
    print("-" * 78)
    for r in recos:
        fit = {True: "✓ tient", False: "✗ large", None: "—"}[r.fits]
        mag = "" if math.isnan(r.target.mag) else "m{:.1f}".format(r.target.mag)
        name = (r.target.name[:26] + "…") if len(r.target.name) > 27 else r.target.name
        print(" {:<9}{:<30}{:>5.0f}°{:>9}{:>7.0f}°{:>9}".format(
            r.target.id, name, r.max_alt, _fmt_t(r.transit_local),
            r.moon_sep, fit))
    print("=" * 78)
    print(" Astuce : 'cadre ✓' = la cible tient dans le champ à {:.0f} mm. ".format(
        args.focal) + "Change --focal pour réévaluer.\n")


if __name__ == "__main__":
    main()
