"""Satellite pass prediction (ISS & others) for NOUT.

Pure-Python on top of `sgp4`. Given a TLE and an observer site, predicts upcoming
passes: rise/max/set times, max elevation, directions, and whether the pass is
*visible* (satellite sunlit while the observer is in the dark).

Offline once a TLE is available. TLEs go stale in days — refresh from Celestrak.
"""
import math
from datetime import datetime, timedelta

_RE_KM = 6378.137          # Earth equatorial radius (km)
_E2 = 6.69437999014e-3     # WGS-84 first eccentricity squared


def _gmst_rad(jd):
    t = (jd - 2451545.0) / 36525.0
    g = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
         + 0.000387933 * t * t - t * t * t / 38710000.0)
    return math.radians(g % 360.0)


def _sun_eci_and_alt(jd, lat, lon):
    """Return (sun unit vector in ECI, Sun altitude at the observer in deg)."""
    n = jd - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    sun = (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))
    gmst = _gmst_rad(jd)
    ha = (gmst + math.radians(lon)) - ra
    la = math.radians(lat)
    alt = math.asin(math.sin(la) * math.sin(dec)
                    + math.cos(la) * math.cos(dec) * math.cos(ha))
    return sun, math.degrees(alt)


def _observer_ecef(lat, lon, h_km=0.3):
    la, lo = math.radians(lat), math.radians(lon)
    nph = _RE_KM / math.sqrt(1.0 - _E2 * math.sin(la) ** 2)
    x = (nph + h_km) * math.cos(la) * math.cos(lo)
    y = (nph + h_km) * math.cos(la) * math.sin(lo)
    z = (nph * (1.0 - _E2) + h_km) * math.sin(la)
    return x, y, z


def _altaz(sat_teme, jd, lat, lon, h_km=0.3):
    """Topocentric alt/az (deg) of a TEME position (km) for a site."""
    theta = _gmst_rad(jd)
    rx, ry, rz = sat_teme
    ex = rx * math.cos(theta) + ry * math.sin(theta)        # TEME -> ECEF
    ey = -rx * math.sin(theta) + ry * math.cos(theta)
    ez = rz
    ox, oy, oz = _observer_ecef(lat, lon, h_km)
    dx, dy, dz = ex - ox, ey - oy, ez - oz
    la, lo = math.radians(lat), math.radians(lon)
    east = -math.sin(lo) * dx + math.cos(lo) * dy
    north = -math.sin(la) * math.cos(lo) * dx - math.sin(la) * math.sin(lo) * dy \
        + math.cos(la) * dz
    up = math.cos(la) * math.cos(lo) * dx + math.cos(la) * math.sin(lo) * dy \
        + math.sin(la) * dz
    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    alt = math.degrees(math.asin(max(-1.0, min(1.0, up / rng))))
    az = math.degrees(math.atan2(east, north)) % 360.0
    return alt, az, rng


def _is_sunlit(sat_teme, sun_hat):
    """Sat in sunlight if not inside Earth's shadow cylinder (ECI ~ TEME)."""
    proj = sum(a * b for a, b in zip(sat_teme, sun_hat))
    if proj > 0:
        return True
    perp = math.sqrt(sum((a - proj * b) ** 2 for a, b in zip(sat_teme, sun_hat)))
    return perp > _RE_KM


def _az_name(az):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((az + 22.5) % 360 // 45)]


def parse_tle(text):
    """Return (name, l1, l2) from a TLE blob (2 or 3 lines)."""
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = lines[i - 1] if i > 0 and not lines[i - 1].startswith(("1 ", "2 ")) else "Satellite"
            return name.strip(), ln, lines[i + 1]
    raise ValueError("No valid TLE (need lines starting with '1 ' and '2 ').")


def topocentric_radec(tle_text, lat, lon, dt=None, h_km=0.3):
    """Topocentric apparent RA/Dec (deg) + alt/az (deg) of a satellite at `dt` (UTC)
    seen from a site. Returns None on error. RA/Dec is observer-referenced, so the
    usual LST->alt/az transform in a star map places it correctly."""
    try:
        from sgp4.api import Satrec, jday
    except Exception:                 # noqa: BLE001
        return None
    if dt is None:
        dt = datetime.utcnow()
    try:
        _name, l1, l2 = parse_tle(tle_text)
        sat = Satrec.twoline2rv(l1, l2)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                      dt.second + dt.microsecond * 1e-6)
        err, r, _v = sat.sgp4(jd, fr)
        if err != 0:
            return None
        jdt = jd + fr
        theta = _gmst_rad(jdt)
        rx, ry, rz = r
        ex = rx * math.cos(theta) + ry * math.sin(theta)      # TEME -> ECEF
        ey = -rx * math.sin(theta) + ry * math.cos(theta)
        ez = rz
        ox, oy, oz = _observer_ecef(lat, lon, h_km)
        dx, dy, dz = ex - ox, ey - oy, ez - oz                # topocentric, ECEF
        # ECEF -> ECI (rotate back by +theta) to get apparent RA/Dec
        ix = dx * math.cos(theta) - dy * math.sin(theta)
        iy = dx * math.sin(theta) + dy * math.cos(theta)
        iz = dz
        ra = math.degrees(math.atan2(iy, ix)) % 360.0
        dec = math.degrees(math.atan2(iz, math.sqrt(ix * ix + iy * iy)))
        # alt/az for convenience
        la, lo = math.radians(lat), math.radians(lon)
        east = -math.sin(lo) * dx + math.cos(lo) * dy
        north = -math.sin(la) * math.cos(lo) * dx - math.sin(la) * math.sin(lo) * dy \
            + math.cos(la) * dz
        up = math.cos(la) * math.cos(lo) * dx + math.cos(la) * math.sin(lo) * dy \
            + math.sin(la) * dz
        rng = math.sqrt(dx * dx + dy * dy + dz * dz)
        alt = math.degrees(math.asin(max(-1.0, min(1.0, up / rng))))
        az = math.degrees(math.atan2(east, north)) % 360.0
        return ra, dec, alt, az
    except Exception:                 # noqa: BLE001
        return None


def predict_passes(tle_text, lat, lon, start=None, hours=48, min_alt=10.0,
                   step_s=30, visible_only=True, h_km=0.3):
    """List upcoming passes for a TLE seen from (lat, lon).

    Returns a list of dicts: rise/max/set datetimes (UTC), max_alt, rise_az, max_az,
    set_az, visible (sunlit + dark observer)."""
    from sgp4.api import Satrec, jday
    name, l1, l2 = parse_tle(tle_text)
    sat = Satrec.twoline2rv(l1, l2)
    if start is None:
        start = datetime.utcnow()
    passes = []
    cur = None
    n_steps = int(hours * 3600 / step_s) + 1
    for k in range(n_steps):
        t = start + timedelta(seconds=k * step_s)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        err, r, _v = sat.sgp4(jd, fr)
        if err != 0:
            continue
        alt, az, _rng = _altaz(r, jd + fr, lat, lon, h_km)
        if alt >= min_alt:
            sun_hat, sun_alt = _sun_eci_and_alt(jd + fr, lat, lon)
            lit = _is_sunlit(r, sun_hat)
            vis = lit and sun_alt < -6.0
            if cur is None:
                cur = {"rise": t, "rise_az": az, "max_alt": alt, "max_az": az,
                       "max": t, "visible": vis}
            else:
                if alt > cur["max_alt"]:
                    cur["max_alt"] = alt; cur["max_az"] = az; cur["max"] = t
                cur["visible"] = cur["visible"] or vis
            cur["set"] = t; cur["set_az"] = az
        elif cur is not None:
            passes.append(cur); cur = None
    if cur is not None:
        passes.append(cur)
    if visible_only:
        passes = [p for p in passes if p.get("visible")]
    for p in passes:
        p["rise_dir"] = _az_name(p["rise_az"]); p["set_dir"] = _az_name(p["set_az"])
        p["max_dir"] = _az_name(p["max_az"])
    return passes
