#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manual_goto.py
==============

"Manual GoTo" for a mount with graduated circles (setting circles) — no motors
needed on the second axis.

Hardware model
--------------
Two rotation axes, assumed perpendicular to each other:

  * AXIS 1 — the mount's polar (RA / hour-angle) axis.  On a Star Adventurer 2i
    it is read on the graduated dial behind the polar scope.  Once polar-aligned
    this axis points at the celestial pole, so turning it changes only the HOUR
    ANGLE of the camera.
  * AXIS 2 — the L-bracket, graduated by hand in degrees.  Its rotation axis is
    perpendicular to axis 1, so turning it changes only the DECLINATION.

The camera sits a few centimetres away from axis 1 (it is *parallel* to it but
offset).  That offset is irrelevant for pointing: a rigid translation of the
camera produces exactly zero pointing error on objects at infinity.  What does
matter is the *direction* of the two axes — polar alignment for axis 1 and the
perpendicularity of axis 2 — see ACCURACY below.

The maths
---------
For a target at right ascension/declination (α, δ) seen at instant *t* from
longitude λ:

    LST = local apparent sidereal time(t, λ)
    H   = LST − α                (hour angle, 0 on the meridian, + to the west)

    axis 1 must be set so the camera's hour angle is H
    axis 2 must be set so the camera's declination is δ

Each axis converts that sky angle into a dial reading with a two-parameter
linear model calibrated once per session (see AxisCal):

    reading = zero + sign × sky_angle / scale        (scale = 15 for an hour dial)

Because the two axes are perpendicular there are always TWO mechanical
positions reaching the same point of the sky (the "meridian flip"):

    A:  (H,        θ₂ = δ)
    B:  (H + 180°, θ₂ = 180° − δ)     ← axis 2 swung over the pole, to the other side

θ₂ is the plate angle counted from the celestial equator.  On a plate graduated
with ZERO AT THE POLE — camera parallel to the polar axis, pointing at Polaris —
the reading is the polar distance and the two solutions are simply symmetric:

    reading = ±(90° − δ)            +48.6° and −48.6° both reach declination 41.4°,
                                     with axis 1 half a turn apart.

Both are returned; pick whichever your bracket can physically reach.

The defaults describe a Star Adventurer 2i as set up here: polar dial graduated
in hours and counting DOWN while the motor tracks (sign −1), L-plate graduated
in degrees with 0 on the pole (sign −1, zero +90).  Only the polar dial's zero
is site/session dependent — one sync on any star fixes it.

ACCURACY
--------
Dial readings are only as good as the alignment behind them:
  * polar alignment error  → same error on the sky (polar scope ≈ 0.2–0.5°)
  * axis-2 non-perpendicularity (cone error) → up to that error near the pole
  * reading a hand-graduated circle ≈ 0.5–1°
So expect ~1° of pointing error, i.e. the target lands inside the frame for any
focal length up to roughly 500 mm on full frame.  Plate-solve to refine.

Coordinates are precessed from J2000 (catalogue) to the date of observation;
that alone is a 0.35° correction in 2026.

Standalone check:
    python manual_goto.py --ra 10.68 --dec 41.27 --lat 48.8566 --lon 2.3522
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

__all__ = ["AxisCal", "DialAngle", "GotoSolution", "ManualGoto",
           "julian_day", "lst_deg", "precess_from_j2000", "radec_to_altaz",
           "pole_offset", "polaris_state", "polaris_apparent", "polar_axis_error",
           "body_track_rate", "rate_summary", "SIDEREAL_RATE_DPS",
           "AlignStar", "AlignResult", "two_star_align", "pair_quality",
           "camera_vec", "aim_angles", "star_vec", "altaz_vec",
           "POLARIS_J2000", "KOCHAB_J2000", "UNIT_SCALE", "UNIT_SPAN"]


# ---------------------------------------------------------------------------
#  Time & coordinates
# ---------------------------------------------------------------------------
def julian_day(dt_utc):
    """Julian day of a naive (or aware) UTC datetime."""
    if dt_utc.tzinfo is not None:
        dt_utc = dt_utc.astimezone(timezone.utc).replace(tzinfo=None)
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    frac = (dt_utc.hour + dt_utc.minute / 60.0
            + (dt_utc.second + dt_utc.microsecond * 1e-6) / 3600.0) / 24.0
    if m <= 2:
        y -= 1; m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5 + frac


def lst_deg(dt_utc, lon_east_deg):
    """Local sidereal time in degrees (same formula as the Sky Map, ~1″/century)."""
    t = julian_day(dt_utc) - 2451545.0
    gmst = 280.46061837 + 360.98564736629 * t
    return (gmst + lon_east_deg) % 360.0


def precess_from_j2000(ra_deg, dec_deg, dt_utc):
    """J2000 catalogue position → mean position of date (IAU 1976 precession).

    Worth doing: by 2026 this already moves a star by ~0.35°, i.e. more than the
    precision of a hand-read dial."""
    t = (julian_day(dt_utc) - 2451545.0) / 36525.0
    # rotation angles, arcseconds -> radians
    as2r = math.pi / (180.0 * 3600.0)
    zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t ** 3) * as2r
    z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t ** 3) * as2r
    theta = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t ** 3) * as2r
    ra0 = math.radians(ra_deg); dec0 = math.radians(dec_deg)
    a = math.cos(dec0) * math.sin(ra0 + zeta)
    b = (math.cos(theta) * math.cos(dec0) * math.cos(ra0 + zeta)
         - math.sin(theta) * math.sin(dec0))
    c = (math.sin(theta) * math.cos(dec0) * math.cos(ra0 + zeta)
         + math.cos(theta) * math.sin(dec0))
    ra = (math.degrees(math.atan2(a, b) + z)) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, c))))
    return ra, dec


def radec_to_altaz(ha_deg, dec_deg, lat_deg):
    """(hour angle, dec) → (altitude, azimuth from N through E), degrees."""
    ha = math.radians(ha_deg); dec = math.radians(dec_deg); lat = math.radians(lat_deg)
    sin_alt = (math.sin(dec) * math.sin(lat)
               + math.cos(dec) * math.cos(lat) * math.cos(ha))
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
    y = -math.cos(dec) * math.cos(lat) * math.sin(ha)
    x = math.sin(dec) - math.sin(math.radians(alt)) * math.sin(lat)
    az = math.degrees(math.atan2(y, x)) % 360.0
    return alt, az


def wrap180(a):
    """Fold an angle into [-180, +180)."""
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
#  Vectors, in the observer's frame: x = north, y = east, z = up
# ---------------------------------------------------------------------------
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def _comb(*pairs):
    """Linear combination: _comb((k1, v1), (k2, v2), ...)."""
    return tuple(sum(k * v[i] for k, v in pairs) for i in range(3))


def altaz_vec(alt_deg, az_deg):
    a, z = math.radians(alt_deg), math.radians(az_deg)
    return (math.cos(a) * math.cos(z), math.cos(a) * math.sin(z), math.sin(a))


def vec_altaz(v):
    return (math.degrees(math.asin(max(-1.0, min(1.0, v[2])))),
            math.degrees(math.atan2(v[1], v[0])) % 360.0)


def star_vec(ra_deg, dec_deg, dt_utc, lat, lon):
    """Where a star is in the sky right now, as a unit vector in the local frame."""
    ha = wrap180(lst_deg(dt_utc, lon) - ra_deg)
    return altaz_vec(*radec_to_altaz(ha, dec_deg, lat))


def sep_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(a, b)))))


def _axis_frame(axis):
    """Two unit vectors spanning the plane the camera swings through, so that the
    axis angle is measured the same way everywhere.  (u, v, axis) is right-handed."""
    ref = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.98 else (1.0, 0.0, 0.0)
    u = _unit(_cross(ref, axis))
    return u, _cross(axis, u)


def camera_vec(axis, theta1_deg, plate_deg):
    """Forward model: where the camera looks, from the two mechanical angles.

    `plate_deg` is the L-plate reading with 0 on the axis, so it is literally the
    angle between the camera and the axis — which is what makes a misaligned
    mount solvable from two stars (each reading is a cone around the axis)."""
    u, v = _axis_frame(axis)
    t, r = math.radians(theta1_deg), math.radians(plate_deg)
    m = _comb((math.cos(t), u), (math.sin(t), v))
    return _comb((math.cos(r), axis), (math.sin(r), m))


def aim_angles(axis, target_vec):
    """Inverse model: the two mechanical angles that put the camera on the target.

    Returns (theta1_deg, plate_deg) for the direct position; the mirrored one is
    (theta1 + 180°, −plate)."""
    u, v = _axis_frame(axis)
    c = max(-1.0, min(1.0, _dot(axis, target_vec)))
    plate = math.degrees(math.acos(c))
    w = _comb((1.0, target_vec), (-c, axis))            # component across the axis
    if math.sqrt(_dot(w, w)) < 1e-12:                   # sitting on the axis itself
        return 0.0, plate
    w = _unit(w)
    return math.degrees(math.atan2(_dot(w, v), _dot(w, u))) % 360.0, plate


# ---------------------------------------------------------------------------
#  Polar alignment: where Polaris has to sit tonight
# ---------------------------------------------------------------------------
#: Polaris (α UMi) and Kochab (β UMi), J2000
POLARIS_J2000 = (37.954560, 89.264109)
KOCHAB_J2000 = (222.676357, 74.155504)

_CLOCK_DIRS = ["straight up", "upper left", "to the left", "lower left",
               "straight down", "lower right", "to the right", "upper right"]


def pole_offset(ra_j2000, dec_j2000, dt_utc, lon_east_deg):
    """Where a near-pole star sits AROUND the celestial pole, in the real sky.

    Returns (hour_angle_hours, polar_distance_deg, clock_hours, direction).

    The position angle is quoted as you see it with the naked eye facing north:
    hour angle 0 puts the star straight above the pole, 6 h to the left (west),
    12 h below, 18 h to the right.  A polar SCOPE may rotate or mirror that —
    which is exactly why the reticle convention has to be checked once against
    a plate-solve rather than reasoned about."""
    ra_d, dec_d = precess_from_j2000(ra_j2000, dec_j2000, dt_utc)
    ha = wrap180(lst_deg(dt_utc, lon_east_deg) - ra_d) / 15.0        # hours
    pdist = 90.0 - dec_d
    clock = (-ha / 2.0) % 12.0                  # 24 h around the pole = 12 clock hours
    direction = _CLOCK_DIRS[int(((ha * 15.0) % 360.0 + 22.5) // 45) % 8]
    return ha, pdist, clock, direction


def polaris_state(dt_utc, lon_east_deg):
    """Polaris' place around the pole right now — see pole_offset."""
    return pole_offset(POLARIS_J2000[0], POLARIS_J2000[1], dt_utc, lon_east_deg)


#: Polaris proper motion (Hipparcos), mas/yr, and distance — it drifts about
#: 0.3′ closer to the pole every year, which matters on a reticle graduated in
#: arcminutes: 39.3′ in 2020, 37.7′ in 2026, 34.8′ in 2040.
POLARIS_PM = (44.48, -11.85, 133.0)


#: (seconds-key, ra, dec, lat, lon) — Polaris' apparent place changes by a fraction
#: of an arcsecond in five minutes, so recomputing it every second is pure waste.
_POLARIS_CACHE = None


def polaris_apparent(dt_utc, lat, lon, height=300.0, max_age_s=300.0):
    """Polaris' APPARENT place: hour angle in hours, polar distance in arcminutes.

    Goes further than the mean position used elsewhere because a polar-scope
    reticle is graduated in arcminutes and the rings are 4′ apart: proper motion,
    nutation and annual aberration together move Polaris by about 0.1′, and the
    secular drift towards the pole by 0.3′ a year. Falls back to precession alone
    when astropy is unavailable, which costs about 0.1′."""
    global _POLARIS_CACHE
    key = julian_day(dt_utc) * 86400.0
    c = _POLARIS_CACHE
    if (c is not None and abs(key - c[0]) <= max_age_s
            and abs(c[3] - lat) < 1e-6 and abs(c[4] - lon) < 1e-6):
        ha = wrap180(lst_deg(dt_utc, lon) - c[1]) / 15.0   # this part IS cheap
        return ha, (90.0 - c[2]) * 60.0
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import astropy.units as u
            from astropy.coordinates import SkyCoord, EarthLocation, CIRS
            from astropy.time import Time
            t = Time(dt_utc)
            loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
            pm_ra, pm_dec, dist = POLARIS_PM
            star = SkyCoord(ra=POLARIS_J2000[0] * u.deg, dec=POLARIS_J2000[1] * u.deg,
                            frame="icrs", pm_ra_cosdec=pm_ra * u.mas / u.yr,
                            pm_dec=pm_dec * u.mas / u.yr, distance=dist * u.pc,
                            obstime=Time("J2000"))
            app = star.apply_space_motion(new_obstime=t).transform_to(
                CIRS(obstime=t, location=loc))
            _POLARIS_CACHE = (key, float(app.ra.deg), float(app.dec.deg), lat, lon)
            ha = wrap180(lst_deg(dt_utc, lon) - float(app.ra.deg)) / 15.0
            return ha, (90.0 - float(app.dec.deg)) * 60.0
    except Exception:               # noqa: BLE001
        ha, pdist, _clock, _d = polaris_state(dt_utc, lon)
        return ha, pdist * 60.0


def polar_axis_error(solved_ra, solved_dec, dt_utc, lon_east_deg):
    """How far the polar axis really points from the pole, from a plate-solve taken
    with the L-plate on 0 (camera parallel to the polar axis).

    Returns (error_deg, hour_angle_of_the_error_hours, direction).  This is the
    only measurement in the whole chain that owes nothing to any dial, reticle or
    convention — it is the ground truth for the polar alignment."""
    err = 90.0 - solved_dec
    ha, _pd, _clock, direction = pole_offset(solved_ra, solved_dec, dt_utc, lon_east_deg)
    return abs(err), ha, direction


# ---------------------------------------------------------------------------
#  Tracking rates — how fast the polar axis must turn for a given target
# ---------------------------------------------------------------------------
SIDEREAL_RATE_DPS = 360.0 / 86164.0905          # degrees of hour angle per second

#: Fallback ratios to sidereal, used only when astropy is unavailable. They are
#: MEAN values: the real rates wander (the Moon's by several percent over a month,
#: a planet's by its whole retrograde loop), which is why they are computed from
#: the ephemeris whenever possible.
MEAN_RATE_RATIO = {
    "sidereal": 1.0,
    "solar": 86164.0905 / 86400.0,              # 0.99727
    "lunar": 0.97633,                           # mean lunar, ~14.685″/s
    "sun": 86164.0905 / 86400.0,
    "moon": 0.97633,
}


_RATE_CACHE = {}


def body_track_rate(body, when_utc, lat, lon, height=300.0, span=1800.0, max_age_s=60.0):
    """How fast the hour angle of `body` changes, in degrees per second.

    A star sits still on the sky, so its hour angle grows at exactly the sidereal
    rate. Everything in the solar system drifts against the stars, so its tracking
    rate is the sidereal rate MINUS its own motion in right ascension:

        dHA/dt = sidereal − dRA/dt

    Measured straight off the ephemeris rather than taken from a table, because
    the sign matters: a planet in retrograde moves westward and therefore needs
    tracking slightly FASTER than sidereal, which no "planetary = sidereal"
    shortcut can express.

    `body` is 'sidereal', or any name astropy knows: sun, moon, jupiter, saturn…
    """
    key = (body or "sidereal").lower()
    if key in ("sidereal", "star", "dso", ""):
        return SIDEREAL_RATE_DPS
    # The Moon's rate wanders 2.4% in a day — 0.003% in a minute. Cheap to cache,
    # and this is called from a two-second poll.
    stamp = julian_day(when_utc) * 86400.0
    hit = _RATE_CACHE.get((key, round(lat, 4), round(lon, 4)))
    if hit is not None and abs(stamp - hit[0]) <= max_age_s:
        return hit[1]
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import astropy.units as u
            from astropy.coordinates import EarthLocation, get_body
            from astropy.time import Time
            from datetime import timedelta
            loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)

            def _ra(t):
                return float(get_body(key, Time(t), loc).ra.deg)

            half = timedelta(seconds=span / 2.0)
            dra = wrap180(_ra(when_utc + half) - _ra(when_utc - half)) / span
            rate = SIDEREAL_RATE_DPS - dra
            _RATE_CACHE[(key, round(lat, 4), round(lon, 4))] = (stamp, rate)
            return rate
    except Exception:               # noqa: BLE001  (no astropy, no ephemeris, bad name)
        return SIDEREAL_RATE_DPS * MEAN_RATE_RATIO.get(key, 1.0)


def body_radec(body, when_utc, lat, lon, height=300.0):
    """Apparent RA/Dec of a solar-system body, at the date, seen from the site.

    Returns None if astropy cannot answer. Worth going to the ephemeris rather than
    reusing a stored position: these objects move (the Moon by 0.5°/h) and their
    coordinates are already of-date, so precessing them from J2000 like a catalogue
    star adds a further 0.37° of pure error."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import astropy.units as u
            from astropy.coordinates import EarthLocation, get_body
            from astropy.time import Time
            loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
            b = get_body(body, Time(when_utc), loc)
            return float(b.ra.deg) % 360.0, float(b.dec.deg)
    except Exception:               # noqa: BLE001
        return None


def rate_summary(body, when_utc, lat, lon):
    """One line describing a tracking rate, for the UI and the logs."""
    omega = body_track_rate(body, when_utc, lat, lon)
    ratio = omega / SIDEREAL_RATE_DPS
    drift = (SIDEREAL_RATE_DPS - omega) * 3600.0            # deg per hour of drift
    return ("{}: {:.6f}°/s = {:.5f}× sidereal ({:+.2f}°/h against the stars{})"
            .format(body, omega, ratio, -drift,
                    ", retrograde" if ratio > 1.0 else ""))


# ---------------------------------------------------------------------------
#  Dial calibration
# ---------------------------------------------------------------------------
UNIT_SCALE = {"hours": 15.0, "degrees": 1.0}    # sky degrees per dial unit
UNIT_SPAN = {"hours": 24.0, "degrees": 360.0}   # dial units in a full turn


@dataclass
class AxisCal:
    """Linear map between a sky angle (degrees) and what is engraved on a dial.

        reading = zero + sign × sky_angle / scale

    `zero` is the reading when the sky angle is 0 (hour angle 0 = meridian for
    axis 1, declination 0 = celestial equator for axis 2).  `sign` is +1 when
    the graduations grow in the same direction as the sky angle, −1 otherwise —
    the one thing you cannot guess from an armchair, so calibrate it (sync on
    two targets) or just try one slew and flip it if the error doubles.
    """
    units: str = "hours"        # "hours" or "degrees"
    zero: float = 0.0           # dial reading at sky angle 0
    sign: float = 1.0           # +1 or -1
    wrap: bool = True           # axis 1 turns all the way round; axis 2 does not
    label: str = "axis"

    # --- unit helpers ------------------------------------------------------
    @property
    def scale(self):
        return UNIT_SCALE.get(self.units, 1.0)

    @property
    def span(self):
        return UNIT_SPAN.get(self.units, 360.0)

    # --- the map and its inverse ------------------------------------------
    def reading(self, sky_deg):
        """Dial reading to set for this sky angle."""
        r = self.zero + self.sign * (sky_deg / self.scale)
        return r % self.span if self.wrap else r

    def sky(self, reading):
        """Sky angle (degrees) corresponding to a dial reading."""
        return (reading - self.zero) * self.scale / self.sign

    # --- calibration -------------------------------------------------------
    def sync(self, sky_deg, reading):
        """One-point sync: you are ON a known object, this is what the dial says.
        Keeps `sign` and solves `zero`."""
        z = reading - self.sign * (sky_deg / self.scale)
        self.zero = z % self.span if self.wrap else z
        return self.zero

    def sync2(self, p1, p2):
        """Two-point sync: [(sky_deg, reading), (sky_deg, reading)] — solves BOTH
        `sign` and `zero`.  Use two targets far apart in the axis' angle."""
        (s1, r1), (s2, r2) = p1, p2
        ds = (s2 - s1) / self.scale
        dr = r2 - r1
        if self.wrap:
            ds = wrap180(ds * self.scale) / self.scale
            dr = (dr + self.span / 2.0) % self.span - self.span / 2.0
        if abs(ds) < 1e-6:
            raise ValueError("the two sync points are at the same angle on this axis")
        self.sign = 1.0 if (dr / ds) >= 0 else -1.0
        self.sync(s1, r1)
        # residual on the second point, as a sanity check for the caller
        resid = self.reading(s2) - r2
        if self.wrap:
            resid = (resid + self.span / 2.0) % self.span - self.span / 2.0
        return self.sign, self.zero, resid * self.scale       # residual in sky degrees

    # --- formatting --------------------------------------------------------
    def fmt(self, reading):
        if self.units == "hours":
            r = reading % 24.0
            h = int(r); m = (r - h) * 60.0
            if m >= 59.95:
                h = (h + 1) % 24; m = 0.0
            return "{:02d} h {:04.1f} m".format(h, m)
        return "{:+.1f}°".format(reading) if not self.wrap else "{:.1f}°".format(reading % 360.0)

    def to_dict(self):
        return {"units": self.units, "zero": self.zero, "sign": self.sign,
                "wrap": self.wrap, "label": self.label}

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {"units", "zero", "sign", "wrap", "label"}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
#  Two-star alignment — works with the polar axis pointing anywhere
# ---------------------------------------------------------------------------
@dataclass
class AlignStar:
    """A star you centred, with what the two dials read at that moment."""
    name: str
    ra_j2000: float
    dec_j2000: float
    when_utc: datetime
    reading1: float                 # polar dial
    reading2: float                 # L-plate, signed (which side of the axis)


@dataclass
class AlignResult:
    ok: bool
    message: str
    axis_alt: float = 0.0           # where the polar axis really points
    axis_az: float = 0.0
    zero1: float = 0.0              # polar-dial reading at axis angle 0
    pole_error: float = 0.0         # how far that is from the celestial pole
    spread: float = 0.0             # the two stars' disagreement, in degrees of sky
    residuals: list = field(default_factory=list)   # per-star pointing error, degrees
    ambiguous: bool = False
    grade: str = "good"            # how well-conditioned the pair was
    why: str = ""


def pair_quality(sep, plate1, plate2):
    """Judge a pair of alignment stars BEFORE spending the night on it.

    Two cones cut cleanly only when they are genuinely different cones.  A star
    sitting near the axis has a tiny cone and pins nothing; two stars at the same
    distance from the axis give near-concentric cones that graze each other.
    Both cases quietly multiply a 0.5° reading error into several degrees, which
    is why this is checked rather than left to luck."""
    dplate = abs(plate1 - plate2)
    near = min(abs(plate1), abs(plate2))
    if near < 20.0:
        return ("bad", "One star is only {:.0f}° from the axis — too close to pin it "
                       "down. Use a star further out.".format(near))
    if sep < 40.0:
        return ("bad", "Only {:.0f}° apart. Use two stars in different parts of the "
                       "sky.".format(sep))
    if dplate < 20.0:
        return ("bad", "Both stars are about the same distance from the axis "
                       "({:.0f}° apart) — their cones barely cross. Pick one high "
                       "and one low.".format(dplate))
    if sep < 60.0 or dplate < 30.0:
        return ("ok", "Usable pair (separation {:.0f}°, plate difference {:.0f}°); "
                      "further apart would be better.".format(sep, dplate))
    return ("good", "Good pair — separation {:.0f}°, plate difference {:.0f}°."
                    .format(sep, dplate))


def two_star_align(stars, lat, lon, axis1, precess=True):
    """Solve where the polar axis actually points, from two centred stars.

    No polar alignment needed.  The trick is that a plate graduated 0 on the axis
    reads the angle between the camera and the axis, so each star pins the axis to
    a cone around itself.  Two cones cut in two directions; the polar dial then
    says which of the two is real, because only one makes both stars agree on
    where the dial's zero is.

    Unknowns are the axis direction (2) and the dial zero (1); two stars give 4
    equations, so the fourth is left over as a genuine check — `spread` and
    `residuals` are that check, not decoration."""
    if len(stars) != 2:
        return AlignResult(False, "Two stars are needed, no more and no less.")
    vecs, rho = [], []
    for st in stars:
        ra, dec = (precess_from_j2000(st.ra_j2000, st.dec_j2000, st.when_utc)
                   if precess else (st.ra_j2000 % 360.0, st.dec_j2000))
        vecs.append(star_vec(ra, dec, st.when_utc, lat, lon))
        rho.append(math.radians(st.reading2))
    s1, s2 = vecs
    g = _dot(s1, s2)
    if abs(g) > 0.999:
        return AlignResult(False, "Those two stars are too close together (or the same "
                                  "one). Pick a pair well apart in the sky — 60° or more.")
    c1, c2 = math.cos(rho[0]), math.cos(rho[1])
    den = 1.0 - g * g
    a = (c1 - g * c2) / den
    b = (c2 - g * c1) / den
    k2 = (1.0 - a * a - b * b - 2.0 * a * b * g) / den
    if k2 < -1e-6:
        return AlignResult(False, "Those readings cannot both be right: the two plate "
                                  "angles are incompatible with the stars' separation. "
                                  "Re-read the dials, or re-centre and start again.")
    k = math.sqrt(max(k2, 0.0))
    cr = _cross(s1, s2)

    best = None
    for sgn in (1.0, -1.0):
        axis = _unit(_comb((a, s1), (b, s2), (sgn * k, cr)))
        zeros = []
        for st, sv in zip(stars, vecs):
            th, _plate = aim_angles(axis, sv)
            if st.reading2 < 0:                 # camera on the other side of the axis
                th = (th + 180.0) % 360.0
            zeros.append((st.reading1 - axis1.sign * (th / axis1.scale)) % axis1.span)
        half = axis1.span / 2.0
        gap = (zeros[0] - zeros[1] + half) % axis1.span - half
        cand = (abs(gap * axis1.scale), axis, zeros, gap)
        if best is None or cand[0] < best[0]:
            best = cand if best is None else best
            best = min([best, cand], key=lambda c: c[0])
    spread, axis, zeros, gap = best

    zero1 = (zeros[1] + gap / 2.0) % axis1.span     # circular mean of the two
    resid = []
    for st, sv in zip(stars, vecs):
        th = (st.reading1 - zero1) * axis1.scale / axis1.sign
        resid.append(sep_deg(sv, camera_vec(axis, th, st.reading2)))
    alt, az = vec_altaz(axis)
    perr = sep_deg(axis, altaz_vec(lat, 0.0))
    grade, why = pair_quality(sep_deg(s1, s2), stars[0].reading2, stars[1].reading2)
    amb = spread > 5.0 or grade == "bad"
    msg = ("Axis solved: it points alt {:+.2f}° az {:.2f}°, which is {:.2f}° from the "
           "celestial pole. Dial zero {:.3f}. The two stars agree to {:.2f}°."
           .format(alt, az, perr, zero1, spread))
    if spread > 5.0:
        msg += (" That disagreement is large — suspect a misread dial, a stale reading, "
                "or the wrong sign on the L-plate.")
    if grade != "good":
        msg += " " + why
    return AlignResult(True, msg, axis_alt=alt, axis_az=az, zero1=zero1,
                       pole_error=perr, spread=spread, residuals=resid, ambiguous=amb,
                       grade=grade, why=why)


# ---------------------------------------------------------------------------
#  A solution: where to put each axis
# ---------------------------------------------------------------------------
@dataclass
class DialAngle:
    """One axis of one solution."""
    sky_deg: float              # hour angle, or declination
    reading: float              # what to line up on the dial
    text: str                   # human-readable reading
    turn_from: float = None     # dial units to turn from the current position
    turn_sky_deg: float = None  # the same, in degrees of sky


@dataclass
class GotoSolution:
    """One of the two mechanical ways to reach the target."""
    flipped: bool
    axis1: DialAngle            # polar / RA axis
    axis2: DialAngle            # L-bracket / Dec axis
    reachable: bool = True      # the axis-2 reading is inside the bracket's travel
    note: str = ""
    pole_dist: float = 0.0      # angle between the camera and the polar axis


@dataclass
class GotoResult:
    """Everything the UI needs for one click on one target."""
    name: str
    ra_j2000: float
    dec_j2000: float
    ra_date: float
    dec_date: float
    lst: float
    hour_angle: float           # [-180, 180), + = west of the meridian
    alt: float
    az: float
    when_utc: datetime = None
    primary: GotoSolution = None
    alternate: GotoSolution = None
    warnings: list = field(default_factory=list)

    @property
    def below_horizon(self):
        return self.alt < 0.0

    @property
    def ha_hours(self):
        return self.hour_angle / 15.0


# ---------------------------------------------------------------------------
#  The engine
# ---------------------------------------------------------------------------
class ManualGoto:
    """Turns (target, time, site) into the two dial readings to set by hand.

        g = ManualGoto(lat=48.8566, lon=2.3522)
        g.axis2.zero = 0.0            # L-bracket reads 0 at the celestial equator
        res = g.solve("M31", 10.6847, 41.269, datetime.utcnow())
        print(res.primary.axis1.text, res.primary.axis2.text)
    """

    #: Targets whose position comes from the ephemeris, never from a catalogue.
    SOLAR_BODIES = ("sun", "moon", "mercury", "venus", "mars", "jupiter",
                    "saturn", "uranus", "neptune")

    #: how far the L-bracket can swing either side of the POLE position, in degrees.
    #: Expressed that way rather than as a min/max reading because it is the physical
    #: limit, and because it stays correct whether the plate is graduated ±180 about
    #: the pole or 0–360 continuously — where "−48.6" and "311.4" are the same place.
    DEFAULT_AXIS2_TRAVEL = 100.0

    def __init__(self, lat=0.0, lon=0.0, axis1=None, axis2=None,
                 axis2_travel=None, precess=True):
        self.lat = float(lat)
        self.lon = float(lon)
        # default polar dial: graduated in hours, counting DOWN as the motor tracks
        # (the hour angle grows while the reading shrinks) -> sign = -1.
        self.axis1 = axis1 or AxisCal(units="hours", zero=0.0, sign=-1.0,
                                      wrap=True, label="RA axis (polar dial)")
        # default L-bracket: graduated 0 at the pole, so the reading is the polar
        # distance 90° − δ (sign = −1, zero = +90 make that exact).
        self.axis2 = axis2 or AxisCal(units="degrees", zero=90.0, sign=-1.0,
                                      wrap=False, label="Dec axis (L-bracket)")
        self.axis2_travel = float(axis2_travel or self.DEFAULT_AXIS2_TRAVEL)
        self.precess = bool(precess)
        #: where the polar axis really points (alt, az), from a two-star align.
        #: None means "assume it is on the celestial pole" — the polar-aligned case.
        self.axis_altaz = None
        #: last position actually dialled in, so we can show "turn by …"
        self.current = None      # (axis1_reading, axis2_reading) or None

    # -- site ---------------------------------------------------------------
    def set_site(self, lat, lon):
        self.lat = float(lat); self.lon = float(lon)

    def axis_error(self):
        """How far the (measured) polar axis is from the celestial pole, in degrees."""
        if self.axis_altaz is None:
            return 0.0
        return sep_deg(altaz_vec(*self.axis_altaz), altaz_vec(self.lat, 0.0))

    def apply_align(self, result):
        """Adopt a two-star solution: the axis direction and the dial zero."""
        if not result.ok:
            return False
        self.axis_altaz = (result.axis_alt, result.axis_az)
        self.axis1.zero = result.zero1
        return True

    def clear_align(self):
        """Go back to trusting the polar alignment (axis assumed on the pole)."""
        self.axis_altaz = None

    def resolve(self, name, ra_j2000, dec_j2000, when_utc):
        """Where the target actually is at `when_utc`, in coordinates of the date.

        A star is a fixed catalogue entry and needs precessing. A planet or the Moon
        is not: its stored position is already of-date and already stale, so it is
        taken from the ephemeris instead."""
        key = (name or "").strip().lower()
        if key in self.SOLAR_BODIES:
            rd = body_radec(key, when_utc, self.lat, self.lon)
            if rd:
                return rd
        if self.precess:
            return precess_from_j2000(ra_j2000, dec_j2000, when_utc)
        return ra_j2000 % 360.0, dec_j2000

    def axis2_pole_reading(self):
        """What the L-plate reads when the camera is parallel to the axis."""
        return self.axis2.reading(90.0)

    def axis2_from_pole(self, reading):
        """Angle between a plate reading and the pole position, 0–180°.

        Going through the shortest way round is what makes a 0–360 graduation work:
        311.4° is not 311° from the pole, it is 48.6° the other way."""
        return abs(wrap180((reading - self.axis2_pole_reading()) * self.axis2.scale))

    def set_axis2_pole_zero(self, sign=-1.0, wrap=False):
        """Calibrate axis 2 from the easiest reference there is: the plate reads 0
        when the camera is parallel to the polar axis (pointing at the pole).

        The reading is then the polar distance, 90° − δ.  `sign` only picks which
        side of the pole counts as positive — and that choice is exactly the choice
        between the two solutions, so getting it 'wrong' costs nothing: the target
        is then at the same number with the opposite sign."""
        self.axis2.units = "degrees"
        self.axis2.wrap = bool(wrap)                   # True for a 0–360 graduation
        self.axis2.sign = 1.0 if sign >= 0 else -1.0
        self.axis2.zero = (-self.axis2.sign * 90.0) % 360.0 if wrap \
            else -self.axis2.sign * 90.0               # reading 0 at δ = +90°
        return self.axis2.zero

    # -- the one call the UI makes -----------------------------------------
    def solve(self, name, ra_j2000, dec_j2000, when_utc=None):
        """Dial readings that put `name` in the centre of the frame at `when_utc`."""
        when_utc = when_utc or datetime.utcnow()
        ra_d, dec_d = self.resolve(name, ra_j2000, dec_j2000, when_utc)
        lst = lst_deg(when_utc, self.lon)
        ha = wrap180(lst - ra_d)
        alt, az = radec_to_altaz(ha, dec_d, self.lat)

        if self.axis_altaz is None:
            # polar-aligned: the axis angle IS the hour angle, the plate angle the dec
            ang1, mech2 = ha, dec_d
        else:
            # axis measured by a two-star align: go through the general model
            axis = altaz_vec(*self.axis_altaz)
            tgt = star_vec(ra_d, dec_d, when_utc, self.lat, self.lon)
            th1, plate = aim_angles(axis, tgt)
            ang1, mech2 = th1, 90.0 - plate

        primary = self._solution(ang1, mech2, flipped=False)
        # the same point of sky with axis 2 swung over the pole to the other side
        alternate = self._solution(ang1 + 180.0, 180.0 - mech2, flipped=True)
        # an unreachable bracket angle demotes a solution
        if not primary.reachable and alternate.reachable:
            primary, alternate = alternate, primary

        warn = []
        if alt < 0:
            warn.append("Below the horizon ({:+.1f}°) — it will not be visible.".format(alt))
        elif alt < 15:
            warn.append("Low on the horizon ({:.1f}°) — haze, and refraction lifts it "
                        "by ~{:.2f}°, so aim slightly low.".format(
                            alt, 0.0167 / math.tan(math.radians(max(alt, 1.0) + 7.31 /
                                                                 (max(alt, 1.0) + 4.4)))))
        if abs(dec_d) > 80:
            warn.append("Close to the pole: the hour-angle dial becomes very "
                        "insensitive and any cone error shows up here.")
        if abs(ha) < 2.0:
            warn.append("Near the meridian — the mount may need a flip within the hour.")
        if self.axis_altaz is not None and self.axis_error() > 1.0:
            warn.append("Axis {:.1f}° off the pole: pointing is corrected for it, but "
                        "TRACKING is not — stars will trail by roughly {:.0f}\u2033 per "
                        "minute of exposure. Fine for finding and framing, not for long "
                        "subs.".format(self.axis_error(),
                                       self.axis_error() * 0.00218 * 3600.0))
        if not primary.reachable:
            warn.append("Both L-plate positions are more than {:.0f}° from the pole, "
                        "which you said the plate cannot reach. Widen the travel, or "
                        "this target is out of the bracket's range."
                        .format(self.axis2_travel))
        return GotoResult(name=name, ra_j2000=ra_j2000 % 360.0, dec_j2000=dec_j2000,
                          ra_date=ra_d, dec_date=dec_d, lst=lst, hour_angle=ha,
                          alt=alt, az=az, when_utc=when_utc,
                          primary=primary, alternate=alternate, warnings=warn)

    def _solution(self, ha_deg, dec_mech, flipped):
        """One mechanical position. `dec_mech` is the plate angle counted from the
        celestial equator: δ for the direct position, 180° − δ for the flipped one."""
        ha_deg = wrap180(ha_deg)
        r1 = self.axis1.reading(ha_deg)
        r2 = self.axis2.reading(dec_mech)
        a1 = DialAngle(ha_deg, r1, self.axis1.fmt(r1))
        a2 = DialAngle(dec_mech, r2, self.axis2.fmt(r2))
        if self.current:
            a1.turn_from, a1.turn_sky_deg = self._turn(self.axis1, self.current[0], r1)
            a2.turn_from, a2.turn_sky_deg = self._turn(self.axis2, self.current[1], r2)
        # what matters is whether the plate can physically swing that far from the
        # pole, not where the number happens to fall on the scale
        reach = self.axis2_from_pole(r2) <= self.axis2_travel
        note = ("L-plate on the other side of the pole (flipped)" if flipped
                else "direct position")
        return GotoSolution(flipped=flipped, axis1=a1, axis2=a2, reachable=reach,
                            note=note, pole_dist=abs(wrap180(90.0 - dec_mech)))

    @staticmethod
    def _turn(cal, frm, to):
        """Shortest turn, in dial units and in sky degrees."""
        d = to - frm
        if cal.wrap:
            d = (d + cal.span / 2.0) % cal.span - cal.span / 2.0
        return d, d * cal.scale

    # -- calibration helpers -----------------------------------------------
    def sync(self, ra_j2000, dec_j2000, reading1, reading2, when_utc=None,
             axis1=True, axis2=True, name=""):
        """You are centred on a known star and the dials read (reading1, reading2):
        solve the zero of each axis you ask for.

        Axis 1 needs this — nothing else can tell it which graduation faces the
        index at hour angle 0.  Axis 2 usually does NOT: a plate graduated 0 on
        the pole is already calibrated, so leave `axis2` off unless you want to
        re-derive it (see axis2_pole_error for the cross-check)."""
        when_utc = when_utc or datetime.utcnow()
        ra_d, dec_d = self.resolve(name, ra_j2000, dec_j2000, when_utc)
        ha = wrap180(lst_deg(when_utc, self.lon) - ra_d)
        z1 = self.axis1.sync(ha, reading1) if axis1 else self.axis1.zero
        z2 = self.axis2.sync(dec_d, reading2) if axis2 else self.axis2.zero
        self.current = (float(reading1), float(reading2))
        return z1, z2

    def axis2_pole_error(self, dec_deg, reading):
        """How far this star says the L-plate zero is from the pole convention.

        A sanity check with real diagnostic value: centred on a star of known
        declination, a plate reading 0 on the pole must show ±(90° − δ).  Any
        gap is the sum of your polar-alignment error, the plate's squareness and
        how well the star was centred — if it is big, something is off."""
        expected = -self.axis2.sign * 90.0          # zero implied by "0 at the pole"
        implied = reading - self.axis2.sign * dec_deg
        return wrap180(implied - expected)

    def mark_current(self, reading1, reading2):
        """Remember where the dials are now, so the next solve says 'turn by …'."""
        self.current = (float(reading1), float(reading2))

    # -- how fast the reading goes stale -----------------------------------
    @staticmethod
    def drift_note(tracking):
        if tracking:
            return ("Tracking on: once set, the RA dial follows the sky — the target "
                    "stays centred.")
        return ("Not tracking: the sky moves 15°/h (1 h of dial per hour), i.e. "
                "0.25°/min — set the dial and start tracking within a minute.")

    # -- persistence --------------------------------------------------------
    def to_dict(self):
        return {"axis1": self.axis1.to_dict(), "axis2": self.axis2.to_dict(),
                "axis2_travel": self.axis2_travel,
                "precess": self.precess, "axis_altaz": self.axis_altaz}

    def load_dict(self, d):
        d = d or {}
        if d.get("axis1"):
            self.axis1 = AxisCal.from_dict(d["axis1"])
        if d.get("axis2"):
            self.axis2 = AxisCal.from_dict(d["axis2"])
        if "axis2_travel" in d:
            self.axis2_travel = float(d["axis2_travel"])
        elif "axis2_max" in d:                          # settings from an older build
            self.axis2_travel = float(d["axis2_max"])
        aa = d.get("axis_altaz")
        self.axis_altaz = (float(aa[0]), float(aa[1])) if aa else None
        self.precess = bool(d.get("precess", self.precess))
        return self


# ---------------------------------------------------------------------------
#  CLI — sanity check without the GUI
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Manual GoTo — dial readings for a target")
    ap.add_argument("--ra", type=float, required=True, help="RA J2000 in degrees")
    ap.add_argument("--dec", type=float, required=True, help="Dec J2000 in degrees")
    ap.add_argument("--lat", type=float, default=48.8566)
    ap.add_argument("--lon", type=float, default=2.3522)
    ap.add_argument("--name", default="target")
    ap.add_argument("--utc", default=None, help="YYYY-MM-DDTHH:MM:SS (default: now)")
    ap.add_argument("--ra-zero", type=float, default=0.0)
    ap.add_argument("--ra-sign", type=float, default=-1.0,
                    help="-1 when the dial counts down as the mount tracks")
    ap.add_argument("--dec-zero", type=float, default=90.0,
                    help="L-plate reading at declination 0 (90 = plate reads 0 at the pole)")
    ap.add_argument("--dec-sign", type=float, default=-1.0)
    a = ap.parse_args()
    when = datetime.fromisoformat(a.utc) if a.utc else datetime.utcnow()
    g = ManualGoto(lat=a.lat, lon=a.lon)
    g.axis1.zero, g.axis1.sign = a.ra_zero, a.ra_sign
    g.axis2.zero, g.axis2.sign = a.dec_zero, a.dec_sign
    r = g.solve(a.name, a.ra, a.dec, when)
    print("{}  RA/Dec of date {:.4f}° {:+.4f}°".format(r.name, r.ra_date, r.dec_date))
    print("UTC {}   LST {:.3f}°   HA {:+.3f}° ({:+.3f} h)".format(
        r.when_utc.strftime("%Y-%m-%d %H:%M:%S"), r.lst, r.hour_angle, r.ha_hours))
    print("Alt {:+.2f}°  Az {:.2f}°".format(r.alt, r.az))
    for tag, s in (("PRIMARY  ", r.primary), ("ALTERNATE", r.alternate)):
        print("{}  axis1 {}   axis2 {}   [{}]".format(
            tag, s.axis1.text, s.axis2.text, s.note))
    for w in r.warnings:
        print("  ! " + w)


if __name__ == "__main__":
    main()
