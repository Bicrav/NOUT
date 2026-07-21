#!/usr/bin/env python3
"""Download the FULL Lynds' Catalogue of Dark Nebulae (VII/7A, ~1802 objects) from VizieR
and write `catalog_ldn.csv` next to this script, in the format NOUT's Sky Map loads:

    id,name,ra_deg,dec_deg,size_arcmin

Run it ONCE (needs Internet):  python3 fetch_ldn.py
Installs astroquery automatically if needed. Coordinates are J2000 (computed by VizieR).
"""
import math
import os
import subprocess
import sys


def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        print("Installing {}…".format(pkg))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])


def _col(row, names):
    for n in names:
        if n in row.colnames:
            try:
                return row[n]
            except Exception:       # noqa: BLE001
                pass
    return None


def main():
    _ensure("astroquery")
    from astroquery.vizier import Vizier

    print("Querying VizieR VII/7A/ldn (Lynds' Dark Nebulae)…")
    v = Vizier(columns=["**", "_RAJ2000", "_DEJ2000"], row_limit=-1)
    cats = v.get_catalogs("VII/7A/ldn")
    if not cats:
        print("No data returned — check your Internet connection."); return
    tab = cats[0]
    print("Columns available:", ", ".join(tab.colnames))

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "catalog_ldn.csv")
    n = 0
    with open(out, "w") as f:
        f.write("id,name,ra_deg,dec_deg,size_arcmin\n")
        for row in tab:
            ra = _col(row, ["_RAJ2000", "RAJ2000"])
            dec = _col(row, ["_DEJ2000", "DEJ2000"])
            if ra is None or dec is None:
                continue
            try:
                ra = float(ra); dec = float(dec)
            except Exception:       # noqa: BLE001
                continue
            ldn = _col(row, ["LDN", "Name", "Seq"])
            try:
                ldn = int(ldn)
            except Exception:       # noqa: BLE001
                continue
            area = _col(row, ["Area", "Cloud_Area", "area"])
            try:
                size = max(4.0, math.sqrt(max(float(area), 0.0)) * 60.0)
            except Exception:       # noqa: BLE001
                size = 20.0
            f.write("LDN{},LDN {},{:.4f},{:.4f},{:.0f}\n".format(ldn, ldn, ra, dec, size))
            n += 1
    print("Wrote {} objects to {}".format(n, out))
    print("Relaunch NOUT — the Sky Map and preview overlay now include all LDN objects.")


if __name__ == "__main__":
    main()
