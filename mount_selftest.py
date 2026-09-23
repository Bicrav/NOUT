#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mount_selftest.py
=================

Staged on-sky check of the Star Adventurer 2i Wi-Fi motor protocol, which the
NOUT code implements from the public SkyWatcher command set but has never had
validated against real hardware.

Run it with the Mac joined to the mount's own Wi-Fi network.

The stages are deliberately ordered so that nothing moves until the stage before
it has proved the protocol is understood:

  probe   read-only. Handshake, counts-per-revolution, timer frequency, axis
          position, axis status. NOTHING MOVES. If this fails, every byte-level
          assumption in the code is wrong and there is no point going further.
  track   starts sidereal tracking and samples the step counter for ~20 s. Motion
          is sidereal, i.e. invisible. This is the strong test: the counter must
          advance at exactly one revolution per sidereal day, which cross-checks
          the counts-per-revolution AND the timer frequency AND ties the counter's
          direction to the sky.
  move    one small GoTo (default 2°) and back. THE MOUNT MOVES. Establishes
          which way the direction bit actually drives the axis.

Usage:
    python mount_selftest.py                      # probe only, cannot move anything
    python mount_selftest.py --stage track
    python mount_selftest.py --stage move --deg 2
    python mount_selftest.py --stage all --deg 2
    python mount_selftest.py --simulate           # against the fake board, no hardware

Safety: keep a hand on the power switch for the `move` stage, make sure the camera
is clear of the tripod legs, and start with a small angle.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIDEREAL_DAY = 86164.0905


def _mount(args):
    from sony_tether_focus import SkyWatcherMount
    if args.simulate:
        from _fake_mount_board import FakeBoard
        m = SkyWatcherMount.__new__(SkyWatcherMount)
        b = FakeBoard()
        m.ip, m.port, m.sock = "sim", 0, None
        m.cpr = m.freq = m.cpr2 = m.freq2 = None
        m.two_axis = False
        m.motor_sign = m.command_sign = None
        m._log = lambda s: print("    " + s)
        m._send = lambda body: b.send(body)
        return m
    return SkyWatcherMount(args.ip, timeout=args.timeout,
                           logfn=lambda s: print("    " + s))


def stage_probe(m):
    """Read-only. Nothing moves."""
    print("\n=== PROBE (read-only) ===")
    info = m.connect(preserve_position=True)     # do not disturb an existing count
    print("  connect -> {}".format(info))
    if not m.cpr or not m.freq:
        print("  FAIL: the board did not report CPR/frequency.")
        return False
    print("  counts per revolution : {}".format(m.cpr))
    print("  timer frequency       : {} Hz".format(m.freq))
    print("  one sidereal day      : {:.1f} counts/s expected while tracking"
          .format(m.cpr / SIDEREAL_DAY))
    try:
        print("  axis position         : {} counts ({:+.3f}°)"
              .format(m.get_counts(), m.counts_to_deg(m.get_counts())))
    except Exception as e:                       # noqa: BLE001
        print("  FAIL reading position (j1): {}".format(e)); return False
    try:
        print("  axis status (f1)      : {}".format(m._send("f1")))
    except Exception as e:                       # noqa: BLE001
        print("  note: status read failed: {}".format(e))
    print("  max slew rate          : {:.3f}°/s  (90° in {:.1f} min)"
          .format(m.max_slew_deg_per_s(256), 90 / m.max_slew_deg_per_s(256) / 60))
    print("  PROBE OK — the protocol is understood.")
    return True


def stage_track(m, seconds):
    """Sidereal tracking only: motion is invisible, but the counter must move."""
    print("\n=== TRACK ({} s, sidereal — motion is invisible) ===".format(seconds))
    m.resume_tracking()
    p0 = m.get_counts(); t0 = time.monotonic()
    time.sleep(seconds)
    p1 = m.get_counts(); dt = time.monotonic() - t0
    dc = p1 - p0
    if dc == 0:
        print("  FAIL: the counter did not move. Tracking is not running, or j1 is "
              "not a live position.")
        return False
    rate = dc / dt
    expect = m.cpr / SIDEREAL_DAY
    err = abs(abs(rate) / expect - 1.0) * 100.0
    print("  counter moved {} counts in {:.1f}s -> {:.2f} counts/s".format(dc, dt, rate))
    print("  expected {:.2f} counts/s (sidereal)  ->  {:.2f}% off".format(expect, err))
    print("  encoder direction vs the sky: {:+.0f}".format(1 if dc > 0 else -1))
    if err > 5.0:
        print("  FAIL: rate is off by more than 5% — CPR or the timer frequency is "
              "being read wrong, so every angle the app computes is wrong too.")
        return False
    print("  TRACK OK — counter, CPR and frequency all agree with the sky.")
    return True


def _blocked(m):
    """Motor-blocked flag from the axis status, if the board reports one.

    Worth reading after every move: a stalled stepper still has its commanded
    steps counted by the firmware, so `j1` would report a move that never
    physically happened. This flag is the only software hint of that."""
    try:
        st = m._send("f1")
    except Exception:                                # noqa: BLE001
        return None, "?"
    if len(st) < 2:
        return None, st
    try:
        return bool(int(st[1], 16) & 0x02), st
    except ValueError:
        return None, st


def stage_move(m, deg, rate_mult=64, measure=True):
    """One small GoTo out and back. THE MOUNT MOVES."""
    rate = m.max_slew_deg_per_s(rate_mult)
    print("\n=== MOVE ({:+.2f}° out and back at {:.3f}°/s — THE MOUNT MOVES) ==="
          .format(deg, rate))
    if measure:
        ms, cs = m.measure_directions(seconds=8.0, test_deg=min(abs(deg), 2.0))
        print("  measured encoder sign {:+.0f}, command sign {:+.0f}".format(ms, cs))
    ms, cs = m.motor_sign, m.command_sign
    before = m.get_counts()
    t0 = time.monotonic()
    m.goto_delta_deg(deg * cs, rate_mult=rate_mult)
    dt = time.monotonic() - t0
    out = m.counts_to_deg(m.get_counts() - before) * ms
    blk, st = _blocked(m)
    print("  asked {:+.2f}° -> counter says {:+.3f}° in {:.1f}s "
          "(≈{:.3f}°/s)   status {}{}".format(
              deg, out, dt, abs(out) / max(dt, 0.01), st,
              "  ** MOTOR BLOCKED **" if blk else ""))
    m.goto_delta_deg(-deg * cs, rate_mult=rate_mult)
    back = m.counts_to_deg(m.get_counts() - before) * ms
    blk2, st2 = _blocked(m)
    print("  returned -> net {:+.3f}° from the start   status {}{}".format(
        back, st2, "  ** MOTOR BLOCKED **" if blk2 else ""))
    ok = abs(out - deg) < max(0.25, abs(deg) * 0.15) and not (blk or blk2)
    print("  MOVE {} (counter-wise). Only your eyes can confirm the axis really "
          "turned — a stalled stepper still gets its steps counted."
          .format("OK" if ok else "SUSPECT"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Staged self-test of the mount protocol")
    ap.add_argument("--ip", default="192.168.4.1")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--stage", default="probe",
                    choices=["probe", "track", "move", "all"])
    ap.add_argument("--deg", type=float, default=2.0, help="test move size")
    ap.add_argument("--track-seconds", type=float, default=20.0)
    ap.add_argument("--rate", type=int, default=64,
                    help="slew rate multiplier (64 is gentle, 256 is brisk)")
    ap.add_argument("--rates", default="", help="comma-separated rates to try in turn")
    ap.add_argument("--simulate", action="store_true", help="fake board, no hardware")
    a = ap.parse_args()

    m = _mount(a)
    try:
        if not stage_probe(m):
            return 1
        if a.stage in ("track", "move", "all"):
            if not stage_track(m, a.track_seconds):
                return 1
        if a.stage in ("move", "all"):
            rates = ([int(r) for r in a.rates.split(",") if r.strip()]
                     if a.rates else [a.rate])
            for i, r in enumerate(rates):
                stage_move(m, a.deg, rate_mult=r, measure=(i == 0))
    except Exception as e:                       # noqa: BLE001
        print("\nERROR: {}".format(e))
        return 1
    finally:
        try:
            m.ensure_tracking()                  # never leave the mount stopped
            print("\ntracking re-asserted.")
        except Exception:                        # noqa: BLE001
            pass
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
