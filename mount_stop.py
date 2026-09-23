#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mount_stop.py — emergency stop, independent of NOUT.

If the axis ever runs away and the app cannot be trusted to stop it, run this
from a terminal. It needs nothing but the Wi-Fi connection to the mount: no
Qt, no camera, no NOUT state.

    python mount_stop.py                 # halt, then confirm it halted
    python mount_stop.py --track         # halt, then restart sidereal tracking

Stopping is not one command and a hope: on a Star Adventurer 2i a single K has
been observed to have no effect on a fast slew, so this repeats it and watches
the position counter until it really is still.
"""

import argparse
import socket
import sys
import time


def _unhex(x):
    pairs = [x[i:i + 2] for i in range(0, len(x), 2)]
    return int("".join(reversed(pairs)) or "0", 16)


def main():
    ap = argparse.ArgumentParser(description="Emergency stop for the mount")
    ap.add_argument("--ip", default="192.168.4.1")
    ap.add_argument("--track", action="store_true",
                    help="restart sidereal tracking once the axis is still")
    a = ap.parse_args()
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sk.settimeout(1.5)

    def cmd(body, tries=6):
        for _ in range(tries):
            try:
                sk.sendto((":" + body + "\r").encode(), (a.ip, 11880))
                r = sk.recvfrom(256)[0].decode("ascii", "ignore").strip()
                return r.lstrip("=").rstrip("\r")
            except Exception:                           # noqa: BLE001
                time.sleep(0.15)
        return None

    if cmd("f1") is None:
        print("Mount not answering at {} — check the Wi-Fi. If it is still moving, "
              "pull the power.".format(a.ip))
        return 2

    print("Halting…")
    for i in range(10):
        cmd("K1")
        time.sleep(0.25)
        p0 = _unhex(cmd("j1") or "0")
        time.sleep(0.6)
        p1 = _unhex(cmd("j1") or "0")
        rate = (p1 - p0) / 0.6
        print("  K1 #{} -> status {}, {:+.0f} steps/s".format(i + 1, cmd("f1"), rate))
        if abs(rate) < 5:
            print("STOPPED.")
            break
    else:
        print("STILL MOVING after 10 tries — pull the power.")
        return 1

    if a.track:
        cpr = _unhex(cmd("a1") or "0")
        freq = _unhex(cmd("b1") or "0")
        if cpr and freq:
            period = int(round(freq * 86164.0905 / cpr))
            h = "{:06X}".format(period)
            cmd("G110")
            cmd("I1" + "".join(reversed([h[i:i + 2] for i in range(0, 6, 2)])))
            cmd("J1")
            print("Sidereal tracking restarted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
