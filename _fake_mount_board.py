"""Simulated SkyWatcher motor board: keeps a real step counter that advances with
time, so the closed loop can be exercised without hardware."""
import time, threading

class FakeBoard:
    CPR = 9024000; FREQ = 64935; SIDEREAL = 86164.0905
    SPEEDUP = 1.0

    def __init__(self, slow_factor=1.0, wrong_dir=False, lag=0.0, cmd_flip=False):
        self.pos = 0x800000
        self.mode = "idle"; self.dirn = 0; self.target_steps = 0; self.period = 1
        self.t = time.monotonic(); self.lock = threading.Lock()
        self.slow = slow_factor          # pretend the board slews slower than asked
        self.sign = -1 if wrong_dir else 1
        self.cmd = -1 if cmd_flip else 1   # direction bit wired the other way
        self.lag = lag
        self.log = []

    def _advance(self):
        now = time.monotonic(); dt = now - self.t; self.t = now
        if self.mode == "track":
            self.pos += self.sign * dt / self.SIDEREAL * self.CPR
        elif self.mode == "goto":
            rate = self.FREQ / max(self.period, 1) / self.slow * self.SPEEDUP
            step = rate * dt
            if step >= self.remaining:
                self.pos += self.sign * self.cmd * (1 if self.dirn == 0 else -1) * self.remaining
                self.remaining = 0; self.mode = "idle"
            else:
                self.pos += self.sign * self.cmd * (1 if self.dirn == 0 else -1) * step
                self.remaining -= step

    def send(self, body):
        with self.lock:
            if self.lag: time.sleep(self.lag)
            self._advance(); self.log.append(body)
            c, ax, arg = body[0], body[1:2], body[2:]
            def hexl(n, d=6):
                s = "{:0{}X}".format(int(round(n)) & ((1 << (4*d))-1), d)
                return "".join(reversed([s[i:i+2] for i in range(0, d, 2)]))
            def unhexl(s):
                pr = [s[i:i+2] for i in range(0, len(s), 2)]
                return int("".join(reversed(pr)) or "0", 16)
            if c == "a": return hexl(self.CPR)
            if c == "b": return hexl(self.FREQ)
            if c == "e": return hexl(0x020300)
            if c == "j": return hexl(int(round(self.pos)))
            if c == "f": return "0" + ("1" if self.mode != "idle" else "0") + "0"
            if c == "E": self.pos = unhexl(arg); return ""
            if c == "F": return ""
            if c == "K": self.mode = "idle"; return ""
            if c == "G": self.gmode, self.dirn = arg[0], int(arg[1]); return ""
            if c == "H": self.remaining = unhexl(arg); return ""
            if c == "I": self.period = unhexl(arg); return ""
            if c == "J":
                self.mode = "goto" if getattr(self, "gmode", "0") == "0" else "track"
                if self.mode == "track": self.dirn = self.dirn
                return ""
            return ""
