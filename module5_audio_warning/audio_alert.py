"""
Module 5 - Audio Alert System
------------------------------
Plays non-blocking beep alerts that react to the current driving decision
and Time-to-Collision (TTC).

Alert types
-----------
  CRITICAL  → rapid beep (high-pitched)  — collision / pedestrian imminent
  WARNING   → slow beep  (medium-pitched) — vehicle approaching / STEER
  SAFE      → silence

Beep speed vs TTC
-----------------
  TTC < 1.0 s  →  beep every 100 ms   (fastest — collision very close)
  TTC < 2.0 s  →  beep every 250 ms
  TTC < 3.0 s  →  beep every 500 ms
  WARNING/STEER →  beep every 1 000 ms  (slow pulse)
  SAFE          →  no sound

Traffic-sign overrides
----------------------
  STOP sign detected → treated as CRITICAL regardless of TTC

Implementation notes
--------------------
  • Uses Python's built-in `winsound` on Windows (no extra install needed).
  • On non-Windows platforms, falls back to a silent stub so the system
    still runs — replace the stub with `playsound`/`pygame` if needed.
  • All beeps run in a daemon background thread so they NEVER block the
    video loop.  A threading.Event gates the timing between beeps.
"""

import threading
import time

try:
    import winsound                    # Windows only — built-in, no install
    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False        # macOS / Linux → silent stub


# ── Beep profile constants ────────────────────────────────────────────────────
BEEP_CRITICAL_FREQ  = 1800   # Hz  — high pitch for critical
BEEP_WARNING_FREQ   = 900    # Hz  — medium pitch for warning
BEEP_DURATION_MS    = 80     # ms  — short burst per beep

# Minimum gap between update() calls that actually changes the interval (ms)
_MIN_INTERVAL_CHANGE = 50


class AudioAlert:
    """
    Non-blocking audio alert engine.

    Usage:
        audio = AudioAlert()
        audio.start()                           # launch background thread

        # Call once per frame — does NOT block
        audio.update(alert_type="CRITICAL", ttc=1.5, sign_alert="STOP AHEAD")

        audio.stop()                            # clean shutdown
    """

    def __init__(self):
        self._alert_type  = "SAFE"   # current alert level
        self._interval_ms = 0        # beep interval; 0 = silent
        self._freq        = BEEP_CRITICAL_FREQ

        self._lock    = threading.Lock()
        self._stop_ev = threading.Event()   # set → thread exits
        self._beep_ev = threading.Event()   # set → wake thread immediately
        self._thread  = threading.Thread(target=self._loop, daemon=True)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def start(self):
        """Launch the background beep thread."""
        self._thread.start()
        print("[AudioAlert] Started.")

    def stop(self):
        """Signal the background thread to exit and wait for it."""
        self._stop_ev.set()
        self._beep_ev.set()          # unblock the wait so thread exits fast
        self._thread.join(timeout=1.0)
        print("[AudioAlert] Stopped.")

    def update(self, alert_type: str = "SAFE",
               ttc: float = None,
               sign_alert: str = ""):
        """
        Update the alert level from the latest decision frame.

        Call this once per video frame — it is cheap (no I/O, no blocking).

        Args:
            alert_type (str)  : "CRITICAL" | "WARNING" | "SAFE"
            ttc        (float): Time-to-Collision in seconds (or None).
            sign_alert (str)  : e.g. "STOP AHEAD" — overrides to CRITICAL.
        """
        # Traffic signs upgrade alert level
        if sign_alert == "STOP AHEAD":
            alert_type = "CRITICAL"
            ttc        = ttc or 1.5    # treat as imminent if no TTC

        new_interval_ms, new_freq = self._compute_beep_params(alert_type, ttc)

        with self._lock:
            changed = (self._interval_ms != new_interval_ms
                       or self._freq != new_freq)
            self._alert_type  = alert_type
            self._interval_ms = new_interval_ms
            self._freq        = new_freq

        # Wake the thread only if something actually changed
        if changed:
            self._beep_ev.set()

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_beep_params(alert_type: str, ttc: float):
        """
        Map alert_type + TTC → (interval_ms, frequency_hz).

        Returns:
            (int, int): interval_ms (0 = silent), frequency_hz
        """
        if alert_type == "SAFE":
            return 0, BEEP_WARNING_FREQ       # silent

        if alert_type == "WARNING":
            return 1000, BEEP_WARNING_FREQ    # slow beep

        # CRITICAL — hyper-fast pulse for imminent collision
        if ttc is None:
            return 400, BEEP_CRITICAL_FREQ

        if ttc < 1.0:
            return 100, BEEP_CRITICAL_FREQ   # hyper-intense
        elif ttc < 2.0:
            return 150, BEEP_CRITICAL_FREQ   # very intense
        elif ttc < 3.5:
            return 300, BEEP_CRITICAL_FREQ
        else:
            return 500, BEEP_CRITICAL_FREQ

    def _beep(self):
        """Play one synchronous beep. Silent on non-Windows platforms."""
        if not _WINSOUND_AVAILABLE:
            return
        with self._lock:
            freq = self._freq
        try:
            winsound.Beep(freq, BEEP_DURATION_MS)
        except Exception:
            pass   # ignore audio errors (e.g. no sound device)

    def _loop(self):
        """
        Background thread: sleeps for interval_ms between beeps.
        When the interval changes it wakes up immediately and re-evaluates.
        """
        while not self._stop_ev.is_set():
            with self._lock:
                interval_ms = self._interval_ms

            if interval_ms == 0:
                # Silent — wait indefinitely until woken by update()
                self._beep_ev.wait()
                self._beep_ev.clear()
                continue

            # Play one beep
            self._beep()

            # Wait for interval_ms OR until update() wakes us
            wait_s = (interval_ms - BEEP_DURATION_MS) / 1000.0
            self._beep_ev.wait(timeout=max(wait_s, 0.01))
            self._beep_ev.clear()
