"""
Module 0 - Video Engine  (v2 — auto-detect + player controls)
--------------------------------------------------------------
Auto-detects video files in sample_videos/ folder.
Falls back to webcam if no files are found.

Controls (in the ADAS display window)
--------------------------------------
  SPACE     → Play / Pause
  1 – 9     → Jump to 10 % – 90 % of total video length
  Q         → Quit  (handled in run_adas.py)

HUD overlay
-----------
  draw_status_overlay(frame) adds a small PLAYING / PAUSED badge
  to the top-left of the frame (called by run_adas.py).
"""

import os
import glob
import cv2


# Folder searched for video files (relative to CWD when run_adas.py runs)
SAMPLE_VIDEO_DIR = "sample_videos"
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv")


def _find_first_video() -> str | None:
    """Return the first video file found in SAMPLE_VIDEO_DIR, or None."""
    if not os.path.isdir(SAMPLE_VIDEO_DIR):
        return None
    for ext in VIDEO_EXTENSIONS:
        pattern = os.path.join(SAMPLE_VIDEO_DIR, ext)
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


class VideoEngine:
    """
    VideoEngine with playback controls.

    Constructor behaviour:
        source = "auto"  → scan sample_videos/, pick first found; else webcam 0
        source = 0/1/2   → specific webcam index
        source = "path"  → specific video file

    Usage:
        engine = VideoEngine()      # auto-detect
        engine.start()
        while engine.is_running():
            key   = cv2.waitKey(1) & 0xFF
            if engine.handle_key(key):   # True → quit
                break
            frame = engine.read_frame()
            if frame is None:
                break
            frame = engine.draw_status_overlay(frame)
            engine.show_frame(frame)
        engine.stop()
    """

    def __init__(self, source="auto", display_width=1280, display_height=720):
        """
        Args:
            source         : "auto" | int (webcam index) | str (file path)
            display_width  : Output window width.
            display_height : Output window height.
        """
        # ── Resolve source ────────────────────────────────────────────
        if source == "auto":
            video_file = _find_first_video()
            if video_file:
                print(f"[VideoEngine] Auto-detected: {video_file}")
                self.source = video_file
            else:
                print("[VideoEngine] No video in sample_videos/ → using webcam 0")
                self.source = 0
        else:
            self.source = source

        self.display_width  = display_width
        self.display_height = display_height
        self.cap            = None
        self._running       = False

        # Playback and frame state
        self._paused        = False
        self._total_frames  = 0      # set after cap is opened
        self._last_frame    = None   # backup for pause buffering

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def start(self):
        """Open the video source."""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[VideoEngine] Cannot open source: '{self.source}'\n"
                "  - For webcam: make sure no other app is using it.\n"
                "  - For a file: check the path is correct."
            )

        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._running      = True
        is_file = not isinstance(self.source, int)
        src_label = self.source if is_file else f"webcam {self.source}"
        duration  = (f"  ({self._total_frames} frames, "
                     f"{self._total_frames / max(self.get_fps(), 1):.1f}s)"
                     if is_file and self._total_frames > 0 else "")
        print(f"[VideoEngine] Started — {src_label}{duration}")

    def is_running(self) -> bool:
        """True while the source is open and active."""
        return self._running and self.cap is not None and self.cap.isOpened()

    def is_paused(self) -> bool:
        """True if playback is currently paused."""
        return self._paused

    def handle_key(self, key: int) -> bool:
        """
        Process a key press.

        Args:
            key (int): cv2.waitKey() return value (masked to 0xFF).

        Returns:
            bool: True if the caller should quit (Q key pressed).
        """
        if key == 255 or key < 0:   # no key / timeout
            return False

        # Q → quit
        if key in (ord("q"), ord("Q")):
            return True

        # SPACE → toggle pause
        if key == ord(" "):
            self._paused = not self._paused
            state = "PAUSED" if self._paused else "PLAYING"
            print(f"\r[VideoEngine] {state}          ", end="", flush=True)
            return False

        # LEFT / RIGHT (Arrows) → seek -5 / +5 seconds
        if key in (81, 2, ord('j')):   # Left arrow
            self.seek_seconds(-5)
            return False
        if key in (83, 3, ord('l')):   # Right arrow
            self.seek_seconds(5)
            return False

        # 1–9 → jump to 10 %–90 % of video
        if ord("1") <= key <= ord("9") and self._total_frames > 0:
            pct   = (key - ord("0")) / 10.0     # 0.1 … 0.9
            frame_pos = int(self._total_frames * pct)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            pct_str = f"{int(pct * 100)}%"
            print(f"\r[VideoEngine] Jumped to {pct_str} "
                  f"(frame {frame_pos}/{self._total_frames})   ",
                  end="", flush=True)
            return False

        return False

    def seek_seconds(self, seconds: float):
        """Seek forward or backward by a number of seconds."""
        if not self.cap or self._total_frames <= 0:
            return
        
        fps = self.get_fps()
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = max(0, min(self._total_frames - 1, current_frame + seconds * fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        
        direction = "Forward" if seconds > 0 else "Backward"
        print(f"\r[VideoEngine] Seek {direction} {abs(seconds)}s          ", end="", flush=True)

    def draw_progress_bar(self, frame):
        """
        Draw a YouTube-style red/white progress bar at the bottom.
        Includes current/total time strings.
        """
        if self._total_frames <= 0:
            return frame

        h, w = frame.shape[:2]
        curr_f, total_f = self.get_progress()
        fps = self.get_fps()
        
        curr_s = curr_f / fps
        total_s = total_f / fps
        
        def format_time(s):
            m, s = divmod(int(s), 60)
            return f"{m:02d}:{s:02d}"

        # ── Geometry ─────────────────────────────────────────────────────
        bar_h = 6
        margin = 15
        bar_y = h - margin - 20
        bar_w = w - (2 * margin)
        
        # ── Background (grey) ────────────────────────────────────────────
        cv2.rectangle(frame, (margin, bar_y), (w - margin, bar_y + bar_h), (80, 80, 80), -1)
        
        # ── Progress (red) ───────────────────────────────────────────────
        pct = curr_f / total_f
        prog_w = int(pct * bar_w)
        if prog_w > 0:
            cv2.rectangle(frame, (margin, bar_y), (margin + prog_w, bar_y + bar_h), (0, 0, 220), -1)
            # Knob (circle at the end of progress)
            cv2.circle(frame, (margin + prog_w, bar_y + bar_h // 2), 6, (0, 0, 240), -1)

        # ── Timestamps ───────────────────────────────────────────────────
        time_str = f"{format_time(curr_s)} / {format_time(total_s)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, time_str, (margin, bar_y - 8), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame

    def read_frame(self):
        """
        Read the next frame, or the last frame if paused (eliminates jitter).

        Returns:
            numpy.ndarray | None: BGR frame, or None when source ends.
        """
        if not self.is_running():
            return None

        if self._paused and self._last_frame is not None:
            return self._last_frame.copy()

        ret, frame = self.cap.read()
        if not ret:
            self._running = False
            return None

        self._last_frame = frame.copy()
        return frame

    def draw_status_overlay(self, frame):
        """
        Draw a small PLAYING / PAUSED badge on the frame (top-right area
        of the top banner so it doesn't overlap the ADAS title).

        Args:
            frame (numpy.ndarray): Annotated ADAS frame.

        Returns:
            numpy.ndarray: Frame with overlay.
        """
        h, w = frame.shape[:2]
        text  = "PAUSED" if self._paused else "PLAYING"
        color = (0, 80, 200) if self._paused else (0, 180, 60)   # red / green
        font  = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
        bx = w // 2 - tw // 2
        by = 7
        cv2.rectangle(frame, (bx - 6, by), (bx + tw + 6, by + th + 8),
                      color, -1)
        cv2.putText(frame, text, (bx, by + th + 2),
                    font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def show_frame(self, frame, window_title="ADAS Output"):
        """Display the frame in an OpenCV window (resized to display size)."""
        resized = cv2.resize(frame, (self.display_width, self.display_height))
        cv2.imshow(window_title, resized)

    def stop(self):
        """Release the source and close all OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self._running = False
        print("\n[VideoEngine] Stopped.")

    def get_fps(self) -> float:
        """Return the FPS of the source (30 as default for webcams)."""
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps and fps > 0 else 30.0
        return 30.0

    def get_progress(self) -> tuple:
        """
        Return (current_frame, total_frames) for progress display.
        Returns (0, 0) for live webcams.
        """
        if self.cap is None or self._total_frames == 0:
            return 0, 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)), self._total_frames
