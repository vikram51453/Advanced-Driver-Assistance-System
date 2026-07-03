"""
Module 3 - Lane Keeper
-----------------------
Computes a steering command based on how far the vehicle
is from the centre of the detected lane.

Logic:
    offset = lane_center - frame_center

    offset < -threshold  →  STEER LEFT   (you drifted right)
    offset >  threshold  →  STEER RIGHT  (you drifted left)
    otherwise            →  CENTER       (you are aligned)

Returns:
    dict with keys:
        "command"    : "STEER LEFT" | "STEER RIGHT" | "CENTER"
        "offset_px"  : integer pixel offset from frame centre
        "offset_pct" : offset as a percentage of frame width
"""


class LaneKeeper:
    """
    Computes a steering correction command from the lane detection output.

    Usage:
        keeper = LaneKeeper(frame_width=1280, offset_threshold=40)
        result = keeper.compute(lane_center, frame_center)
        # result["command"]    → "STEER LEFT" / "STEER RIGHT" / "CENTER"
        # result["offset_px"]  → signed pixel offset
        # result["offset_pct"] → signed percentage offset
    """

    def __init__(self, frame_width=1280, offset_threshold=40):
        """
        Args:
            frame_width      (int): Width of the video frame in pixels.
            offset_threshold (int): Dead-zone in pixels; offsets within
                                    ±threshold are treated as centred.
        """
        self.frame_width      = frame_width
        self.frame_center     = frame_width // 2
        self.offset_threshold = offset_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, lane_center):
        """
        Calculate steering command from the detected lane centre X position.

        Args:
            lane_center (int | None): X-pixel of the detected lane centre.
                                      Pass None if no lane was found.

        Returns:
            dict: command, offset_px, offset_pct
        """
        if lane_center is None:
            return {
                "command"   : "UNKNOWN",
                "offset_px" : 0,
                "offset_pct": 0.0,
            }

        # Positive offset → lane centre is to the RIGHT of frame centre
        # → car is drifting LEFT → steer RIGHT to correct
        offset_px  = lane_center - self.frame_center
        offset_pct = round((offset_px / self.frame_center) * 100, 1)

        if offset_px < -self.offset_threshold:
            command = "STEER RIGHT"   # lane is left of centre → steer right
        elif offset_px > self.offset_threshold:
            command = "STEER LEFT"    # lane is right of centre → steer left
        else:
            command = "CENTER"

        return {
            "command"   : command,
            "offset_px" : offset_px,
            "offset_pct": offset_pct,
        }

    def update_frame_width(self, frame_width):
        """
        Update geometry if frame size changes at runtime.

        Args:
            frame_width (int): New frame width in pixels.
        """
        self.frame_width  = frame_width
        self.frame_center = frame_width // 2
