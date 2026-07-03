"""
Module 1 - Lane Detector
------------------------
Detects lane lines in a frame using classical computer vision:
  1. Convert to grayscale
  2. Gaussian blur
  3. Canny edge detection
  4. Region-of-Interest (ROI) masking
  5. Hough Line Transform
  6. Draw detected lines on the frame

Returns:
    - Lane center X position (pixels from left)
    - Simple curvature description (LEFT / STRAIGHT / RIGHT)
"""

import cv2
import numpy as np


class LaneDetector:
    """
    Detects lane lines using Canny edges + Hough Transform.

    Usage:
        detector = LaneDetector()
        result   = detector.detect(frame)
        # result["frame"]        → frame with drawn lane lines
        # result["lane_center"]  → X pixel of estimated lane center
        # result["curvature"]    → "LEFT" / "STRAIGHT" / "RIGHT"
        # result["left_line"]    → (x1,y1,x2,y2) or None
        # result["right_line"]   → (x1,y1,x2,y2) or None
    """

    def __init__(self):
        # ── Canny thresholds ──────────────────────────────────────────
        self.canny_low  = 50
        self.canny_high = 150

        # ── Gaussian blur kernel size (must be odd) ───────────────────
        self.blur_kernel = (5, 5)

        # ── Hough Transform parameters ───────────────────────────────
        self.hough_rho        = 1          # distance resolution (px)
        self.hough_theta      = np.pi / 180  # angle resolution (rad)
        self.hough_threshold  = 50         # minimum votes
        self.hough_min_length = 80         # minimum line length (px)
        self.hough_max_gap    = 150        # maximum gap between segments

        # ── Curvature sensitivity (px from frame center) ─────────────
        self.curve_threshold = 50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame):
        """
        Run the full lane-detection pipeline on one frame.

        Args:
            frame (numpy.ndarray): BGR input frame.

        Returns:
            dict: Keys → frame, lane_center, curvature, left_line, right_line
        """
        height, width = frame.shape[:2]
        frame_center  = width // 2

        # Step 1-4: Build an edge image with ROI applied
        edges = self._preprocess(frame)

        # Step 5: Hough lines
        lines = cv2.HoughLinesP(
            edges,
            rho       = self.hough_rho,
            theta     = self.hough_theta,
            threshold = self.hough_threshold,
            minLineLength = self.hough_min_length,
            maxLineGap    = self.hough_max_gap
        )

        # Step 6: Separate into left / right and average
        left_line, right_line = self._average_lines(frame, lines)

        # Step 7: Draw on the frame
        output_frame = self._draw_lines(frame.copy(), left_line, right_line)

        # Step 8: Estimate lane center and curvature
        lane_center = self._estimate_center(left_line, right_line, frame_center, height)
        curvature   = self._estimate_curvature(lane_center, frame_center)

        # Draw lane center marker
        if lane_center is not None:
            cv2.circle(output_frame, (lane_center, int(height * 0.75)), 8, (0, 255, 255), -1)
            cv2.line(output_frame, (frame_center, height - 10),
                     (frame_center, height - 40), (255, 255, 255), 2)

        return {
            "frame"      : output_frame,
            "lane_center": lane_center,
            "curvature"  : curvature,
            "left_line"  : left_line,
            "right_line" : right_line,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame):
        """Grayscale → Blur → Canny → ROI mask."""
        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian blur reduces noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # 3. Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 4. Apply ROI mask (trapezoidal region in the lower half)
        mask  = self._roi_mask(edges)
        return mask

    def _roi_mask(self, image):
        """
        Keep only the trapezoidal region where lane lines are likely.
        The trapezoid covers roughly the bottom 55 % of the frame.
        """
        height, width = image.shape[:2]
        mask = np.zeros_like(image)

        # Define trapezoid vertices (clockwise from bottom-left)
        roi_vertices = np.array([[
            (int(width * 0.05), height),                    # bottom-left
            (int(width * 0.45), int(height * 0.58)),        # top-left
            (int(width * 0.55), int(height * 0.58)),        # top-right
            (int(width * 0.95), height),                    # bottom-right
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi_vertices, 255)
        return cv2.bitwise_and(image, mask)

    def _average_lines(self, frame, lines):
        """
        Separate raw Hough lines into left/right by slope sign,
        then fit a single averaged line for each side.

        Returns:
            left_line  (tuple | None): (x1, y1, x2, y2)
            right_line (tuple | None): (x1, y1, x2, y2)
        """
        height = frame.shape[0]
        left_pts  = []   # [slope, intercept]
        right_pts = []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:          # vertical line → skip
                continue

            slope     = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Ignore nearly horizontal lines (|slope| < 0.3)
            if abs(slope) < 0.3:
                continue

            if slope < 0:         # negative slope → left lane
                left_pts.append((slope, intercept))
            else:                 # positive slope → right lane
                right_pts.append((slope, intercept))

        left_line  = self._make_line(left_pts,  height)
        right_line = self._make_line(right_pts, height)
        return left_line, right_line

    def _make_line(self, pts, height):
        """
        Average a list of (slope, intercept) pairs and return
        a line segment stretching from 60 % of frame height to the bottom.
        """
        if not pts:
            return None

        slope, intercept = np.mean(pts, axis=0)

        y1 = height                          # bottom of frame
        y2 = int(height * 0.60)             # ~60 % from top

        # x = (y - b) / m
        if slope == 0:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return (x1, y1, x2, y2)

    def _draw_lines(self, frame, left_line, right_line):
        """Draw the detected lane lines and a filled polygon between them."""
        overlay = frame.copy()

        # ── 1. Draw the filled lane area (transparent green) ──────────
        if left_line and right_line:
            pts = np.array([
                [left_line[0],  left_line[1]],
                [left_line[2],  left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]],
            ])
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # ── 2. Draw the lane edges (Thick Green with White Glow) ──────
        # We draw a thin white line underneath a thick green line for contrast
        def draw_styled_line(img, line, color, thickness):
            if line:
                # White "shadow" for contrast
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), thickness + 2)
                # Actual Green line
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)

        draw_styled_line(frame, left_line,  (0, 255, 0), 6)
        draw_styled_line(frame, right_line, (0, 255, 0), 6)

        return frame

    def _estimate_center(self, left_line, right_line, frame_center, height):
        """
        Estimate the X coordinate of the lane center at ~75 % frame height.

        Returns:
            int | None: Lane center X, or None if both lines are missing.
        """
        y_ref = int(height * 0.75)

        if left_line and right_line:
            # Interpolate X for each line at y_ref
            lx = self._x_at_y(left_line,  y_ref)
            rx = self._x_at_y(right_line, y_ref)
            return (lx + rx) // 2

        if left_line:
            lx = self._x_at_y(left_line, y_ref)
            return lx + int(frame_center * 0.5)   # guess right lane position

        if right_line:
            rx = self._x_at_y(right_line, y_ref)
            return rx - int(frame_center * 0.5)

        return None   # no lanes detected

    @staticmethod
    def _x_at_y(line, y):
        """Return X on the line at a given Y using linear interpolation."""
        x1, y1, x2, y2 = line
        if y2 == y1:
            return x1
        t  = (y - y1) / (y2 - y1)
        return int(x1 + t * (x2 - x1))

    def _estimate_curvature(self, lane_center, frame_center):
        """
        Simple curvature from horizontal lane-center offset.

        Returns:
            str: "LEFT" | "STRAIGHT" | "RIGHT"
        """
        if lane_center is None:
            return "UNKNOWN"

        offset = lane_center - frame_center

        if offset < -self.curve_threshold:
            return "RIGHT"    # lane curves right → steer right
        elif offset > self.curve_threshold:
            return "LEFT"     # lane curves left  → steer left
        else:
            return "STRAIGHT"
