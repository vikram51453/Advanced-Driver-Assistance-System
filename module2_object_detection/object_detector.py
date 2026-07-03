"""
Module 2 - Object Detector  (v3 — pedestrian + priority)
----------------------------------------------------------
Detects vehicles AND pedestrians using YOLOv8.

Priority levels
---------------
  HIGH   : person  (pedestrian)
  MEDIUM : car, truck, bus, motorcycle

Braking thresholds (consumed by DecisionController)
----------------------------------------------------
  HIGH   (person) : BRAKE if TTC < 3 s  OR  distance < 8 m
  MEDIUM (vehicle): BRAKE if TTC < 2 s  OR  distance < 5 m

Drawing
-------
  Pedestrians → solid RED box  +  "PERSON (HIGH PRIORITY)" badge
  Vehicles    → class colour   +  "#ID  label  conf  dist" badge

Each detected object dict contains:
  object_id      int    : stable centroid-tracker ID
  class_label    str    : human-readable name ("person", "car", …)
  confidence     float  : 0.0 – 1.0
  bbox           tuple  : (x1, y1, x2, y2)
  center         tuple  : (cx, cy)
  distance       float  : estimated metres (pinhole model)
  prev_distance  float  : previous-frame distance (None if new)
  priority_level str    : "HIGH" | "MEDIUM"
"""

import math
import cv2
from ultralytics import YOLO


# ── Real-world widths for distance estimation (metres) ───────────────────────
REAL_WIDTHS = {
    "car"        : 1.8,
    "truck"      : 2.5,
    "bus"        : 2.6,
    "motorcycle" : 0.8,
    "person"     : 0.5,   # shoulder width approximation
}

# Focal length in pixels (calibrate for your camera; 800 suits 1280-wide frame)
FOCAL_LENGTH_PX = 800

# Centroid match radius (pixels) for tracker
TRACK_MAX_DIST = 80

# Priority map
PRIORITY = {
    "person"     : "HIGH",
    "car"        : "MEDIUM",
    "truck"      : "MEDIUM",
    "bus"        : "MEDIUM",
    "motorcycle" : "MEDIUM",
}

# Draw colours (BGR)
COLOR_PERSON  = (0, 0, 220)          # red for pedestrians
CLASS_COLORS  = {
    "car"        : (0,   200, 255),  # yellow
    "truck"      : (255, 100,   0),  # blue
    "bus"        : (255,   0, 150),  # purple
    "motorcycle" : (0,   255, 100),  # green
}
DEFAULT_COLOR = (200, 200, 200)


class ObjectDetector:
    """
    Vehicle + pedestrian detector powered by YOLOv8, with centroid
    tracking, distance estimation, and priority classification.

    Usage:
        detector = ObjectDetector(model_path="yolov8n.pt")
        result   = detector.detect(frame)
        # result["frame"]   → annotated frame
        # result["objects"] → list of object dicts
    """

    TARGET_CLASSES = {"car", "truck", "bus", "motorcycle", "person"}

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.40):
        self.confidence_threshold = confidence_threshold
        print(f"[ObjectDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("[ObjectDetector] Model ready.")

        # Tracker: object_id → {"center": (cx,cy), "distance": float}
        self._tracked: dict[int, dict] = {}
        self._next_id: int = 1

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def detect(self, frame):
        """
        Run inference, assign priority, estimate distances, update tracker.

        Args:
            frame (numpy.ndarray): BGR input frame.

        Returns:
            dict:
                "frame"   → annotated BGR frame
                "objects" → list of object dicts
        """
        results      = self.model(frame, verbose=False)[0]
        raw_detects  = []
        output_frame = frame.copy()

        # ── 1. Parse YOLO detections ──────────────────────────────────
        for box in results.boxes:
            class_id   = int(box.cls[0])
            label      = self.model.names[class_id]
            confidence = float(box.conf[0])

            if label not in self.TARGET_CLASSES:
                continue
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx       = (x1 + x2) // 2
            cy       = (y1 + y2) // 2
            bbox_w   = x2 - x1
            bbox_h   = y2 - y1

            # ── 1.1 Ego-vehicle filter ────────────────────────────────
            # Ignore objects that are in the extreme bottom of the frame
            # (likely the hood/dashboard of the car)
            h, w = frame.shape[:2]
            if y2 > 0.90 * h and bbox_w > 0.25 * w:
                # This check catches wide objects sticking out from the bottom
                continue
            if cy > 0.94 * h:
                # This catches anything whose center is essentially off-road/on-hood
                continue

            distance = self._estimate_distance(label, bbox_w)
            priority = PRIORITY.get(label, "MEDIUM")

            raw_detects.append({
                "class_label"  : label,
                "confidence"   : confidence,
                "bbox"         : (x1, y1, x2, y2),
                "center"       : (cx, cy),
                "distance"     : distance,
                "priority_level": priority,
            })

        # ── 2. Centroid tracking ──────────────────────────────────────
        matched_ids: set[int] = set()
        detected_objects      = []

        for det in raw_detects:
            oid, prev_dist = self._match_or_create(det["center"],
                                                    det["distance"])
            matched_ids.add(oid)
            det["object_id"]     = oid
            det["prev_distance"] = prev_dist
            detected_objects.append(det)

            # Draw
            x1, y1, x2, y2 = det["bbox"]
            self._draw_box(
                output_frame,
                det["class_label"], det["confidence"],
                oid, det["distance"], det["priority_level"],
                x1, y1, x2, y2,
            )

        # ── 3. Prune stale tracks ─────────────────────────────────────
        stale = [k for k in self._tracked if k not in matched_ids]
        for k in stale:
            del self._tracked[k]

        return {
            "frame"  : output_frame,
            "objects": detected_objects,
        }

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _estimate_distance(self, label: str, bbox_width_px: int) -> float:
        """Pinhole model: distance = (real_width × focal_length) / bbox_px."""
        if bbox_width_px <= 0:
            return 999.0
        real_w = REAL_WIDTHS.get(label, 1.8)
        dist   = (real_w * FOCAL_LENGTH_PX) / bbox_width_px
        return round(max(0.5, min(200.0, dist)), 2)

    def _match_or_create(self, center: tuple, distance: float):
        """Match detection to nearest track or create a new one."""
        cx, cy        = center
        best_id       = None
        best_dist_px  = TRACK_MAX_DIST + 1

        for oid, info in self._tracked.items():
            tx, ty     = info["center"]
            pixel_dist = math.hypot(cx - tx, cy - ty)
            if pixel_dist < best_dist_px:
                best_dist_px = pixel_dist
                best_id      = oid

        if best_id is not None:
            prev_distance = self._tracked[best_id]["distance"]
            self._tracked[best_id] = {"center": center, "distance": distance}
            return best_id, prev_distance
        else:
            new_id = self._next_id
            self._next_id += 1
            self._tracked[new_id] = {"center": center, "distance": distance}
            return new_id, None

    def _draw_box(self, frame, label, confidence, oid, distance,
                  priority, x1, y1, x2, y2):
        """Draw bounding box + label badge with priority-aware styling."""
        is_person = (label == "person")

        # ── Box ───────────────────────────────────────────────────────
        color     = COLOR_PERSON if is_person else CLASS_COLORS.get(label, DEFAULT_COLOR)
        thickness = 3 if is_person else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Add a second inner rectangle for HIGH priority (double border)
        if is_person:
            cv2.rectangle(frame, (x1 + 3, y1 + 3),
                          (x2 - 3, y2 - 3), (255, 255, 255), 1)

        # ── Badge text ────────────────────────────────────────────────
        if is_person:
            text = f"#{oid} PERSON (HIGH PRIORITY)  {distance:.1f}m"
        else:
            text = f"#{oid} {label} {confidence:.0%}  {distance:.1f}m"

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 2)
        badge_y = max(y1 - th - baseline - 4, 0)

        cv2.rectangle(frame,
                      (x1, badge_y),
                      (x1 + tw + 8, badge_y + th + baseline + 6),
                      color, -1)
        cv2.putText(frame, text,
                    (x1 + 4, badge_y + th + 2),
                    font, scale,
                    (255, 255, 255) if is_person else (0, 0, 0), 2)
