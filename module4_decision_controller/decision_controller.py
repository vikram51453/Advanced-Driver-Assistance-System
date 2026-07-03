"""
Module 4 - Decision Controller  (v3 — pedestrian priority + sign awareness)
-----------------------------------------------------------------------------
Fuses lane-keeping, object detection (with priority), and traffic-sign
results into a single driving decision.

Priority braking thresholds
----------------------------
  Person  (HIGH)   : BRAKE if TTC < 3 s  OR  distance < 8 m
  Vehicle (MEDIUM) : BRAKE if TTC < 2 s  OR  distance < 5 m

  Pedestrians override all other logic — even if a vehicle is inside the
  lane, a pedestrian with a critical TTC/distance will always win.

Traffic-sign integration
------------------------
  STOP sign detected        → adds "STOP AHEAD" alert to result
  SPEED LIMIT sign detected → adds detected speed string to result
  TURN sign detected        → noted in reason string

Decision priority order
-----------------------
  BRAKE  (pedestrian)  >  BRAKE (vehicle)  >  STEER  >  SAFE

Returns
-------
  decision_state  : "SAFE" | "STEER" | "BRAKE"
  ttc             : float | None
  scene_type      : "TRAFFIC" | "CLEAR"
  vehicle_count   : int
  in_lane_count   : int
  pedestrian_count: int
  sign_alert      : str    e.g. "STOP AHEAD" | "SPEED LIMIT 30" | ""
  speed_limit     : str    e.g. "30" | ""
  reason          : str
  summary         : str
"""


class DecisionController:
    """
    High-level decision controller — pedestrian-aware, sign-aware.

    Usage:
        ctrl   = DecisionController()
        result = ctrl.decide(lane_result, keep_result,
                             detect_result, sign_result, fps)
    """

    def __init__(self,
                 traffic_threshold: int    = 5,
                 # Vehicle thresholds
                 ttc_brake_vehicle: float  = 2.0,
                 dist_brake_vehicle: float = 5.0,
                 # Pedestrian thresholds (stricter)
                 ttc_brake_person: float   = 3.0,
                 dist_brake_person: float  = 8.0):
        """
        Args:
            traffic_threshold  (int)  : # vehicles ≥ this → TRAFFIC scene.
            ttc_brake_vehicle  (float): TTC threshold (s) for vehicles.
            dist_brake_vehicle (float): Distance threshold (m) for vehicles.
            ttc_brake_person   (float): TTC threshold (s) for pedestrians.
            dist_brake_person  (float): Distance threshold (m) for pedestrians.
        """
        self.traffic_threshold   = traffic_threshold
        self.ttc_brake_vehicle   = ttc_brake_vehicle
        self.dist_brake_vehicle  = dist_brake_vehicle
        self.ttc_brake_person    = ttc_brake_person
        self.dist_brake_person   = dist_brake_person

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def decide(self,
               lane_result:   dict,
               keep_result:   dict,
               detect_result: dict,
               sign_result:   dict  = None,
               fps:           float = 30.0) -> dict:
        """
        Compute the driving decision for one frame.

        Args:
            lane_result   (dict): LaneDetector.detect() output.
            keep_result   (dict): LaneKeeper.compute() output.
            detect_result (dict): ObjectDetector.detect() output.
                                  Each object has: label/class_label,
                                  priority_level, bbox, center, distance,
                                  prev_distance, object_id.
            sign_result   (dict): TrafficSignDetector.detect() output
                                  (or None / {} if signs disabled).
            fps           (float): Pipeline FPS for speed calc.

        Returns:
            dict — see module docstring for all keys.
        """
        fps        = max(fps, 1.0)
        sign_result = sign_result or {"signs": []}

        # ── 1. Lane info ──────────────────────────────────────────────
        left_line  = lane_result.get("left_line")
        right_line = lane_result.get("right_line")
        curvature  = lane_result.get("curvature", "UNKNOWN")

        command   = keep_result.get("command",   "UNKNOWN")
        offset_px = keep_result.get("offset_px",  0)

        # ── 2. Separate pedestrians from vehicles ─────────────────────
        all_objects = detect_result.get("objects", [])

        pedestrians = [o for o in all_objects
                       if o.get("priority_level") == "HIGH"
                       or self._label(o) == "person"]
        vehicles    = [o for o in all_objects
                       if o not in pedestrians]

        vehicle_count    = len(all_objects)
        pedestrian_count = len(pedestrians)
        scene_type = "TRAFFIC" if vehicle_count >= self.traffic_threshold else "CLEAR"

        # ── 3. Lane-aware filtering (vehicles only) ───────────────────
        in_lane_vehicles = self._filter_in_lane(vehicles, left_line, right_line)
        in_lane_count    = len(in_lane_vehicles)

        # Pedestrians are ALWAYS considered — they may be crossing
        in_lane_pedestrians = pedestrians   # no lane filter for people

        # ── 4. Collision analysis ─────────────────────────────────────
        brake_triggered = False
        brake_reason    = ""
        brake_from_person = False    # flag: was BRAKE caused by a person?

        min_ttc              = None
        min_ttc_id           = None
        in_lane_closest_dist = None
        overall_closest_dist = None
        closest_speed        = None

        # -- 4a. Pedestrians (override, stricter) ----------------------
        for obj in in_lane_pedestrians:
            dist      = obj.get("distance")
            prev_dist = obj.get("prev_distance")
            oid       = obj.get("object_id", "?")
            if dist is None:
                continue

            # (already handles in lines 158-163 below)
            pass

            if dist < self.dist_brake_person:
                brake_triggered   = True
                brake_from_person = True
                brake_reason      = f"PEDESTRIAN #{oid} too close ({dist:.1f}m)"

            # Overall closest tracking
            if overall_closest_dist is None or dist < overall_closest_dist:
                overall_closest_dist = dist
            
            # Pedestrians are treated as in-lane for safety
            if in_lane_closest_dist is None or dist < in_lane_closest_dist:
                in_lane_closest_dist = dist

            if prev_dist is not None:
                rel_speed = (prev_dist - dist) * fps
                if rel_speed > 0.1:
                    ttc = dist / rel_speed
                    if min_ttc is None or ttc < min_ttc:
                        min_ttc       = ttc
                        min_ttc_id    = oid
                        closest_speed = rel_speed

                    if ttc < self.ttc_brake_person:
                        brake_triggered   = True
                        brake_from_person = True
                        brake_reason      = (
                            f"PEDESTRIAN #{oid} TTC={ttc:.1f}s "
                            f"({rel_speed:.1f}m/s)"
                        )

        # -- 4b. In-lane vehicles (standard thresholds) ----------------
        for obj in in_lane_vehicles:
            dist      = obj.get("distance")
            prev_dist = obj.get("prev_distance")
            oid       = obj.get("object_id", "?")
            lbl       = self._label(obj)
            if dist is None:
                continue

            if overall_closest_dist is None or dist < overall_closest_dist:
                overall_closest_dist = dist

            if in_lane_closest_dist is None or dist < in_lane_closest_dist:
                in_lane_closest_dist = dist

            if dist < self.dist_brake_vehicle:
                brake_triggered = True
                brake_reason    = brake_reason or f"#{oid} {lbl} close ({dist:.1f}m)"

            if prev_dist is not None:
                rel_speed = (prev_dist - dist) * fps
                if rel_speed > 0.1:
                    ttc = dist / rel_speed
                    if min_ttc is None or ttc < min_ttc:
                        min_ttc       = ttc
                        min_ttc_id    = oid
                        closest_speed = rel_speed

                    if ttc < self.ttc_brake_vehicle:
                        brake_triggered = True
                        brake_reason    = brake_reason or (
                            f"#{oid} {lbl} TTC={ttc:.1f}s "
                            f"({rel_speed:.1f}m/s)"
                        )

        # ── 5. Traffic sign processing ────────────────────────────────
        signs       = sign_result.get("signs", [])
        sign_alert  = ""
        speed_limit = ""

        for sign in signs:
            lbl = sign.get("label", "")
            if lbl == "STOP":
                sign_alert = "STOP AHEAD"
                # A stop sign also triggers a brake-like alert
                if not brake_triggered:
                    brake_triggered = True
                    brake_reason    = brake_reason or "STOP sign detected"
            elif lbl == "SPEED LIMIT":
                extra = sign.get("extra", "")
                speed_limit = extra
                sign_alert  = sign_alert or f"SPEED LIMIT {extra}".strip()
            elif lbl == "TURN":
                sign_alert = sign_alert or "TURN AHEAD"

        # ── 6. Final decision ─────────────────────────────────────────
        needs_steering = command in ("STEER LEFT", "STEER RIGHT")

        if brake_triggered:
            decision_state = "BRAKE"
        elif needs_steering:
            decision_state = "STEER"
        else:
            decision_state = "SAFE"

        # ── 6b. Alert type (consumed by AudioAlert) ───────────────────
        # CRITICAL: BRAKE condition — especially if pedestrian-triggered
        # WARNING : STEER or object present but not yet critical
        # SAFE    : nothing of concern
        if decision_state == "BRAKE":
            alert_type = "CRITICAL"
        elif decision_state == "STEER" or (min_ttc is not None and min_ttc < 5.0):
            alert_type = "WARNING"
        else:
            alert_type = "SAFE"

        # ── 7. Reason + summary strings ───────────────────────────────
        reason_parts = []
        if brake_triggered:
            reason_parts.append(brake_reason)
        if sign_alert:
            reason_parts.append(sign_alert)
        if needs_steering:
            reason_parts.append(f"Lane offset {offset_px:+d}px → {command}")
        if curvature in ("LEFT", "RIGHT"):
            reason_parts.append(f"Curve: {curvature}")
        if scene_type == "TRAFFIC":
            reason_parts.append(f"{vehicle_count} vehicles in scene")

        reason  = " | ".join(reason_parts) if reason_parts else "All clear"
        ttc_str = f"{min_ttc:.1f}s" if min_ttc is not None else "N/A"

        summary = (
            f"Scene: {scene_type} ({vehicle_count}v, "
            f"{pedestrian_count}p, {in_lane_count} in-lane) | "
            f"Lane: {command} | "
            f"TTC: {ttc_str} | "
            f"Sign: {sign_alert or 'none'} | "
            f"Decision: {decision_state}"
        )

        return {
            "decision_state"       : decision_state,
            "alert_type"           : alert_type,
            "ttc"                  : min_ttc,
            "min_ttc_id"           : min_ttc_id,
            "closest_dist"         : in_lane_closest_dist,
            "overall_closest_dist" : overall_closest_dist,
            "relative_speed"       : closest_speed,
            "scene_type"           : scene_type,
            "vehicle_count"    : vehicle_count,
            "pedestrian_count" : pedestrian_count,
            "in_lane_count"    : in_lane_count,
            "brake_from_person": brake_from_person,
            "sign_alert"       : sign_alert,
            "speed_limit"      : speed_limit,
            "reason"           : reason,
            "summary"          : summary,
            "lane_result"      : lane_result,
            # backward-compat aliases
            "scene"            : scene_type,
            "decision"         : decision_state,
        }

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _label(obj: dict) -> str:
        """Return the object class label (supports both key names)."""
        return obj.get("class_label", obj.get("label", "unknown"))

    @staticmethod
    def _filter_in_lane(objects: list, left_line, right_line) -> list:
        """
        Keep only objects whose centre is between the two lane lines.
        Falls back to all objects when lanes are not detected.
        """
        if left_line is None or right_line is None:
            return objects

        in_lane = []
        for obj in objects:
            cx, cy = obj["center"]
            lx = DecisionController._x_at_y(left_line,  cy)
            rx = DecisionController._x_at_y(right_line, cy)
            lane_left  = min(lx, rx)
            lane_right = max(lx, rx)
            if lane_left <= cx <= lane_right:
                in_lane.append(obj)

        return in_lane

    @staticmethod
    def _x_at_y(line: tuple, y: int) -> int:
        """Linearly interpolate X on (x1,y1,x2,y2) at a given Y."""
        x1, y1, x2, y2 = line
        if y2 == y1:
            return x1
        t = (y - y1) / (y2 - y1)
        return int(x1 + t * (x2 - x1))
