"""
Utils - HUD Overlay  (v5 — Split Panels + Cyber-Neon visualization)
------------------------------------------------------------
Draws a professional, balanced HUD on every ADAS frame.

Features:
  - Dual Side Panels (Left & Right) to reduce clutter.
  - Cyber-Neon Lane Visualization (Glowing Cyan).
  - No full-screen flashing borders.
  - Smoothed data display (EMA).
  - Clean labels (no redundant warnings in panels).
"""

import cv2
import numpy as np

# ── Colour palette (BGR) ─────────────────────────────────────────────────────
C_WHITE    = (255, 255, 255)
C_BLACK    = (0,   0,   0)
C_GREEN    = (0,   210, 80)
C_RED      = (30,  30,  220)
C_YELLOW   = (0,   220, 255)
C_CYAN     = (255, 255, 0)       # Bright Neon Cyan
C_GREY     = (25,  25,  25)      # Darker panel background
C_ORANGE   = (0,   140, 255)
C_BRAKE    = (0,   0,   240)
C_PEDBRAKE = (180, 0,   220)

def draw_hud(frame, decision_result: dict, keep_result: dict, fps: float = 0):
    """
    Draw the full ADAS HUD with split dashboards.
    """
    h, w = frame.shape[:2]

    # ── 1. Unpack & Clean Data ───────────────────────────────────────
    decision          = decision_result.get("decision_state", "SAFE")
    scene             = decision_result.get("scene_type",    "CLEAR")
    v_count           = decision_result.get("vehicle_count",    0)
    p_count           = decision_result.get("pedestrian_count", 0)
    i_count           = decision_result.get("in_lane_count",    0)
    ttc               = decision_result.get("ttc")
    brake_from_person = decision_result.get("brake_from_person", False)
    sign_alert        = decision_result.get("sign_alert",  "")
    speed_limit       = decision_result.get("speed_limit", "")
    alert_type        = decision_result.get("alert_type",  "SAFE")

    command    = keep_result.get("command",    "STRAIGHT")
    offset_pct = keep_result.get("offset_pct", 0.0)

    # ── 2. Decision Badge Colour ──────────────────────────────────────
    if decision == "BRAKE":
        dec_color = C_PEDBRAKE if brake_from_person else C_BRAKE
    elif decision == "STEER":
        dec_color = C_ORANGE
    else:
        dec_color = C_GREEN

    # ── 3. Cyber-Neon Lane Visualization ─────────────────────────────
    _draw_cyber_lanes(frame, decision_result.get("lane_result", {}))

    # ── 4. Split Dashboard Panels ─────────────────────────────────────
    panel_w = 320
    panel_h = 160
    margin  = 12
    font    = cv2.FONT_HERSHEY_SIMPLEX
    
    # ── 4a. Left Panel (Environment) ──
    # Fields: SCENE, PERSONS, SPEED LIMIT, SIGN
    left_rows = [
        ("SCENE",   f"{scene} ({v_count}v / {i_count}L)", C_CYAN),
        ("PEDEST",  f"{p_count} detected",               (0, 100, 255) if p_count > 0 else C_WHITE),
        ("SIGN",    sign_alert if sign_alert else "NONE", C_YELLOW if sign_alert else C_WHITE),
        ("LIMIT",   f"{speed_limit} km/h" if speed_limit else "---", C_ORANGE if speed_limit else C_WHITE),
    ]
    _draw_side_panel(frame, 10, h - panel_h - 10 - 25, panel_w, panel_h, "ENVIRONMENT", left_rows)

    # ── 4b. Right Panel (Control) ──
    # Fields: COMMAND, OFFSET, TTC, ALERT, FPS
    if alert_type == "CRITICAL": alert_color = C_BRAKE
    elif alert_type == "WARNING": alert_color = C_YELLOW
    else: alert_color = C_GREEN

    right_rows = [
        ("COMMAND", command,                            C_YELLOW),
        ("OFFSET",  f"{offset_pct:+.1f}%",             C_WHITE),
        ("TTC",     f"{ttc:.1f} s" if ttc else "N/A",  C_RED if (ttc and ttc < 3) else C_WHITE),
        ("ALERT",   alert_type,                        alert_color),
        ("SYSTEM",  f"{fps:.1f} FPS",                  C_WHITE),
    ]
    _draw_side_panel(frame, w - panel_w - 10, h - panel_h - 10 - 25, panel_w, panel_h, "CONTROL", right_rows)

    # ── 5. Top Bar & Badge ───────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 38), (15, 15, 15), -1)
    cv2.putText(frame, "ADAS v5 | Advanced Driver Assistance System",
                (15, 26), font, 0.65, C_CYAN, 2, cv2.LINE_AA)

    # Decision Badge (Top Right)
    badge_text = f" {decision} "
    (tw, th), _ = cv2.getTextSize(badge_text, font, 0.75, 2)
    bx_right = w - 10
    bx_left  = w - tw - 25
    cv2.rectangle(frame, (bx_left, 4), (bx_right, 34), dec_color, -1)
    cv2.putText(frame, badge_text, (bx_left + 5, 26), font, 0.75, C_WHITE, 2)

    # Sign Alert (placed just to the left of the decision badge)
    if sign_alert:
        (saw, sah), _ = cv2.getTextSize(sign_alert, font, 0.6, 2)
        cv2.putText(frame, sign_alert, (bx_left - saw - 20, 26), font, 0.6, C_YELLOW, 2, cv2.LINE_AA)


    return frame

def _draw_side_panel(frame, x, y, w, h, title, rows):
    """Utility to draw a rounded-ish translucent side panel with rows."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_GREY, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    
    # Accent bar
    cv2.rectangle(frame, (x, y), (x + w, y + 3), C_CYAN, -1)
    
    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, title, (x + 8, y + 18), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Rows
    fy = y + 42
    row_h = 24
    for key, val, color in rows:
        cv2.putText(frame, f"{key}:", (x + 8, fy), font, 0.45, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, str(val),  (x + 100, fy), font, 0.5, color, 2, cv2.LINE_AA)
        fy += row_h

def _draw_cyber_lanes(frame, lane_result):
    """Draws Glowing Neon Cyan lane guide lines."""
    left  = lane_result.get("left_line")
    right = lane_result.get("right_line")
    
    if not left and not right:
        return

    # 1. Filled area (dark subtle glow)
    overlay = frame.copy()
    if left and right:
        pts = np.array([[left[0], left[1]], [left[2], left[3]], 
                        [right[2], right[3]], [right[0], right[1]]])
        cv2.fillPoly(overlay, [pts], (255, 100, 0)) # Blueprint blue
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # 2. Glowing Edges
    def draw_glow_line(img, line, color):
        if not line: return
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        # Outer thick glow
        cv2.line(img, p1, p2, color, 12, cv2.LINE_AA)
        # Inner core
        cv2.line(img, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

    # Temporary surface for blur
    glow_surf = np.zeros_like(frame)
    if left:  cv2.line(glow_surf, (left[0], left[1]),   (left[2], left[3]),   C_CYAN, 8)
    if right: cv2.line(glow_surf, (right[0], right[1]), (right[2], right[3]), C_CYAN, 8)
    
    # Fast "glow" via weight integration
    cv2.addWeighted(frame, 1.0, glow_surf, 0.6, 0, frame)
    
    # Draw sharp core lines
    if left:  cv2.line(frame, (left[0], left[1]),   (left[2], left[3]),   C_WHITE, 2, cv2.LINE_AA)
    if right: cv2.line(frame, (right[0], right[1]), (right[2], right[3]), C_WHITE, 2, cv2.LINE_AA)
