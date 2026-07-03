"""
run_adas.py  —  Advanced Driver Assistance System  (v5)
========================================================
Entry point — wires all modules together.

Pipeline per frame
------------------
  VideoEngine
      ↓
  LaneDetector       → lane lines, curvature, center
      ↓
  ObjectDetector     → vehicles + pedestrians (ID, distance, priority)
      ↓
  TrafficSignDetector → STOP / SPEED LIMIT / TURN signs
      ↓
  LaneKeeper         → STEER LEFT / CENTER / RIGHT
      ↓
  DecisionController → BRAKE / STEER / SAFE  (+TTC, sign alerts)
      ↓
  HUD overlay  →  Display / save

Usage
-----
  python run_adas.py                       # auto-detect video or webcam
  python run_adas.py --source 0            # explicit webcam
  python run_adas.py --source video.mp4    # specific file
  python run_adas.py --source auto         # same as default
  python run_adas.py --source video.mp4 --save out.mp4
  python run_adas.py --no-yolo             # lane-only (no objects/signs)
  python run_adas.py --no-signs            # disable traffic-sign detection
  python run_adas.py --mute               # silence audio alerts

Player controls (in display window)
-------------------------------------
  SPACE   → Play / Pause
  1–9     → Jump to 10 %–90 % of video
  Q       → Quit

Press Q to quit.
"""

import argparse
import time
import cv2

# ── Module imports ────────────────────────────────────────────────────────────
from module0_base.video_engine                         import VideoEngine
from module1_lane_detection.lane_detector              import LaneDetector
from module2_object_detection.object_detector          import ObjectDetector
from module3_lane_keeping.lane_keeper                  import LaneKeeper
from module4_decision_controller.decision_controller   import DecisionController
from module5_audio_warning.audio_alert                 import AudioAlert
from module6_traffic_sign.traffic_sign_detector        import TrafficSignDetector
from utils.hud                                         import draw_hud


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SOURCE         = "auto"   # auto-detect sample_videos/ or webcam
DISPLAY_WIDTH          = 1280
DISPLAY_HEIGHT         = 720
YOLO_MODEL             = "yolov8n.pt"
YOLO_CONFIDENCE        = 0.40
LANE_OFFSET_THRESHOLD  = 40
TRAFFIC_THRESHOLD      = 5

# Vehicle braking thresholds
TTC_BRAKE_VEHICLE      = 2.0    # seconds
DIST_BRAKE_VEHICLE     = 5.0    # metres

# Pedestrian braking thresholds (stricter)
TTC_BRAKE_PERSON       = 3.0    # seconds
DIST_BRAKE_PERSON      = 8.0    # metres


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="ADAS – Advanced Driver Assistance System")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help=("'auto' = scan sample_videos/ then webcam; "
                              "or int webcam index; or path to file."))
    parser.add_argument("--save", default=None, metavar="OUTPUT.mp4",
                        help="Save annotated output to this file.")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Disable object detection (lane-only mode).")
    parser.add_argument("--no-signs", action="store_true",
                        help="Disable traffic-sign detection.")
    parser.add_argument("--mute", action="store_true",
                        help="Disable audio alerts (run silently).")
    args = parser.parse_args()
    # Convert numeric string to int ONLY for explicit numeric webcam index
    # Leave "auto" and file paths as strings
    if str(args.source).lstrip('-').isdigit():
        args.source = int(args.source)
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Source Selector
# ─────────────────────────────────────────────────────────────────────────────
def select_source(requested_source):
    """
    If requested_source is "auto", scan sample_videos/ and prompt user if
    multiple files exist. Otherwise returns the requested source.
    """
    if requested_source != "auto":
        return requested_source

    import glob
    import os
    
    sample_dir = "sample_videos"
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(sample_dir, ext)))
    
    files = sorted(files)
    
    if not files:
        print("[ADAS] No videos found in sample_videos/ → falling back to webcam 0")
        return 0
    
    if len(files) == 1:
        print(f"[ADAS] Auto-selected: {files[0]}")
        return files[0]
    
    # Multiple files → Choice based selection
    print("\n[ADAS] Multiple videos found in sample_videos/:")
    for i, f in enumerate(files):
        print(f"  [{i + 1}] {os.path.basename(f)}")
    
    print(f"  [0] Use Webcam")
    
    while True:
        try:
            choice = input(f"\nSelect a video [0-{len(files)}]: ").strip()
            if not choice:
                idx = 1 # default to first
            else:
                idx = int(choice)
            
            if idx == 0:
                print("[ADAS] Using webcam.")
                return 0
            if 1 <= idx <= len(files):
                print(f"[ADAS] selected: {files[idx-1]}")
                return files[idx-1]
        except ValueError:
            pass
        print(f"Invalid selection. Please enter 0 or 1-{len(files)}.")

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    
    # ── Choice based source selection ─────────────────────────────────────
    args.source = select_source(args.source)

    print("=" * 64)
    print("  ADAS — Advanced Driver Assistance System  v4")
    print("=" * 64)
    print(f"  Source        : {args.source}")
    print(f"  YOLO          : {'disabled' if args.no_yolo else YOLO_MODEL}")
    print(f"  Traffic Signs : {'disabled' if args.no_signs else 'enabled'}")
    print(f"  Audio         : {'muted' if args.mute else 'enabled'}")
    print(f"  Save to       : {args.save or 'not saving'}")
    print(f"  Vehicle TTC   : <{TTC_BRAKE_VEHICLE}s / <{DIST_BRAKE_VEHICLE}m → BRAKE")
    print(f"  Person  TTC   : <{TTC_BRAKE_PERSON}s / <{DIST_BRAKE_PERSON}m → BRAKE")
    print("  Press Q to quit")
    print("=" * 64)

    # ── Initialise modules ────────────────────────────────────────────────
    video_engine = VideoEngine(
        source         = args.source,  # already resolved by select_source
        display_width  = DISPLAY_WIDTH,
        display_height = DISPLAY_HEIGHT,
    )
    lane_detector = LaneDetector()

    object_detector = None
    if not args.no_yolo:
        object_detector = ObjectDetector(
            model_path           = YOLO_MODEL,
            confidence_threshold = YOLO_CONFIDENCE,
        )

    sign_detector = None
    if not args.no_yolo and not args.no_signs:
        sign_detector = TrafficSignDetector(
            model_path           = YOLO_MODEL,
            confidence_threshold = YOLO_CONFIDENCE,
        )

    lane_keeper = LaneKeeper(
        frame_width      = DISPLAY_WIDTH,
        offset_threshold = LANE_OFFSET_THRESHOLD,
    )

    decision_controller = DecisionController(
        traffic_threshold   = TRAFFIC_THRESHOLD,
        ttc_brake_vehicle   = TTC_BRAKE_VEHICLE,
        dist_brake_vehicle  = DIST_BRAKE_VEHICLE,
        ttc_brake_person    = TTC_BRAKE_PERSON,
        dist_brake_person   = DIST_BRAKE_PERSON,
    )

    # ── Audio alert module ─────────────────────────────────────────
    audio_alert = AudioAlert()
    if not args.mute:
        audio_alert.start()

    # ── Open video ────────────────────────────────────────────────────────
    video_engine.start()

    # ── Optional writer ───────────────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.save, fourcc, video_engine.get_fps(),
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        )
        print(f"[ADAS] Recording → {args.save}")

    # ── Smoothing state (EMA) ─────────────────────────────────────────────
    # helps stop the HUD values from jumping around
    smooth_ttc   = None
    smooth_dist  = None
    ema_alpha    = 0.3   # lower = smoother but more lag

    # ── FPS tracking ──────────────────────────────────────────────────────
    prev_time = time.time()
    fps       = 30.0

    print("[ADAS] Running… press Q in the display window to stop.\n")

    # ── Frame loop ────────────────────────────────────────────────────────
    while video_engine.is_running():

        frame = video_engine.read_frame()
        if frame is None:
            print("[ADAS] No more frames.")
            break

        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # ── 1. Lane Detection ─────────────────────────────────────────
        lane_result = lane_detector.detect(frame)
        frame       = lane_result["frame"]

        # ── 2. Object Detection (vehicles + pedestrians) ──────────────
        if object_detector is not None:
            detect_result = object_detector.detect(frame)
            frame         = detect_result["frame"]
        else:
            detect_result = {"frame": frame, "objects": []}

        # ── 3. Traffic Sign Detection ─────────────────────────────────
        if sign_detector is not None:
            sign_result = sign_detector.detect(frame)
            frame       = sign_result["frame"]
        else:
            sign_result = {"frame": frame, "signs": []}

        # ── 4. Lane Keeping ───────────────────────────────────────────
        keep_result = lane_keeper.compute(lane_result["lane_center"])

        # ── 5. Decision Controller ────────────────────────────────────
        decision_result = decision_controller.decide(
            lane_result   = lane_result,
            keep_result   = keep_result,
            detect_result = detect_result,
            sign_result   = sign_result,
            fps           = fps,
        )

        # ── 5.5 Smoothing ─────────────────────────────────────────────
        new_ttc  = decision_result.get("ttc")
        new_dist = decision_result.get("closest_dist")  # in-lane
        
        if new_ttc is not None:
            smooth_ttc = new_ttc if smooth_ttc is None else (ema_alpha * new_ttc + (1 - ema_alpha) * smooth_ttc)
        else:
            smooth_ttc = None

        if new_dist is not None:
            smooth_dist = new_dist if smooth_dist is None else (ema_alpha * new_dist + (1 - ema_alpha) * smooth_dist)
        else:
            smooth_dist = None

        # Update decision result with smoothed values for HUD/Log
        decision_result["ttc"]          = smooth_ttc
        decision_result["closest_dist"] = smooth_dist

        # ── 6. Audio alert (non-blocking, lane-aware, proximity guard) ────
        if not args.mute:
            # Beeping starts at 10m. 
            # INTENSE beep only if object is IN-LANE and < 2m.
            # WARNING beep if object is anywhere < 10m.
            
            in_lane_dist = decision_result.get("closest_dist") # smoothed in-lane
            overall_dist = decision_result.get("overall_closest_dist", 999)
            sign_alert   = decision_result.get("sign_alert", "")
            
            # Use 999 for None
            d_lane = in_lane_dist if in_lane_dist is not None else 999.0
            d_all  = overall_dist if overall_dist is not None else 999.0
            
            alert_type_final = "SAFE"
            if d_lane < 2.0 or sign_alert == "STOP AHEAD":
                alert_type_final = "CRITICAL"
            elif d_lane < 10.0 or d_all < 5.0:
                alert_type_final = "WARNING"
            
            # Pass to audio module
            audio_alert.update(
                alert_type = alert_type_final,
                ttc        = smooth_ttc,
                sign_alert = sign_alert,
            )

        # ── Console log ───────────────────────────────────────────────
        ttc_val  = decision_result.get("ttc")
        ttc_log  = f"{ttc_val:.1f}s" if ttc_val is not None else " N/A "
        sign_log = decision_result.get("sign_alert", "") or "none"
        print(
            f"\r  [{decision_result['decision_state']:^5}] "
            f"TTC:{ttc_log:<7}| "
            f"Sign:{sign_log:<14}| "
            f"{decision_result['summary']}   ",
            end="", flush=True,
        )

        # ── HUD overlay ───────────────────────────────────────────────
        frame = draw_hud(frame, decision_result, keep_result, fps=fps)

        # ── Playback status badge + Progress Bar ──────────────────────
        frame = video_engine.draw_status_overlay(frame)
        frame = video_engine.draw_progress_bar(frame)

        # ── Display + save ────────────────────────────────────────────
        video_engine.show_frame(frame, window_title="ADAS Output")
        if writer is not None:
            writer.write(frame)

        # ── Key handler (SPACE/1-9 delegated to VideoEngine, Q quits) ───
        key = cv2.waitKey(1) & 0xFF
        if video_engine.handle_key(key):
            print("\n[ADAS] Q pressed — stopping.")
            break

    # ── Clean up ──────────────────────────────────────────────────────────
    if not args.mute:
        audio_alert.stop()              # gracefully end beep thread
    if writer is not None:
        writer.release()
        print(f"[ADAS] Saved → {args.save}")

    video_engine.stop()
    print("[ADAS] Done.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
