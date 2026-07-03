# Advanced Driver Assistance System (ADAS) Pipeline

A modular, real-time Python computer vision framework for smart vehicle assistance. Integrates classical CV techniques, YOLOv8 object detection, and deterministic sensor fusion to deliver situational awareness, collision forecasting, and driver guidance through a live HUD.

---

## Pipeline Architecture

Each video frame passes through five sequential modules. Every module is independently upgradeable without touching the rest of the pipeline.

- **Video Ingestion & State Engine** (`module0_base`)
  - Handles video I/O via OpenCV pipelines
  - Custom frame-buffer architecture eliminates OpenCV's native jitter when pausing or seeking
  - Manages the playback loop and tracks frame index on a graphical timeline progress bar

- **Spatial Lane Detection** (`module1_lane_detection`)
  - Identifies road boundaries and calculates lateral vehicle offset
  - Preprocessing chain:
    - Grayscale conversion
    - Gaussian blur for noise suppression
    - Canny edge detection
  - Trapezoidal ROI mask zeroes out sky and background, isolating the road surface
  - Hough Line Transform extracts raw geometric vectors
    - Filtered by slope magnitude to separate left and right lane boundaries
    - Synthesized into dynamic UI overlays

- **Object Detection & Tracking** (`module2_object_detection`)
  - YOLOv8 for real-time visual inference
  - Centroid tracking assigns persistent IDs to vehicles across frames
    - Uses Euclidean distance between bounding boxes over time
  - Pinhole camera model approximates real-world distance to targets
    - Based on focal length constants and bounding box pixel width calibration

- **Lane Keeping** (`module3_lane_keeping`)
  - Computes steering commands based on vehicle offset from the detected lane center
  - Generates `STEER LEFT`, `STEER RIGHT`, or `CENTER` commands
  - Calculates proportional pixel and percentage offsets for HUD telemetry

- **Decision Logic & Sensor Fusion** (`module4_decision_controller`)
  - Fuses lane geometry with object detection output
  - In-lane geometric filtering checks whether a detected object falls within the calculated lane boundaries
  - Time-to-Collision (TTC) calculated continuously as delta distance over delta time
    - TTC under 2.0 seconds triggers a critical state override

- **Multi-Threaded Audio Warnings** (`module5_audio_warning`)
  - Non-blocking audio feedback via Python OS threading
  - Warning beep frequency scales with hazard proximity
    - Critical thresholds set at 10m and 2m

- **Traffic Sign Recognition** (`module6_traffic_sign`)
  - Multi-tier detection strategy using CNN, YOLO fallback, and classical CV segmentation
  - Identifies critical signs (STOP, SPEED LIMIT, TURN) to override standard collision logic

---

## HUD Features

Telemetry panels overlaid directly on the video feed:

- **Environment telemetry** — scene data, active road signs, detected entity counts
- **Ego telemetry** — driver commands (`STEER LEFT`, `BRAKE`, `SAFE`), lane offset in pixels, live TTC values
- **EMA smoothing** — Exponential Moving Average applied to all numeric outputs to prevent flickering from camera noise

---

## Repository Structure

```
adas-pipeline/
├── module0_base/
├── module1_lane_detection/
├── module2_object_detection/
├── module3_lane_keeping/
├── module4_decision_controller/
├── module5_audio_warning/
├── module6_traffic_sign/
├── utils/
├── sample_videos/          ← place your .mp4 / .avi files here
├── run_adas.py
├── requirements.txt
├── yolov8n.pt
└── README.md
```

---

## Setup

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Add a video to sample_videos/

Place any dashcam or road footage (`.mp4`, `.avi`) into the `sample_videos/` folder. The pipeline auto-detects all video files in that folder at startup — no hardcoded paths needed. See `sample_videos/README.md` for details.

The code auto detects sample videos and allows choosing of a specific video from the folder.

### Step 3 — Run

```bash
python run_adas.py
```

The YOLOv8 weights file (`yolov8n.pt`) is bundled in the repository — no first-run download needed.

---

## Keyboard Controls

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `Left Arrow` | Seek backward |
| `Right Arrow` | Seek forward |
| `1` – `9` | Jump to 10%–90% through the video |
| `Q` | Quit |

---

## Notes

- Distance estimation via the pinhole model is an approximation — accuracy depends on how well the focal length and pixel width calibration values match the actual camera used
  - If the input footage changes (different camera or mount position), recalibrate these constants in the config before trusting TTC values

---

## Developer Notes

- The 2.0-second TTC threshold and the 10m/2m audio thresholds are hardcoded
  - These should live in a central config file so they can be tuned without digging through module source files
- EMA smoothing factor is not exposed anywhere visible — if the HUD feels too laggy or too jittery, that is the value to adjust