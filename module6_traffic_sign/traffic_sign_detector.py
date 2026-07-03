"""
Module 6 - Traffic Sign Detector  (v2 — CNN + classical fallback)
------------------------------------------------------------------
Detection strategy
------------------
  Priority 1 — Trained CNN (sign_model.pth)
    Load once at startup.  For each red-blob or yellow-blob ROI
    extracted by colour segmentation, run the CNN classifier.
    Fast (<2 ms per crop on CPU).

  Priority 2 — YOLOv8 (COCO stop-sign class 11)
    Still used as a reliable fallback for STOP signs even without
    a trained model.

  Priority 3 — Classical CV only
    If no model file exists, fall back to shape/colour rules and
    print a one-time reminder to train the model.

Model loading
-------------
  Expects:  module6_traffic_sign/sign_model.pth
  Train with:  python module6_traffic_sign/train_sign_model.py

Returns per sign
----------------
    label       str   : "STOP" | "SPEED LIMIT 30/50/80" | "TURN LEFT/RIGHT"
    confidence  float : 0.0 – 1.0
    bbox        tuple : (x1, y1, x2, y2)
    extra       str   : speed value or ""
"""

import os
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ultralytics import YOLO


# ── Path to trained model ────────────────────────────────────────────────────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_MODULE_DIR, "sign_model.pth")

# ── CNN definition (must match train_sign_model.py) ──────────────────────────
if _TORCH_AVAILABLE:
    class _SignCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

# ── Constants ─────────────────────────────────────────────────────────────────
IOU_MERGE_THRESHOLD = 0.30
MIN_CONTOUR_AREA    = 800
COCO_STOP_CLASS_ID  = 11

SIGN_COLORS = {
    "STOP"           : (0,   0,   220),
    "SPEED LIMIT 30" : (0,   180, 255),
    "SPEED LIMIT 50" : (0,   160, 255),
    "SPEED LIMIT 80" : (0,   140, 255),
    "TURN LEFT"      : (0,   200, 100),
    "TURN RIGHT"     : (0,   200, 100),
    "SPEED LIMIT"    : (0,   180, 255),
    "TURN"           : (0,   200, 100),
}


class TrafficSignDetector:
    """
    Traffic sign detector with CNN + YOLO + classical CV fallback.

    Usage:
        tsd    = TrafficSignDetector()
        result = tsd.detect(frame)
        # result["frame"] → annotated frame
        # result["signs"] → list of sign dicts
    """

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self._cnn        = None
        self._id_to_label = {}
        self._img_size    = 32
        self._preprocess  = None
        self._device      = "cpu"

        # ── Try loading trained CNN ───────────────────────────────────
        if _TORCH_AVAILABLE and os.path.isfile(MODEL_PATH):
            self._load_cnn()
        else:
            if not _TORCH_AVAILABLE:
                print("[TrafficSignDetector] PyTorch not found — using classical CV only.")
            else:
                print("[TrafficSignDetector] sign_model.pth not found.")
                print("  → Run:  python module6_traffic_sign/train_sign_model.py")
                print("  → Falling back to YOLO + classical CV.\n")

        # ── Load YOLO (shared weights) ────────────────────────────────
        print(f"[TrafficSignDetector] Loading YOLO: {model_path}")
        self.yolo = YOLO(model_path)
        print("[TrafficSignDetector] Ready.")

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect traffic signs. Returns annotated frame + list of sign dicts.
        """
        output_frame = frame.copy()
        signs        = []

        # ── YOLO: stop signs ──────────────────────────────────────────
        signs.extend(self._detect_yolo(frame))

        # ── CNN or classical: speed-limit + turn ──────────────────────
        if self._cnn is not None:
            signs.extend(self._detect_with_cnn(frame))
        else:
            signs.extend(self._detect_classical(frame))

        # ── Deduplicate ───────────────────────────────────────────────
        signs = self._deduplicate(signs)

        # ── Draw ──────────────────────────────────────────────────────
        for sign in signs:
            self._draw_sign(output_frame, sign)

        return {"frame": output_frame, "signs": signs}

    # ──────────────────────────────────────────────────────────────────
    # CNN loading + inference
    # ──────────────────────────────────────────────────────────────────

    def _load_cnn(self):
        """Load trained CNN weights from sign_model.pth."""
        try:
            checkpoint         = torch.load(MODEL_PATH, map_location="cpu",
                                            weights_only=False)
            num_classes        = checkpoint["num_classes"]
            self._id_to_label  = checkpoint["id_to_label"]
            self._img_size     = checkpoint.get("img_size", 32)
            self._device       = "cuda" if torch.cuda.is_available() else "cpu"

            model = _SignCNN(num_classes=num_classes)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            model.to(self._device)
            self._cnn = model

            # Normalisation transform (same as training)
            self._preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self._img_size, self._img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])
            print(f"[TrafficSignDetector] CNN loaded ({num_classes} classes) "
                  f"from {MODEL_PATH}")
        except Exception as e:
            print(f"[TrafficSignDetector] Failed to load CNN: {e}")
            self._cnn = None

    def _classify_crop(self, crop: np.ndarray):
        """
        Run CNN on a BGR crop. Returns (label, confidence) or (None, 0).
        """
        if self._cnn is None or crop.size == 0:
            return None, 0.0

        rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inp   = self._preprocess(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._cnn(inp)
            probs  = torch.softmax(logits, dim=1)[0]
            conf, cls = probs.max(0)

        label = self._id_to_label.get(cls.item(), "UNKNOWN")
        return label, float(conf)

    def _detect_with_cnn(self, frame: np.ndarray) -> list:
        """
        Extract colour-blob ROIs, classify each with the CNN.
        Returns sign dicts whose CNN confidence ≥ 0.70.
        """
        signs = []
        rois  = self._extract_rois(frame)

        for (x, y, w, h) in rois:
            # Clamp to frame bounds
            x1 = max(x, 0);  y1 = max(y, 0)
            x2 = min(x + w, frame.shape[1])
            y2 = min(y + h, frame.shape[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label, conf = self._classify_crop(crop)
            if label is None or conf < 0.70:
                continue

            # Parse speed value from label e.g. "SPEED LIMIT 50"
            extra = ""
            if "SPEED LIMIT" in label:
                parts = label.split()
                extra = parts[-1] if parts[-1].isdigit() else ""

            signs.append({
                "label"     : label,
                "confidence": round(conf, 2),
                "bbox"      : (x1, y1, x2, y2),
                "extra"     : extra,
            })

        return signs

    def _extract_rois(self, frame: np.ndarray) -> list:
        """
        Colour segmentation to find candidate sign regions.
        Returns list of (x, y, w, h) bounding boxes.
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rois = []

        # Red blobs (stop / speed-limit)
        m1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 120, 70), (180, 255, 255))
        red_mask = cv2.morphologyEx(
            cv2.bitwise_or(m1, m2),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Yellow blobs (turn/warning)
        yel_mask = cv2.morphologyEx(
            cv2.inRange(hsv, (18, 100, 100), (35, 255, 255)),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        for mask in (red_mask, yel_mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if 0.5 < w / max(h, 1) < 2.0:
                    # Add small padding around the ROI
                    pad = 6
                    rois.append((x - pad, y - pad, w + 2*pad, h + 2*pad))

        return rois

    # ──────────────────────────────────────────────────────────────────
    # YOLO (stop signs)
    # ──────────────────────────────────────────────────────────────────

    def _detect_yolo(self, frame: np.ndarray) -> list:
        results = self.yolo(frame, verbose=False)[0]
        signs   = []
        for box in results.boxes:
            if int(box.cls[0]) != COCO_STOP_CLASS_ID:
                continue
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            signs.append({"label": "STOP", "confidence": round(conf, 2),
                          "bbox": (x1, y1, x2, y2), "extra": ""})
        return signs

    # ──────────────────────────────────────────────────────────────────
    # Classical CV fallback (no model)
    # ──────────────────────────────────────────────────────────────────

    def _detect_classical(self, frame: np.ndarray) -> list:
        return self._detect_red_circle(frame) + self._detect_yellow_diamond(frame)

    def _detect_red_circle(self, frame):
        signs = []
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1    = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        m2    = cv2.inRange(hsv, (160, 120, 70), (180, 255, 255))
        mask  = cv2.morphologyEx(cv2.bitwise_or(m1, m2), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0 or 4*np.pi*cv2.contourArea(cnt)/(peri*peri) < 0.55:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.6 < w / max(h,1) < 1.6:
                signs.append({"label": "SPEED LIMIT", "confidence": 0.70,
                               "bbox": (x, y, x+w, y+h), "extra": ""})
        return signs

    def _detect_yellow_diamond(self, frame):
        signs = []
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.morphologyEx(cv2.inRange(hsv, (18,100,100),(35,255,255)),
                                  cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
            if len(approx) not in (4, 5):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.6 < w / max(h,1) < 1.6:
                signs.append({"label": "TURN", "confidence": 0.65,
                               "bbox": (x, y, x+w, y+h), "extra": ""})
        return signs

    # ──────────────────────────────────────────────────────────────────
    # Deduplication + drawing
    # ──────────────────────────────────────────────────────────────────

    def _deduplicate(self, signs):
        keep = []
        for s in signs:
            dominated = False
            for i, k in enumerate(keep):
                if self._iou(s["bbox"], k["bbox"]) > IOU_MERGE_THRESHOLD:
                    if s["confidence"] > k["confidence"]:
                        keep[i] = s
                    dominated = True
                    break
            if not dominated:
                keep.append(s)
        return keep

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        return inter / (max(1,(a[2]-a[0])*(a[3]-a[1]))
                        + max(1,(b[2]-b[0])*(b[3]-b[1])) - inter)

    def _draw_sign(self, frame, sign):
        lbl  = sign["label"]
        conf = sign["confidence"]
        x1, y1, x2, y2 = sign["bbox"]
        extra = sign.get("extra", "")
        color = SIGN_COLORS.get(lbl, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        return frame
