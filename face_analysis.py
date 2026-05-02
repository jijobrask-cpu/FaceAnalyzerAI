"""
face_analysis.py - Core facial analysis engine for Face Analyzer AI

Uses the modern MediaPipe FaceLandmarker Tasks API (compatible with
mediapipe >= 0.10.0), which replaced the deprecated mp.solutions API.

All scores are geometric estimates only — NOT scientific measurements.
"""

import cv2
import numpy as np
import urllib.request
import os
import threading
from collections import deque
from utils import (
    euclidean_distance,
    calculate_image_sharpness,
    calculate_image_brightness,
    smooth_score,
)

# ─── Model download ───────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

def _ensure_model():
    """Download the FaceLandmarker model file if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("[FaceAnalyzer] Baixando modelo MediaPipe (~6 MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[FaceAnalyzer] Modelo baixado com sucesso.")
        except Exception as e:
            raise RuntimeError(
                f"Falha ao baixar o modelo MediaPipe: {e}\n"
                f"Baixe manualmente em:\n  {MODEL_URL}\n"
                f"E salve como: {MODEL_PATH}"
            )

# ─── Key Landmark Indices ────────────────────────────────────────────────────
LEFT_EYE_OUTER   = 33
LEFT_EYE_INNER   = 133
RIGHT_EYE_OUTER  = 362
RIGHT_EYE_INNER  = 263
LEFT_EYE_TOP     = 159
LEFT_EYE_BOTTOM  = 145
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
NOSE_TIP   = 1
NOSE_BASE  = 2
NOSE_LEFT  = 64
NOSE_RIGHT = 294
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
CHIN       = 152
FOREHEAD   = 10
FACE_LEFT  = 234
FACE_RIGHT = 454
JAW_LEFT   = 172
JAW_RIGHT  = 397
BROW_INNER = 70

SYMMETRY_PAIRS = [
    (33, 362), (133, 263), (159, 386), (145, 374),
    (61, 291), (172, 397), (234, 454), (70, 300),
    (105, 334), (46, 276),
]


class FaceAnalyzer:
    """
    Multi-criteria facial geometry analyzer using MediaPipe FaceLandmarker.
    Compatible with mediapipe >= 0.10.0 (Tasks API).
    """

    def __init__(self):
        _ensure_model()
        self._landmarker = None
        self._lock = threading.Lock()
        self._init_landmarker()

        self._smoothed = {k: 5.0 for k in [
            "symmetry", "proportions", "harmony",
            "jaw_structure", "image_quality", "centralization", "stability", "total"
        ]}
        self._detection_history = deque(maxlen=20)

        self.WEIGHTS = {
            "symmetry":       0.25,
            "proportions":    0.20,
            "harmony":        0.15,
            "jaw_structure":  0.15,
            "image_quality":  0.10,
            "centralization": 0.10,
            "stability":      0.05,
        }

    def _init_landmarker(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # ─── PUBLIC API ───────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray) -> dict:
        import mediapipe as mp

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._lock:
            result = self._landmarker.detect(mp_image)

        face_detected = bool(result.face_landmarks)
        self._detection_history.append(1 if face_detected else 0)
        stability_score = (sum(self._detection_history) / max(len(self._detection_history), 1)) * 10.0
        quality_score = (calculate_image_sharpness(frame)*0.6 + calculate_image_brightness(frame)*0.4) * 10.0

        if not face_detected:
            raw = {k: 0.0 for k in self.WEIGHTS}
            raw["image_quality"] = quality_score
            raw["stability"]     = stability_score
            raw["total"]         = 0.0
            scores = self._smooth_all(raw)
            scores["face_detected"] = False
            scores["landmarks_px"]  = []
            scores["feedback"]      = self._get_feedback(scores, False)
            return scores

        lm_list = result.face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]

        raw = {
            "symmetry":       self._score_symmetry(pts, w),
            "proportions":    self._score_proportions(pts),
            "harmony":        self._score_harmony(pts),
            "jaw_structure":  self._score_jaw_structure(pts),
            "image_quality":  quality_score,
            "centralization": self._score_centralization(pts, w, h),
            "stability":      stability_score,
        }
        raw["total"] = min(sum(raw[k]*v for k, v in self.WEIGHTS.items()), 10.0)

        scores = self._smooth_all(raw)
        scores["face_detected"] = True
        scores["landmarks_px"]  = pts
        scores["feedback"]      = self._get_feedback(scores, True)
        return scores

    def draw_overlay(self, frame: np.ndarray, landmarks_px: list) -> np.ndarray:
        if not landmarks_px:
            return frame
        output = frame.copy()
        for (x, y) in landmarks_px:
            cv2.circle(output, (x, y), 1, (0, 220, 120), -1)
        self._draw_feature_lines(output, landmarks_px)
        return output

    # ─── METRICS ──────────────────────────────────────────────────────────────

    def _score_symmetry(self, pts, frame_w):
        try:
            nose_x = pts[NOSE_TIP][0]
            devs = []
            for li, ri in SYMMETRY_PAIRS:
                if li >= len(pts) or ri >= len(pts):
                    continue
                lx, ly = pts[li]; rx, ry = pts[ri]
                l_d = abs(lx - nose_x); r_d = abs(rx - nose_x)
                h_sym = 1.0 - abs(l_d - r_d) / (l_d + r_d + 1e-6)
                face_h = abs(pts[CHIN][1] - pts[FOREHEAD][1]) + 1
                v_sym = 1.0 - min(abs(ly - ry) / (face_h * 0.1), 1.0)
                devs.append((h_sym + v_sym) / 2.0)
            return float(np.clip(np.mean(devs)*10.0, 0, 10)) if devs else 5.0
        except Exception:
            return 5.0

    def _score_proportions(self, pts):
        try:
            face_h = euclidean_distance(pts[FOREHEAD], pts[CHIN])
            if face_h < 10: return 5.0
            t = euclidean_distance(pts[FOREHEAD], pts[BROW_INNER])
            m = euclidean_distance(pts[BROW_INNER], pts[NOSE_BASE])
            b = euclidean_distance(pts[NOSE_BASE], pts[CHIN])
            total = t + m + b + 1e-6; ideal = total / 3.0
            thirds_score = 1.0 - min((abs(t-ideal)+abs(m-ideal)+abs(b-ideal))/total, 1.0)
            face_w = euclidean_distance(pts[FACE_LEFT], pts[FACE_RIGHT])
            le_w = euclidean_distance(pts[LEFT_EYE_OUTER], pts[LEFT_EYE_INNER])
            re_w = euclidean_distance(pts[RIGHT_EYE_OUTER], pts[RIGHT_EYE_INNER])
            avg_eye_w = (le_w + re_w) / 2.0
            eye_score   = 1.0 - min(abs(avg_eye_w/(face_w/5.0+1e-6)-1.0), 1.0)
            mouth_w     = euclidean_distance(pts[MOUTH_LEFT], pts[MOUTH_RIGHT])
            mouth_score = 1.0 - min(abs(mouth_w/(avg_eye_w*1.5+1e-6)-1.0), 1.0)
            return float(np.clip((thirds_score*0.4+eye_score*0.3+mouth_score*0.3)*10, 0, 10))
        except Exception:
            return 5.0

    def _score_harmony(self, pts):
        try:
            face_w = euclidean_distance(pts[FACE_LEFT], pts[FACE_RIGHT])
            face_h = euclidean_distance(pts[FOREHEAD], pts[CHIN])
            if face_w < 10 or face_h < 10: return 5.0
            pl = ((pts[LEFT_EYE_OUTER][0]+pts[LEFT_EYE_INNER][0])/2,
                  (pts[LEFT_EYE_OUTER][1]+pts[LEFT_EYE_INNER][1])/2)
            pr = ((pts[RIGHT_EYE_OUTER][0]+pts[RIGHT_EYE_INNER][0])/2,
                  (pts[RIGHT_EYE_OUTER][1]+pts[RIGHT_EYE_INNER][1])/2)
            ipd = euclidean_distance(pl, pr)
            ipd_score   = 1.0 - min(abs(ipd/face_w-0.46)/0.2, 1.0)
            nose_h      = euclidean_distance(pts[BROW_INNER], pts[NOSE_TIP])
            nose_score  = 1.0 - min(abs(nose_h/face_h-0.30)/0.15, 1.0)
            fy, cy      = pts[FOREHEAD][1], pts[CHIN][1]
            my          = (pts[MOUTH_TOP][1]+pts[MOUTH_BOTTOM][1])/2
            mp_ratio    = (my-fy)/(cy-fy+1e-6)
            mouth_score = 1.0 - min(abs(mp_ratio-0.75)/0.15, 1.0)
            return float(np.clip((ipd_score*0.4+nose_score*0.3+mouth_score*0.3)*10, 0, 10))
        except Exception:
            return 5.0

    def _score_jaw_structure(self, pts):
        try:
            face_w = euclidean_distance(pts[FACE_LEFT], pts[FACE_RIGHT])
            face_h = euclidean_distance(pts[FOREHEAD], pts[CHIN])
            if face_w < 10: return 5.0
            jaw_w        = euclidean_distance(pts[JAW_LEFT], pts[JAW_RIGHT])
            jaw_score    = 1.0 - min(abs(jaw_w/face_w-0.70)/0.25, 1.0)
            cx           = (pts[JAW_LEFT][0]+pts[JAW_RIGHT][0])/2
            taper        = abs(pts[CHIN][0]-cx)/(jaw_w/2+1e-6)
            taper_score  = 1.0 - min(taper/0.3, 1.0)
            aspect       = face_h/(face_w+1e-6)
            aspect_score = 1.0 - min(abs(aspect-1.45)/0.5, 1.0)
            return float(np.clip((jaw_score*0.4+taper_score*0.25+aspect_score*0.35)*10, 0, 10))
        except Exception:
            return 5.0

    def _score_centralization(self, pts, frame_w, frame_h):
        try:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2
            dx = abs(cx-frame_w/2)/(frame_w/2); dy = abs(cy-frame_h/2)/(frame_h/2)
            center_score = 1.0 - min((dx+dy)/2, 1.0)
            ratio = (max(xs)-min(xs))/frame_w
            if 0.20 <= ratio <= 0.60:   size_score = 1.0
            elif ratio < 0.20:          size_score = ratio/0.20
            else:                       size_score = max(0, 1.0-(ratio-0.60)/0.40)
            return float(np.clip((center_score*0.6+size_score*0.4)*10, 0, 10))
        except Exception:
            return 5.0

    # ─── HELPERS ──────────────────────────────────────────────────────────────

    def _smooth_all(self, raw):
        out = {}
        for k, v in raw.items():
            if k in self._smoothed:
                self._smoothed[k] = smooth_score(self._smoothed[k], v)
                out[k] = round(self._smoothed[k], 2)
            else:
                out[k] = v
        return out

    def _get_feedback(self, scores, face_detected):
        if not face_detected:
            return ["⚠️  Nenhum rosto detectado — posicione-se na frente da câmera"]
        tips = []
        if scores.get("centralization", 10) < 6:
            tips.append("📐 Centralize o rosto na câmera para maior precisão")
        if scores.get("image_quality", 10) < 5:
            tips.append("💡 Melhore a iluminação para resultados mais precisos")
        if scores.get("stability", 10) < 6:
            tips.append("🔄 Mantenha o rosto estável para melhor detecção")
        if scores.get("symmetry", 10) < 5:
            tips.append("↔️  Vire o rosto de frente para a câmera")
        if scores.get("image_quality", 10) < 4:
            tips.append("🔍 Imagem com pouca nitidez — reduza o movimento")
        if not tips:
            tips.append("✅ Condições ideais — análise estável")
        return tips

    def _draw_feature_lines(self, frame, pts):
        def line(i, j, color, t=1):
            if i < len(pts) and j < len(pts):
                cv2.line(frame, pts[i], pts[j], color, t)
        ec = (100, 220, 255)
        line(LEFT_EYE_OUTER, LEFT_EYE_TOP,    ec); line(LEFT_EYE_TOP,    LEFT_EYE_INNER,   ec)
        line(LEFT_EYE_INNER, LEFT_EYE_BOTTOM, ec); line(LEFT_EYE_BOTTOM, LEFT_EYE_OUTER,   ec)
        line(RIGHT_EYE_OUTER,RIGHT_EYE_TOP,   ec); line(RIGHT_EYE_TOP,   RIGHT_EYE_INNER,  ec)
        line(RIGHT_EYE_INNER,RIGHT_EYE_BOTTOM,ec); line(RIGHT_EYE_BOTTOM,RIGHT_EYE_OUTER,  ec)
        nc = (255, 180, 50)
        line(NOSE_TIP, NOSE_LEFT, nc); line(NOSE_TIP, NOSE_RIGHT, nc)
        mc = (255, 100, 150)
        line(MOUTH_LEFT, MOUTH_TOP, mc);  line(MOUTH_TOP,    MOUTH_RIGHT,  mc)
        line(MOUTH_RIGHT,MOUTH_BOTTOM,mc);line(MOUTH_BOTTOM, MOUTH_LEFT,   mc)
        jc = (80, 255, 180)
        line(JAW_LEFT, CHIN, jc, 2); line(CHIN, JAW_RIGHT, jc, 2)

    def release(self):
        if self._landmarker:
            self._landmarker.close()
