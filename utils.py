"""
utils.py - Utility functions for Face Analyzer AI
Helper functions for image processing, formatting, and general utilities.
"""

import cv2
import numpy as np
from datetime import datetime
import os


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a frame while preserving aspect ratio with letterboxing."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Create black canvas and center the image
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def calculate_image_sharpness(frame: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher values = sharper image.
    Returns a normalized score 0.0 to 1.0.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: values above 300 are considered very sharp
    score = min(laplacian_var / 300.0, 1.0)
    return float(score)


def calculate_image_brightness(frame: np.ndarray) -> float:
    """
    Estimate image brightness/lighting quality.
    Ideal brightness is in the middle range (not too dark or overexposed).
    Returns a score 0.0 to 1.0.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()  # 0-255

    # Ideal range: 80-180. Penalize extremes.
    if mean_brightness < 80:
        score = mean_brightness / 80.0
    elif mean_brightness > 180:
        score = 1.0 - ((mean_brightness - 180) / 75.0)
    else:
        score = 1.0

    return float(max(0.0, min(1.0, score)))


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def score_to_color(score: float) -> tuple:
    """
    Convert a 0-10 score to an RGB color tuple.
    Low = red, mid = yellow, high = green.
    """
    normalized = score / 10.0
    if normalized < 0.5:
        r = 255
        g = int(normalized * 2 * 255)
        b = 0
    else:
        r = int((1 - normalized) * 2 * 255)
        g = 255
        b = 0
    return (r, g, b)


def score_to_hex(score: float) -> str:
    """Convert a score to a hex color string for Tkinter."""
    r, g, b = score_to_color(score)
    return f"#{r:02x}{g:02x}{b:02x}"


def score_to_label(score: float) -> str:
    """Convert numeric score to a human-readable label."""
    if score >= 9.0:
        return "Excepcional"
    elif score >= 8.0:
        return "Excelente"
    elif score >= 7.0:
        return "Muito Bom"
    elif score >= 6.0:
        return "Bom"
    elif score >= 5.0:
        return "Regular"
    elif score >= 4.0:
        return "Abaixo da Média"
    else:
        return "Baixo"


def save_screenshot(frame: np.ndarray, scores: dict, output_dir: str = "screenshots") -> str:
    """
    Save a screenshot with score overlay to disk.
    Returns the file path saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"face_analysis_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Create annotated copy
    annotated = frame.copy()
    total_score = scores.get("total", 0.0)

    # Draw score overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    color = score_to_color(total_score)
    cv2.putText(annotated, f"Face Score: {total_score:.1f}/10",
                (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    y = 80
    for key, val in scores.items():
        if key != "total":
            label = key.replace("_", " ").title()
            cv2.putText(annotated, f"{label}: {val:.1f}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20

    cv2.imwrite(filepath, annotated)
    return filepath


def normalize_landmarks(landmarks, frame_w: int, frame_h: int) -> list:
    """Convert MediaPipe normalized landmarks to pixel coordinates."""
    points = []
    for lm in landmarks:
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        points.append((x, y))
    return points


def smooth_score(current: float, new_value: float, alpha: float = 0.15) -> float:
    """
    Exponential moving average smoothing for scores.
    alpha: smoothing factor (lower = smoother but slower to react).
    """
    return current * (1 - alpha) + new_value * alpha
