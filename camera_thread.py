"""
camera_thread.py - Webcam capture thread for Face Analyzer AI

Runs OpenCV frame capture in a dedicated background thread,
decoupled from both the analysis pipeline and the GUI.
This prevents the UI from blocking on camera I/O.
"""

import cv2
import threading
import time
import numpy as np
from typing import Optional, Callable


class CameraThread(threading.Thread):
    """
    Background thread that continuously reads frames from a webcam.

    Usage:
        cam = CameraThread(camera_index=0)
        cam.start()
        frame = cam.get_latest_frame()  # safe from any thread
        cam.stop()
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 30,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(daemon=True, name="CameraThread")
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.on_error = on_error  # optional callback for error reporting

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._is_running = False
        self._frame_count = 0
        self._error: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Thread lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Main thread loop: open camera and continuously grab frames."""
        self._stop_event.clear()

        try:
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if self._is_windows() else 0)
        except Exception:
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            msg = f"Não foi possível abrir a câmera (índice {self.camera_index})"
            self._set_error(msg)
            return

        # Set preferred resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._is_running = True
        frame_interval = 1.0 / self.target_fps

        while not self._stop_event.is_set():
            t_start = time.time()

            ret, frame = self._cap.read()
            if not ret:
                # Retry a few times before giving up
                self._error = "Falha na leitura do frame"
                time.sleep(0.1)
                continue

            # Flip horizontally (mirror view, more natural for front-cam)
            frame = cv2.flip(frame, 1)

            with self._lock:
                self._frame = frame
                self._frame_count += 1
                self._error = None

            # Throttle to target FPS
            elapsed = time.time() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._is_running = False
        if self._cap:
            self._cap.release()

    def stop(self):
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        self.join(timeout=2.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API (thread-safe)
    # ─────────────────────────────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recently captured frame (or None if unavailable)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_frame_count(self) -> int:
        """Return total number of frames captured so far."""
        with self._lock:
            return self._frame_count

    def is_running(self) -> bool:
        """Return True if the camera is actively streaming."""
        return self._is_running

    def get_error(self) -> Optional[str]:
        """Return the last error message, or None if no error."""
        return self._error

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _set_error(self, msg: str):
        self._error = msg
        if self.on_error:
            self.on_error(msg)

    @staticmethod
    def _is_windows() -> bool:
        import platform
        return platform.system() == "Windows"


class ImageLoader:
    """
    Loads a static image file and exposes it via the same interface
    as CameraThread, so the analysis pipeline works identically for
    both webcam and uploaded images.
    """

    def __init__(self, filepath: str):
        self._frame: Optional[np.ndarray] = None
        self._error: Optional[str] = None
        self._load(filepath)

    def _load(self, filepath: str):
        frame = cv2.imread(filepath)
        if frame is None:
            self._error = f"Não foi possível carregar: {filepath}"
        else:
            self._frame = frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        return self._frame.copy() if self._frame is not None else None

    def is_running(self) -> bool:
        return self._frame is not None

    def get_error(self) -> Optional[str]:
        return self._error

    def stop(self):
        pass  # No-op for static images
