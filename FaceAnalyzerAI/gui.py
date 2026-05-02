"""
gui.py - Modern Tkinter GUI for Face Analyzer AI

Renders the main application window with:
  - Real-time webcam display with facial mesh overlay
  - Animated Face Score (0-10)
  - Individual metric bars
  - Feedback tips panel
  - Control buttons (Start, Pause, Screenshot, Upload Image)

The GUI never blocks: all heavy work runs in background threads.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional

from camera_thread import CameraThread, ImageLoader
from face_analysis import FaceAnalyzer
from utils import save_screenshot, score_to_hex, score_to_label


# ─── Color palette ────────────────────────────────────────────────────────────
BG_DARK    = "#0d0f14"
BG_PANEL   = "#13161e"
BG_CARD    = "#1a1e2a"
ACCENT     = "#00e5ff"
ACCENT2    = "#7c4dff"
TEXT_WHITE = "#f0f4ff"
TEXT_GRAY  = "#8892a4"
SUCCESS    = "#00e676"
WARNING    = "#ffab40"
DANGER     = "#ff5252"

METRIC_LABELS = {
    "symmetry":       "Simetria Facial",
    "proportions":    "Proporções",
    "harmony":        "Harmonia",
    "jaw_structure":  "Estrutura Mandibular",
    "image_quality":  "Qualidade da Imagem",
    "centralization": "Centralização",
    "stability":      "Estabilidade",
}

METRIC_WEIGHTS = {
    "symmetry":       25,
    "proportions":    20,
    "harmony":        15,
    "jaw_structure":  15,
    "image_quality":  10,
    "centralization": 10,
    "stability":       5,
}


class FaceAnalyzerApp:
    """Main application window."""

    DISPLAY_W = 640
    DISPLAY_H = 480
    UPDATE_MS  = 33   # ~30 fps GUI update

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Analyzer AI")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # State
        self._camera: Optional[CameraThread] = None
        self._image_loader: Optional[ImageLoader] = None
        self._analyzer = FaceAnalyzer()
        self._running = False
        self._paused = False
        self._show_mesh = True

        # Threading
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._score_queue: queue.Queue = queue.Queue(maxsize=4)
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_stop = threading.Event()

        # Smoothed display values
        self._display_scores = {k: 0.0 for k in METRIC_LABELS}
        self._display_total = 0.0

        self._status_var = tk.StringVar(value="Pronto — clique em Iniciar Câmera")
        self._build_ui()
        self._schedule_gui_update()

    # ─────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all UI widgets."""
        # ── Title bar ──
        title_frame = tk.Frame(self.root, bg=BG_DARK, pady=8)
        title_frame.pack(fill="x", padx=16)

        tk.Label(
            title_frame,
            text="⬡  FACE ANALYZER AI",
            font=("Courier New", 18, "bold"),
            fg=ACCENT, bg=BG_DARK,
        ).pack(side="left")

        tk.Label(
            title_frame,
            text="Estimativa geométrica — não científica",
            font=("Courier New", 9),
            fg=TEXT_GRAY, bg=BG_DARK,
        ).pack(side="right", pady=4)

        # ── Main layout: camera left, panel right ──
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill="both", padx=16, pady=(0, 8))

        self._build_camera_panel(main_frame)
        self._build_score_panel(main_frame)

        # ── Controls bottom ──
        self._build_controls()

        # ── Status bar ──
        status_bar = tk.Label(
            self.root,
            textvariable=self._status_var,
            font=("Courier New", 9),
            fg=TEXT_GRAY, bg="#0a0c11",
            anchor="w", padx=12, pady=4,
        )
        status_bar.pack(fill="x")

    def _build_camera_panel(self, parent: tk.Frame):
        """Camera feed display."""
        cam_frame = tk.Frame(parent, bg=BG_PANEL, bd=1, relief="flat")
        cam_frame.pack(side="left", padx=(0, 8))

        # Canvas for video
        self._canvas = tk.Canvas(
            cam_frame,
            width=self.DISPLAY_W,
            height=self.DISPLAY_H,
            bg="#000000",
            highlightthickness=0,
        )
        self._canvas.pack()

        # Draw placeholder
        self._canvas.create_text(
            self.DISPLAY_W // 2, self.DISPLAY_H // 2,
            text="📷  Câmera inativa",
            fill=TEXT_GRAY,
            font=("Courier New", 14),
        )

        # Mesh toggle
        mesh_frame = tk.Frame(cam_frame, bg=BG_PANEL)
        mesh_frame.pack(fill="x", padx=8, pady=4)

        self._mesh_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            mesh_frame,
            text="Mostrar malha facial",
            variable=self._mesh_var,
            command=self._toggle_mesh,
            fg=TEXT_GRAY, bg=BG_PANEL,
            selectcolor=BG_CARD,
            activebackground=BG_PANEL,
            font=("Courier New", 9),
        ).pack(side="left")

        self._face_status_label = tk.Label(
            mesh_frame,
            text="● Sem detecção",
            font=("Courier New", 9),
            fg=DANGER, bg=BG_PANEL,
        )
        self._face_status_label.pack(side="right")

    def _build_score_panel(self, parent: tk.Frame):
        """Right panel: score + metrics + feedback."""
        panel = tk.Frame(parent, bg=BG_PANEL, width=320)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)

        # ── Big score display ──
        score_card = tk.Frame(panel, bg=BG_CARD, pady=16)
        score_card.pack(fill="x", padx=8, pady=(8, 6))

        tk.Label(
            score_card,
            text="FACE SCORE",
            font=("Courier New", 10, "bold"),
            fg=TEXT_GRAY, bg=BG_CARD,
        ).pack()

        self._score_label = tk.Label(
            score_card,
            text="—",
            font=("Courier New", 52, "bold"),
            fg=ACCENT, bg=BG_CARD,
        )
        self._score_label.pack()

        self._score_sublabel = tk.Label(
            score_card,
            text="Aguardando análise...",
            font=("Courier New", 10),
            fg=TEXT_GRAY, bg=BG_CARD,
        )
        self._score_sublabel.pack()

        # ── Metric bars ──
        metrics_header = tk.Frame(panel, bg=BG_PANEL)
        metrics_header.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(
            metrics_header,
            text="MÉTRICAS INDIVIDUAIS",
            font=("Courier New", 8, "bold"),
            fg=TEXT_GRAY, bg=BG_PANEL,
        ).pack(side="left")

        self._bar_widgets: dict = {}
        metrics_frame = tk.Frame(panel, bg=BG_PANEL)
        metrics_frame.pack(fill="x", padx=8)

        for key in METRIC_LABELS:
            row = tk.Frame(metrics_frame, bg=BG_PANEL)
            row.pack(fill="x", pady=2)

            label_text = f"{METRIC_LABELS[key]} ({METRIC_WEIGHTS[key]}%)"
            tk.Label(
                row,
                text=label_text,
                font=("Courier New", 8),
                fg=TEXT_GRAY, bg=BG_PANEL,
                width=30, anchor="w",
            ).pack(side="left")

            val_label = tk.Label(
                row,
                text="0.0",
                font=("Courier New", 8, "bold"),
                fg=ACCENT, bg=BG_PANEL,
                width=4,
            )
            val_label.pack(side="right")

            bar_frame = tk.Frame(row, bg=BG_CARD, height=6)
            bar_frame.pack(fill="x", pady=1)
            bar_frame.pack_propagate(False)

            bar_fill = tk.Frame(bar_frame, bg=ACCENT2, height=6, width=0)
            bar_fill.place(x=0, y=0, relheight=1)

            self._bar_widgets[key] = {"fill": bar_fill, "frame": bar_frame, "val": val_label}

        # ── Feedback panel ──
        tk.Label(
            panel,
            text="DICAS",
            font=("Courier New", 8, "bold"),
            fg=TEXT_GRAY, bg=BG_PANEL,
        ).pack(anchor="w", padx=8, pady=(8, 2))

        self._feedback_text = tk.Text(
            panel,
            height=5,
            bg=BG_CARD,
            fg=TEXT_WHITE,
            font=("Courier New", 9),
            relief="flat",
            state="disabled",
            padx=6, pady=4,
            wrap="word",
        )
        self._feedback_text.pack(fill="x", padx=8, pady=(0, 8))

    def _build_controls(self):
        """Bottom button bar with camera selector."""
        ctrl_frame = tk.Frame(self.root, bg=BG_DARK, pady=8)
        ctrl_frame.pack(fill="x", padx=16)

        btn_cfg = dict(
            font=("Courier New", 10, "bold"),
            relief="flat",
            padx=16, pady=8,
            cursor="hand2",
        )

        # ── Camera selector ──
        cam_frame = tk.Frame(ctrl_frame, bg=BG_DARK)
        cam_frame.pack(side="left", padx=(0, 8))

        tk.Label(
            cam_frame, text="Câmera:",
            font=("Courier New", 9), fg=TEXT_GRAY, bg=BG_DARK,
        ).pack(side="left", padx=(0, 4))

        self._cam_var = tk.StringVar()
        self._cam_combo = ttk.Combobox(
            cam_frame,
            textvariable=self._cam_var,
            state="readonly",
            width=22,
            font=("Courier New", 9),
        )
        self._cam_combo.pack(side="left")

        # Style the combobox
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TCombobox",
            fieldbackground=BG_CARD,
            background=BG_CARD,
            foreground=TEXT_WHITE,
            selectbackground=ACCENT2,
            selectforeground=TEXT_WHITE,
        )

        tk.Button(
            cam_frame, text="🔄",
            font=("Courier New", 10), relief="flat",
            bg=BG_CARD, fg=ACCENT, cursor="hand2",
            padx=6, pady=6,
            command=self._refresh_cameras,
        ).pack(side="left", padx=4)

        # Populate camera list
        self._refresh_cameras(silent=True)

        self._btn_start = tk.Button(
            ctrl_frame, text="▶  INICIAR CÂMERA",
            bg=ACCENT2, fg=TEXT_WHITE,
            command=self._start_camera, **btn_cfg,
        )
        self._btn_start.pack(side="left", padx=4)

        self._btn_pause = tk.Button(
            ctrl_frame, text="⏸  PAUSAR",
            bg=BG_CARD, fg=TEXT_WHITE,
            command=self._toggle_pause, state="disabled", **btn_cfg,
        )
        self._btn_pause.pack(side="left", padx=4)

        tk.Button(
            ctrl_frame, text="📷  SCREENSHOT",
            bg=BG_CARD, fg=TEXT_WHITE,
            command=self._take_screenshot, **btn_cfg,
        ).pack(side="left", padx=4)

        tk.Button(
            ctrl_frame, text="🖼  ABRIR IMAGEM",
            bg=BG_CARD, fg=TEXT_WHITE,
            command=self._upload_image, **btn_cfg,
        ).pack(side="left", padx=4)

        tk.Button(
            ctrl_frame, text="✕  SAIR",
            bg=DANGER, fg=TEXT_WHITE,
            command=self._quit, **btn_cfg,
        ).pack(side="right", padx=4)

    # ─────────────────────────────────────────────────────────────────────────
    # Camera & Analysis control
    # ─────────────────────────────────────────────────────────────────────────

    def _start_camera(self, source=None):
        """Start the webcam (or image loader) and analysis thread."""
        self._stop_all()

        if source is not None:
            # Static image mode
            self._image_loader = ImageLoader(source)
            if self._image_loader.get_error():
                messagebox.showerror("Erro", self._image_loader.get_error())
                return
        else:
            # Webcam mode
            cam_index = self._get_selected_camera_index()
            self._camera = CameraThread(camera_index=cam_index, on_error=self._on_camera_error)
            self._camera.start()

        self._running = True
        self._paused = False
        self._analysis_stop.clear()
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True,
            name="AnalysisThread",
        )
        self._analysis_thread.start()

        self._btn_start.config(state="disabled")
        self._btn_pause.config(state="normal")
        self._set_status("Câmera ativa — análise em execução")

    def _stop_all(self):
        """Stop camera and analysis threads."""
        self._running = False
        self._analysis_stop.set()

        if self._camera:
            self._camera.stop()
            self._camera = None

        self._image_loader = None

        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=1.0)
        self._analysis_thread = None

        self._btn_start.config(state="normal")
        self._btn_pause.config(state="disabled")

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._btn_pause.config(text="▶  RETOMAR")
            self._set_status("Análise pausada")
        else:
            self._btn_pause.config(text="⏸  PAUSAR")
            self._set_status("Análise retomada")

    def _toggle_mesh(self):
        self._show_mesh = self._mesh_var.get()

    def _upload_image(self):
        """Open file dialog and load an image for analysis."""
        filepath = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")],
        )
        if filepath:
            self._start_camera(source=filepath)
            self._set_status(f"Imagem carregada: {filepath.split('/')[-1]}")

    def _take_screenshot(self):
        """Save the current frame with score overlay."""
        source = self._camera or self._image_loader
        if source is None:
            messagebox.showinfo("Info", "Nenhuma câmera ativa para screenshot")
            return

        frame = source.get_latest_frame()
        if frame is None:
            return

        try:
            scores = dict(self._display_scores)
            scores["total"] = self._display_total
            path = save_screenshot(frame, scores)
            self._set_status(f"Screenshot salvo: {path}")
            messagebox.showinfo("Screenshot", f"Salvo em:\n{path}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))


    def _detect_cameras(self) -> list:
        """
        Probe camera indices 0-9 and return a list of dicts with
        {index, label} for every device that opens successfully.
        Runs quickly by setting a short read timeout.
        """
        import platform
        found = []
        for i in range(10):
            try:
                if platform.system() == "Windows":
                    cap = __import__("cv2").VideoCapture(i, __import__("cv2").CAP_DSHOW)
                else:
                    cap = __import__("cv2").VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        w = int(cap.get(__import__("cv2").CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(__import__("cv2").CAP_PROP_FRAME_HEIGHT))
                        found.append({"index": i, "label": f"Câmera {i}  ({w}x{h})"})
                cap.release()
            except Exception:
                pass
        return found

    def _refresh_cameras(self, silent: bool = False):
        """Re-scan for cameras and populate the dropdown."""
        self._set_status("Procurando câmeras...")
        self.root.update_idletasks()

        cameras = self._detect_cameras()
        if not cameras:
            cameras = [{"index": 0, "label": "Câmera 0 (padrão)"}]

        labels = [c["label"] for c in cameras]
        self._cameras_data = cameras
        self._cam_combo["values"] = labels
        self._cam_combo.current(0)

        if not silent:
            self._set_status(f"{len(cameras)} câmera(s) encontrada(s)")
        else:
            self._set_status("Pronto — selecione a câmera e clique em Iniciar")

    def _get_selected_camera_index(self) -> int:
        """Return the OpenCV index of the currently selected camera."""
        sel = self._cam_combo.current()
        if hasattr(self, "_cameras_data") and 0 <= sel < len(self._cameras_data):
            return self._cameras_data[sel]["index"]
        return 0

    def _quit(self):
        self._stop_all()
        self._analyzer.release()
        self.root.quit()
        self.root.destroy()

    def _on_camera_error(self, msg: str):
        self.root.after(0, lambda: messagebox.showerror("Erro de Câmera", msg))
        self.root.after(0, lambda: self._set_status(f"Erro: {msg}"))

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis thread
    # ─────────────────────────────────────────────────────────────────────────

    def _analysis_loop(self):
        """
        Background thread: reads frames from camera, runs FaceAnalyzer,
        pushes results to score_queue for the GUI update loop.
        """
        is_static_image = self._image_loader is not None
        source = self._image_loader if is_static_image else self._camera

        while not self._analysis_stop.is_set() and self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            frame = source.get_latest_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            try:
                scores = self._analyzer.analyze(frame)
            except Exception as e:
                time.sleep(0.05)
                continue

            # Draw mesh overlay
            annotated = frame.copy()
            if self._show_mesh and scores.get("face_detected") and scores.get("landmarks_px"):
                annotated = self._analyzer.draw_overlay(annotated, scores["landmarks_px"])

            # Push to GUI queue (drop old frames if queue full)
            try:
                self._score_queue.put_nowait((annotated, scores))
            except queue.Full:
                try:
                    self._score_queue.get_nowait()
                    self._score_queue.put_nowait((annotated, scores))
                except queue.Empty:
                    pass

            # For static images, analyze continuously (small sleep)
            if is_static_image:
                time.sleep(0.05)

    # ─────────────────────────────────────────────────────────────────────────
    # GUI update loop (runs on main thread via after())
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_gui_update(self):
        self.root.after(self.UPDATE_MS, self._gui_update)

    def _gui_update(self):
        """Drain the score queue and refresh all GUI elements."""
        # Process latest result from the analysis thread
        latest_frame = None
        latest_scores = None

        while True:
            try:
                latest_frame, latest_scores = self._score_queue.get_nowait()
            except queue.Empty:
                break

        if latest_frame is not None and latest_scores is not None:
            self._update_video(latest_frame)
            self._update_scores(latest_scores)
            self._update_feedback(latest_scores.get("feedback", []))
            self._update_face_status(latest_scores.get("face_detected", False))

        self._schedule_gui_update()

    def _update_video(self, frame: np.ndarray):
        """Render a BGR frame onto the Tkinter canvas."""
        try:
            h, w = frame.shape[:2]
            scale = min(self.DISPLAY_W / w, self.DISPLAY_H / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)

            # Center on canvas
            x_off = (self.DISPLAY_W - nw) // 2
            y_off = (self.DISPLAY_H - nh) // 2

            self._canvas.delete("all")
            self._canvas.create_image(x_off, y_off, anchor="nw", image=photo)
            self._canvas._photo = photo  # prevent GC
        except Exception:
            pass

    def _update_scores(self, scores: dict):
        """Update score display and metric bars."""
        total = scores.get("total", 0.0)
        self._display_total = total

        hex_color = score_to_hex(total)
        self._score_label.config(
            text=f"{total:.1f}",
            fg=hex_color,
        )
        self._score_sublabel.config(
            text=score_to_label(total),
            fg=hex_color,
        )

        # Update each metric bar
        bar_max_w = 280  # pixels
        for key, widgets in self._bar_widgets.items():
            val = scores.get(key, 0.0)
            self._display_scores[key] = val
            bar_w = int((val / 10.0) * bar_max_w)
            c = score_to_hex(val)
            widgets["fill"].config(bg=c, width=bar_w)
            widgets["val"].config(text=f"{val:.1f}", fg=c)

    def _update_feedback(self, tips: list):
        """Update the feedback text widget."""
        self._feedback_text.config(state="normal")
        self._feedback_text.delete("1.0", "end")
        for tip in tips:
            self._feedback_text.insert("end", tip + "\n")
        self._feedback_text.config(state="disabled")

    def _update_face_status(self, detected: bool):
        if detected:
            self._face_status_label.config(text="● Rosto detectado", fg=SUCCESS)
        else:
            self._face_status_label.config(text="● Sem detecção", fg=DANGER)

    def _set_status(self, msg: str):
        self._status_var.set(msg)
