"""
main.py - Entry point for Face Analyzer AI

Run with:
    python main.py

Requirements:
    pip install -r requirements.txt
"""

import tkinter as tk
import sys
import os


def check_dependencies():
    """Verify that required packages are installed before launching."""
    missing = []
    packages = {
        "cv2":       "opencv-python",
        "mediapipe": "mediapipe",
        "numpy":     "numpy",
        "PIL":       "Pillow",
    }
    for module, pkg in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("=" * 50)
        print("DEPENDÊNCIAS FALTANDO:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstale com:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 50)
        sys.exit(1)


def main():
    check_dependencies()

    # Ensure we're running from the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from gui import FaceAnalyzerApp

    root = tk.Tk()
    root.title("Face Analyzer AI")

    # Center window on screen
    window_w, window_h = 1010, 640
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - window_w) // 2
    y = (screen_h - window_h) // 2
    root.geometry(f"{window_w}x{window_h}+{x}+{y}")

    app = FaceAnalyzerApp(root)

    # Handle window close button
    root.protocol("WM_DELETE_WINDOW", app._quit)

    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        input("\nPressione ENTER para sair...")
