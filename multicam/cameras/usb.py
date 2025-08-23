import os
import cv2
import time

from .base import BaseCamera
from ..utils.overlay import overlay_text

def try_open_cam(index, backend=None, w=None, h=None, fps=None, test_frames=1):
    if backend is None:
        backend = cv2.CAP_MSMF if os.name == "nt" else None
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    if w:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    if h:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    ok = False
    for _ in range(test_frames):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.02)
    if not ok:
        cap.release()
        return None
    return cap

class USBCamera(BaseCamera):
    def __init__(self, index, w=None, h=None, fps=None):
        backend = cv2.CAP_DSHOW if os.name == "nt" else None  # Windows için DSHOW dene
        self.cap = try_open_cam(index, backend, w, h, fps)
        self.index = index

    def read_frame(self):
        if self.cap is None:
            return None
        ok, f = self.cap.read()
        return f if ok else None

    def draw_detections(self, frame, dets, color=(255, 0, 0)):
        vis = super().draw_detections(frame, dets, color=color)
        if vis is not None:
            from ..utils.overlay import overlay_text
            overlay_text(vis, f"USB {self.index}", (10, 25), 0.7, (255, 255, 0))
        return vis

    def close(self):
        if self.cap is not None:
            self.cap.release()

def pick_two_usb(candidates, blacklist=None, w=None, h=None, fps=None):
    blacklist = set(blacklist or [])
    picked = []
    for idx in candidates:
        if idx in blacklist:
            print(f"[INFO] skip blacklisted index {idx}")
            continue
        cam = USBCamera(idx, w, h, fps)
        if cam.cap is not None:
            picked.append(cam)
            print(f"[OK] opened USB cam index {idx}")
            if len(picked) == 2:
                break
        else:
            print(f"[X] failed USB cam index {idx}")
    return picked
