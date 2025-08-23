import numpy as np
import cv2

from .base import BaseCamera
from ..utils.overlay import overlay_text

class RealSenseCamera(BaseCamera):
    def __init__(self, w=1280, h=720, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def read_aligned(self, timeout_ms=1000):
        try:
            frameset = self.pipe.wait_for_frames(timeout_ms)
        except Exception:
            return False, None, None, None
        if frameset is None:
            return False, None, None, None
        frameset = self.align.process(frameset)
        depth = frameset.get_depth_frame()
        color = frameset.get_color_frame()
        if not depth or not color:
            return False, None, None, None
        rs_frame = np.asanyarray(color.get_data())
        depth_img = np.asanyarray(depth.get_data())
        return True, rs_frame, depth_img, self.depth_scale

    def draw_detections(self, frame_bgr, depth_img, dets, depth_scale=None):
        if frame_bgr is None:
            return None
        vis = frame_bgr.copy()
        for (x1, y1, x2, y2), cid, sc in dets:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
            lbl = f"{cid}:{sc:.2f}"
            if depth_scale is not None and depth_img is not None:
                d_cm = self.median_depth_cm(depth_img, x1, y1, x2, y2, depth_scale)
                if d_cm is not None:
                    lbl += f" {d_cm:.0f}cm"
            overlay_text(vis, lbl, (p1[0], max(0, p1[1] - 8)), 0.6, (0, 255, 255))
        return vis

    @staticmethod
    def median_depth_cm(depth_frame, x1, y1, x2, y2, scale):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if x2 <= x1 or y2 <= y1:
            return None
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        k = 10
        x1s, y1s, x2s, y2s = max(cx - k, 0), max(cy - k, 0), cx + k, cy + k
        roi = depth_frame[y1s:y2s, x1s:x2s].astype(np.float32)
        if roi.size == 0:
            return None
        vals = roi[roi > 0]
        if vals.size == 0:
            return None
        return float(np.median(vals) * scale * 100.0)

    def close(self):
        self.pipe.stop()
