import cv2
import numpy as np

from ..utils.overlay import overlay_text

class BaseCamera:
    def read_frame(self):
        raise NotImplementedError

    def draw_detections(self, frame, dets, color=(0, 255, 0)):
        if frame is None:
            return None
        vis = frame.copy()
        for (x1, y1, x2, y2), cid, sc in dets:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(vis, p1, p2, color, 2)
            overlay_text(vis, f"{cid}:{sc:.2f}", (p1[0], max(0, p1[1] - 8)), 0.6, (255, 255, 255))
        return vis

    def close(self):
        pass
