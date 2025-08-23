import cv2
import numpy as np
from ..utils.overlay import overlay_text

class WindowManager:
    def __init__(self, title, window_size=""):
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if window_size:
            try:
                W, H = map(int, window_size.lower().split("x"))
                cv2.resizeWindow(self.title, W, H)
            except Exception:
                pass

    def compose_canvas(self, rs_vis, cam2_vis, cam3_vis, rs_scale=1.0, aux_scale=0.5, window_scale=1.0):
        base = rs_vis if rs_vis is not None else (cam2_vis if cam2_vis is not None else cam3_vis)
        if base is None:
            base = np.zeros((480, 640, 3), np.uint8)
        rs_img = base if rs_vis is None else (rs_vis if rs_scale == 1.0 else cv2.resize(rs_vis, (0, 0), fx=rs_scale, fy=rs_scale))
        h, w = rs_img.shape[:2]
        sw, sh = int(w * aux_scale), int(h * aux_scale)

        def norm(x, label):
            if x is None:
                blk = np.zeros((sh, sw, 3), np.uint8)
                overlay_text(blk, label, (20, 40), 0.8, (0, 0, 255))
                return blk
            return cv2.resize(x, (sw, sh))

        right = np.vstack([norm(cam2_vis, "No signal (Cam2)"), norm(cam3_vis, "No signal (Cam3)")])
        canvas = np.hstack([rs_img, right])
        if window_scale != 1.0:
            canvas = cv2.resize(canvas, (0, 0), fx=window_scale, fy=window_scale)
        return canvas

    def show(self, canvas):
        cv2.imshow(self.title, canvas)

    def read_key(self):
        return cv2.waitKey(1) & 0xFF

    def toggle_fullscreen(self):
        fs = cv2.getWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, 1 if fs == 0 else 0)

    def close(self):
        cv2.destroyAllWindows()
