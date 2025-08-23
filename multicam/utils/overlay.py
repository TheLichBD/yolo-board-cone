import cv2
import numpy as np

def overlay_text(img, text, org=(10, 30), scale=0.8, color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def draw_boxes(img, dets, color=(0, 255, 0), show_scores=True):
    vis = img.copy()
    for (x1, y1, x2, y2), cid, sc in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, color, 2)
        if show_scores:
            t = f"{cid}:{sc:.2f}"
            overlay_text(vis, t, (p1[0], max(0, p1[1] - 8)), 0.6, (255, 255, 0))
    return vis
