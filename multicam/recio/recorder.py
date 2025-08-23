import cv2
from pathlib import Path
from datetime import datetime

def ensure_writer(path_str, fps, frame_shape, fourcc_str="mp4v"):
    h, w = frame_shape[:2]
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    wr = cv2.VideoWriter(str(p), fourcc, fps, (w, h))
    return wr

def auto_record_name(prefix="multicam"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / "record"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{prefix}_{ts}.mp4")
