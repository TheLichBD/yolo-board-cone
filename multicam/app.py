import argparse, os, time
from pathlib import Path
import cv2

from multicam.utils.img import preprocess_bgr
from multicam.utils.overlay import overlay_text
from multicam.detect.onnx_detector import ONNXDetector
from multicam.detect.ort_detector import ORTDetector
from multicam.cameras.realsense import RealSenseCamera
from multicam.cameras.usb import USBCamera
from multicam.ui.window import WindowManager
from multicam.recio.recorder import ensure_writer, auto_record_name

_same_feed_hits = 0  # RS kopya USB kameralarda sayaç


def parse_indices(csv):
    if not csv:
        return []
    out = []
    for t in csv.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except:
            pass
    return out


def build_argparser():
    ap = argparse.ArgumentParser(description="RealSense main + 2 USB helper (ONNX/ORT)")
    ap.add_argument("--onnx", type=str, default="yolov5/runs/train/model_two/weights/best.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)

    ap.add_argument("--rs_w", type=int, default=1280)
    ap.add_argument("--rs_h", type=int, default=720)
    ap.add_argument("--rs_fps", type=int, default=30)

    ap.add_argument("--aux_w", type=int, default=640)
    ap.add_argument("--aux_h", type=int, default=480)
    ap.add_argument("--aux_fps", type=int, default=15)

    ap.add_argument("--auto_usb", action="store_true")
    ap.add_argument("--usb_candidates", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--blacklist", type=str, default="", nargs="?", const="")

    ap.add_argument("--show_depth_cm", action="store_true")
    ap.add_argument("--infer_stride", type=int, default=1)

    ap.add_argument("--record", type=str, default="")
    ap.add_argument("--fourcc", type=str, default="mp4v")
    ap.add_argument("--rec_fps", type=int, default=25)

    ap.add_argument("--rs_scale", type=float, default=1.0)
    ap.add_argument("--aux_scale", type=float, default=0.5)
    ap.add_argument("--window_scale", type=float, default=1.0)
    ap.add_argument("--window_size", type=str, default="")
    ap.add_argument("--list_cams", action="store_true")
    ap.add_argument("--backend", type=str, default="cpu",
                    choices=["cpu", "cuda", "cuda_fp16", "ort_cpu", "ort_cuda"])
    return ap


def list_cameras():
    backend = cv2.CAP_DSHOW if os.name == "nt" else None
    print("Index | Open | First read")
    for i in range(10):
        cap = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
        opened = cap.isOpened()
        ok = False
        if opened:
            ok, _ = cap.read()
        print(f"{i:5d} | {str(opened):5s} | {str(ok):10s}")
        cap.release()


def _same_feed(a, b, thresh=2.0):
    if a is None or b is None:
        return False
    ah, aw = a.shape[:2]
    a_small = cv2.resize(a, (aw // 4, ah // 4))
    b_small = cv2.resize(b, (aw // 4, ah // 4))
    ag = cv2.cvtColor(a_small, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b_small, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ag, bg).mean()
    return diff < thresh


def run():
    args = build_argparser().parse_args()

    if args.list_cams:
        list_cameras()
        return

    if not Path(args.onnx).exists():
        print(f"ERROR: onnx not found: {Path(args.onnx).resolve()}")
        return

    # detector seçimi
    if args.backend in ("ort_cpu", "ort_cuda"):
        detector = ORTDetector(args.onnx, imgsz=args.imgsz, conf_th=args.conf, nms_th=args.nms,
                               use_cuda=(args.backend == "ort_cuda"))
    else:
        detector = ONNXDetector(args.onnx, imgsz=args.imgsz, conf_th=args.conf, nms_th=args.nms,
                                backend=args.backend)

    try:
        rs = RealSenseCamera(args.rs_w, args.rs_h, args.rs_fps)
    except Exception as e:
        print("pyrealsense2 missing or init error:", e)
        return

    cap2 = cap3 = None
    queue = []

    if args.auto_usb:
        cand = parse_indices(args.usb_candidates)
        bl = set(parse_indices(args.blacklist))
        queue = [i for i in cand if i not in bl]

        # RS prob frame
        ok_probe, rs_probe, _, _ = rs.read_aligned()
        rs_probe = rs_probe if ok_probe else None

        chosen = []
        while queue and len(chosen) < 2:
            idx = queue.pop(0)
            cam = USBCamera(idx, args.aux_w, args.aux_h, args.aux_fps)
            if cam.cap is None:
                print(f"[X] failed USB cam index {idx}")
                continue
            f = cam.read_frame()
            if rs_probe is not None and f is not None and _same_feed(rs_probe, f):
                print(f"[INFO] USB cam {idx} seems to be RS UVC, skipping")
                cam.close()
                continue
            print(f"[OK] opened USB cam index {idx}")
            chosen.append(cam)

        cap2 = chosen[0] if len(chosen) > 0 else None
        cap3 = chosen[1] if len(chosen) > 1 else None

    win = WindowManager("Multi-Camera (RS + 2xUSB)", args.window_size)
    writer = None
    want_record = bool(args.record)
    out_path = args.record if args.record else auto_record_name()

    stride = max(1, args.infer_stride)
    frame_idx, t0, frames = 0, time.time(), 0
    last_rs, last2, last3 = [], [], []

    try:
        while True:
            ok_rs, rs_color, rs_depth, depth_scale = rs.read_aligned()
            f2 = cap2.read_frame() if cap2 else None
            f3 = cap3.read_frame() if cap3 else None

            if frame_idx % stride == 0:
                if ok_rs and rs_color is not None:
                    blob, r, d = preprocess_bgr(rs_color, args.imgsz)
                    last_rs = detector.infer(blob, r, d, rs_color.shape)
                if f2 is not None:
                    blob2, r2, d2 = preprocess_bgr(f2, args.imgsz)
                    last2 = detector.infer(blob2, r2, d2, f2.shape)
                if f3 is not None:
                    blob3, r3, d3 = preprocess_bgr(f3, args.imgsz)
                    last3 = detector.infer(blob3, r3, d3, f3.shape)

            vis_rs = rs.draw_detections(rs_color, rs_depth, last_rs,
                                        depth_scale if (args.show_depth_cm and ok_rs) else None) if ok_rs else None
            vis2 = cap2.draw_detections(f2, last2) if f2 is not None else None
            vis3 = cap3.draw_detections(f3, last3) if f3 is not None else None

            canvas = win.compose_canvas(vis_rs, vis2, vis3,
                                        args.rs_scale, args.aux_scale, args.window_scale)

            frames += 1
            if frames % 30 == 0:
                fps = frames / (time.time() - t0 + 1e-9)
                overlay_text(canvas, f"FPS: {fps:.1f}", (10, 30), 0.9, (255, 255, 255))

            if want_record and writer is None:
                writer = ensure_writer(out_path, args.rec_fps, canvas.shape, args.fourcc)

            if writer is not None and want_record:
                writer.write(canvas)

            win.show(canvas)
            key = win.read_key()
            if key in (27, ord("q"), ord("Q")):
                break

            frame_idx += 1
    finally:
        if writer: writer.release()
        rs.close()
        if cap2: cap2.close()
        if cap3: cap3.close()
        win.close()


if __name__ == "__main__":
    run()
