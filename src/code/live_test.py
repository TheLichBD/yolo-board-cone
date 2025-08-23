# live_test.py
from pathlib import Path
from ultralytics import YOLO
import argparse, shutil, sys, time
import numpy as np
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def find_best_weights(runs_detect: Path) -> Path | None:
    cands = list(runs_detect.rglob("weights/best.pt"))
    if not cands: return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]

def pick_one_image(data_root: Path) -> Path | None:
    for split in ("val", "test", "train"):
        d = data_root / "images" / split
        if d.exists():
            for im in d.rglob("*.*"):
                if im.suffix.lower() in IMG_EXTS:
                    return im
    return None

def ensure_example_image(dst_dir: Path, example: Path | None) -> Path | None:
    if example is None: return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / example.name
    if not out.exists():
        try: shutil.copy2(example, out)
        except Exception: return example
    return out

def run_val(model, data_yaml, args):
    chosen_split = args.split
    try:
        print(f"\n=== VAL (split={chosen_split}) ===")
        metrics = model.val(
            data=str(data_yaml),
            split=chosen_split,
            imgsz=args.imgsz,
            conf=args.val_conf,
            iou=args.iou,
            batch=args.batch,
            device=args.device,
            plots=True,
            save_json=True,
            verbose=True
        )
    except FileNotFoundError:
        if chosen_split != "val":
            print("[WARN] Istenen split bulunamadi, 'val' ile tekrar deniyorum.")
            metrics = model.val(
                data=str(data_yaml),
                split="val",
                imgsz=args.imgsz,
                conf=args.val_conf,
                iou=args.iou,
                batch=args.batch,
                device=args.device,
                plots=True,
                save_json=True,
                verbose=True
            )
        else:
            raise
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")

def run_single_image_infer(model, root, args):
    example = pick_one_image(root / "data")
    example = ensure_example_image(root / "src" / "_sample", example)
    if example is None:
        print("\n[WARN] Ornek resim bulunamadi, inference atlandi.")
        return
    print(f"\n=== INFERENCE ===\n[INFO] image: {example}")
    _ = model.predict(
        source=str(example),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        verbose=True
    )
    print("[INFO] Cikti: runs/detect/predict*/ altinda.")

def run_webcam(model, args):
    print("\n=== WEBCAM ===  (Cikis: Q veya ESC)")
    stream = model.predict(
        source=args.webcam,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False
    )
    writer = None
    t0, frames = time.time(), 0
    for result in stream:
        frame = result.plot()
        frames += 1
        dt = time.time() - t0
        if dt > 0:
            fps = frames / dt
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if args.record:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, fourcc, 30, (w, h))
            writer.write(frame)
        cv2.imshow("YOLO Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
    if writer is not None: writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam bitti.")

def median_depth_cm(depth_frame, x1, y1, x2, y2, scale):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    if x2 <= x1 or y2 <= y1: return None
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    k = 10
    x1s, y1s, x2s, y2s = max(cx - k, 0), max(cy - k, 0), cx + k, cy + k
    depth_roi = depth_frame[y1s:y2s, x1s:x2s].astype(np.float32)
    if depth_roi.size == 0: return None
    vals = depth_roi[depth_roi > 0]
    if vals.size == 0: return None
    return float(np.median(vals) * scale * 100.0)

def run_realsense(model, args):
    print("\n=== REALSENSE ===  (Cikis: Q veya ESC)")
    try:
        import pyrealsense2 as rs
    except Exception as e:
        print("pyrealsense2 import hatasi:", e, file=sys.stderr)
        print("pip install pyrealsense2", file=sys.stderr)
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    writer = None
    t0, frames = time.time(), 0
    try:
        while True:
            frameset = pipeline.wait_for_frames()
            frameset = align.process(frameset)
            depth = frameset.get_depth_frame()
            color = frameset.get_color_frame()
            if not depth or not color:
                continue
            color_image = np.asanyarray(color.get_data())
            depth_image = np.asanyarray(depth.get_data())

            # labels=False: kutular çizilsin, ama varsayılan yazılar olmasın
            results = model.predict(
                source=color_image,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False
            )
            plotted = results[0].plot(labels=False)

            boxes = results[0].boxes
            names = results[0].names
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
                    d_cm = median_depth_cm(depth_image, x1, y1, x2, y2, depth_scale)
                    label = names.get(int(c), str(int(c)))
                    txt = f"{label} {cf:.2f}"
                    if d_cm is not None:
                        txt += f"  {d_cm:.0f}cm"

                    # yazıyı kutunun üstüne, arka planla koy
                    x1i, y1i = int(x1), int(y1)
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    pad = 4
                    y_text = max(0, y1i - 6)
                    y_bg1  = max(0, y_text - th - pad*2)
                    y_bg2  = y_text
                    x_bg1  = max(0, x1i)
                    x_bg2  = min(plotted.shape[1]-1, x1i + tw + pad*2)

                    cv2.rectangle(plotted, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
                    cv2.putText(plotted, txt, (x1i + pad, y_text - pad),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            frames += 1
            dt = time.time() - t0
            if dt > 0:
                fps = frames / dt
                cv2.putText(plotted, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = plotted.shape[:2]
                    writer = cv2.VideoWriter(args.record, fourcc, 30, (w, h))
                writer.write(plotted)

            cv2.imshow("YOLO + RealSense", plotted)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        if writer is not None: writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense bitti.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--val_conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--skip_val", action="store_true")
    ap.add_argument("--skip_pred", action="store_true")
    ap.add_argument("--webcam", type=str, default="")
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--record", type=str, default="")
    args = ap.parse_args()

    root = find_project_root()
    runs_detect = root / "runs" / "detect"
    data_yaml = (root / "data.yaml")
    if not data_yaml.exists():
        alt = root / "data" / "data.yaml"
        if alt.exists(): data_yaml = alt

    if args.weights:
        weights_path = Path(args.weights)
    else:
        # hem root\runs hem root\data\runs bak
        weights_path = find_best_weights(runs_detect)
        if weights_path is None:
            alt_runs = root / "data" / "runs" / "detect"
            weights_path = find_best_weights(alt_runs) if alt_runs.exists() else None
        if weights_path is None:
            y11 = root / "yolo11n.pt"
            if y11.exists(): weights_path = y11
    if weights_path is None or not weights_path.exists():
        print("HATA: weights bulunamadi. --weights ile yol ver.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] weights: {weights_path}")
    print(f"[INFO] data.yaml: {data_yaml if data_yaml.exists() else 'BULUNAMADI'}")

    model = YOLO(str(weights_path))

    if args.realsense:
        run_realsense(model, args)
        return

    if args.webcam:
        run_webcam(model, args)
        return

    if not args.skip_val and data_yaml.exists():
        run_val(model, data_yaml, args)
    if not args.skip_pred:
        run_single_image_infer(model, root, args)
    print("\n[OK] bitti.")

if __name__ == "__main__":
    main()

#python live_test.py --realsense --weights "..\data\runs\detect\train\weights\best.pt" --device 0 --record rs_out.mp4
#python live_test.py --realsense --weights "..\data\runs\detect\train\weights\best.pt" --device 0