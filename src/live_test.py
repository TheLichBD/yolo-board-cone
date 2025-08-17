# live_test.py
from pathlib import Path
from ultralytics import YOLO
import argparse, shutil, sys, time
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
            print("[WARN] İstenen split bulunamadı, 'val' ile tekrar deniyorum.")
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
        print("\n[WARN] Örnek resim bulunamadı, inference atlandı.")
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
    print("[INFO] Çıktı: runs/detect/predict*/ altında.")

def run_webcam(model, args):
    print("\n=== WEBCAM ===  (Çıkış: Q veya ESC)")
    # Ultralytics'in stream çıktısını kullanarak kutuları çiziyoruz
    stream = model.predict(
        source=args.webcam,      # 0,1,... ya da rtsp/http url
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
        if key in (27, ord('q')):  # ESC veya q
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam bitti.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--device", type=str, default=None)   # 0,1 veya 'cpu'
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--val_conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--skip_val", action="store_true")
    ap.add_argument("--skip_pred", action="store_true")
    ap.add_argument("--webcam", type=str, default="")     # "0", "1", "rtsp://...", "http://..."
    ap.add_argument("--record", type=str, default="")     # "out.mp4" kaydetmek için
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
        weights_path = find_best_weights(runs_detect) or (root / "yolo11n.pt")
    if not weights_path.exists():
        print("HATA: weights bulunamadı. --weights ile yol ver.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] weights: {weights_path}")
    print(f"[INFO] data.yaml: {data_yaml if data_yaml.exists() else 'BULUNAMADI'}")

    model = YOLO(str(weights_path))

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
