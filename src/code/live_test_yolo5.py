import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ========================
# Yolov5 klasörünü Python path'e ekle
# ========================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # proje kökü (model_two)
YOLOV5_DIR = ROOT / "yolov5"

if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

# detect.py ve val.py içinden run fonksiyonlarını al
from detect import run as detect_run
from val import run as val_run

# YOLOv5 backend utils (paketsiz import)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox  # <-- ÖNEMLİ: oran korumalı ön işleme

# ========================
# Fonksiyonlar
# ========================

def run_val(weights, data_yaml, args):
    """Validation: mAP değerlerini hesaplar"""
    print(f"\n=== VAL split={args.split} ===")
    val_run(
        weights=weights,
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch_size=args.batch,
        conf_thres=args.val_conf,
        iou_thres=args.iou,
        device=args.device,
        task=args.split,
        plots=True,
        save_json=True,
        verbose=True,
    )


def run_single_image(weights, img_path, args):
    """Tek bir resim üzerinde tahmin yapar"""
    print(f"\n=== SINGLE IMAGE === {img_path}")
    detect_run(
        weights=weights,
        source=str(img_path),
        imgsz=args.imgsz,
        conf_thres=args.conf,
        device=args.device,
        save_txt=False,
        save_conf=True,
        project="runs/detect",
        name="predict_one",
        exist_ok=True,
    )


def run_webcam(weights, args):
    """Webcam stream üzerinde canlı inference"""
    print("\n=== WEBCAM === (Q/ESC çıkış)")
    detect_run(
        weights=weights,
        source=args.webcam if args.webcam else "0",
        imgsz=args.imgsz,
        conf_thres=args.conf,
        device=args.device,
        project="runs/detect",
        name="webcam",
        exist_ok=True,
        view_img=True,
    )


def run_realsense(weights, args):
    """RealSense kamera ile inference (renk + derinlik) — LETTERBOX DÜZELTİLMİŞ"""
    print("\n=== REALSENSE === (Q/ESC çıkış)")
    try:
        import pyrealsense2 as rs
    except Exception as e:
        print("pyrealsense2 yok:", e)
        print("Kurulum: pip install pyrealsense2")
        return

    # RealSense pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # YOLOv5 model backend
    device = select_device(args.device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = args.imgsz

    try:
        while True:
            frameset = pipe.wait_for_frames()
            frameset = align.process(frameset)
            depth = frameset.get_depth_frame()
            color = frameset.get_color_frame()
            if not depth or not color:
                continue

            im0 = np.asanyarray(color.get_data())      # orijinal BGR (H,W,3)
            depth_img = np.asanyarray(depth.get_data())

            # --------- DÜZELTME: LETTERBOX ÖN İŞLEME ---------
            # Oran koru + pad ekle; YOLOv5 girişine uygun hazırla
            # (auto=True -> stride'a uygun pad)
            lb_img = letterbox(im0, imgsz, stride=stride, auto=True)[0]  # H,W,3 (BGR)
            img = lb_img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device).float() / 255.0
            if img.ndim == 3:
                img = img[None, ...]
            # -----------------------------------------------

            # İnferens + NMS
            pred = model(img)
            pred = non_max_suppression(pred, args.conf, args.iou)

            # Kutu geri ölçekleme (letterbox dikkate alınır)
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = f"{names[int(cls)]} {conf:.2f}"
                        # derinlik (kutu merkezinde cm)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if 0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]:
                            dist = depth_img[cy, cx] * depth_scale * 100
                            if dist > 0:
                                label += f" {dist:.0f}cm"
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            im0, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )

            cv2.imshow("YOLOv5 + RealSense", im0)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense bitti.")


# ========================
# Main
# ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--val_conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--skip_val", action="store_true")
    ap.add_argument("--skip_pred", action="store_true")
    ap.add_argument("--webcam", type=str, default="")
    ap.add_argument("--realsense", action="store_true")
    args = ap.parse_args()

    data_yaml = ROOT / "data.yaml"

    if args.realsense:
        run_realsense(args.weights, args)
    elif args.webcam:
        run_webcam(args.weights, args)
    else:
        if not args.skip_val and data_yaml.exists():
            run_val(args.weights, data_yaml, args)
        if not args.skip_pred:
            # örnek tek resim
            example_dir = ROOT / "data" / "images" / "val"
            if example_dir.exists():
                first_img = next(example_dir.glob("*.*"), None)
                if first_img:
                    run_single_image(args.weights, first_img, args)


if __name__ == "__main__":
    main()
