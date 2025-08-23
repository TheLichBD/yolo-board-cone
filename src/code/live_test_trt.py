# live_test_trt.py
# Ultralytics'siz (TensorRT) canlı test aracı
# - Engine: best_fp16.engine
# - Kaynaklar: tek resim, webcam, Intel RealSense
# - "val" modu: mAP hesaplamaz; val görüntüleri üzerinde tahmin/overlay yapar

from __future__ import annotations
from pathlib import Path
import argparse, sys, time, shutil
import cv2
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except Exception as _e:
    TRT_AVAILABLE = False

# ----- Kullanıcı ayarları / Varsayılanlar -----
DEFAULT_ENGINE = "best_fp16.engine"
DEFAULT_IMG_SIZE = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_NAMES = ["board", "cone"]  # data.yaml okunamazsa kullanılır

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------- Yardımcılar -----------------
def find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent  # .../model_two


def find_best_engine(start_dir: Path) -> Path | None:
    cands = list(start_dir.rglob("*.engine"))
    if not cands: 
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
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
    if example is None: 
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / example.name
    if not out.exists():
        try:
            shutil.copy2(example, out)
        except Exception:
            return example
    return out


def load_names_from_yaml(data_yaml: Path) -> list[str]:
    try:
        import yaml  # type: ignore
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):
            # {0:"cls0",1:"cls1",...} olabilir
            names = [names[k] for k in sorted(names)]
        if isinstance(names, list) and all(isinstance(n, str) for n in names):
            return names
    except Exception:
        pass
    return DEFAULT_NAMES[:]


# --------------- Ön/son işleme -----------------
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]  # h, w
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess_bgr(img_bgr: np.ndarray, img_size: int):
    img, r, (dw, dh) = letterbox(img_bgr, img_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))[None, ...]  # 1x3xH xW
    return img_chw, r, (dw, dh)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_th=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_th]
    return keep


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-6)


def postprocess(pred: np.ndarray, r, dwdh, orig_shape, conf_th=0.25, iou_th=0.45):
    """
    Beklenen YOLOv5 ONNX/TRT çıkışı: (1, N, 5+nc)  -> [x,y,w,h, obj, class_scores...]
    """
    if pred.ndim == 3:
        pred = pred[0]
    boxes_xywh = pred[:, :4]
    obj = pred[:, 4:5]
    cls = pred[:, 5:]
    cls_id = np.argmax(cls, axis=1)
    cls_conf = cls.max(axis=1, keepdims=True)
    scores = (obj * cls_conf).squeeze()

    m = scores >= conf_th
    boxes_xywh, cls_id, scores = boxes_xywh[m], cls_id[m], scores[m]
    if boxes_xywh.shape[0] == 0:
        return []

    # xywh -> xyxy
    boxes = boxes_xywh.copy()
    boxes[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2)
    boxes[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2)
    boxes[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2)
    boxes[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2)

    # letterbox'u geri al
    dw, dh = dwdh
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r

    # clip
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

    # NMS
    keep = nms(boxes, scores, iou_th)
    return [(boxes[i], int(cls_id[i]), float(scores[i])) for i in keep]


def draw_overlays(frame: np.ndarray, detections, names: list[str]):
    for (x1, y1, x2, y2), cid, sc in detections:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        label = f"{names[cid] if cid < len(names) else cid}:{sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, p1[1] - 6)
        cv2.rectangle(frame, (p1[0], max(0, y_text - th - 6)), (p1[0] + tw + 8, y_text), (0, 0, 0), -1)
        cv2.putText(frame, label, (p1[0] + 4, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# --------------- TensorRT çatı sınıfı ---------------
class TRTInfer:
    def __init__(self, engine_path: Path):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT/PyCUDA mevcut değil. Jetson veya TRT kurulu bir ortamda çalıştırın.")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.context.get_binding_shape(i)
            size = int(np.prod(shape))
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            if self.engine.binding_is_input(i):
                self.inputs.append({"dtype": dtype, "shape": shape, "device": device_mem})
            else:
                self.outputs.append({"dtype": dtype, "shape": shape, "device": device_mem})
            self.bindings.append(int(device_mem))

    def infer(self, host_input: np.ndarray) -> np.ndarray:
        inp = self.inputs[0]
        out = self.outputs[0]
        cuda.memcpy_htod_async(inp["device"], host_input, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        host_out = np.empty(out["shape"], dtype=out["dtype"])
        cuda.memcpy_dtoh_async(host_out, out["device"], self.stream)
        self.stream.synchronize()
        return host_out


# --------------- RealSense yardımcıları ---------------
def median_depth_cm(depth_frame: np.ndarray, x1, y1, x2, y2, scale):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    k = 10
    x1s, y1s, x2s, y2s = max(cx - k, 0), max(cy - k, 0), cx + k, cy + k
    depth_roi = depth_frame[y1s:y2s, x1s:x2s].astype(np.float32)
    if depth_roi.size == 0:
        return None
    vals = depth_roi[depth_roi > 0]
    if vals.size == 0:
        return None
    return float(np.median(vals) * scale * 100.0)


# --------------- Koşu modları ---------------
def run_single_image_infer(engine: TRTInfer, names, root: Path, args):
    data_root = root / "data"
    example = pick_one_image(data_root)
    example = ensure_example_image(root / "src" / "_sample_trt", example)
    if example is None:
        print("\n[WARN] Örnek resim bulunamadı, inference atlandı.")
        return
    print(f"\n=== INFERENCE (single image) ===\n[INFO] image: {example}")
    frame = cv2.imread(str(example))
    if frame is None:
        print("Görsel açılamadı.")
        return
    inp, r, dwdh = preprocess_bgr(frame, args.imgsz)
    pred = engine.infer(inp.astype(np.float32).ravel())
    dets = postprocess(pred, r, dwdh, frame.shape, args.conf, args.iou)
    draw_overlays(frame, dets, names)
    out_dir = root / "runs" / "detect_trt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{example.stem}_trt.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"[OK] Çıktı: {out_path}")


def run_webcam(engine: TRTInfer, names, args):
    print("\n=== WEBCAM (Q/ESC ile çıkış) ===")
    cap = cv2.VideoCapture(0 if args.webcam == "" else int(args.webcam))
    if not cap.isOpened():
        print("Webcam açılamadı.")
        return
    writer = None
    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        inp, r, dwdh = preprocess_bgr(frame, args.imgsz)
        pred = engine.infer(inp.astype(np.float32).ravel())
        dets = postprocess(pred, r, dwdh, frame.shape, args.conf, args.iou)
        draw_overlays(frame, dets, names)

        frames += 1
        dt = time.time() - t0
        if dt > 0:
            fps = frames / (dt + 1e-9)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if args.record:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, fourcc, 30, (w, h))
            writer.write(frame)

        cv2.imshow("TRT Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam bitti.")


def run_realsense(engine: TRTInfer, names, args):
    print("\n=== REALSENSE (Q/ESC ile çıkış) ===")
    try:
        import pyrealsense2 as rs
    except Exception as e:
        print("pyrealsense2 import hatası:", e, file=sys.stderr)
        print("pip install pyrealsense2", file=sys.stderr)
        return

    pipeline = rs.pipeline()
    config = rs.config()
    # Renk + Derinlik
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

            inp, r, dwdh = preprocess_bgr(color_image, args.imgsz)
            pred = engine.infer(inp.astype(np.float32).ravel())
            dets = postprocess(pred, r, dwdh, color_image.shape, args.conf, args.iou)

            plotted = color_image.copy()
            draw_overlays(plotted, dets, names)

            # Derinlik etiketi (kutu merkezinde medyan)
            for (x1, y1, x2, y2), cid, sc in dets:
                d_cm = median_depth_cm(depth_image, x1, y1, x2, y2, depth_scale)
                if d_cm is not None:
                    txt = f"{names[cid] if cid < len(names) else cid}: {sc:.2f}  {d_cm:.0f}cm"
                    x1i, y1i = int(x1), int(y1)
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    pad = 4
                    y_text = max(0, y1i - 6)
                    y_bg1 = max(0, y_text - th - pad * 2)
                    y_bg2 = y_text
                    x_bg1 = max(0, x1i)
                    x_bg2 = min(plotted.shape[1] - 1, x1i + tw + pad * 2)
                    cv2.rectangle(plotted, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
                    cv2.putText(plotted, txt, (x1i + pad, y_text - pad),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            frames += 1
            dt = time.time() - t0
            if dt > 0:
                fps = frames / (dt + 1e-9)
                cv2.putText(plotted, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = plotted.shape[:2]
                    writer = cv2.VideoWriter(args.record, fourcc, 30, (w, h))
                writer.write(plotted)

            cv2.imshow("TRT + RealSense", plotted)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        if writer is not None:
            writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense bitti.")


def run_val(engine: TRTInfer, names, data_yaml: Path, img_size: int, conf: float, iou: float, split="val"):
    """
    Basit "val" modu: mAP hesaplamaz.
    data/images/{split} altındaki görüntüleri okur, çıktı overlay'lerini runs/detect_trt/{split}/ altına yazar.
    """
    root = data_yaml.parent if data_yaml.exists() else find_project_root() / "data"
    img_dir = root / "images" / split
    if not img_dir.exists():
        print(f"[WARN] {img_dir} yok. Val atlandı.")
        return
    out_dir = find_project_root() / "runs" / "detect_trt" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    ims = [p for p in img_dir.rglob("*.*") if p.suffix.lower() in IMG_EXTS]
    if not ims:
        print(f"[WARN] {img_dir} içinde görüntü yok.")
        return
    t0 = time.time()
    for i, p in enumerate(ims, 1):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        inp, r, dwdh = preprocess_bgr(frame, img_size)
        pred = engine.infer(inp.astype(np.float32).ravel())
        dets = postprocess(pred, r, dwdh, frame.shape, conf, iou)
        draw_overlays(frame, dets, names)
        cv2.imwrite(str(out_dir / f"{p.stem}.jpg"), frame)
        if i % 50 == 0:
            print(f"[VAL] {i}/{len(ims)} işlendi...")
    dt = time.time() - t0
    print(f"[OK] {len(ims)} görsel işlendi. Süre: {dt:.1f}s (ort: {dt/max(1,len(ims)):.3f}s/img). Çıkış: {out_dir}")


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=str, default=DEFAULT_ENGINE)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--webcam", type=str, default="")      # "" veya "0" gibi
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--record", type=str, default="")      # mp4 kaydı
    ap.add_argument("--val", action="store_true")          # data/images/val üstünde koş
    ap.add_argument("--split", type=str, default="val")    # val/test/train
    args = ap.parse_args()

    root = find_project_root()
    data_yaml = (root / "data.yaml")
    if not data_yaml.exists():
        alt = root / "data" / "data.yaml"
        if alt.exists():
            data_yaml = alt

    # sınıf isimleri
    names = load_names_from_yaml(data_yaml)

    # engine yolu
    engine_path = Path(args.engine)
    if not engine_path.exists():
        # proje içinde en yeni .engine'i bulmayı dene
        cand = find_best_engine(root)
        if cand is not None:
            engine_path = cand
    if not engine_path.exists():
        print(f"HATA: engine bulunamadı: {args.engine}. \n"
              f"Örnek üretim: /usr/src/tensorrt/bin/trtexec --onnx=best.onnx "
              f"--saveEngine=best_fp16.engine --fp16 --workspace=2048 --shapes=images:1x3x{args.imgsz}x{args.imgsz}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] engine: {engine_path}")
    print(f"[INFO] data.yaml: {data_yaml if data_yaml.exists() else 'BULUNAMADI'}")
    print(f"[INFO] classes: {names}")

    engine = TRTInfer(engine_path)

    if args.realsense:
        run_realsense(engine, names, args)
        return

    if args.webcam != "":
        run_webcam(engine, names, args)
        return

    if args.val:
        run_val(engine, names, data_yaml, args.imgsz, args.conf, args.iou, args.split)

    # tek resim inference (örnek)
    run_single_image_infer(engine, names, root, args)
    print("\n[OK] bitti.")


if __name__ == "__main__":
    main()
