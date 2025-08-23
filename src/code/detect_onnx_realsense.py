import argparse, cv2, numpy as np, time, os
from pathlib import Path
from datetime import datetime

# --- Ayarlar ---
ONNX_PATH   = r"/yolov5/runs/train/model_two_v5n/weights/best.onnx"
IMG_SIZE    = 640
CONF_THRES  = 0.25
NMS_THRES   = 0.45
DRAW_DEPTH_CM = True  # derinlikten cm etiketi yazsın mı?
DEFAULT_FPS = 30
DEFAULT_FOURCC = "mp4v"  # 'mp4v' ya da 'XVID'

def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape/h, new_shape/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2; dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def preprocess_bgr(img_bgr):
    img, r, (dw, dh) = letterbox(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    blob = np.transpose(img_rgb, (2,0,1))[None, ...]  # 1x3xHxW
    return blob, r, (dw, dh)

def postprocess(out, r, dwdh, orig_shape):
    # YOLOv5 ONNX: (1, N, 5+nc) -> [x,y,w,h, obj, c1, c2, ...]
    if out.ndim == 3: out = out[0]
    boxes_xywh = out[:, :4]
    obj = out[:, 4:5]
    cls = out[:, 5:]
    cls_id = np.argmax(cls, axis=1)
    cls_conf = cls.max(axis=1, keepdims=True)
    scores = (obj * cls_conf).squeeze()

    m = scores >= CONF_THRES
    boxes_xywh, cls_id, scores = boxes_xywh[m], cls_id[m], scores[m]
    if boxes_xywh.shape[0] == 0: return []

    # xywh -> xyxy (letterbox'u geri al)
    boxes = boxes_xywh.copy()
    boxes[:, 0] = boxes_xywh[:,0] - boxes_xywh[:,2]/2.0
    boxes[:, 1] = boxes_xywh[:,1] - boxes_xywh[:,3]/2.0
    boxes[:, 2] = boxes_xywh[:,0] + boxes_xywh[:,2]/2.0
    boxes[:, 3] = boxes_xywh[:,1] + boxes_xywh[:,3]/2.0

    dw, dh = dwdh
    boxes[:, [0,2]] -= dw
    boxes[:, [1,3]] -= dh
    boxes /= r

    h, w = orig_shape[:2]
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, w-1)
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, h-1)

    # OpenCV NMSBoxes x,y,w,h ister
    bbs_xywh = [[float(x1), float(y1), float(x2-x1), float(y2-y1)] for x1,y1,x2,y2 in boxes]
    idxs = cv2.dnn.NMSBoxes(bbs_xywh, list(map(float, scores)), CONF_THRES, NMS_THRES)
    idxs = idxs.flatten() if len(idxs) else []
    return [(boxes[i], int(cls_id[i]), float(scores[i])) for i in idxs]

def median_depth_cm(depth_frame, x1, y1, x2, y2, scale):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    if x2 <= x1 or y2 <= y1: return None
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    k = 10  # kutu merkezinde 20x20 piksel pencere
    x1s, y1s, x2s, y2s = max(cx - k, 0), max(cy - k, 0), cx + k, cy + k
    roi = depth_frame[y1s:y2s, x1s:x2s].astype(np.float32)
    if roi.size == 0: return None
    vals = roi[roi > 0]
    if vals.size == 0: return None
    return float(np.median(vals) * scale * 100.0)  # m -> cm

def ensure_runs_dir():
    out_dir = Path("../runs") / "record"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def make_auto_filename():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_runs_dir() / f"rec_{ts}.mp4"


def run_multi_camera(args):
    import pyrealsense2 as rs

    # --- RealSense ---
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # --- Yardımcı kameralar (ID 0 ve 1) ---
    cap2 = cv2.VideoCapture(0)
    cap3 = cv2.VideoCapture(1)

    net = cv2.dnn.readNetFromONNX(args.onnx)

    while True:
        # RealSense
        fs = pipe.wait_for_frames()
        fs = align.process(fs)
        depth = fs.get_depth_frame()
        color = fs.get_color_frame()
        if not depth or not color:
            continue
        frame_rs = np.asanyarray(color.get_data())
        depth_img = np.asanyarray(depth.get_data())

        # Inference RS
        blob, r, dwdh = preprocess_bgr(frame_rs)
        net.setInput(blob)
        out = net.forward()
        dets = postprocess(out, r, dwdh, frame_rs.shape)
        # çizimler (tam nesne seti)
        frame_rs = draw_detections(frame_rs, depth_img, dets, depth_scale)

        # Kamera 2
        ret2, frame2 = cap2.read()
        if ret2:
            blob, r, dwdh = preprocess_bgr(frame2)
            net.setInput(blob)
            out = net.forward()
            dets = postprocess(out, r, dwdh, frame2.shape)
            # sadece "cone" sınıfı
            dets = [d for d in dets if d[1] == CONE_CLASS_ID]
            frame2 = draw_detections(frame2, None, dets, None)

        # Kamera 3
        ret3, frame3 = cap3.read()
        if ret3:
            blob, r, dwdh = preprocess_bgr(frame3)
            net.setInput(blob)
            out = net.forward()
            dets = postprocess(out, r, dwdh, frame3.shape)
            dets = [d for d in dets if d[1] == CONE_CLASS_ID]
            frame3 = draw_detections(frame3, None, dets, None)

        # --- Pencere düzeni ---
        # RS büyük, sağda iki küçük
        h, w = frame_rs.shape[:2]
        frame2 = cv2.resize(frame2, (w // 2, h // 2))
        frame3 = cv2.resize(frame3, (w // 2, h // 2))
        top_right = np.zeros_like(frame2) if frame2 is None else frame2
        bot_right = np.zeros_like(frame3) if frame3 is None else frame3
        right = np.vstack([top_right, bot_right])
        vis = np.hstack([frame_rs, right])

        cv2.imshow("Multi-Camera View", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default=ONNX_PATH, help="ONNX model yolu")
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--conf", type=float, default=CONF_THRES)
    ap.add_argument("--nms", type=float, default=NMS_THRES)
    ap.add_argument("--record", type=str, default="", help="Video kaydı dosya yolu (örn: out.mp4)")
    ap.add_argument("--fourcc", type=str, default=DEFAULT_FOURCC, help="Codec (mp4v/XVID vb.)")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Kayıt FPS")
    args = ap.parse_args()

    # RealSense başlat
    try:
        import pyrealsense2 as rs
    except Exception as e:
        print("pyrealsense2 import hatası:", e)
        print("Çözüm: pip install pyrealsense2")
        return

    pipe = rs.pipeline()
    cfg = rs.config()
    # Renk + derinlik akışları (30 FPS, 1280x720)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # ONNX modeli yükle
    net = cv2.dnn.readNetFromONNX(args.onnx)

    # Kayıt ayarları
    writer = None
    recording = False
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)

    # Eğer --record verilmişse otomatik başlat
    if args.record:
        out_path = Path(args.record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # writer, ilk frame geldikten sonra (boyutu bilince) oluşturulacak
        recording = True
        print(f"[REC] Kayıt açık: {out_path}")

    t0 = time.time(); frames = 0
    try:
        while True:
            frameset = pipe.wait_for_frames()
            frameset = align.process(frameset)
            depth = frameset.get_depth_frame()
            color = frameset.get_color_frame()
            if not depth or not color:
                continue
            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())

            blob, r, dwdh = preprocess_bgr(color_img)
            net.setInput(blob)
            out = net.forward()
            dets = postprocess(out, r, dwdh, color_img.shape)

            # çizimler
            vis = color_img.copy()
            for (x1,y1,x2,y2), cid, sc in dets:
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(vis, p1, p2, (0,255,0), 2)
                label = f"{cid}:{sc:.2f}"
                if DRAW_DEPTH_CM:
                    d_cm = median_depth_cm(depth_img, x1, y1, x2, y2, depth_scale)
                    if d_cm is not None:
                        label += f"  {d_cm:.0f}cm"
                # arka planlı yazı
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_text = max(0, p1[1]-6)
                cv2.rectangle(vis, (p1[0], max(0, y_text-th-6)), (p1[0]+tw+8, y_text), (0,0,0), -1)
                cv2.putText(vis, label, (p1[0]+4, y_text-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # FPS
            frames += 1
            if frames % 30 == 0:
                fps = frames / (time.time() - t0 + 1e-9)
                cv2.putText(vis, f"FPS: {fps:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # --- KAYIT: İlk frame’de writer aç (boyut belli olunca) ---
            if recording and writer is None:
                if args.record:
                    out_file = Path(args.record)
                else:
                    out_file = make_auto_filename()
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out_file), fourcc, args.fps, (w, h))
                if not writer.isOpened():
                    print("[REC] VideoWriter açılamadı! Yol/codec kontrol et.")
                    recording = False
                else:
                    print(f"[REC] Yazıyor -> {out_file}")

            # Kayıt yaz
            if recording and writer is not None:
                writer.write(vis)
                # köşeye kırmızı nokta
                cv2.circle(vis, (20, 20), 6, (0,0,255), -1)

            # Görüntüle
            cv2.imshow("ONNX + RealSense", vis)
            key = cv2.waitKey(1) & 0xFF

            # 'r' ile kayıt toggle
            if key in (ord('r'), ord('R')):
                recording = not recording
                if recording:
                    print("[REC] Kayıt ON (R ile kapatabilirsin)")
                    writer = None  # yeni isim için sonraki frame’de aç
                else:
                    print("[REC] Kayıt OFF")
                    if writer is not None:
                        writer.release()
                        writer = None

            # Çıkış
            if key in (27, ord('q'), ord('Q')):
                break
    finally:
        if writer is not None:
            writer.release()
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
