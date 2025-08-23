import argparse, cv2, numpy as np, time, os
from pathlib import Path
from datetime import datetime

# ---------------- helpers ----------------
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

def preprocess_bgr(img_bgr, img_size):
    img, r, (dw, dh) = letterbox(img_bgr, img_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    blob = np.transpose(img_rgb, (2,0,1))[None, ...]
    return blob, r, (dw, dh)

def postprocess(out, r, dwdh, orig_shape, conf_th=0.25, nms_th=0.45):
    if out.ndim == 3: out = out[0]
    boxes_xywh = out[:, :4]; obj = out[:, 4:5]; cls = out[:, 5:]
    cls_id = np.argmax(cls, axis=1); cls_conf = cls.max(axis=1, keepdims=True)
    scores = (obj * cls_conf).squeeze()
    m = scores >= conf_th
    boxes_xywh, cls_id, scores = boxes_xywh[m], cls_id[m], scores[m]
    if boxes_xywh.shape[0] == 0: return []
    boxes = boxes_xywh.copy()
    boxes[:,0] = boxes_xywh[:,0] - boxes_xywh[:,2]/2.0
    boxes[:,1] = boxes_xywh[:,1] - boxes_xywh[:,3]/2.0
    boxes[:,2] = boxes_xywh[:,0] + boxes_xywh[:,2]/2.0
    boxes[:,3] = boxes_xywh[:,1] + boxes_xywh[:,3]/2.0
    dw, dh = dwdh
    boxes[:,[0,2]] -= dw; boxes[:,[1,3]] -= dh; boxes /= r
    h, w = orig_shape[:2]
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, w-1)
    boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, h-1)
    bbs_xywh = [[float(x1),float(y1),float(x2-x1),float(y2-y1)] for x1,y1,x2,y2 in boxes]
    idxs = cv2.dnn.NMSBoxes(bbs_xywh, list(map(float, scores)), conf_th, nms_th)
    idxs = idxs.flatten() if len(idxs) else []
    return [(boxes[i], int(cls_id[i]), float(scores[i])) for i in idxs]

def median_depth_cm(depth_frame, x1, y1, x2, y2, scale):
    x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
    if x2<=x1 or y2<=y1: return None
    cx, cy = (x1+x2)//2, (y1+y2)//2
    k=10
    x1s,y1s,x2s,y2s = max(cx-k,0), max(cy-k,0), cx+k, cy+k
    roi = depth_frame[y1s:y2s, x1s:x2s].astype(np.float32)
    if roi.size==0: return None
    vals = roi[roi>0]
    if vals.size==0: return None
    return float(np.median(vals)*scale*100.0)

def overlay_text(img, text, org=(10,30), scale=0.8, color=(255,255,255)):
    (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x,y = org
    cv2.rectangle(img, (x-6, y-th-6), (x+tw+6, y+6), (0,0,0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def parse_indices(csv):
    if not csv: return []
    out=[]
    for t in csv.split(","):
        t=t.strip()
        if not t: continue
        try: out.append(int(t))
        except: pass
    return out

def try_open_cam(index, backend=None, w=None, h=None, fps=None, test_frames=3):
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    if w: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    if h: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fps: cap.set(cv2.CAP_PROP_FPS, fps)
    ok=False
    for _ in range(test_frames):
        ok, _ = cap.read()
        if ok: break
        time.sleep(0.02)
    if not ok:
        cap.release(); return None
    return cap

def auto_pick_two_usb(candidates, blacklist=None, w=None, h=None, fps=None):
    blacklist=set(blacklist or [])
    picked=[]
    backend = cv2.CAP_DSHOW if os.name=="nt" else None
    for idx in candidates:
        if idx in blacklist:
            print(f"[INFO] skip blacklisted index {idx}")
            continue
        cap = try_open_cam(idx, backend, w, h, fps)
        if cap is not None:
            picked.append((idx, cap))
            print(f"[OK] opened USB cam index {idx}")
            if len(picked)==2: break
        else:
            print(f"[X]  failed USB cam index {idx}")
    return picked

def ensure_writer(path_str, fps, frame_shape, fourcc_str="mp4v"):
    h,w = frame_shape[:2]
    p=Path(path_str); p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    wr = cv2.VideoWriter(str(p), fourcc, fps, (w,h))
    return wr

def auto_record_name(prefix="multicam"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs")/"record"; out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir/f"{prefix}_{ts}.mp4")

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser(description="RealSense ana + 2 USB yardımcı (ONNX)")
    ap.add_argument("--onnx", type=str, default="models/best.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms",  type=float, default=0.45)
    ap.add_argument("--rs_w", type=int, default=1280)
    ap.add_argument("--rs_h", type=int, default=720)
    ap.add_argument("--rs_fps", type=int, default=30)
    ap.add_argument("--aux_w", type=int, default=1280)
    ap.add_argument("--aux_h", type=int, default=720)
    ap.add_argument("--aux_fps", type=int, default=30)
    ap.add_argument("--auto_usb", action="store_true")
    ap.add_argument("--usb_candidates", type=str, default="0,1,2,3,4")
    ap.add_argument("--blacklist", type=str, default="0", help="laptop iç kamera genelde 0")
    ap.add_argument("--show_depth_cm", action="store_true")
    ap.add_argument("--infer_stride", type=int, default=1)
    ap.add_argument("--record", type=str, default="")
    ap.add_argument("--fourcc", type=str, default="mp4v")
    ap.add_argument("--rec_fps", type=int, default=25)
    # pencere/boyutlandırma
    ap.add_argument("--rs_scale", type=float, default=1.0)
    ap.add_argument("--aux_scale", type=float, default=0.5)
    ap.add_argument("--window_scale", type=float, default=1.0)
    ap.add_argument("--window_size", type=str, default="", help="örn 1600x900")
    # tanılama
    ap.add_argument("--list_cams", action="store_true", help="0-9 arası kameraları listele ve çık")
    args = ap.parse_args()

    # Kamera listesini isteyip çıkmak için
    if args.list_cams:
        backend = cv2.CAP_DSHOW if os.name=="nt" else None
        print("Index | Open | First read")
        for i in range(10):
            cap = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
            opened = cap.isOpened()
            ok = False
            if opened: ok,_ = cap.read()
            print(f"{i:5d} | {str(opened):5s} | {str(ok):10s}")
            cap.release()
        return

    # ONNX
    if not Path(args.onnx).exists():
        print(f"HATA: ONNX bulunamadı: {Path(args.onnx).resolve()}"); return
    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # RealSense
    try:
        import pyrealsense2 as rs
    except Exception as e:
        print("pyrealsense2 import hatası:", e); print("pip install pyrealsense2"); return
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.rs_w, args.rs_h, rs.format.bgr8, args.rs_fps)
    cfg.enable_stream(rs.stream.depth, args.rs_w, args.rs_h, rs.format.z16, args.rs_fps)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # USB cams
    cap2 = cap3 = None
    if args.auto_usb:
        cand = parse_indices(args.usb_candidates)
        bl   = parse_indices(args.blacklist)
        picked = auto_pick_two_usb(cand, bl, args.aux_w, args.aux_h, args.aux_fps)
        if len(picked)>=1: _, cap2 = picked[0]
        if len(picked)>=2: _, cap3 = picked[1]
        if cap2 is None or cap3 is None:
            print("[WARN] Yeterli USB kamera bulunamadı. --usb_candidates/--blacklist ayarlarını kontrol et.")
    # pencere ayarı
    win_name = "Multi-Camera (RS + 2xUSB)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if args.window_size:
        try:
            W,H = map(int, args.window_size.lower().split("x"))
            cv2.resizeWindow(win_name, W, H)
        except Exception:
            pass

    writer=None; want_record = bool(args.record)
    out_path = args.record if args.record else auto_record_name()

    stride = max(1, args.infer_stride)
    last_rs=[]; last2=[]; last3=[]
    frame_idx=0; t0=time.time(); frames=0

    try:
        while True:
            # RS
            frameset = pipe.wait_for_frames()
            frameset = align.process(frameset)
            depth = frameset.get_depth_frame()
            color = frameset.get_color_frame()
            if not depth or not color: continue
            rs_frame = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())

            # USB
            ret2, f2 = (cap2.read() if cap2 else (False,None))
            ret3, f3 = (cap3.read() if cap3 else (False,None))

            # infer stride
            if frame_idx % stride == 0:
                blob, r, dwdh = preprocess_bgr(rs_frame, args.imgsz)
                net.setInput(blob); out = net.forward()
                last_rs = postprocess(out, r, dwdh, rs_frame.shape, args.conf, args.nms)
                if ret2 and f2 is not None:
                    blob,r2,d2 = preprocess_bgr(f2, args.imgsz)
                    net.setInput(blob); out = net.forward()
                    last2 = postprocess(out, r2, d2, f2.shape, args.conf, args.nms)
                if ret3 and f3 is not None:
                    blob,r3,d3 = preprocess_bgr(f3, args.imgsz)
                    net.setInput(blob); out = net.forward()
                    last3 = postprocess(out, r3, d3, f3.shape, args.conf, args.nms)

            # draw
            vis_rs = rs_frame.copy()
            for (x1,y1,x2,y2), cid, sc in last_rs:
                p1,p2=(int(x1),int(y1)),(int(x2),int(y2))
                cv2.rectangle(vis_rs,p1,p2,(0,255,0),2)
                lbl=f"{cid}:{sc:.2f}"
                if args.show_depth_cm:
                    d_cm = median_depth_cm(depth_img, x1,y1,x2,y2, depth_scale)
                    if d_cm is not None: lbl += f" {d_cm:.0f}cm"
                overlay_text(vis_rs, lbl, (p1[0], max(0,p1[1]-8)), 0.6, (0,255,255))

            def ensure_frame(img, text):
                if img is None:
                    img = np.zeros_like(vis_rs)
                    overlay_text(img, text, (20,40), 0.8, (0,0,255))
                return img

            vis2 = f2.copy() if (ret2 and f2 is not None) else None
            if vis2 is not None:
                for (x1,y1,x2,y2), cid, sc in last2:
                    cv2.rectangle(vis2, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
                    overlay_text(vis2, f"{cid}:{sc:.2f}", (int(x1), max(0,int(y1)-8)), 0.6, (255,255,255))
            vis3 = f3.copy() if (ret3 and f3 is not None) else None
            if vis3 is not None:
                for (x1,y1,x2,y2), cid, sc in last3:
                    cv2.rectangle(vis3, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
                    overlay_text(vis3, f"{cid}:{sc:.2f}", (int(x1), max(0,int(y1)-8)), 0.6, (255,255,255))

            # stack: RS sol büyük, sağ üst/alt USB
            rs_scale = args.rs_scale; aux_scale=args.aux_scale; win_scale=args.window_scale
            rs_vis = vis_rs if rs_scale==1.0 else cv2.resize(vis_rs, (0,0), fx=rs_scale, fy=rs_scale)
            h,w = rs_vis.shape[:2]; sw,sh = int(w*aux_scale), int(h*aux_scale)

            def sz(x, label):
                if x is None:
                    blk = np.zeros((sh, sw, 3), np.uint8)
                    overlay_text(blk, label, (20,40), 0.8, (0,0,255))
                    return blk
                return cv2.resize(x, (sw, sh))

            right = np.vstack([sz(vis2,"No signal (Cam2)"), sz(vis3,"No signal (Cam3)")])
            canvas = np.hstack([rs_vis, right])
            if win_scale != 1.0:
                canvas = cv2.resize(canvas, (0,0), fx=win_scale, fy=win_scale)

            # FPS
            frames += 1
            if frames % 30 == 0:
                fps = frames / (time.time()-t0+1e-9)
                overlay_text(canvas, f"FPS: {fps:.1f}", (10,30), 0.9, (255,255,255))

            # record
            if want_record and writer is None:
                writer = ensure_writer(args.record if args.record else auto_record_name(),
                                       args.rec_fps, canvas.shape, args.fourcc)
                if not writer.isOpened():
                    print("[REC] Writer açılamadı, kayıt kapalı."); want_record=False
                else:
                    print("[REC] Yazıyor...")
            if writer is not None and want_record:
                writer.write(canvas); cv2.circle(canvas,(25,25),7,(0,0,255),-1)

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')): break
            if key in (ord('r'), ord('R')):
                want_record = not want_record
                if not want_record and writer is not None:
                    writer.release(); writer=None; print("[REC] OFF")
                elif want_record:
                    print("[REC] ON")
            if key in (ord('f'), ord('F')):
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                      1 if cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)==0 else 0)

            frame_idx += 1

    finally:
        if writer is not None: writer.release()
        pipe.stop()
        for cap in [locals().get('cap2'), locals().get('cap3')]:
            try:
                if cap is not None: cap.release()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
