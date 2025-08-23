import cv2, numpy as np, time

ONNX_PATH = r"/yolov5/runs/train/model_two_v5n/weights/best.onnx"
SOURCE = 1            # 0: webcam, ya da "video.mp4"
IMG_SIZE = 640
CONF_THRES = 0.25
NMS_THRES = 0.45

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
    # YOLOv5 ONNX cikisi: (1, N, 5+nc)  -> [x,y,w,h, obj, c1, c2, ...]
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

    # xywh -> xyxy, letterbox'u geri al
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

    # OpenCV NMSBoxes x,y,w,h istiyor
    bbs_xywh = []
    for x1,y1,x2,y2 in boxes:
        bbs_xywh.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
    idxs = cv2.dnn.NMSBoxes(bbs_xywh, list(map(float, scores)),
                            score_threshold=CONF_THRES, nms_threshold=NMS_THRES)
    idxs = idxs.flatten() if len(idxs) else []
    return [(boxes[i], int(cls_id[i]), float(scores[i])) for i in idxs]

def main():
    net = cv2.dnn.readNetFromONNX(ONNX_PATH)
    cap = cv2.VideoCapture(SOURCE)
    t0 = time.time(); frames = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        blob, r, dwdh = preprocess_bgr(frame)
        net.setInput(blob)
        out = net.forward()
        dets = postprocess(out, r, dwdh, frame.shape)
        for (x1,y1,x2,y2), cid, sc in dets:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2)
            cv2.putText(frame, f"{cid}:{sc:.2f}", (p1[0], max(0,p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        frames += 1
        if frames % 30 == 0:
            fps = frames / (time.time() - t0 + 1e-9)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("ONNX DNN detect", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
