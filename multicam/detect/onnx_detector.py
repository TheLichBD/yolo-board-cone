import cv2
import numpy as np

def _has_cuda_dnn():
    # basit kontrol: pip wheel'lerde yoktur; Jetson/ozel build varsa >0 doner
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

class ONNXDetector:
    def __init__(self, onnx_path, imgsz=640, conf_th=0.25, nms_th=0.45, backend="cpu"):
        self.imgsz = imgsz
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

        b = (backend or "cpu").lower()
        try:
            if b in ("cuda", "cuda_fp16") and _has_cuda_dnn():
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                tgt = cv2.dnn.DNN_TARGET_CUDA_FP16 if b == "cuda_fp16" else cv2.dnn.DNN_TARGET_CUDA
                self.net.setPreferableTarget(tgt)
                print("[DNN] using CUDA backend:", b)
            else:
                # fallback
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                if b != "cpu":
                    print("[DNN] CUDA not available; falling back to CPU")
        except Exception as e:
            # kesin fallback
            print("[DNN] backend set failed, falling back to CPU:", e)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _postprocess(self, out, r, dwdh, orig_shape):
        if out.ndim == 3:
            out = out[0]
        boxes_xywh = out[:, :4]
        obj = out[:, 4:5]
        cls = out[:, 5:]
        cls_id = np.argmax(cls, axis=1)
        cls_conf = cls.max(axis=1, keepdims=True)
        scores = (obj * cls_conf).squeeze()
        m = scores >= self.conf_th
        boxes_xywh, cls_id, scores = boxes_xywh[m], cls_id[m], scores[m]
        if boxes_xywh.shape[0] == 0:
            return []
        boxes = boxes_xywh.copy()
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        dw, dh = dwdh
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r
        h, w = orig_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        bbs_xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes]
        idxs = cv2.dnn.NMSBoxes(bbs_xywh, list(map(float, scores)), self.conf_th, self.nms_th)
        idxs = idxs.flatten() if len(idxs) else []
        return [(boxes[i], int(cls_id[i]), float(scores[i])) for i in idxs]

    def infer(self, blob, r, dwdh, orig_shape):
        self.net.setInput(blob)
        out = self.net.forward()
        return self._postprocess(out, r, dwdh, orig_shape)
