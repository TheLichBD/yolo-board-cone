import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime is not installed. Install onnxruntime-gpu or onnxruntime.") from e


class ORTDetector:
    def __init__(self, onnx_path, imgsz=640, conf_th=0.25, nms_th=0.45, use_cuda=True):
        self.imgsz = imgsz
        self.conf_th = conf_th
        self.nms_th = nms_th

        # Provider secimi (GPU varsa CUDA, yoksa CPU'ya dus)
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in wanted if p in available]
        if not providers:
            # zorunlu fallback
            providers = ["CPUExecutionProvider"]
            print("[ORT] CUDA provider not available; falling back to CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # Çıktı isimlerini okumak zorunda değiliz; run(None, ...) ile alacağız

    def _postprocess(self, out, r, dwdh, orig_shape):
        # YOLOv5 tipik çıkış: [N, 85] (x,y,w,h,obj,cls...)
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

        # xywh -> xyxy + letterbox geri dönüşü
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

        # NMS
        bbs_xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes]
        idxs = cv2.dnn.NMSBoxes(bbs_xywh, list(map(float, scores)), self.conf_th, self.nms_th)
        idxs = idxs.flatten() if len(idxs) else []

        return [(boxes[i], int(cls_id[i]), float(scores[i])) for i in idxs]

    def infer(self, blob, r, dwdh, orig_shape):
        # blob: (1,3,H,W) float32 [0..1]
        # ORT FP16/FP32 otomatik handle eder, ama tip float32 olsun
        out = self.session.run(None, {self.input_name: blob.astype(np.float32)})[0]
        return self._postprocess(out, r, dwdh, orig_shape)
