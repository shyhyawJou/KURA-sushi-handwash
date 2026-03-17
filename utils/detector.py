import numpy as np
from pathlib import Path as p
import cv2
from time import time
import onnxruntime as ort
from loguru import logger

try:
    from tensorflow.lite.python.interpreter import Interpreter
except:
    from tflite_runtime.interpreter import Interpreter



class RTMDet:
    def __init__(self, path, score_thresh, iou_thresh, input_wh, classes, agnostic_nms=False):
        self.model = None
        self.input_wh = input_wh
        self.agnostic_nms = agnostic_nms
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.strides = [8, 16, 32]
        self.mean = np.asarray([103.53, 116.28, 123.675], 'float32')
        self.std = np.asarray([57.375, 57.12, 58.395], 'float32')
        self.grids = self._make_grids().astype('float32')
        self.is_onnx = p(path).suffix.lower() == '.onnx'
        self.classes = classes
        self._load_model(path)
        self._warmup()

    def __call__(self, img):
        x, scale = self._preprocess(img)
        y = self._forward(x)
        boxes, scores = np.split(y, [4], 1)
        boxes = self._distance2bbox(boxes)
        scores, boxes, pred_labels = self._postprocess(scores, boxes, scale)
        return scores, boxes, pred_labels

    def _forward(self, x):
        raise NotImplementedError

    def _preprocess(self, img):
        dst_w, dst_h = self.input_wh
        h, w = img.shape[:2]
        scale_w, scale_h = dst_w / w, dst_h / h
        scale = min(scale_w, scale_h)
        new_h, new_w = int(h * scale), int(w * scale)

        img = cv2.resize(img, (new_w, new_h))
        x = np.full((dst_h, dst_w, 3), 114, 'uint8')
        x[:new_h, :new_w] = img

        x = (x - self.mean) / self.std
        if self.is_onnx:
            x = x.transpose(2, 0, 1)[None].astype('float32')
        else:
            x = x[None].astype('float32')
        return x, scale

    def _postprocess(self, scores, boxes, scale):
        boxes, scores = boxes[0].T, scores[0].T
        scores, pred_labels = scores.max(1), scores.argmax(1)

        ids = (scores >= self.score_thresh)
        scores, boxes, pred_labels = scores[ids], boxes[ids], pred_labels[ids]
        boxes_nms = boxes.copy()
        boxes_nms[:, 2:4] -= boxes[:, 0:2]

        if self.agnostic_nms:
            ids = cv2.dnn.NMSBoxes(boxes_nms, scores, 0, self.iou_thresh)
        else:
            OFFSET_WH = 4096
            offset = OFFSET_WH * pred_labels
            boxes_nms[:, 0] += offset
            ids = cv2.dnn.NMSBoxes(boxes_nms, scores, 0, self.iou_thresh)
        
        if len(ids) > 0:
            scores, boxes, pred_labels = scores[ids], boxes[ids], pred_labels[ids]
            boxes[:, :4] /= scale
        else:
            scores = np.empty((0,), 'float32')
            boxes = np.empty((0, 4), 'float32')
            pred_labels = np.empty((0,), 'int32')
        return scores, boxes, pred_labels

    def _make_grids(self):
        feat_hw = []
        for s in self.strides:
            feat_hw.append((self.input_wh[1] // s, self.input_wh[0] // s))
        grids = MlvlPointGenerator(self.strides, 0).grid_priors(feat_hw)
        grids = np.concatenate(grids, 0) 
        return grids

    def _distance2bbox(self, distance):
        distance = distance.reshape(4, -1).transpose(1, 0)

        assert self.grids.shape[0] == distance.shape[0], f'{self.grids.shape}, {distance.shape}'
        assert self.grids.shape[-1] == 2
        assert distance.shape[-1] == 4

        points = self.grids
        max_shape = self.input_wh[::-1]

        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]

        bboxes = np.stack([x1, y1, x2, y2], -1)
        bboxes = bboxes.transpose(1, 0).reshape(1, 4, -1)

        if max_shape is not None:
            # speed up
            bboxes[:, 0::2].clip(min=0, max=max_shape[1])
            bboxes[:, 1::2].clip(min=0, max=max_shape[0])
            return bboxes

        return bboxes

    def _load_model(self):
        raise NotImplementedError

    def _warmup(self):
        if self.is_onnx:
            x = np.random.randn(1, 3, self.input_wh[1], self.input_wh[0]).astype('float32')
        else:
            x = np.random.randn(1, self.input_wh[1], self.input_wh[0], 3).astype('float32')
        
        t0 = time()
        for _ in range(1):
            y = self._forward(x)
        t1 = time()
        logger.info(f'warmup: {t1 - t0:.3f} (s)')


class RTMDet_ONNX(RTMDet):
    def __init__(self, path, score_thresh, iou_thresh, input_wh, classes, agnostic_nms=False):
        super().__init__(path, score_thresh, iou_thresh, input_wh, classes, agnostic_nms)

    def _forward(self, x):
        y = self.model.run(None, {self.input_name: x})[0]
        return y
    
    def _load_model(self, path):
        providers = ['CUDAExecutionProvider']
        self.model = ort.InferenceSession(path, providers=providers)
        self.input_name = self.model.get_inputs()[0].name


class RTMDet_TFLITE(RTMDet):
    def __init__(self, path, score_thresh, iou_thresh, input_wh, classes, agnostic_nms=False):
        super().__init__(path, score_thresh, iou_thresh, input_wh, classes, agnostic_nms)

    def _forward(self, x):
        self.model.set_tensor(self.input_id, x)
        self.model.invoke()
        y = self.model.get_tensor(self.output_id)
        return y
    
    def _load_model(self, path):
        self.model = Interpreter(path)
        self.input_id = self.model.get_input_details()[0]['index']
        self.output_id = self.model.get_output_details()[0]['index']
        self.model.allocate_tensors()


class RTMDet_DLA(RTMDet):
    def __init__(self, path, score_thresh, iou_thresh, input_wh, classes, agnostic_nms=False):
        super().__init__(path, score_thresh, iou_thresh, input_wh, classes, agnostic_nms)

    def _forward(self, x):
        y = self.model.run(x).reshape(1, -1, self.grids.shape[0])
        return y
    
    def _load_model(self, path):
        from utils import AI
        self.model = AI.RTMDET(path)


class MlvlPointGenerator:
    def __init__(self, strides, offset=0.) -> None:
        self.strides = [(stride, stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    @property
    def num_base_priors(self):
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major: bool = True):
        yy, xx = np.meshgrid(y, x, indexing='ij')
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self, featmap_sizes, with_stride = False):

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self, featmap_size, level_idx: int, with_stride = False):
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (np.arange(0, feat_w) + self.offset) * stride_w
        shift_y = (np.arange(0, feat_h) + self.offset) * stride_h
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = np.stack([shift_xx, shift_yy], axis=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = np.full((shift_xx.shape[0], ), stride_w).astype('float32')
            stride_h = np.full((shift_yy.shape[0], ), stride_h).astype('float32')
            shifts = np.stack([shift_xx, shift_yy, stride_w, stride_h], axis=-1)
        all_points = shifts
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape):
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w))
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self, featmap_size, valid_size):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = np.zeros(feat_w, dtype='bool')
        valid_y = np.zeros(feat_h, dtype='bool')
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self, prior_idxs, featmap_size, level_idx: int):
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = np.stack([x, y], 1).astype('float32')
        return prioris
