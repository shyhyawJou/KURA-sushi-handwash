import numpy as np
from datetime import datetime, timezone
import json
from .image import keep_scale_geometry



def get_iou(boxes_A, boxes_B):
    x1 = np.maximum(boxes_A[:, None, 0], boxes_B[None, :, 0])
    y1 = np.maximum(boxes_A[:, None, 1], boxes_B[None, :, 1])
    x2 = np.minimum(boxes_A[:, None, 2], boxes_B[None, :, 2])
    y2 = np.minimum(boxes_A[:, None, 3], boxes_B[None, :, 3])
    inter = np.maximum((x2 - x1) * (y2 - y1), 0)
    area_A = np.prod(boxes_A[:, 2:4] - boxes_A[:, 0:2], axis=1)
    area_B = np.prod(boxes_B[:, 2:4] - boxes_B[:, 0:2], axis=1)
    return inter / ((area_A[:, None] + area_B[None] - inter) + 1e-16)


def get_now_str(now, utc=True):
    if now is None:
        return ''
    if utc:
        now = datetime.fromtimestamp(now, timezone.utc)
        now = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    else:
        now = datetime.fromtimestamp(now)
        now = now.strftime("%Y%m%d %H%M%S.%f")[:-3]
    return now


def get_utc_offset() -> int:
    """ 單位: 小時 """
    offset = datetime.now().astimezone().utcoffset()
    hour = int(offset.total_seconds() / 3600)
    return hour


def get_boxes_outside(boxes: np.ndarray, roi: tuple) -> np.ndarray:
    boxes = np.asarray(boxes)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 計算中心點
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = roi

    # 判斷中心點是否在 ROI 內 (包含邊界)
    inside_roi = (cx >= rx1) & (cx <= rx2) & (cy >= ry1) & (cy <= ry2)
    return ~inside_roi


def parse_lateral_flags(stages: list[dict]) -> tuple[dict, dict]:
    """
    回傳 (need_lateral_count, need_lateral_time)
    兩個 list 的順序跟 stages 順序一致（即所有 step）
    """
    need_lateral_count = {}
    need_lateral_time = {}

    for stage in stages:
        joined = ''.join(stage['items'])

        lateral_count = ('左手回数' in joined) or ('右手回数' in joined)
        lateral_time = ('左手時間' in joined) or ('右手時間' in joined)

        # 有左右回數，時間也一定要分左右
        if lateral_count:
            lateral_time = True

        need_lateral_count[stage['id']] = lateral_count
        need_lateral_time[stage['id']] = lateral_time

    return need_lateral_count, need_lateral_time


def origin_rel_to_frame_px(box_rel, origin_hw, crop_xyxy, frame_hw):
    """
    把「未裁切原圖」的相對座標 [x1, y1, x2, y2] (0~1) 換算成「AI 吃的 frame」的像素座標。
    換算流程跟 Camera 一致: 原圖 -> 裁切 (crop_xyxy, 原圖像素) -> resize_keep_scale(corner) -> frame

    Args
        origin_hw: 原圖 (h, w)
        crop_xyxy: 原圖像素座標的裁切範圍; None 代表沒裁切
        frame_hw: frame (h, w), 也就是 resize 的目標大小
    Returns
        np.ndarray (int) [x1, y1, x2, y2], 已 clip 在 frame 範圍內
    """
    oh, ow = origin_hw
    fh, fw = frame_hw
    cx1, cy1, cx2, cy2 = (0, 0, ow, oh) if crop_xyxy is None else [int(v) for v in crop_xyxy]
    cw, ch = cx2 - cx1, cy2 - cy1

    # 原圖像素 -> 裁切圖像素
    box = np.asarray(box_rel, dtype='float64') * [ow, oh, ow, oh] - [cx1, cy1, cx1, cy1]

    # 裁切圖 -> frame (等比縮放 + offset)
    new_w, new_h, x_off, y_off = keep_scale_geometry((ch, cw), (fw, fh), 'corner')
    box = box * [new_w / cw, new_h / ch, new_w / cw, new_h / ch] + [x_off, y_off, x_off, y_off]

    box = np.round(box).astype(int)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, fw - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, fh - 1)
    return box
