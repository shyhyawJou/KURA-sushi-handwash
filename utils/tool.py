import numpy as np
from datetime import datetime, timezone



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


def get_utc_offset_str():
    """ 單位: 小時 """
    offset = datetime.now().astimezone().utcoffset()
    hour = int(offset.total_seconds() / 3600)
    return hour