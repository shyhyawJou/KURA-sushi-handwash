import cv2
import numpy as np



def keep_scale_geometry(src_hw, size, mode='corner'):
    """
    resize_keep_scale 的幾何計算 (獨立出來, 讓其他地方可以做座標換算)

    Returns
        (new_w, new_h, x_offset, y_offset): 縮放後影像的大小, 以及貼進畫布的位置
    """
    h, w = src_hw
    target_w, target_h = size

    # 1. 計算縮放比例 (取寬高比最小的那個，確保不會超出邊界)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 2. 計算置中位置
    if mode == 'center':
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
    elif mode == 'corner':
        x_offset = 0
        y_offset = 0
    else:
        raise ValueError(f'invaild resize mode: {mode}')

    return new_w, new_h, x_offset, y_offset


def resize_keep_scale(img, size, mode='corner'):
    """
    Args
        mode: "corner" | "center"
    """
    if img.shape[:2] == size[::-1]:
        return img

    target_w, target_h = size
    new_w, new_h, x_offset, y_offset = keep_scale_geometry(img.shape[:2], size, mode)

    # 縮放影像
    resized_frame = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 建立黑色背景 (畫布), 並將縮放後的影像放入
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    return canvas


def paste(img1, img2, roi, feather=20):
    x1, y1, x2, y2 = roi
    merged = img2.copy()
    crop = img1[y1:y2, x1:x2]
    h, w = crop.shape[:2]

    # 建立 alpha mask
    mask = np.ones((h, w), dtype=np.float32)

    # 四邊 feather
    for i in range(feather):
        alpha = i / feather
        mask[i, :] *= alpha
        mask[h - i - 1, :] *= alpha
        mask[:, i] *= alpha
        mask[:, w - i - 1] *= alpha

    # 避免全黑
    mask = np.clip(mask, 0, 1)

    # 擴成三通道
    mask = mask[:, :, None]

    # blending
    roi_dst = merged[y1:y2, x1:x2]
    blended = (
        crop.astype(np.float32) * mask +
        roi_dst.astype(np.float32) * (1 - mask)
    )
    merged[y1:y2, x1:x2] = blended.astype(np.uint8)
    return merged