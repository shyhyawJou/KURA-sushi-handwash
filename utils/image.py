import cv2
import numpy as np



def resize_keep_scale(img, size, mode='corner'):
    """
    Args
        mode: "corner" | "center"
    """

    h, w = img.shape[:2]
    target_w, target_h = size
    
    # 1. 計算縮放比例 (取寬高比最小的那個，確保不會超出邊界)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 2. 縮放影像
    resized_frame = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 3. 建立黑色背景 (畫布)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 4. 計算置中位置
    if mode == 'center':
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
    elif mode == 'corner':
        x_offset = 0
        y_offset = 0
    else:
        raise ValueError(f'invaild resize mode: {mode}')

    # 5. 將縮放後的影像放入畫布中央
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    return canvas