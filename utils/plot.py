import cv2
import numpy as np
from typing import Sequence
from loguru import logger



COLORS = [
    '5F5FF3',  # 藍紫
    'F3A1EB',  # 粉紅
    'FCFA98',  # 淡黃
    '96DCF8',  # 淺藍
    '00FF00',  # 亮綠
    'FF0000',  # 純紅
    'FFA500',  # 橘色
    '800080',  # 紫色
    '00FFFF',  # 青色 (Cyan)
    'FFD700',  # 金黃
    'DC143C',  # 猩紅
    '00CED1',  # 深青綠
    'ADFF2F',  # 黃綠
    '1E90FF',  # 道奇藍
    'FF69B4',  # 粉紅紅
    '8B4513',  # 深棕
    'A52A2A',  # 棕紅
    'FF1493',  # 深粉紅
    '7FFF00',  # 查特綠
    '9932CC',  # 紫水晶
    '40E0D0',  # 綠松石
    'B22222',  # 火磚紅
    '2E8B57',  # 海綠
    'D2691E',  # 巧克力棕
    'E6E6FA',  # 薰衣草紫
    '0000FF',  # 純藍
    '008000',  # 深綠
    'FFFF00',  # 純黃
    'FF00FF',  # 洋紅
    '00FF7F',  # 春綠
    '4682B4',  # 鋼藍
    '6A5ACD',  # 石板藍
    'C71585',  # 中紫紅
    '191970',  # 午夜藍
    '228B22',  # 森林綠
    'B8860B',  # 深金
    'FF4500',  # 橘紅
    '2F4F4F',  # 深灰綠
    '6495ED',  # 矢車菊藍
    'FFB6C1',  # 淺粉紅
    '20B2AA',  # 淺海綠
    'CD5C5C',  # 印度紅
    'BA55D3',  # 中蘭花紫
    '3CB371',  # 中海綠
    'DB7093',  # 淺紫紅
    '87CEEB',  # 天空藍
    '6B8E23',  # 橄欖綠
    'FF8C00',  # 深橘
    '483D8B',  # 深石板藍
    '708090'  # 石板灰
]


DISTANCE_COLOR = [
    (0, 0, 255),      # 01. 紅色 (Red) - 初始顏色
    (0, 255, 0),      # 02. 亮綠 (Green)
    (255, 0, 255),    # 03. 品紅 (Magenta)
    (0, 255, 255),    # 04. 黃色 (Yellow)
    (255, 255, 0),    # 05. 青色 (Cyan)
    (0, 165, 255),    # 06. 橘色 (Orange)
    (255, 128, 0),    # 07. 亮蔚藍 (Bright Azure) - 用來替代純藍，但更偏青
    (128, 0, 128),    # 08. 紫色 (Purple)
    (0, 250, 154),    # 09. 中春綠 (Medium Spring Green)
    (203, 192, 255),  # 10. 粉紅 (Pink)
    (0, 128, 0),      # 11. 深綠 (Dark Green)
    (255, 255, 255),  # 12. 白色 (White)
    (42, 42, 165),    # 13. 棕色 (Brown)
    (128, 128, 0),    # 14. 深青 (Teal)
    (0, 0, 128),      # 15. 深紅 (Maroon)
    (209, 206, 0),    # 16. 天藍 (Sky Blue)
    (0, 215, 255),    # 17. 金色 (Gold)
    (130, 0, 255),    # 18. 霓虹紫 (Neon Purple)
    (128, 128, 128),  # 19. 灰色 (Gray)
    (30, 105, 210)    # 20. 巧克力色 (Chocolate)
]


class Result:
    def __init__(self, mode, stay_time, num_block):
        self.mode = mode
        self.stay_time = stay_time
        self.num_block = num_block

        if self.mode != 'center':
            raise ValueError(f'"{self.mode}" is the unknown mode of drawing result !')
        
    def draw_step(self, img, texts: Sequence):
        img_h, img_w = img.shape[:2]

        if self.mode == 'center':
            x = np.linspace(0, img_w - 1, self.num_block * 2 + 1, endpoint=True)[1:-1:2]
            y = np.linspace(0, img_h - 1, 5, endpoint=True)[1:2]
            x, y = np.meshgrid(x, y, indexing='xy')
            points = np.stack([x.ravel(), y.ravel()], axis=-1).astype(int)
            assert len(texts) == len(points)

            for text, point in zip(texts, points):
                # 陰影
                shadow_offset = 2
                cv2.putText(img, text, point + shadow_offset, cv2.FONT_HERSHEY_SIMPLEX,
                            1., (0, 0, 0), 2)

                cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, 1.,
                            (0, 255, 0), 2)

    def draw_region(self, img, bboxes, text):
        bboxes = bboxes.astype(int)
        for box in bboxes:
            cv2.putText(img, text, box[2:4], cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 2)

    def _make_grids(self):
        pass


def hex_to_rgb(hex_str):
    # 1. 去除可能存在的 '#' 或空白
    hex_str = hex_str.lstrip('#').strip()
    
    # 2. 分段切片並轉換為 10 進位整數
    # int(x, 16) 代表將 x 視為 16 進位進行轉換
    b = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    r = int(hex_str[4:6], 16)
    
    return b, g, r


def get_color(label):
    return hex_to_rgb(COLORS[label])


def plot_bbox(img, 
              boxes, 
              pred_labels, 
              scores, 
              classes, 
              bbox_thickness=2, 
              omit_classes=set(), 
              plot_score=True,
              font_scale=0.55,
              font_thickness=1,
              text_padding=2):
    
    # bbox
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    assert boxes.shape[-1] == 4
    boxes = boxes.astype(int).reshape(-1, 4)
    
    # predict labels
    if isinstance(pred_labels, (int, str)):
        pred_labels = [pred_labels]

    # 取得影像維度以進行邊界檢查
    img_h, img_w, _ = img.shape
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 忽略的類別
    omit_classes = set(omit_classes)

    for score, box, pred_lb in zip(scores, boxes, pred_labels):
        cls = classes[pred_lb]
        if cls in omit_classes:
            continue

        # 畫 bbox
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), get_color(pred_lb), bbox_thickness)

        text = f'{cls} {score * 100:.1f}%' if plot_score else cls
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        label_h = text_h + baseline + text_padding * 2
        label_w = text_w + text_padding * 2

        x1, y1, x2, y2 = map(int, box[:4])

        # --- 自動位置偵測邏輯 ---
        # 預設位置：上方外側 (Top-Outside)
        ty = y1 - label_h
        
        # 情況 A：上方超出影像邊界，嘗試切換到下方外側 (Bottom-Outside)
        if ty < 0:
            if y2 + label_h < img_h:
                ty = y2
            else:
                # 情況 B：上下都沒空間（例如物體佔滿垂直空間），強行放在內部頂端 (Top-Inside)
                ty = y1 + bbox_thickness

        # 水平邊界修正：確保標籤不會超出左右邊緣
        tx = max(0, min(x1, img_w - label_w))

        # 背景與文字繪製
        bg_top_left = (tx, ty)
        bg_bottom_right = (tx + label_w, ty + label_h)

        cv2.rectangle(img, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
        cv2.putText(img, text, (tx + text_padding, ty + text_h + text_padding),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


def plot_distance(img, boxA, boxB, distance, color):
    """ 畫在 boxA 附近 """
    distance = int(distance)
    ctr_A = ((boxA[0:2] + boxA[2:4]) / 2.).astype(int)
    ctr_B = ((boxB[0:2] + boxB[2:4]) / 2.).astype(int)
    org = np.int64((ctr_A + ctr_B) / 2.)
    cv2.line(img, ctr_A, ctr_B, (0, 255, 0), 2)
    cv2.putText(img, str(distance), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def plot_xy(img, xy, org):
    xy = np.array(xy).astype(int)
    org = np.array(org).astype(int)
    cv2.putText(img, f'{xy}', org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


def plot_timeout(img, name, elapsed, org, color):
    org = np.array(org).astype(int)
    cv2.putText(img, f'[{name}] {elapsed:.1f}', org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_timestamp(img, timestamp_str, font_scale=0.8, thickness=2, shadow_offset=2):
    """
    在影像右下角繪製帶有陰影的紅色時間戳
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 取得文字寬高以便計算座標 (右下角)
    (text_w, text_h), baseline = cv2.getTextSize(timestamp_str, font, font_scale, thickness)
    
    # 設定位置 (距離邊界 20 pixel)
    x = w - text_w - 20
    y = h - 20
    
    # 1. 畫陰影 (黑色，偏移 2 pixel)
    shadow_offset = 2
    cv2.putText(img, timestamp_str, (x + shadow_offset, y + shadow_offset), 
                font, font_scale, (0, 0, 0), thickness)
    
    # 2. 畫主文字 (紅色 BGR: 0, 0, 255)
    cv2.putText(img, timestamp_str, (x, y), 
                font, font_scale, (0, 0, 255), thickness)