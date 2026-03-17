import cv2



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


def plot_bbox(img, boxes, labels, thickness=3):
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    if isinstance(labels, (int, str)):
        labels = [labels]

    assert boxes.shape[-1] == 4
    formatted_boxes = boxes.astype(int).reshape(-1, 2, 2)
    
    for box, lb in zip(formatted_boxes, labels):
        cv2.rectangle(img, tuple(box[0]), tuple(box[1]), get_color(lb), thickness)


def plot_class(img, texts, classes, pred_labels, boxes, bbox_thickness=2):
    if isinstance(texts, str):
        texts = [texts]
    if boxes.ndim == 1:
        boxes = boxes[None, :]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 2  

    for text, box, pred_lb in zip(texts, boxes, pred_labels):
        if not isinstance(text, str):
            text = f'{classes[pred_lb]} {text * 100:.1f}'

        # 取得文字大小
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # 取得左上角座標
        x1, y1 = int(box[0]), int(box[1])

        # 內縮處理
        inner_x = x1 + bbox_thickness
        inner_y = y1 + bbox_thickness

        # 背景框範圍
        bg_top_left = (inner_x, inner_y)
        bg_bottom_right = (
            inner_x + text_w + padding * 2,
            inner_y + text_h + baseline + padding * 2
        )

        # 畫背景與文字
        cv2.rectangle(img, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
        cv2.putText(
            img,
            text,
            (inner_x + padding, inner_y + text_h + padding),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )