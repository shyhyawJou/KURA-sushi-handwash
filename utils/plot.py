import cv2
from .handwash import HandWashTracker



COLORS = [
    "4242BF",  # 紅色系
    "00F200",  # 綠色
    "F20000",  # 藍色
    "F2DA00",  # 青色
    "24C9F2",  # 橙黃色
    "C100F2",  # 洋紅色
    "BF5D1C",  # 藍紫色
    "6DBF1C",  # 黃綠色
    "0000F2",  # 紅色(深)
    "F25493",  # 粉藍色
    "00BF85",  # 黃綠(暗)
    "7900F2",  # 紫紅色
    "428DBF",  # 橙棕色
    "0079F2",  # 橙色
    "BF8D42",  # 藍棕色
    "9942BF",  # 紫色
    "D2F254",  # 青綠色
    "F200DA",  # 粉紫色
    "26BF00",  # 綠色(亮)
    "00F2DA",  # 黃綠(亮)
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


def draw_timestamp(img, timestamp_str, font_scale=0.8, thickness=2, shadow_offset=2):
    """
    在影像右下角繪製帶有陰影的紅色時間戳
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 取得文字寬高以便計算座標
    (text_w, text_h), baseline = cv2.getTextSize(timestamp_str, font, font_scale, thickness)
    
    # 設定位置 (距離邊界 10 pixel)
    #x = w - text_w - 5
    #y = h - 20
    x = 10
    y = text_h + 10
    
    # 1. 畫陰影 (黑色，偏移 2 pixel)
    shadow_offset = 2
    cv2.putText(img, timestamp_str, (x + shadow_offset, y + shadow_offset), 
                font, font_scale, (0, 0, 0), thickness)
    
    # 2. 畫主文字 (紅色 BGR: 0, 0, 255)
    cv2.putText(img, timestamp_str, (x, y), 
                font, font_scale, (0, 0, 255), thickness)
    

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#').strip()
    b = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    r = int(hex_str[4:6], 16)
    return b, g, r


def draw_debug_panel(img, tracker_l, tracker_r):
    """
    在畫面底部繪製雙手的洗手步驟監控面板
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # --- 參數配置 ---
    font_size = 0.31     # 緊湊字體
    line_height = 12     # 緊湊行高
    panel_height = 190   # 面板總高度 (從底部往上算)
    text_start_x_offset = 20
    
    def draw_text_with_shadow(image, text, org, color=(255, 255, 255), size=0.31, thickness=1):
        # 繪製黑色陰影
        cv2.putText(image, text, (org[0] + 1, org[1] + 1), font, size, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # 繪製主文字
        cv2.putText(image, text, org, font, size, color, thickness, cv2.LINE_AA)

    def draw_zone_debug(tracker: HandWashTracker, start_x):
        d = tracker.debug_info
        cfg = tracker.cfg
        sys_cfg = tracker.sys_cfg
        scr = tracker.scrub_count_ratio
        steps = {i: data for i, data in d['detected_steps'].items() if data is not None} 

        # step
        saved_step = d['saved_steps']
        if saved_step:
            text = f'[{tracker.zone_name}] Saved step {",".join(str(s) for s in saved_step)}'
            pos_y = h - panel_height + 3
            t_size = cv2.getTextSize(text, font, font_size, 1)[0]
            cv2.rectangle(img, (start_x - 2, pos_y - 9), (start_x + t_size[0] + 25, pos_y + 3), (0, 165, 255), -1)
            #cv2.putText(img, text, (start_x, pos_y), font, font_size, (0, 0, 0), 1, cv2.LINE_AA)
            draw_text_with_shadow(img, text, (start_x, pos_y), (0, 255, 0), 0.35, 1)

        # 登入狀態
        status_color = (0, 255, 0) if d['is_login'] else (0, 0, 255)
        text = 'Login' if d['is_login'] else 'Logout'
        draw_text_with_shadow(img, f"[{tracker.zone_name}] {text}", 
                              (start_x, h - panel_height), status_color, font_size)

        # 手狀態
        status_color = (0, 255, 255) if d['status'] == "Hand Detected" else (150, 150, 150)
        draw_text_with_shadow(img, f"[{tracker.zone_name}] {d['status']}", 
                              (start_x, h - panel_height + 15), status_color, font_size)

        # 步驟狀態
        for i in range(1, 13):
            recorded = steps.get(i)

            if recorded is not None:
                frame = recorded.frame
                left_frame = recorded.left_frame
                right_frame = recorded.right_frame
                duration = recorded.duration
                left_duration = recorded.left_duration
                right_duration = recorded.right_duration
                count = recorded.count
                left_count = recorded.left_count
                right_count = recorded.right_count
            else:
                frame = d['frames'][i]
                left_frame = d['left_frames'][i]
                right_frame = d['right_frames'][i]
                duration = d['durations'][i]
                left_duration = d['left_durations'][i]
                right_duration = d['right_durations'][i]
                count = d['counts'][i]
                left_count = d['left_counts'][i]
                right_count = d['right_counts'][i]

            # 重要資訊
            min_frame = cfg['action_frame'][i]
            min_duration = sys_cfg[i-1]['washtimemax']
            min_count = sys_cfg[i-1]['washcountmax']

            if tracker.time_need_lr[i]:
                time_ok = left_duration >= min_duration and right_duration >= min_duration
            else:
                time_ok = duration >= min_duration

            if tracker.count_need_lr[i]:
                count_ok = left_count >= min_count and right_count >= min_count
            else:
                count_ok = count >= min_count

            is_done = time_ok and count_ok and recorded is not None
            # 分左右的 step 進行中動的是 left_frame/right_frame, frame 本身不會動, 三個都要檢查
            is_active = (frame > 0 or left_frame > 0 or right_frame > 0) and not is_done
            is_detecting = i == d['detecting_step']
            
            # 文字, 背景
            if is_detecting:
                bg_color = (180, 105, 255)  # 粉色底
                text_color = (0, 0, 0)      # 黑字
                suffix = "[DETECTING] !"
            elif is_done:
                bg_color = None  # 沒背景
                text_color = (0, 255, 0)
                suffix = "[DONE]"
            elif is_active:
                bg_color = (0, 255, 255)  # 黃底
                text_color = (0, 0, 0)  # 黑字
                suffix = "[ACTIVE]"
            else:
                bg_color = None  # 沒背景
                text_color = (255, 255, 255)  # 綠字
                suffix = ""

            frame_info = f'{frame},{left_frame},{right_frame}/{min_frame}'
            count_info = f'  {count},{left_count},{right_count}/{min_count}/{scr[i]}'
            duration_info = f'  {duration:.1f},{left_duration:.1f},{right_duration:.1f}/{min_duration:.1f}'
            line_text = f"Step {i:<2}: {frame_info}{count_info}{duration_info}  {suffix}"
            
            # y 座標隨 panel_height 自動計算，讓列表貼合底部
            pos_y = h - (panel_height - 30) + (i-1) * line_height

            if bg_color:
                t_size = cv2.getTextSize(line_text, font, font_size, 1)[0]
                # 畫色塊背景
                cv2.rectangle(img, (start_x - 2, pos_y - 9), (start_x + t_size[0] + 5, pos_y + 3), bg_color, -1)
                # 在色塊上畫字 (不帶陰影以增加清晰度)
                cv2.putText(img, line_text, (start_x, pos_y), font, font_size, text_color, 1, cv2.LINE_AA)
            else:
                draw_text_with_shadow(img, line_text, (start_x, pos_y), text_color, font_size)

    def render_rich_mqtt_on_frame(tracker, start_x, start_y, title_color):
        d = tracker.debug_info
        sent_msg = d['sent_msg']
        if not sent_msg:
            return
            
        # 1. 準備 Token 縮排結構
        msg_str = str(sent_msg).strip("'\"")
        items = msg_str.replace("{", "").replace("}", "").split(",")
        
        lines = []
        lines.append(f"● {tracker.zone_name.upper()} MQTT SENT:")
        lines.append("{")
        for item in items:
            if item.strip():
                lines.append("    " + item.strip())
        lines.append("}")

        # 2. 計算這坨字卡需要的背景寬高
        max_w = 0
        card_font_size = 0.30
        card_line_height = 12
        for line in lines:
            w_size = cv2.getTextSize(line, font, card_font_size, 1)[0][0]
            if w_size > max_w:
                max_w = w_size
                
        card_w = max_w + 20
        card_h = len(lines) * card_line_height + 15

        # 3. 在 Frame 上畫一塊微透明黑底襯托文字，防止背景太亮干擾閱讀
        #overlay = img.copy()
        cv2.rectangle(img, (start_x - 5, start_y - 12), (start_x + card_w, start_y + card_h - 10), (10, 10, 10), -1)
        #cv2.addWeighted(img, 0.65, img, 0.35, 0, img)
        # 外邊框裝飾
        cv2.rectangle(img, (start_x - 5, start_y - 12), (start_x + card_w, start_y + card_h - 10), (80, 80, 80), 1)

        # 4. 逐行繪製帶縮排的文字
        curr_y = start_y
        for idx, line in enumerate(lines):
            if idx == 0:
                color = title_color # 標題使用專屬黃色/紫色
            elif line.strip() in ("{", "}"):
                color = (220, 220, 220) # 結構符號用白灰色
            else:
                color = (0, 255, 0) # 內容欄位用經典富文字綠色
                
            draw_text_with_shadow(img, line, (start_x, curr_y), color, 0.32)
            curr_y += card_line_height

    # 畫 mqtt 訊息 
    render_rich_mqtt_on_frame(tracker_l, start_x=text_start_x_offset, start_y=25, title_color=(0, 255, 255))
    render_rich_mqtt_on_frame(tracker_r, start_x=int(w * 0.52), start_y=25, title_color=(0, 255, 255))

    # 呼叫左右區域繪製 (使用 w 比例確保對齊)
    draw_zone_debug(tracker_l, text_start_x_offset)
    draw_zone_debug(tracker_r, int(w * 0.52))