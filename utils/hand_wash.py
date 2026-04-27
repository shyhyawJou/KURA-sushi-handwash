import numpy as np
from datetime import datetime
from loguru import logger

class HandWashTracker:
    def __init__(self, zone_name="Left", logic_cfg=None):
        self.zone_name = zone_name
        if logic_cfg is None:
            raise KeyError(f"[{zone_name}] logic_cfg must be provided in config.yaml under AI.handwash.logic")
        
        self.cfg = logic_cfg
        self.last_active_time = datetime.now()
        self.reset()

    def reset(self):
        self.current_step = 0  
        self.start_time = None
        self.flags = [0] * 12
        self.times = [""] * 12
        self.counts = [-1] * 12
        for i in range(2, 7): self.counts[i] = 0
        
        self.buffer_count = 0
        self.alcohol_in_progress = False
        self.last_active_time = datetime.now()
        logger.info(f"[{self.zone_name}] Tracker Reset.")

    def _is_collided(self, boxA, boxB):
        """判斷兩個 BBox 是否有重疊"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        return interArea > 0

    def _is_contained(self, boxA, boxB):
        """
        Check if boxA is completely inside boxB.
        box format: [x1, y1, x2, y2]
        """
        is_inside = (boxA[0] >= boxB[0] and  # x1
                     boxA[1] >= boxB[1] and  # y1
                     boxA[2] <= boxB[2] and  # x2
                     boxA[3] <= boxB[3])     # y2
        return is_inside

    def _get_dist(self, boxA, boxB):
        cA = [(boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2]
        cB = [(boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2]
        return np.linalg.norm(np.array(cA) - np.array(cB))

    def _get_point_to_line_dist(self, p, l1, l2):
        p, l1, l2 = np.array(p), np.array(l1), np.array(l2)
        line_vec = l2 - l1
        p_vec = p - l1
        line_len_sq = np.sum(line_vec**2)
        t = max(0, min(1, np.dot(p_vec, line_vec) / (line_len_sq + 1e-6)))
        projection = l1 + t * line_vec
        return np.linalg.norm(p - projection)

    def update(self, detections):
        now = datetime.now()
        hands = [d for d in detections if d['label'] in ['left', 'right']]
        gloved_hands = [d for d in detections if d['label'] == 'gloved hand']
        dev = {d['label']: d['box'] for d in detections if d['label'] not in ['left', 'right', 'gloved hand']}
        all_labels = [d['label'] for d in detections]
        now_str = f'{now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z'
        
        scrub_labels = ['palm to palm', 'palm over back', 'interlaced fingers', 'thumb or wrist scrub', 'scrub under nails']
        next_step = self.current_step + 1

        # --- 1. 超時重置 ---
        if (now - self.last_active_time).total_seconds() > self.cfg['reset_time']:
            if self.current_step > 0:
                logger.warning(f"[{self.zone_name}] Session timeout, resetting from step {self.current_step}...")
                res = self._get_final_data(now_str)
                self.reset()
                return now, res # 回傳未完成的記錄
            self.reset()

        if len(hands) > 0:
            self.last_active_time = now

        # --- Step 1 & 8: 沖水 ---
        if next_step in [1, 8]:
            target = 'faucet' if next_step == 1 else 'sink'
            is_active = False
            
            if target in dev and len(hands) > 0:
                target_box = dev[target]
                target_mid_y = (target_box[1] + target_box[3]) / 2 # 設備的垂直中心
                
                # 判定 A: 碰撞 (觸碰到設備)
                # 判定 B: 在下方且距離夠近 (手中心 y > 設備中心 y)
                for h in hands:
                    h_box = h['box']
                    h_mid_y = (h_box[1] + h_box[3]) / 2
                    
                    # 必須滿足 (碰撞 OR 距離近) 且 手在下方
                    if (self._is_collided(h_box, target_box) or \
                        self._get_dist(h_box, target_box) < self.cfg['faucet_dist_thresh']):
                        
                        if h_mid_y > target_mid_y: # 關鍵：手部中心必須低於水龍頭中心
                            is_active = True
                            break
                
                # 額外條件：沖水時不能正在做搓揉動作
                #if is_active and any(lb in all_labels for lb in scrub_labels):
                #    is_active = False
                #is_active &= any(lb in all_labels for lb in scrub_labels)
            
            self._handle_buffer(is_active, next_step, now_str)
            if is_active and next_step == 1 and self.start_time is None:
                self.start_time = now_str

        # --- 3. Step 2: 洗手液 (碰撞) ---
        elif next_step == 2:
            is_active = ('soap dispenser' in dev and 
                         any(self._is_collided(h['box'], dev['soap dispenser']) for h in hands))
            self._handle_buffer(is_active, 2, now_str)

        # --- 4. Step 3-7: 搓揉 ---
        elif 3 <= next_step <= 7:
            target_label = scrub_labels[next_step - 3]
            has_gloved = any(self._is_collided(g_h['box'], h['box']) for g_h in gloved_hands for h in hands)
            if target_label in all_labels and not has_gloved:
                self.buffer_count += 1
                self.counts[next_step-1] += 1
                if self.buffer_count >= self.cfg['scrub_verify_frames']:
                    self._trigger(next_step, now_str)
            else:
                self.buffer_count = 0

        # --- 5. Step 9: 擦手紙 (與手交集且在 sink 範圍) ---
        elif next_step == 9:
            paper_boxes = [d['box'] for d in detections if d['label'] == 'paper towel']
            is_active = False
            if 'sink' in dev and len(hands) > 0 and paper_boxes:
                for p_box in paper_boxes:
                    # 檢查紙張是否與手碰撞，且紙張是否位於水槽範圍內 (碰撞判定)
                    if any(self._is_collided(h['box'], p_box) for h in hands) and \
                       self._is_contained(p_box, dev['sink']):
                        is_active = True
                        break
            self._handle_buffer(is_active, 9, now_str)

        # --- 6. Step 10: 殺菌燈 ---
        elif next_step == 10:
            if 'air outlet' in dev and 'brush tray' in dev and len(hands) > 0:
                p1 = [(dev['air outlet'][0]+dev['air outlet'][2])/2, (dev['air outlet'][1]+dev['air outlet'][3])/2]
                p2 = [(dev['brush tray'][0]+dev['brush tray'][2])/2, (dev['brush tray'][1]+dev['brush tray'][3])/2]
                is_under_lamp = any(self._get_point_to_line_dist([(h['box'][0]+h['box'][2])/2, (h['box'][1]+h['box'][3])/2], p1, p2) < 40 for h in hands)
                self._handle_buffer(is_under_lamp, 10, now_str)

        # --- 7. Step 11: 酒精噴灑 ---
        elif next_step == 11:
            if 'alcohol nozzle' in dev and len(hands) > 0:
                box = dev['alcohol nozzle']
                short_edge = min(box[2] - box[0], box[3] - box[1])
                if any(self._get_dist(h['box'], box) < short_edge for h in hands):
                    self.buffer_count += 1
                    if self.buffer_count >= self.cfg['alcohol_trigger_frames']:
                        self._trigger(11, now_str)
                        self.alcohol_in_progress = True
            else: self.buffer_count = 0

        # --- 8. Step 12: 酒精搓揉 ---
        elif next_step == 12:
            is_active = self.alcohol_in_progress and any(lb in all_labels for lb in scrub_labels)
            self._handle_buffer(is_active, 12, now_str)

        if self.current_step == 12:
            res = self._get_final_data(now_str)
            self.reset()
            return now, res
        
        return now, None

    def _handle_buffer(self, condition, step_num, time_str):
        if condition:
            self.buffer_count += 1
            if self.buffer_count >= self.cfg['trigger_buffer']:
                self._trigger(step_num, time_str)
        else:
            self.buffer_count = 0

    def _trigger(self, step_num, time_str):
        self.flags[step_num-1] = 1
        self.times[step_num-1] = time_str
        self.current_step = step_num
        self.buffer_count = 0 
        logger.info(f"[{self.zone_name}] Step {step_num} Done")

    def _get_final_data(self, end_time):
        res = {"Store ID": 1, "Start Time": self.start_time, "End Time": end_time}
        for i in range(1, 13):
            res[f"Step{i} flag"] = self.flags[i-1]
            res[f"Step{i} time"] = self.times[i-1]
            res[f"Step{i} count"] = self.counts[i-1]
        return res