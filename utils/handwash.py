import numpy as np
from datetime import datetime
from loguru import logger
from .plot import plot_distance, plot_xy, plot_timeout, COLORS, DISTANCE_COLOR



class HandWashTracker:
    def __init__(self, zone_name="Left", logic_cfg=None, ai_class=None, devices=None):
        self.zone_name = zone_name
        if logic_cfg is None:
            raise KeyError(f"[{zone_name}] logic_cfg must be provided in config.yaml under AI.handwash.logic")
        
        self.cfg = logic_cfg['parameter']
        self.no_hand_start_time = datetime.now()
        self.idle_start_time = datetime.now()
        self.logic_classes = logic_cfg['class']
        self.ai_classes = ai_class
        self.devices = devices
        self.logic_labels = {cls: np.array([ai_class.index(name) for name in names]) 
                             for cls, names in self.logic_classes.items()}

        self._check()

        logger.info(f'Tracker logic classes: {self.logic_classes}')
        logger.info(f'Tracker logic labels: {self.logic_labels}')

        self.reset()

    #def update(self, detections):
    def update(self, detections, img):
        # 分組
        hands = detections['box'][np.isin(detections['label'], self.logic_labels['hand'])]
        gloved_hands = detections['box'][np.isin(detections['label'], self.logic_labels['gloved hand'])]
        mask = np.isin(detections['label'], self.logic_labels['handwash'])
        handwashes = detections['box'][mask]
        handwash_labels = detections['label'][mask]
        nail_brush = detections['box'][np.isin(detections['label'], self.logic_labels['nail brush'])]
        paper_towel = detections['box'][np.isin(detections['label'], self.logic_labels['paper towel'])]
        alcohol = detections['box'][np.isin(detections['label'], self.logic_labels['alcohol nozzle'])]
        air_outlet = self.devices['air outlet'][0]
        brush_tray = self.devices['brush tray'][0]
        faucet = self.devices['faucet'][0]
        soap_dispenser = self.devices['soap dispenser'][0]
        sink = self.devices['sink'][0]

        # 時間
        now = datetime.now()
        now_str = f'{now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z'

        # step
        next_step = self.current_step + 1
        if self.current_step == 0:
            self.no_hand_start_time = now
            self.idle_start_time = now
    
        # 沒手超時重置
        elapsed = (now - self.no_hand_start_time).total_seconds()
        plot_timeout(img, 'no hand', elapsed, sink[0:2] + [40, -80], (0, 0, 255))
        timeout = self.cfg['no_hand_timeout']
        if self.current_step > 0 and elapsed > timeout:
            logger.warning(f"[{self.zone_name}] Session timeout (no hand exceed {timeout} seconds), "
                           f"resetting from step {self.current_step}...")
            res = self._get_final_data(now_str)
            self.reset()
            return now, res

        # step 超時重置
        elapsed = (now - self.idle_start_time).total_seconds()
        plot_timeout(img, 'step idle', elapsed, sink[0:2] + [40, -50], (0, 165, 255))
        timeout = self.cfg['step_idle_timeout']
        if self.current_step > 0 and elapsed > timeout:
            logger.warning(f"[{self.zone_name}] Session timeout (step idle {timeout} seconds), "
                           f"resetting from step {self.current_step}...")
            res = self._get_final_data(now_str)
            self.reset()
            return now, res

        # 重置沒手開始時間
        if len(hands) > 0:
            self.no_hand_start_time = now

        # --- Step 1: 沖水(浸濕手) ---
        if next_step == 1:
            
            # 每隻手都要符合以下條件:
            # 1. 手在水槽內
            # 2. 手在水龍頭下方
            # 3. 手離水龍頭夠近
            # 4. 手上不能有 nail brush
            is_active = False
            faucet_mid_y = (faucet[1] + faucet[3]) / 2
            if len(hands) > 0:
                for hand in hands:
                    hand_mid_y = (hand[1] + hand[3]) / 2
                    dist = self._get_dist(hand, faucet)
                    if not (self._is_contained(hand, sink) and \
                            hand_mid_y > faucet_mid_y and \
                            all(not self._is_collided(hand, brush) for brush in nail_brush) and \
                            dist < self.cfg['faucet_dist_thresh']):
                        break
                else:
                    is_active = True

            self._handle_buffer(is_active, next_step, now_str)
            if is_active and self.start_time is None:
                self.start_time = now_str

            # 可視化
            for i, hand in enumerate(hands):
                dist = self._get_dist(hand, faucet)
                plot_distance(img, hand, faucet, dist, DISTANCE_COLOR[i])

        # --- Step 2: 洗手液 ---
        elif next_step == 2:

            # 任一手符合以下條件:
            # 1. 手碰觸洗手液
            # 2. 手要在洗手液下方
            # 3. 手要距離洗手液夠近
            is_active = False
            soap_mid_y = (soap_dispenser[1] + soap_dispenser[3]) / 2
            for hand in hands:
                hand_mid_y = (hand[1] + hand[3]) / 2
                dist = self._get_dist(hand, soap_dispenser)
                if (soap_mid_y < hand_mid_y and \
                    dist < self.cfg['soap_dist_thresh'] and \
                    self._is_collided(hand, soap_dispenser)):
                    is_active = True
                    break

            self._handle_buffer(is_active, next_step, now_str)

            # 可視化
            for i, hand in enumerate(hands):
                dist = self._get_dist(hand, soap_dispenser)
                plot_distance(img, hand, soap_dispenser, dist, DISTANCE_COLOR[i])

        # --- Step 3-6: 搓手動作 ---
        elif 3 <= next_step <= 6 and len(handwash_labels) > 0:

            # 每隻手要符合以下條件:
            # 1. 在手槽內
            is_active = False
            if len(hands) > 0:
                for hand in hands:
                    if not self._is_contained(hand, sink):
                        break
                else:
                    is_active = True

            # 每個洗手動作要符合:
            # 1. 只有一個洗手動作
            # 2. 洗手動作在水槽內
            label = handwash_labels[0]
            is_active &= len(handwash_labels) == 1
            is_active &= self.ai_classes[label] == self.logic_classes['handwash'][next_step - 3]

            self._handle_buffer(is_active, next_step, now_str)

        # --- Step 7: 洗指甲動作 ---
        elif next_step == 7 and len(handwash_labels) > 0:

            # 每隻手要符合以下條件:
            # 1. 在手槽內
            # 2. 手有接觸 nail brush
            is_active = False
            if len(hands) > 0:
                for hand in hands:
                    if not (self._is_contained(hand, sink) and \
                            any(self._is_collided(hand, brush) for brush in nail_brush)):
                        break
                else:
                    is_active = True

            # 每個洗手動作要符合:
            # 1. 只有一個洗手動作
            # 2. 洗手動作在水槽內
            label = handwash_labels[0]
            is_active &= len(handwash_labels) == 1
            is_active &= self.ai_classes[label] == self.logic_classes['handwash'][next_step - 3]

            self._handle_buffer(is_active, next_step, now_str)

        # --- Step 8: 沖水(沖洗洗手液) ---
        if next_step == 8:
            
            # 每隻手都要符合以下條件:
            # 1. 手在水槽內
            # 2. 手在水龍頭下方
            # 3. 手離水龍頭夠近
            # 4. 手上不能有 nail brush
            # 5. 做洗手動作且只有一個
            is_active = False
            faucet_mid_y = (faucet[1] + faucet[3]) / 2
            if len(hands) > 0:
                for hand in hands:
                    hand_mid_y = (hand[1] + hand[3]) / 2
                    dist = self._get_dist(hand, faucet)
                    if not (hand_mid_y > faucet_mid_y and \
                            dist < self.cfg['faucet_dist_thresh'] and \
                            self._is_contained(hand, sink) and \
                            all(not self._is_collided(hand, brush) for brush in nail_brush) and \
                            len(handwashes) == 1 and \
                            self._is_collided(hand, handwashes[0])):
                        break
                else:
                    is_active = True

            self._handle_buffer(is_active, next_step, now_str)

            # 可視化
            for i, hand in enumerate(hands):
                dist = self._get_dist(hand, faucet)
                plot_distance(img, hand, faucet, dist, DISTANCE_COLOR[i])

        # --- Step 9: 擦手紙 ---
        elif next_step == 9:

            # 每隻手都要符合以下條件:
            # 1. 任一衛生紙在水龍頭下方
            # 2. 手有接觸擦手紙
            is_active = False
            if len(hands) > 0:
                faucet_y = (faucet[1] + faucet[3]) / 2.
                for hand in hands:
                    is_valid = False
                    for paper in paper_towel:
                        paper_y = (paper[1] + paper[3]) / 2.
                        if paper_y > faucet_y and self._is_collided(paper, hand):
                            is_valid = True
                            break
                    
                    if not is_valid:
                        break
                else:
                    is_active = True

            self._handle_buffer(is_active, next_step, now_str)

        ## --- Step 10: 殺菌燈 ---
        elif next_step == 10:

            # 手要符合以下條件:
            # 1. 距離 air outlet 要夠近 (每隻手)
            # 2. 與殺菌燈有交集 (任一手)
            is_active = False
            any_valid = False
            if len(hands) > 0:
                for hand in hands:
                    dist = self._get_dist(hand, air_outlet)
                    any_valid |= self._is_collided(hand, air_outlet)
                    if not dist < self.cfg['air_outlet_dist_thresh']:
                        break
                else:
                    is_active = True

            is_active &= any_valid
            self._handle_buffer(is_active, next_step, now_str)

            # 可視化
            for i, hand in enumerate(hands):
                dist = self._get_dist(hand, air_outlet)
                plot_distance(img, hand, air_outlet, dist, DISTANCE_COLOR[i])

        # --- Step 11: 酒精噴灑 ---
        elif next_step == 11:

            # 要符合以下條件:
            # 1. 任一手有接觸酒精
            # 2. 手和任一酒精都要在水龍頭下方的範圍內
            is_active = False
            is_valid = False

            if len(hands) > 0:
                faucet_y = (faucet[1] + faucet[3]) / 2.
                                
                for hand in hands:
                    hand_y = (hand[1] + hand[3]) / 2.
                    if not hand_y > faucet_y:
                        break
                    
                    if not is_valid:
                        for alc in alcohol:
                            alcohol_y = (alc[1] + alc[3]) / 2.
                            if alcohol_y > faucet_y and self._is_collided(hand, alc):
                                is_valid = True
                                break
                else:
                    is_active = True

            is_active &= is_valid
            self._handle_buffer(is_active, next_step, now_str)

        # --- 8. Step 12: 酒精搓揉 ---
        elif next_step == 12:

            # 要符合以下條件:
            # 1. 在做洗手動作
            # 2. 手要在水龍頭下方
            is_active = False
            is_valid = False

            if len(hands) > 0:
                faucet_y = (faucet[1] + faucet[3]) / 2.
                for hand in hands:
                    hand_y = (hand[1] + hand[3]) / 2.
                    if not (len(handwashes) > 0 and \
                            hand_y > faucet_y and \
                            self._is_collided(handwashes[0], hand)):
                        break
                else:
                    is_active = True

            is_active &= len(handwashes) == 1
            self._handle_buffer(is_active, 12, now_str)

        if self.current_step == 12:
            res = self._get_final_data(now_str)
            self.reset()
            return now, res
        
        return now, None

    def reset(self):
        self.current_step = 0  
        self.start_time = None
        self.flags = [0] * 12
        self.times = [""] * 12
        self.counts = [-1] * 12
        for i in range(2, 7): self.counts[i] = 0
        
        self.buffer_count = 0
        self.alcohol_in_progress = False
        self.no_hand_start_time = datetime.now()
        self.idle_start_time = datetime.now()
        logger.info(f"[{self.zone_name}] Tracker Reset.")

    def _handle_buffer(self, condition, step_num, time_str):
        if condition:
            self.idle_start_time = datetime.now()  # 重置
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

    def _is_collided(self, boxA, boxB):
        """ 判斷兩個 BBox 是否有重疊 """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        return interArea > 0

    def _is_contained(self, boxA, boxB):
        """ 判斷 boxB 是否有完全包含 boxA """
        is_inside = (boxA[0] >= boxB[0] and  # x1
                     boxA[1] >= boxB[1] and  # y1
                     boxA[2] <= boxB[2] and  # x2
                     boxA[3] <= boxB[3])     # y2
        return is_inside

    def _get_dist(self, boxA, boxB):
        cA = (boxA[0:2] + boxA[2:4]) / 2.
        cB = (boxB[0:2] + boxB[2:4]) / 2.
        return np.linalg.norm(cA - cB)

    def _get_point_to_line_dist(self, p, l1, l2):
        p, l1, l2 = np.array(p), np.array(l1), np.array(l2)
        line_vec = l2 - l1
        p_vec = p - l1
        line_len_sq = np.sum(line_vec**2)
        t = max(0, min(1, np.dot(p_vec, line_vec) / (line_len_sq + 1e-6)))
        projection = l1 + t * line_vec
        return np.linalg.norm(p - projection)

    def _check(self):
        # 檢查 device
        for dev, boxes in self.devices.items():
            if len(boxes) != 1:
                logger.error(f'amount of {dev} should be 1 but got {len(boxes)} ! ')

