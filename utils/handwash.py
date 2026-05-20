import numpy as np
from numpy.linalg import norm as np_norm
from datetime import datetime, timezone
from time import time
from loguru import logger



class HandWashTracker:
    def __init__(self, zone_name="Left", logic_cfg=None, ai_class=None, devices=None,
                 mqtt = None, pub_freq=10):
        
        self.zone_name = zone_name
        self.cfg = logic_cfg['parameter']
        self.ai_classes = ai_class
        self.devices = devices
        
        # 類別索引
        self.logic_classes = logic_cfg['class']
        self.label_bare_hand = [ai_class.index(n) for n in self.logic_classes['hand']]
        self.label_gloved_hand = [ai_class.index(n) for n in self.logic_classes['gloved hand']]
        
        # 映射 AI 標籤索引
        ai_labels = self.logic_classes['ai_logic_labels']
        self.idx_step1_8 = ai_class.index(ai_labels['step1_8'])
        self.idx_step2 = ai_class.index(ai_labels['step2'])
        self.idx_step9 = ai_class.index(ai_labels['step9'])
        self.idx_step10 = ai_class.index(ai_labels['step10'])
        self.idx_step11 = ai_class.index(ai_labels['step11'])
        
        # Step 3-7
        scrub_names = self.logic_classes['handwash']
        self.label_to_step = {ai_class.index(name): i + 3 for i, name in enumerate(scrub_names)}

        # mqtt
        self.mqtt = mqtt
        self.pub_period = 1. / pub_freq
        self.pub_time = float('-inf')

        #
        self.is_no_hand_timeout = False

        self.reset()

    def reset(self):
        self.start_time = None
        self.flags = [0] * 12
        self.trigger_times = [""] * 12
        self.counts = [0] * 12
        
        for i in range(2, 7): self.counts[i] = 0
        
        self.step_sequence = []
        self.last_step_trigger_times = {}
        self.gloved_sequence = []
        
        # Buffer 初始化
        self.collision_buffers = {i: 0 for i in range(1, 13)}
        self.durations = {i: 0.0 for i in range(1, 13)}
        self.max_durations = {i: 0.0 for i in range(1, 13)}
        self.step_start_times = {i: 0.0 for i in range(1, 13)}

        self.current_scrub_label = None
        self.scrub_frame_counter = 0
        
        self.no_hand_start_time = datetime.now(timezone.utc)
        self.has_soaped = False 
        self.temp_continuous_collisions = [0] * 12 
        self.now_dt = None       
        self.finish_reason = None

        self._reset_scrub_vars()

        #self._clear_debug_info()
        
    def update(self, detections, img):
        self.now_dt = datetime.now(timezone.utc)
        
        self._clear_debug_info()

        h, w = img.shape[:2]
        frame_area = h * w
        
        # 1. 抓出手部與過濾
        hand_mask = np.isin(detections['label'], self.label_bare_hand + self.label_gloved_hand)
        hands = detections['box'][hand_mask]
        
        hand_areas = (hands[:, 2] - hands[:, 0]) * (hands[:, 3] - hands[:, 1])
        valid_mask = (hand_areas / frame_area) <= self.cfg['max_hand_ratio']
        valid_hands = hands[valid_mask]

        if len(valid_hands) > 0:
            self.debug_info['status'] = "Hand Detected"
            self.no_hand_start_time = self.now_dt
            if not self.start_time:
                self.start_time = self._get_utc_now()
        else:
            self.debug_info['status'] = "No Hand"
            elapsed = (self.now_dt - self.no_hand_start_time).total_seconds()
            if self.start_time and elapsed > self.cfg['no_hand_timeout']:
                self.finish_reason = 'no hand timeout'
                self.update_debug_info()
                
                # 如果 no hand timeout 的狀態沒解除, 不連續發送 reset
                #if not self.is_no_hand_timeout:
                #    self._publish_status(self.mqtt.pub_topics['system'], 'Reset')
                self._publish_status(self.mqtt.pub_topics['system'], 'Reset')
                return self.now_dt, self._finalize_session()
            return self.now_dt, None

        is_gloved, _ = self._verify_glove(detections)

        # --- 以下為各別實現的 AI 邏輯 ---
        self._logic_step_1_8(detections, is_gloved)
        self._logic_step_2(detections, is_gloved)
        #self._logic_step_3_7(valid_hands, detections, is_gloved)
        self._logic_step_3_7(detections, is_gloved)
        self._logic_step_9(detections, is_gloved)
        self._logic_step_10(detections, is_gloved)
        self._logic_step_11(detections, is_gloved)
        self._logic_step_12(valid_hands, is_gloved) # 依賴雙手重疊，暫不更動

        # 更新 debug 資料
        self.update_debug_info()

        # 自動結案
        if all(f == 1 for f in self.flags):
            self.finish_reason = 'all flags are 1'
            # 發送訊息給應用端
            self._publish_status(self.mqtt.pub_topics['system'], 'Reset')
            return self.now_dt, self._finalize_session()

        # 發送訊息給應用端
        self._publish_status(self.mqtt.pub_topics['process'], 'status')

        return self.now_dt, None

    def _finalize_session(self):
        """ 結束 Session 並回傳資料，若完全無進度則放棄寫入 """
        end_time_str = self._get_utc_now()
        
        # 需求：過濾完全沒更新的紀錄
        # 如果 flags 全部都是 0，代表 12 個步驟一個都沒達成
        if sum(self.flags) == 0:
            logger.info(f"[{self.zone_name}] Session timed out with no progress. Skipping CSV write.")
            self.reset()
            return None 

        # 正常寫入邏輯
        final_data = self._get_final_data(end_time_str)

        # 結果
        n_valid = sum(self.flags)
        if n_valid == len(self.flags):
            logger.success(f"[{self.zone_name}] Session completed. Saving to CSV.")
        else:
            logger.warning(f"[{self.zone_name}] Session completed with only {n_valid} steps! "
                           f"Saving to CSV.")
        
        # 重置 
        self.reset()
        return final_data

    def _logic_step_1_8(self, detections, is_gloved):
        """ Step 1 & 8: 基於 AI 偵測到 handwash """
        step = 8 if self.has_soaped else 1
        if self.idx_step1_8 in detections['label']:
            self.collision_buffers[step] += 1
            self.compute_step_duration(step)
            if self.collision_buffers[step] == self.cfg['trigger_step1_8_buffer']:
                self._update_record(step, is_gloved)
        else:
            # 洗掉肥皂
            num_collision = self.collision_buffers[step]
            if self.has_soaped and num_collision >= self.cfg['trigger_step1_8_buffer']:
                self.has_soaped = False
            self.collision_buffers[step] = 0  # 重置
            self.durations[step] = 0.0
            self.step_start_times[step] = 0.0

    def _logic_step_2(self, detections, is_gloved):
        """ Step 2: 基於 AI 偵測到 trigger soap dispenser """
        if self.idx_step2 in detections['label']:
            self.collision_buffers[2] += 1
            self.compute_step_duration(2)
            if self.collision_buffers[2] == self.cfg['trigger_step2_buffer']:
                self._update_record(2, is_gloved)
                self.has_soaped = True 
        else:
            self.collision_buffers[2] = 0
            self.durations[2] = 0.0
            self.step_start_times[2] = 0.0

    #def _logic_step_3_7(self, hands, detections, is_gloved):
    #    if len(hands) == 0:
    #        #self._reset_scrub_vars()
    #        return
#
    #    # 依照信心分數排序
    #    #hand_indices = hand_indices[np.argsort(detections['score'][hand_indices])[::-1]]
#
    #    # 判定參考框
    #    if len(hands) >= 2:
    #        h1, h2 = hands[0], hands[1]
    #    else:
    #        h1 = hands[0]
    #        h2 = hands[0] # 傳入相同框觸發單手位移邏輯
#
    #    target_step = None
    #    
    #    # 洗手動作
    #    mask = np.isin(detections['label'], list(self.label_to_step.keys()))
    #    handwash = detections['box'][mask]
    #    label = detections['label'][mask]
#
    #    # 必須唯一
    #    if len(handwash) != 1:
    #        return
#
    #    # 是否確實在做洗手 (iou 判斷)
    #    handwash = handwash[0]
    #    label = label[0]
    #    if all(self.get_iou(h, handwash) > self.cfg['hand_wash_iou'] for h in hands):
    #        target_step = label
#
    #    # 狀態機與計次觸發
    #    if target_step:
    #        if target_step == self.current_scrub_label:
    #            self.scrub_frame_counter += 1
    #        else:
    #            self.scrub_frame_counter = 1
    #            self.current_scrub_label = target_step
    #            self._reset_scrub_vars()
#
    #        if self.scrub_frame_counter >= self.cfg['scrub_min_frames']:
    #            self._do_scrub_count(target_step, h1, h2, is_gloved)
    #    else:
    #        pass
    #        #self.scrub_frame_counter = 0
    #        #self.current_scrub_label = None
    #        #self._reset_scrub_vars()

    def _logic_step_3_7(self, detections, is_gloved):
        """ 
        Step 3-7: 基於幀數的連續計數機制
        """
        # 1. 找出當前畫面中是否有屬於 Step 3-7 的 Label
        mask = np.isin(detections['label'], list(self.label_to_step.keys()))
        active_labels = detections['label'][mask]

        target_step = None
        if len(active_labels) > 0:
            # 取第一個（通常是信心分數最高者）
            detected_label = active_labels[0] 
            target_step = self.label_to_step[detected_label]

        # 2. 更新狀態機
        if target_step is not None:
            # 如果換了動作，上一波的連續計數要中斷
            if target_step != self.current_scrub_label:
                if self.current_scrub_label is not None:
                    self.temp_continuous_collisions[self.current_scrub_label - 1] = 0
                self.current_scrub_label = target_step
            
            # 呼叫計數邏輯（傳入 True 代表偵測中）
            self._do_scrub_count(target_step, is_gloved, detected=True)
            
            # 重置其他步驟的 Buffer 與連續計數（嚴格模式：一次只能做一件事）
            for s in range(3, 8):
                if s != target_step:
                    self.collision_buffers[s] = 0
                    self.temp_continuous_collisions[s-1] = 0
                    self.durations[s] = 0.0
                    self.step_start_times[s] = 0.0
        else:
            # 畫面上沒東西，重置當前動作的連續計數
            if self.current_scrub_label is not None:
                self.temp_continuous_collisions[self.current_scrub_label - 1] = 0
                self.collision_buffers[self.current_scrub_label] = 0
                self.durations[self.current_scrub_label] = 0.0
                self.step_start_times[self.current_scrub_label] = 0.0
            self.current_scrub_label = None

    def _logic_step_9(self, detections, is_gloved):
        """ Step 9: 基於 AI 偵測到 wipe hands with tissue """
        if self.idx_step9 in detections['label']:
            self.collision_buffers[9] += 1
            self.compute_step_duration(9)
            if self.collision_buffers[9] == self.cfg['trigger_step9_buffer']:
                self._update_record(9, is_gloved)
        else:
            self.collision_buffers[9] = 0
            self.durations[9] = 0.0
            self.step_start_times[9] = 0.0

    def _logic_step_10(self, detections, is_gloved):
        """ Step 10: 基於 AI 偵測到 UV sterilization """
        if self.idx_step10 in detections['label']:
            self.collision_buffers[10] += 1
            self.compute_step_duration(10)
            if self.collision_buffers[10] == self.cfg['trigger_step10_buffer']:
                self._update_record(10, is_gloved)
        else:
            self.collision_buffers[10] = 0
            self.durations[10] = 0.0
            self.step_start_times[10] = 0.0

    def _logic_step_11(self, detections, is_gloved):
        """ Step 11: 基於 AI 偵測到 spray alcohol """
        if self.idx_step11 in detections['label']:
            self.collision_buffers[11] += 1
            self.compute_step_duration(11)
            if self.collision_buffers[11] == self.cfg['trigger_step11_buffer']:
                self._update_record(11, is_gloved)
        else:
            self.collision_buffers[11] = 0
            self.durations[11] = 0.0
            self.step_start_times[11] = 0.0

    def _logic_step_12(self, hands, is_gloved):
        """ 
        Step 12: 依舊維持雙手重疊判定 
        修正：必須緊接在 Step 11 之後觸發才有效 (中間不可穿插其他步驟)
        """
        # 條件 1: 檢查 step_sequence 的最後一個步驟是否為 11
        # 這樣可以確保 Step 12 是緊接在 11 之後，且中間沒有別的有效步驟
        is_after_step11 = len(self.step_sequence) > 0 and self.step_sequence[-1] == 11
        
        if is_after_step11 and len(hands) >= 2:
            if self.get_iou(hands[0], hands[1]) > self.cfg['scrub_overlap_thresh']:
                self.collision_buffers[12] += 1
                self.compute_step_duration(12)
                if self.collision_buffers[12] == self.cfg['trigger_step12_buffer']:
                    self._update_record(12, is_gloved)
                return
        self.collision_buffers[12] = 0
        self.durations[12] = 0.0
        self.step_start_times[12] = 0.0

    #def _do_scrub_count(self, step_num, box1, box2, is_gloved):
    #    """ 需求 10: 完整的連續最高次數計數邏輯 """
    #    _, move_detected = self._calculate_movement(box1, box2)
    #    
    #    if move_detected:
    #        # 這裡的邏輯是：方向切換一次算 0.5 次，來回算 1 次
    #        # 為了簡化，您可以根據需求調整
    #        self.temp_continuous_counts[step_num-1] += 1
    #        
    #        # 需求 10: 只有目前這波「連續次數」超過歷史最高，才更新 counts
    #        if self.temp_continuous_counts[step_num-1] > self.counts[step_num-1]:
    #            self.counts[step_num-1] = self.temp_continuous_counts[step_num-1]
    #            
    #        # 檢查是否達到 Flag 門檻
    #        if self.counts[step_num-1] == self.cfg['scrub_flag_count']:
    #            self._update_record(step_num, is_gloved)
    #    else:
    #        # 動作停下來了，連續計數中斷
    #        self.temp_continuous_counts[step_num-1] = 0
    #        #self._reset_scrub_vars()

    def _do_scrub_count(self, step_num, is_gloved, detected=False):
        """ 
        保留 temp_count 機制：
        temp_continuous_collisions 記錄「這一波」連續偵測到的幀數。
        self.counts 記錄該步驟「歷史最高」的連續幀數。
        """
        if detected:
            # 增加 Buffer (用於判斷是否正在觸發)
            self.collision_buffers[step_num] += 1
            
            # 計算持續時間
            self.compute_step_duration(step_num)

            # 增加這一波的連續計數
            idx = step_num - 1
            self.temp_continuous_collisions[idx] += 1
            
            # 需求 10：只有目前這波超過歷史最高，才更新 counts
            name = f'step{step_num}_frame_scrub_ratio'
            current_count = self.temp_continuous_collisions[idx] // self.cfg[name]
            if current_count > self.counts[idx]:
                self.counts[idx] = current_count
                
            # 檢查是否達到 Flag 門檻
            if self.counts[idx] == self.cfg[f'step{step_num}_min_scrub']:
                self._update_record(step_num, is_gloved)
        else:
            # 偵測中斷，該波計數歸零
            #self.temp_continuous_collisions[step_num - 1] = 0
            #self.collision_buffers[step_num - 1] = 0
            pass

    def _calculate_movement(self, box1, box2):
        """
        支援雙手相對位移 或 單手自身位移 (修正類型衝突 Bug)
        """
        # 1. 取得當前的特徵值 (Current Value)
        if np.array_equal(box1, box2):
            # 單手模式：特徵值是中心點座標 [x, y] (ndarray)
            curr_val = (box1[:2] + box1[2:]) / 2
            # 存入 debug 資訊 (單手存座標)
            self.debug_info['hand_dist'] = 0.0 
            self.debug_info['hand_center'] = curr_val.tolist()
        else:
            # 雙手模式：特徵值是兩手中心點的「距離」 (float)
            c1 = (box1[:2] + box1[2:]) / 2
            c2 = (box2[:2] + box2[2:]) / 2
            curr_val = float(np_norm(c1 - c2))
            # 存入 debug 資訊 (雙手存距離)
            self.debug_info['hand_dist'] = curr_val
            self.debug_info['hand_center'] = None

        move_detected = False
        
        # 2. 與上一幀比較
        if self.prev_dist is not None:
            # --- 關鍵修正：檢查類型是否一致，若不一致則重置並跳過 ---
            if type(curr_val) != type(self.prev_dist):
                self.prev_dist = curr_val
                self.last_dir = 0
                self.move_acc = 0.0
                return curr_val, False

            # 3. 判斷差異 (diff)
            if isinstance(curr_val, np.ndarray):
                # 單手模式：計算座標位移向量的長度
                diff_vec = curr_val - self.prev_dist
                diff = np_norm(diff_vec)
                # 為了判定方向 (last_dir)，我們取位移最大的軸向 (以 y 軸為例)
                cur_dir = 1 if diff_vec[1] > self.cfg['move_dir_thresh'] else (-1 if diff_vec[1] < -self.cfg['move_dir_thresh'] else 0)
            else:
                # 雙手模式：計算距離的變化量
                diff = curr_val - self.prev_dist
                cur_dir = 1 if diff > self.cfg['move_dir_thresh'] else (-1 if diff < -self.cfg['move_dir_thresh'] else 0)
            
            # 4. 累積位移與方向判定
            if cur_dir != 0:
                if cur_dir != self.last_dir:
                    # 方向改變，清空累積位移重新計算
                    self.move_acc = 0.0
                
                self.move_acc += abs(diff)
                self.last_dir = cur_dir

                # 位移量達標判定 (門檻維持 5.0)
                if self.move_acc > 5.0: 
                    move_detected = True
                    
        # 5. 更新緩存
        self.prev_dist = curr_val
        return curr_val, move_detected

    def _update_record(self, step_num, is_gloved):
        utc_now_dt = self.now_dt
        utc_now_str = self._get_utc_now()

        if is_gloved:
            if not self.gloved_sequence or self.gloved_sequence[-1] != step_num:
                self.gloved_sequence.append(step_num)
            return

        # 1. 檢查是否與序列最後一個步驟相同
        is_same_as_last = len(self.step_sequence) > 0 and self.step_sequence[-1] == step_num
        
        if is_same_as_last:
            # 2. 如果相同，檢查時間間隔
            last_time = self.last_step_trigger_times.get(step_num)
            if last_time:
                elapsed = (utc_now_dt - last_time).total_seconds()
                if elapsed < self.cfg['min_repeat_interval']:
                    # 連續且間隔太短 -> 忽略
                    self.debug_info['is_same_as_last_and_fast'] = True
                    return
                
        if self.flags[step_num-1] == 0:
            self.flags[step_num-1] = 1
            self.trigger_times[step_num-1] = utc_now_str
        
        if not self.step_sequence or self.step_sequence[-1] != step_num:
            self.step_sequence.append(step_num)
            logger.info(f"[{self.zone_name}] VALIDATED: Step {step_num}")

    def _verify_glove(self, detections):
        g_boxes = detections['box'][np.isin(detections['label'], self.label_gloved_hand)]
        b_boxes = detections['box'][np.isin(detections['label'], self.label_bare_hand)]
        
        if len(g_boxes) == 0: 
            return False, np.array([False] * len(b_boxes))
        
        gloved_mask = []
        for gb in g_boxes:
            for bb in b_boxes:
                gloved_mask.append(self._calculate_iou(gb, bb) > self.cfg['glove_iou_thresh'])
        gloved_mask = np.array(gloved_mask).reshape(len(g_boxes), -1).any(0)
        return gloved_mask.any(), gloved_mask

    def _calculate_iou(self, b1, b2):
        xA, yA, xB, yB = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _get_final_data(self, end_time):
        res = {"Store ID": "test", "Start Time": self.start_time, "End Time": end_time,
               "Actual Sequence": self.step_sequence, "Gloved Action Sequence": self.gloved_sequence}
        for i in range(1, 13):
            res[f"Step{i} flag"] = self.flags[i-1]
            res[f"Step{i} time"] = self.trigger_times[i-1]
            res[f"Step{i} count"] = self.counts[i-1]
        for i in range(3, 8):
            res[f'Step{i} min count'] = self.cfg[f'step{i}_min_scrub']
        res['Finish reason'] = self.finish_reason
        return res
    
    def _get_utc_now(self):
        return self.now_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _clear_debug_info(self):
        """ 每一幀開始時重置 Debug 資訊 """
        self.debug_info = {
            "status": "No Hand",
            "move_acc": self.move_acc,
            "flags": self.flags.copy(),
            "counts": self.counts.copy(),
            "active_buffers": self.collision_buffers.copy(),
            "durations": self.durations.copy(),
            "max_durations": self.max_durations.copy(),
            "hand_dist": 0.0,
            "hand_center": None,
            "is_same_as_last_and_fast": False,
            "sent_msg": None
        }

    def update_debug_info(self):
        self.debug_info['move_acc'] = self.move_acc
        self.debug_info['flags'] = self.flags.copy()
        self.debug_info['counts'] = self.counts.copy()
        self.debug_info['active_buffers'] = self.collision_buffers.copy()

    @staticmethod
    def get_iou(box1, box2):
        """ 工具函式：計算兩框之交集 / 聯集 (IoU) """
        xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
        xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    @staticmethod
    def is_contained(outer_box, inner_box_or_pt):
        """ 檢查 outer_box 是否包含指定的點或框 """
        if len(inner_box_or_pt) == 2: # 點 [x, y]
            x, y = inner_box_or_pt
            return outer_box[0] <= x <= outer_box[2] and outer_box[1] <= y <= outer_box[3]
        else: # 框 [x1, y1, x2, y2]
            return (inner_box_or_pt[0] >= outer_box[0] and inner_box_or_pt[1] >= outer_box[1] and 
                    inner_box_or_pt[2] <= outer_box[2] and inner_box_or_pt[3] <= outer_box[3])

    def _reset_scrub_vars(self):
        """ 重置位移判定相關變數 (需求 8) """
        self.prev_dist = None
        self.last_dir = 0
        self.move_acc = 0.0

    def _get_intersection_ratio(self, box_hand, box_device, mode='device'):
        """ 計算交集佔比。mode='device' 表示 交集/設備面積 """
        xA, yA = max(box_hand[0], box_device[0]), max(box_hand[1], box_device[1])
        xB, yB = min(box_hand[2], box_device[2]), min(box_hand[3], box_device[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if mode == 'device':
            denom = (box_device[2] - box_device[0]) * (box_device[3] - box_device[1])
        else:
            denom = (box_hand[2] - box_hand[0]) * (box_hand[3] - box_hand[1])
        return inter / max(1.0, denom)

    def compute_step_duration(self, step_num):
        """ 計算步驟的持續時間 """
        now = time()
        
        # 如果這個步驟剛開始（起點為 0），就記錄現在的時間點
        if self.step_start_times[step_num] == 0.0:
            self.step_start_times[step_num] = now
            
        # 總持續時間 = 當前時間 - 開始時間點
        self.durations[step_num] = now - self.step_start_times[step_num]
        if self.durations[step_num] > self.max_durations[step_num]:
            self.max_durations[step_num] = self.durations[step_num]

    def _create_mqtt_message(self, cmd):
        if cmd == 'Reset':
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'Alarm':
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'AlarmCancel':
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'status':
            # 找最多的 collision buffer 的 step, 為了要一次只會傳一個 step 的狀態
            step = max(self.collision_buffers, key=self.collision_buffers.get)
            if self.collision_buffers[step] == 0:
                return
            
            #logger.info(f'the most continuous buffers in this frame is step{step} !')

            msgs = {
                "step_id": f"Step{step}",
                "washcount": str(self.counts[step - 1]),
                "washtime": str(self.durations[step]),
                "side": self.zone_name.lower()
            }
        else:
            logger.error(f'unknow command: {cmd} !')

        return msgs

    def _publish_status(self, topic, cmd):
        msg = self._create_mqtt_message(cmd)
        now = time()
        if msg and now - self.pub_time >= self.pub_period:
            self.debug_info['sent_msg'] = self.mqtt.publish(topic, msg)
            self.pub_time = now
