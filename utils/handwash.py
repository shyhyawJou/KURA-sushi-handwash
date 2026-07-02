import numpy as np
from numpy.linalg import norm as np_norm
from time import time, sleep
from threading import Event
from loguru import logger
from .tool import get_iou, get_now_str
from .step import Step_History, MyDict



class HandWashTracker:
    def __init__(self, zone_name, logic_cfg, sys_cfg, ai_class, mqtt = None, pub_freq=10):
        # config
        self.cfg = logic_cfg['handwash_parameter']
        self.time_cfg = logic_cfg['time_parameter']
        self.sys_cfg = sys_cfg['stages']
        self.login_mode = logic_cfg['login']

        # 重要變數
        self.zone_name = zone_name
        self.ai_classes = ai_class
        self.logic_classes = logic_cfg['class']
        self.label_bare_hand = [ai_class.index(n) for n in self.logic_classes['hand']]
        self.label_gloved_hand = [ai_class.index(n) for n in self.logic_classes['gloved hand']]
        self.label_scrub_hand = [ai_class.index(self.cfg['step_name'][i]) 
                                 for i, cfg in enumerate(self.sys_cfg, 1) 
                                 if cfg['washcountmax'] > 0]
        self.step_labels = {i: ai_class.index(name) for i, name in self.cfg['step_name'].items() 
                            if name in ai_class}
        self.srcub_steps = {i for i, cfg in enumerate(self.sys_cfg, 1) if cfg['washcountmax'] > 0}
        assert self.srcub_steps == self.cfg['scrub_count_ratio'].keys()
        self.scrub_count_ratio = {i: (self.cfg['scrub_count_ratio'][i] if i in self.srcub_steps else -1) 
                                  for i in range(1, 13)}
        self.is_no_hand = False

        # mqtt
        self.mqtt = mqtt
        self.pub_period = 1. / pub_freq

        # 初始化
        self.reset()

    def reset(self):
        self.has_soaped = False 
        self.now = time()       
        self.finish_reason = None
        self.detecting_step = self.cfg['init_step_num']
        self.pub_time = float('-inf')
        self.debug_info = {}
        self.is_login = False
        self.sent_msg = None
        self.no_hand_elapsed = -1
        self.has_hand_elapsed = -1
        self.no_hand_start_time = None
        self.has_hand_start_time = None
        self.pub_no_hand = False
        self.pub_no_hand_zero = False
        self.saved_steps = []
        self.is_final = False  # 重置訊號
        
        # 使用者資訊
        self.user_id = None
        self.user_name = None
        
        # 洗手歷史資訊
        self.steps = Step_History()

        # 當下洗手資訊
        self.frames = MyDict({i: 0 for i in range(1, 13)})
        self.idle_frames = MyDict({i: 0 for i in range(1, 13)})
        self.counts = MyDict({i: 0 for i in range(1, 13)})
        self.start_times = MyDict({i: None for i in range(1, 13)})
        self.step_confirmed_times = MyDict({i: None for i in range(1, 13)})
        self.end_times = MyDict({i: None for i in range(1, 13)})
        self.step_confirmed = MyDict({i: False for i in range(1, 13)})
        self.last_start_times = MyDict({i: None for i in range(1, 13)})
        self.durations = MyDict({i: 0 for i in range(1, 13)})
        self.is_detecting_steps = MyDict({i: False for i in range(1, 13)})
        self.is_switch_step = False
        self.next_step = None

        # debug info
        self._update_debug_info()
        logger.info(f'[{self.zone_name}] Reset all handwash information ! '
                    f'Detecting step become: {self.detecting_step} !')

    def update(self, detections, img, now):
        self.now = now
        
        export_data = None
        self.sent_msg = None
        self.saved_steps = []

        # barcode 模式下且沒登入
        if self.login_mode == 'barcode' and not self.is_login:
            return None

        # 手
        hand_mask = np.isin(detections['label'], self.label_bare_hand + self.label_gloved_hand)
        hands = detections['box'][hand_mask]

        # 檢測每個步驟
        self._check_step1_to_11(detections, hands)
        self._check_step12(detections, hands)

        # 有手沒手
        if len(hands) == 0:
            # reset
            self.has_hand_start_time = None
            self.has_hand_elapsed = -1

            # no hand start time
            if self.no_hand_start_time is None:
                self.no_hand_start_time = self.now

            # no hand elapsed
            self.no_hand_elapsed = self.now - self.no_hand_start_time
            
            # publish no hand
            if self.is_login and self.no_hand_elapsed >= self.time_cfg['pub_hand_delay']:
                self._publish_status(self.mqtt.pub_topics['system'], 'Reset', fatal=not self.pub_no_hand)

            # UI become logout
            if self.is_login and self.pub_no_hand_zero:
                self.is_final = True
                self.finish_reason = 'No hand'
        else:
            # reset
            self.no_hand_start_time = None
            self.no_hand_elapsed = -1

            # has hand start time
            if self.has_hand_start_time is None:
                self.has_hand_start_time = self.now

            # has hand elapsed
            self.has_hand_elapsed = self.now - self.has_hand_elapsed

            # publish has hand
            delay = self.time_cfg['pub_hand_delay']
            if self.is_login and self.pub_no_hand and self.has_hand_elapsed >= delay:
                self._publish_status(self.mqtt.pub_topics['system'], 'ResetCancel', fatal=True)

            # AI login
            is_hand_login = self.login_mode == 'hand'
            if not self.is_login and is_hand_login and self.has_hand_elapsed >= delay:
                self._publish_status(self.mqtt.pub_topics['system'], 'AILogin', fatal=True)
                self.is_login = True

        # 狀態
        if self.is_login:
            self._publish_status(self.mqtt.pub_topics['process'], 'status', fatal=False)

        # 更新 debug 資料
        self._update_debug_info(hands)

        # 切換步驟
        if self.is_switch_step:
            self._switch_step()
            self.is_switch_step = False
            self.next_step = None

        # 是否結束
        if self.is_final:
            export_data = self.stop()

        return export_data

    def _check_step1_to_11(self, detections, hands):
        mask = np.isin(detections['label'], list(self.step_labels.values()))
        step_boxes = detections['box'][mask]
        step_labels = detections['label'][mask]

        for i in range(1, 12):
            # step 1 和 8 只需要判斷其中一個, 用是否有沾過肥皂區分
            if i == (1 if self.has_soaped else 8):
                continue

            mask = step_labels == self.step_labels[i]
            if len(hands) > 0 and np.any(mask):
                self._do_step(i)
                self._do_scrub_count(i)
            else:
                self._undo_step(i)

    def _check_step12(self, detections, hands):
        # 忽略不是剛噴完酒精
        if len(self.steps) == 0 or self.steps[-1].id != 11:
            return
        
        scrub_hand_mask = np.isin(detections['label'], self.label_scrub_hand)
        
        # 有做洗手動作或兩手有交集都算有效
        if len(hands) > 0 and np.any(scrub_hand_mask):
            self._do_step(12)
        elif len(hands) >= 2:
            ious = get_iou(hands, hands)
            np.fill_diagonal(ious, 0)
            if np.any(ious >= self.cfg['scrub_overlap_thresh']):
                self._do_step(12)
        else:
            self._undo_step(12)
        
    def _do_step(self, step_id):
        self.frames[step_id] += 1
        self.end_times[step_id] = self.now
        self.idle_frames[step_id] = 0

        # 第一次
        if self.frames[step_id] == 1:
            self.start_times[step_id] = self.now
            self.is_detecting_steps[step_id] = step_id == self.detecting_step

        # 滿足動作確認條件
        if self.frames[step_id] == self.cfg['action_frame']:
            self.step_confirmed[step_id] = True
            self.step_confirmed_times[step_id] = self.now
            logger.debug(f'[{self.zone_name}] Step {step_id}: action confirmed !')
            
        # 計算做了多久
        self._compute_step_duration(step_id)  

    def _undo_step(self, step_id):        
        # 必要處理
        self.last_start_times[step_id] = None

        # 跳過不處理
        if self.frames[step_id] == 0:
            return
        if step_id == self.detecting_step and self.step_confirmed[step_id]:
        #if step_id == self.detecting_step:
            return

        # idle 處理
        self.idle_frames[step_id] += 1
        if self.idle_frames[step_id] == self.cfg['action_frame'] // 4:
            # 儲存
            if self.step_confirmed[step_id]:
                self._update_record(step_id)

                # 肥皂
                if step_id == 2:
                    self.has_soaped = True
                    logger.info(f'[{self.zone_name}] Soaped !')
                elif step_id == 8:
                    self.has_soaped = False
                    logger.info(f'[{self.zone_name}] Rinsed !')

            # reset
            self.reset_step_info(step_id)

    def _do_scrub_count(self, step_id):
        if step_id not in self.srcub_steps:
            return
        frame = max(self.frames[step_id] - self.cfg['action_frame'], 0)
        self.counts[step_id] = frame // self.scrub_count_ratio[step_id]

    def _compute_step_duration(self, step_id):
        if self.last_start_times[step_id] is None:
            self.last_start_times[step_id] = self.now
        is_confirmed = self.step_confirmed[step_id]
        confirmed_time = self.step_confirmed_times[step_id]
        if is_confirmed and confirmed_time != self.now:
            delta_t = max(self.now - self.last_start_times[step_id], 0)
        else:
            delta_t = 0
        self.durations[step_id] += delta_t
        self.last_start_times[step_id] = self.now

    def _update_record(self, step_id):
        if not self.step_confirmed[step_id]:
            return
        self.steps.append(step_id, self.counts[step_id], self.start_times[step_id], 
                          self.end_times[step_id], self.step_confirmed_times[step_id], 
                          self.durations[step_id], self.frames[step_id], 
                          int(self.is_detecting_steps[step_id]))
        self.saved_steps.append(step_id)
        logger.info(f'[{self.zone_name}] Add Step {step_id} into step sequence !')
        logger.debug(f'[{self.zone_name}] last step in sequence: {self.steps[-1]}')

    def _get_final_data(self):
        if len(self.steps) == 0:
            return {}

        res = {
            "Store ID": "test", 
            "User ID": '' if self.login_mode == 'hand' else str(self.user_id),
            "User Name": '' if self.login_mode == 'hand' else str(self.user_name),
            "Step Sequence": self.steps.ids.copy(),
            "Start Time": [get_now_str(t) for t in self.steps.start_times], 
            "Action Confirmed Time": [get_now_str(t) for t in self.steps.step_confirmed_times], 
            "End Time": [get_now_str(t) for t in self.steps.end_times],
            "Step Count": self.steps.counts.copy(),
            "Is Detecting Step": self.steps.is_detecting_steps.copy(),
            'Duration': self.steps.durations.copy()
        }
        res['Finish reason'] = self.finish_reason
        res['Region'] = self.zone_name.lower()
        for i in range(1, 13):
            res[f'Step{i} min count'] = self.sys_cfg[i-1]['washcountmax']
            res[f'Step{i} min time'] = self.sys_cfg[i-1]['washtimemax']
        return res

    def _finalize_session(self):
        """ 結束 Session 並回傳資料 """
        final_data = self._get_final_data()
        n_step = len(self.steps)
        if n_step == 0:
            logger.debug(f'[{self.zone_name}] Completed with no any step! Skip to save CSV !')
        else:
            logger.info(f'[{self.zone_name}] Completed with {n_step} steps! ')
        return final_data

    def _update_debug_info(self, hands=[]):
        self.debug_info['status'] = 'Hand Detected' if len(hands) > 0 else 'No Hand'
        self.debug_info['frames'] = self.frames
        self.debug_info['counts'] = self.counts
        self.debug_info['durations'] = self.durations
        self.debug_info['start_times'] = self.start_times
        self.debug_info['step_confirmed_times'] = self.step_confirmed_times
        self.debug_info['step_confirmed'] = self.step_confirmed
        self.debug_info['last_start_times'] = self.last_start_times
        self.debug_info['detected_steps'] = self.steps.detected_steps
        self.debug_info['detecting_step'] = self.detecting_step
        self.debug_info['sent_msg'] = self.sent_msg
        self.debug_info['saved_steps'] = self.saved_steps
        self.debug_info['now'] = self.now

    def reset_step_info(self, step_id):
        self.frames[step_id] = 0
        self.idle_frames[step_id] = 0
        self.counts[step_id] = 0
        self.start_times[step_id] = None
        self.end_times[step_id] = None
        self.step_confirmed_times[step_id] = None
        self.step_confirmed[step_id] = False
        self.last_start_times[step_id] = None
        self.durations[step_id] = 0
        self.is_detecting_steps[step_id] = False
        logger.debug(f"[{self.zone_name}] Detecting step {step_id}'s data is reset !")

    def _publish_status(self, topic, cmd, fatal=False):
        msg = self._create_mqtt_message(cmd)
        if msg and (fatal or self.now - self.pub_time >= self.pub_period):
            if fatal or msg.get('cmd') == 'Reset' and float(msg.get('time')) == 0:
                if msg.get('cmd') == 'Reset' and float(msg.get('time')) == 0:
                    level = 'WARNING'
                else:
                    level = 'INFO'
            else:
                level = 'TRACE'
            self.sent_msg = self.mqtt.publish(topic, msg, level) 
            
            # 有發送訊息
            if self.sent_msg is not None:
                if cmd == 'Reset':
                    self.pub_no_hand = True
                    if float(msg['time']) == 0:
                        self.pub_no_hand_zero = True
                elif cmd == 'ResetCancel':
                    self.pub_no_hand = False
                elif cmd == 'status':
                    if self.detecting_step == 12 and msg['trigger']:
                        self.is_final = True
                        self.finish_reason = 'all completed'

            # 控制發送頻率
            if cmd == 'status':
                self.pub_time = self.now

    def _create_mqtt_message(self, cmd):
        if cmd == 'Reset':
            timeout = self.sys_cfg[max(self.detecting_step-1, 0)]['timeoutmax']
            delay = self.time_cfg['pub_hand_delay']
            remain = max(timeout - (self.no_hand_elapsed - delay), 0)
            msgs = {"cmd": cmd, "side": self.zone_name.lower(), "time": str(remain)}
        elif cmd == 'ResetCancel':
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'BackLogin':  # 12 步驟 reset
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'AILogin':
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'Alarm':
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'AlarmCancel':
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'status':
            if not self.durations.get(self.detecting_step):
                return
            step = self.detecting_step
            msgs = {
                "step_id": f"Step{step}",
                "washcount": str(self.counts[step]),
                "washtime": str(self.durations[step]),
                "side": self.zone_name.lower(),
                "trigger": self.step_confirmed[step]
            }
        else:
            msgs = None
            logger.error(f'[{self.zone_name}] unknow command: {cmd} !')

        return msgs

    def stop(self):
        """
        強制停止當前 session。
        把當下所有 step_confirmed 但尚未寫入 self.steps 的步驟，
        依 start_time 排序後補入 self.steps，再執行 finalize。
        """
        # 收集所有「已確認但尚未寫入」的步驟
        pending = []
        for step_id in range(1, 13):
            if not self.step_confirmed[step_id]:
                continue
            if self.start_times[step_id] is None:
                continue
            # 跳過相同
            already_recorded = any(
                (
                    s.id == step_id and 
                    s.start_time == self.start_times[step_id] and 
                    s.end_time == self.end_times[step_id]
                )
                for s in self.steps
            )
            if already_recorded:
                continue
            pending.append(step_id)

        # 依 step confirmed time 排序後寫入
        pending.sort(key=lambda sid: self.end_times[sid])
        for step_id in pending:
            self._update_record(step_id)

        export_data = self._finalize_session()
        # check
        if export_data:
            detecting_steps = set()
            data = zip(export_data['Step Sequence'], export_data['Is Detecting Step'])
            for step_id, is_detecting_step in data:
                if is_detecting_step:
                    detecting_steps.add(step_id)

        logger.info(f'[{self.zone_name}] Session stop because of {self.finish_reason} !')

        self.reset()
        return export_data

    def _switch_step(self):
        if self.detecting_step == self.next_step:
            return
        old = self.detecting_step
        self.detecting_step = self.next_step
        self.reset_step_info(self.detecting_step)
        logger.success(f'[{self.zone_name}] Detecting step switched, {old} -> {self.detecting_step} !')

    def switch_step_callback(self, cmd):
        self.is_switch_step = True
        self.next_step = int(cmd['step_id'].replace('Step', ''))

    def login_callback(self, cmd):
        self.is_login = True
        self.user_name = cmd['user']
        self.user_id = cmd['id']
        logger.info(f'[{self.zone_name}] UI became login, '
                    f'[User ID]: {self.user_id}, '
                    f'[User Name]: {self.user_name} !')

    def logout_callback(self, cmd):
        #self.is_final = True
        logger.warning(f'[{self.zone_name}] UI became logout !')
