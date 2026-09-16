import numpy as np
from time import time
from queue import Queue
from loguru import logger
from .tool import get_iou, get_now_str, get_utc_offset, parse_lateral_flags
from .step import Step_History, MyDict
from .clip import Clip
from .cfg import CFG
from .hand_presence import HandPresenceGate



class HandWashTracker:
    def __init__(self, zone_name, logic_cfg, sys_cfg, ai_class, mqtt=None, pub_freq=10,
                 hand_trigger_logger=None):
        # config
        self.cfg = logic_cfg['handwash_parameter']
        self.time_cfg = logic_cfg['time_parameter']
        self.sys_cfg = sys_cfg['stages']
        self.login_mode = logic_cfg['login'][sys_cfg['TriggerMode']]
        self.valid_login_modes = set(logic_cfg['login'].values())
        logger.warning(f'[{zone_name}] current login mode is "{self.login_mode}"')

        # 手觸發登入/登出的時間記錄器 (跟主要洗手紀錄 CSV 完全分開)
        self.hand_trigger_logger = hand_trigger_logger
        self._no_login_wash_streak = 0
        self._no_login_wash_idle = 0

        # check config
        assert self.cfg['alarm_frame'] > 0

        # 重要變數
        self.zone_name = zone_name
        self.ai_classes = ai_class
        self.label_bare_hand = [ai_class.index(n) for n in logic_cfg['class']['hand']]
        self.label_gloved_hand = [ai_class.index(n) for n in logic_cfg['class']['gloved hand']]
        self.label_scrub_hand = [ai_class.index(name)
                                 for i, cfg in enumerate(self.sys_cfg, 1) if cfg['washcountmax'] > 0
                                 for name in self.cfg['step_name'][i]]
        self.step_labels = {i: [ai_class.index(name) for name in (names or []) if name in ai_class] 
                            for i, names in self.cfg['step_name'].items()}
        self.step_labels_1d = [l for labels in self.step_labels.values() for l in labels]
        self.srcub_steps = {i for i, cfg in enumerate(self.sys_cfg, 1) if cfg['washcountmax'] > 0}
        assert self.srcub_steps == self.cfg['scrub_count_ratio'].keys()
        self.scrub_count_ratio = {i: (self.cfg['scrub_count_ratio'][i] if i in self.srcub_steps else -1) 
                                  for i in range(1, 13)}
        self.count_need_lr, self.time_need_lr = parse_lateral_flags(self.sys_cfg)
        assert len(self.step_labels) == len(self.count_need_lr), f'{len(self.step_labels), len(self.count_need_lr)}'

        # mqtt
        self.mqtt = mqtt
        self.pub_period = 1. / pub_freq

        # 手觸發登入/登出去彈跳狀態機: 只反映真實手的出現/消失, 
        self.hand_trigger_gate = HandPresenceGate(
            self.time_cfg['pub_hand_delay'],
            exit_timeout_fn=lambda now: self.sys_cfg[max(self.detecting_step - 1, 0)]['timeoutmax'],
        )
        self._any_step_done_since_hand_trigger = False
        self._hand_trigger_enter_time = None

        # 初始化
        self.reset()

        # 錄影
        self.origin_clip = Clip(**CFG['clip']['origin'], tag=f'{self.zone_name}_Origin')
        self.result_clip = Clip(**CFG['clip']['result'], tag=f'{self.zone_name}_Result')

        logger.debug(f'step labels: {self.step_labels}\n'
                     f'label scrub hand: {self.label_scrub_hand}')

    def reset(self):
        self.now = time()       
        self.login_reason = None
        self.finish_reason = None
        self.detecting_step = self.cfg['init_step_num']
        self.pub_time = float('-inf')
        self.debug_info = {}
        self.is_login = False
        #self.is_alarm = False
        #self.multi_step_frame = 0
        self.sent_msg = None
        self.saved_steps = []

        # 手計數狀態機:
        self.login_gate = HandPresenceGate(
            self.time_cfg['pub_hand_delay'],
            exit_timeout_fn=lambda now: self.sys_cfg[max(self.detecting_step - 1, 0)]['timeoutmax'],
        )
        self.is_final = False  # 重置訊號
        self.is_paused = False  # 完成 12 步驟後, 等待 UI 回到首頁後發出通知
        self.login_time = None
        
        # 使用者資訊
        self.user_id = None
        self.user_name = None
        
        # 洗手歷史資訊
        self.steps = Step_History()

        # 當下洗手資訊
        new_dict = lambda value=0: MyDict({i: value for i in range(1, 13)})
        self.frames = new_dict()
        self.left_frames = new_dict()
        self.right_frames = new_dict()
        self.idle_frames = new_dict()
        # ----------------------------------------------------------------------------
        self.counts = new_dict()
        self.left_counts = new_dict()
        self.right_counts = new_dict()
        self.categories = new_dict(None)
        # ----------------------------------------------------------------------------
        self.start_times = new_dict(None)
        self.step_confirmed_times = new_dict(None)
        self.end_times = new_dict(None)
        self.last_start_times = new_dict(None)
        self.durations = new_dict()
        self.left_durations = new_dict()
        self.right_durations = new_dict()
        self.step_confirmed = new_dict(False)
        # ----------------------------------------------------------------------------
        self.is_detecting_steps = new_dict(False)
        self.is_switch_step = False
        self.next_step = None
        self.cmd_queue = Queue()

        # debug info
        self._update_debug_info()
        logger.info(f'[{self.zone_name}] Reset all handwash information ! '
                    f'Detecting step become: {self.detecting_step} !')

    def update(self, detections, frame, now):
        # 時間戳
        self.now = now

        # 處理 mqtt cmd
        self._drain_events()

        # reset variables
        export_data = None
        self.sent_msg = None
        self.saved_steps = []

        # 如果在 paused 狀態下, 不進行檢測
        if self.is_paused:
            return

        # 手
        hand_mask = np.isin(detections['label'], self.label_bare_hand + self.label_gloved_hand)
        hands = detections['box'][hand_mask]
        has_hand = len(hands) > 0

        # 手觸發登入/登出時間記錄: 跟真正的登入狀態無關, 只是借用同一套去彈跳規則
        self.hand_trigger_gate.update(self.now, has_hand)
        if self.hand_trigger_gate.entered:
            self._any_step_done_since_hand_trigger = False
            self._hand_trigger_enter_time = get_now_str(self.now, utc=True)
            if self.login_mode == 'scanner':
                self._publish_status(self.mqtt.pub_topics['system'], 'AIDetection', fatal=True)
                
        if self.hand_trigger_gate.exited and self.hand_trigger_logger is not None:
            if self._any_step_done_since_hand_trigger:
                self.hand_trigger_logger.log(self.zone_name.lower(), 'Login', self._hand_trigger_enter_time)
                self.hand_trigger_logger.log(self.zone_name.lower(), 'Logout', get_now_str(self.now, utc=True))

        # scanner 模式下且沒登入
        if self.login_mode == 'scanner' and not self.is_login:
            # 沒掃 barcode, 但如果畫面上「穩定」偵測到洗手動作
            if np.any(np.isin(detections['label'], self.step_labels_1d)):
                self._no_login_wash_streak += 1
                self._no_login_wash_idle = 0
            else:
                self._no_login_wash_idle += 1
                if self._no_login_wash_idle > 5:
                    self._no_login_wash_streak = 0

            if self._no_login_wash_streak >= 10:
                self._any_step_done_since_hand_trigger = True

            self._update_debug_info(hands)
            return

        # 同時有 2 個以上的洗手動作出現, 發出警告並暫停檢測
        #if self.is_login:
        #    step_mask = np.isin(detections['label'], list(self.step_labels.values()))
        #    if step_mask.sum() >= 2:
        #        self.multi_step_frame += 1
        #    else:
        #        self.multi_step_frame = 0
#
        #    if self.multi_step_frame == self.cfg['alarm_frame']:
        #        steps = [self.ai_classes[i] for i in detections['label'][step_mask]]
        #        self._publish_status(self.mqtt.pub_topics['system'], 'Alarm', fatal=True)
        #        self.is_alarm = True
        #        logger.warning(f'there are {len(steps)} handwash {steps}, detection is paused !')
#
        #    if self.is_alarm and self.multi_step_frame == 0:
        #        self._publish_status(self.mqtt.pub_topics['system'], 'AlarmCancel', fatal=True)
        #        self.is_alarm = False
        #        logger.success(f'multiple handwash is gone, detection restarted !')
#
        #    if self.is_alarm:
        #        self._update_debug_info(hands)
        #        return

        # 檢測每個步驟
        self._check_step1_to_11(detections, hands)
        self._check_step12(detections, hands)

        # 觸發登出
        if self.is_login:
            self.login_gate.update(self.now, has_hand)

            if self.login_gate.exit_cancelled:
                self._publish_status(self.mqtt.pub_topics['system'], 'ResetCancel', fatal=True)

            if self.login_gate.exiting or self.login_gate.exited:
                self._publish_status(self.mqtt.pub_topics['system'], 'Reset',
                                     fatal=self.login_gate.exit_started)

            if self.login_gate.exited:
                self.is_login = False
                self._become_final('No hand')
        # 觸發 AI 自動登入
        elif self.login_mode == 'hand':
            self.login_gate.update(self.now, has_hand)
            if self.login_gate.entered:
                self._publish_status(self.mqtt.pub_topics['system'], 'AILogin', fatal=True)
                self.is_login = True
                self.login_time = get_now_str(self.now, utc=True)
                self.origin_clip.start()
                self.result_clip.start()

        # 即時狀態和錄影
        if self.is_login and not self.is_final:
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
        mask = np.isin(detections['label'], self.step_labels_1d)
        step_boxes = detections['box'][mask]
        step_labels = detections['label'][mask]

        for i in range(1, 12):
            # step 1 和 8 只需要檢測其中一個
            if i == (1 if self.detecting_step > 2 else 8):  # 做完肥皂後的洗手視為 step8
                continue

            mask = np.isin(step_labels, self.step_labels[i])

            # 是否要區分左右
            if np.any(mask) and (self.time_need_lr[i] or self.count_need_lr[i]):
                category = self.ai_classes[step_labels[mask][0]].split()[0]
                self.categories[i] = category

            if len(hands) > 0 and np.any(mask):
                self._do_step(i)
                self._do_scrub_count(i, self.categories[i])
            else:
                self._undo_step(i)

    def _check_step12(self, detections, hands):
        # 忽略不是剛噴完酒精
        if len(self.steps) == 0 or (self.steps[-1].id != 11 and self.detecting_step != 12):
            return
        
        scrub_hand_mask = np.isin(detections['label'], self.label_scrub_hand)
        
        # 有做洗手動作
        if len(hands) > 0 and np.any(scrub_hand_mask):
            self._do_step(12)
        elif self.detecting_step != 12:  # 如果正在檢測 step12, 允許累加
            self._undo_step(12)
        
    def _do_step(self, step_id):
        self._bucket(step_id, self.frames, self.left_frames, self.right_frames)[step_id] += 1
        self.end_times[step_id] = self.now
        self.idle_frames[step_id] = 0

        left_frame = self.left_frames[step_id]
        right_frame = self.right_frames[step_id]
        frame = self.frames[step_id]

        # 第一次
        if frame + left_frame + right_frame == 1:
            self.start_times[step_id] = self.now
            self.is_detecting_steps[step_id] = step_id == self.detecting_step

        # 滿足動作確認條件
        is_confirmed = self.step_confirmed[step_id]
        action_frame = self.cfg['action_frame'][step_id]
        if not is_confirmed and frame + left_frame + right_frame >= action_frame:
            self.step_confirmed[step_id] = True
            self.step_confirmed_times[step_id] = self.now
            logger.debug(f'[{self.zone_name}] Step {step_id}: action confirmed !')

            # step12 一滿足就 reset
            if step_id == 12 and step_id != self.detecting_step:
                self._undo_step(step_id, force=True)

        # 計算做了多久
        self._compute_step_duration(step_id)  

    def _undo_step(self, step_id, force=False):        
        # 跳過不處理
        frame = self.frames[step_id]
        left_frame = self.left_frames[step_id]
        right_frame = self.right_frames[step_id]
        
        if frame == 0 and left_frame == 0 and right_frame == 0:
            self.last_start_times[step_id] = None
            return

        # 如果是檢測中的步驟, 即使中斷一定的幀數, 仍累積洗手次數和時長 
        if step_id == self.detecting_step and self.step_confirmed[step_id]:
            self.idle_frames[step_id] += 1
            if self.idle_frames[step_id] <= self.cfg['valid_idle_frame']:
                self._bucket(step_id, self.frames, self.left_frames, self.right_frames)[step_id] += 1
                self.end_times[step_id] = self.now
                self._compute_step_duration(step_id)
            else:
                self.last_start_times[step_id] = None  # reset
            return

        # 非檢測中步驟或未滿動作確認幀數, 進行 reset
        self.last_start_times[step_id] = None

        # 強制結束
        if force:
            self._update_record(step_id)
            self.reset_step_info(step_id)
        else:
            # idle 處理
            self.idle_frames[step_id] += 1

            # reset
            if self.idle_frames[step_id] >= self.cfg['action_frame'][step_id] // 2:
                # 儲存
                if self.step_confirmed[step_id]:
                    self._update_record(step_id)

                # reset
                self.reset_step_info(step_id)

    def _do_scrub_count(self, step_id, side=None):
        if step_id not in self.srcub_steps:
            return

        frame_dict = self._select(side, self.frames, self.left_frames, self.right_frames)
        count_dict = self._select(side, self.counts, self.left_counts, self.right_counts)
        frame = max(frame_dict[step_id] - self.cfg['action_frame'][step_id], 0)
        count_dict[step_id] = frame // self.scrub_count_ratio[step_id]

    def _compute_step_duration(self, step_id):
        if self.last_start_times[step_id] is None:
            self.last_start_times[step_id] = self.now
        is_confirmed = self.step_confirmed[step_id]
        confirmed_time = self.step_confirmed_times[step_id]
        if is_confirmed and confirmed_time != self.now:
            delta_t = max(self.now - self.last_start_times[step_id], 1e-6)
        else:
            delta_t = 0

        self._bucket(step_id, self.durations, self.left_durations, self.right_durations)[step_id] += delta_t
        self.last_start_times[step_id] = self.now

    def _select(self, side, base, left, right):
        """依 side ('left' / 'right' / None) 選出對應要操作的字典"""
        if side == 'left':
            return left
        elif side == 'right':
            return right
        elif side is None:
            return base
        else:
            raise ValueError(f'[{self.zone_name}] unknown side: {side}')

    def _bucket(self, step_id, base, left, right):
        """依 self.categories[step_id] 選出該 step 目前要操作的字典"""
        return self._select(self.categories[step_id], base, left, right)

    def _update_record(self, step_id):
        if not self.step_confirmed[step_id]:
            return
        self.steps.append(step_id, self.counts[step_id], self.left_counts[step_id], 
                          self.right_counts[step_id], self.start_times[step_id], 
                          self.end_times[step_id], self.step_confirmed_times[step_id], 
                          self.durations[step_id], self.left_durations[step_id],
                          self.right_durations[step_id], self.frames[step_id], 
                          self.left_frames[step_id], self.right_frames[step_id],
                          int(self.is_detecting_steps[step_id]))
        self.saved_steps.append(step_id)
        self._any_step_done_since_hand_trigger = True
        logger.info(f'[{self.zone_name}] Add Step {step_id} into step sequence !')

    def _get_final_data(self):
        if len(self.steps) == 0 and (self.login_mode == 'hand' or self.user_id is None):
            return {}

        # sort
        self.steps.sort_by_start_time()

        res = {
            "Store ID": "test", 
            "User ID": str(self.user_id) if self.user_id is not None else '',
            "User Name": str(self.user_name) if self.user_name is not None else '',
            "UTC Offset": get_utc_offset(),
            "Login Mode": self.login_mode,
            "Step Sequence": self.steps.ids.copy(),
            "Finished Step": sum(self.steps.is_detecting_steps),
            "Login Time": self.login_time,
            "Start Time": [get_now_str(t) for t in self.steps.start_times], 
            "Action Confirmed Time": [get_now_str(t) for t in self.steps.step_confirmed_times], 
            "End Time": [get_now_str(t) for t in self.steps.end_times],
            "Step Count": self.steps.counts.copy(),
            "Left Step Count": self.steps.left_counts.copy(),
            "Right Step Count": self.steps.right_counts.copy(),
            "Is Detecting Step": self.steps.is_detecting_steps.copy(),
            'Duration': self.steps.durations.copy(),
            'Left Duration': self.steps.left_durations.copy(),
            'Right Duration': self.steps.right_durations.copy(),
            'Frame': self.steps.frames.copy(),
            'Left Frame': self.steps.left_frames.copy(),
            'Right Frame': self.steps.right_frames.copy(),
            'Step Length': len(self.steps)
        }
        res['Login reason'] = self.login_reason
        res['Finish reason'] = self.finish_reason
        res['Region'] = self.zone_name.lower()
        for i in range(1, 13):
            res[f'Step{i} min count'] = self.sys_cfg[i-1]['washcountmax']
            res[f'Step{i} min time'] = self.sys_cfg[i-1]['washtimemax']
        res['Left Right Count Steps'] = [i for i in range(1, 13) if self.count_need_lr[i]]
        res['Left Right Time Steps'] = [i for i in range(1, 13) if self.time_need_lr[i]]
        return res

    def _finalize_session(self, is_interrupted):
        """ 結束 Session 並回傳資料 """
        # finish reason
        if is_interrupted:  # 被 kill
            self._become_final('killed')

        final_data = self._get_final_data()
        n_step = len(self.steps)
        #if n_step == 0:
        #    logger.debug(f'[{self.zone_name}] Completed with no any step! Skip to save CSV !')
        #else:
        #    logger.info(f'[{self.zone_name}] Completed with {n_step} steps! ')
        logger.info(f'[{self.zone_name}] Completed with {n_step} steps! ')

        # clip
        save_video = n_step != 0 or (self.login_mode == 'scanner' and self.user_id is not None)
        self.origin_clip.stop(save_video, is_interrupted)
        self.result_clip.stop(save_video, is_interrupted)
        return final_data

    def _update_debug_info(self, hands=[]):
        self.debug_info.update({
            'status': 'Hand Detected' if len(hands) > 0 else 'No Hand',
            'frames': self.frames, 'left_frames': self.left_frames, 'right_frames': self.right_frames,
            'counts': self.counts, 'left_counts': self.left_counts, 'right_counts': self.right_counts,
            'durations': self.durations, 'left_durations': self.left_durations, 'right_durations': self.right_durations,
            'start_times': self.start_times,
            'step_confirmed_times': self.step_confirmed_times,
            'step_confirmed': self.step_confirmed,
            'last_start_times': self.last_start_times,
            'detected_steps': self.steps.detected_steps,
            'detecting_step': self.detecting_step,
            'sent_msg': self.sent_msg,
            'saved_steps': self.saved_steps,
            'now': self.now,
            #'is_alarm': self.is_alarm,
            'is_login': self.is_login,
        })

    def reset_step_info(self, step_id):
        self.categories[step_id] = None
        for d in (self.frames, self.left_frames, self.right_frames, self.idle_frames,
                  self.counts, self.left_counts, self.right_counts,
                  self.durations, self.left_durations, self.right_durations):
            d[step_id] = 0
        for d in (self.start_times, self.end_times, self.step_confirmed_times, self.last_start_times):
            d[step_id] = None
        self.step_confirmed[step_id] = False
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
                if cmd == 'status':
                    if self.detecting_step == 12 and msg['trigger']:
                        logger.info('handwashing completed, waiting for the UI to return to the homepage...')
                        self.cmd_queue.put(('completed', None))

            # 控制發送頻率
            if cmd == 'status':
                self.pub_time = self.now

    def _create_mqtt_message(self, cmd):
        if cmd == 'Reset':
            msgs = {"cmd": cmd, "side": self.zone_name.lower(), "time": str(self.login_gate.remain)}
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
        elif cmd == 'AIDetection':
            msgs = {"cmd": cmd, "side": self.zone_name.lower()}
        elif cmd == 'status':
            step = self.detecting_step
            count = self._bucket(step, self.counts, self.left_counts, self.right_counts)[step]
            duration = self._bucket(step, self.durations, self.left_durations, self.right_durations)[step]
            if self.count_need_lr[step]:
                duration = self.left_durations[step] + self.right_durations[step]

            if duration == 0:
                return

            msgs = {
                "step_id": f"Step{step}",
                "washcount": str(count),
                "washtime": str(duration),
                "side": self.zone_name.lower(),
                "category": self.categories[step],
                "trigger": self.step_confirmed[step]
            }
        else:
            msgs = None
            logger.error(f'[{self.zone_name}] unknow command: {cmd} !')

        return msgs

    def stop(self, is_interrupted=False):
        """
        強制停止當前 session, 只寫入 step_confirmed 的步驟
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

        # 依 end time 排序後寫入
        for step_id in pending:
            self._update_record(step_id)

        # 輸出最終結果
        export_data = self._finalize_session(is_interrupted)
        logger.warning(f'[{self.zone_name}] Session stop because of "{self.finish_reason}" !')

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
        self.cmd_queue.put(('switch_step', cmd))

    def login_callback(self, cmd):
        """ QR code 登入 """
        if self.is_login:
            logger.warning('current status is login, so ignored the command of "login" !')
        else:
            self.cmd_queue.put(('login', cmd))

    def button_login_callback(self, cmd):
        """ UI Button 登入 """
        if self.is_login:
            logger.warning('current status is login, so ignored the command of "UI button login" !')
        else:
            self.cmd_queue.put(('bn_login', cmd))

    def logout_callback(self, cmd):
        self.cmd_queue.put(('logout', cmd))

    def switch_login_mode_callback(self, cmd):
        self.cmd_queue.put(('switch_login_mode', cmd))

    def _become_final(self, reason):
        self.is_final = True
        self.is_paused = False
        self.finish_reason = reason

    def _drain_events(self):
        while not self.cmd_queue.empty():
            kind, cmd = self.cmd_queue.get_nowait()

            if kind == 'switch_step':
                self.is_switch_step = True
                self.next_step = int(cmd['step_id'].replace('Step', ''))

            elif kind == 'login' and not self.is_login:
                self.is_login = True
                self.is_final = False
                self.login_reason = 'QR code'
                self.finish_reason = None
                self.user_name = cmd['user']
                self.user_id = cmd['id']
                self.login_gate.force_active(True)
                logger.info(f'[{self.zone_name}] UI became login, '
                            f'[User ID]: {self.user_id}, '
                            f'[User Name]: {self.user_name} !')
                self.origin_clip.start()
                self.result_clip.start()
                self.login_time = get_now_str(self.now, utc=True)

            elif kind == 'bn_login' and not self.is_login:
                self.is_login = True
                self.is_final = False
                self.login_reason = 'UI button'
                self.finish_reason = None
                self.login_gate.force_active(True)
                logger.info(f'[{self.zone_name}] UI became login, because of button')
                self.origin_clip.start()
                self.result_clip.start()
                self.login_time = get_now_str(self.now, utc=True)
                
            elif kind == 'logout':
                logger.warning(f'[{self.zone_name}] UI became logout !')
                self._become_final('all completed')

            elif kind == 'completed':
                self.is_paused = True

            elif kind == 'switch_login_mode':
                if cmd['mode'] not in self.valid_login_modes:
                    logger.error(f'[{self.zone_name}] Invaild login mode: {cmd["mode"]}')
                    return

                if self.login_mode == cmd['mode']:
                    logger.warning(f'[{self.zone_name}] login mode is the same as original, '
                                f'ignored the command !')
                else:
                    logger.warning(f'[{self.zone_name}] login mode became {cmd["mode"]}, '
                                f'original mode is {self.login_mode} !')
                    self.login_mode = cmd['mode']