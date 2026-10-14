import signal
import sys
import yaml
import traceback
import socket
from loguru import logger
from tqdm import tqdm
from time import time, sleep
from datetime import datetime
from pathlib import Path as p
import re
import os
from os.path import dirname
import cv2
import numpy as np
import socket
import subprocess
from threading import Thread
from utils import (Mjpeg_Streamer, 
                   RTMDet_DLA, 
                   Camera,
                   Video,
                   setup_logger, MY_LOGGER,
                   Csv_Manager,
                   HandWashTracker,
                   HandTriggerLogger,
                   draw_timestamp, draw_debug_panel, plot_bbox,
                   draw_zone_divider, draw_step_roi,
                   Timer,
                   MQTT,
                   get_now_str, get_utc_offset, get_boxes_outside,
                   CFG, SYS_CFG)


VIDEO_PATH = None


class App_HandWash:
    def __init__(self):
        self.is_running = False

        self.camera = Camera(**CFG['camera'])
        self.ai_model = RTMDet_DLA(**CFG['AI']['handwash'])
        self.streamer = Mjpeg_Streamer(**CFG['streamer'])
        self.origin_video = Video(**CFG['video']['origin'])
        self.result_video = Video(**CFG['video']['result'])
        self.predict_video = Video(**CFG['video']['predict'])
        self.csv_manager = Csv_Manager(**CFG['csv'])
        self.mqtt_manager = MQTT(**CFG['mqtt'])
        self.hand_trigger_logger = HandTriggerLogger(save_dir=CFG['csv_hand_trigger']['save_dir'])

        # 檢測洗手
        self.screen = np.asarray(CFG['roi']['screen']) if CFG['roi']['screen'] is not None else None
        self.tracker_left = HandWashTracker("Left", CFG['logic'], SYS_CFG, self.ai_model.classes, 
                                            self.mqtt_manager, CFG['mqtt']['pub_freq'],
                                            hand_trigger_logger=self.hand_trigger_logger)
        self.tracker_right = HandWashTracker("Right", CFG['logic'], SYS_CFG, self.ai_model.classes, 
                                             self.mqtt_manager, CFG['mqtt']['pub_freq'],
                                             hand_trigger_logger=self.hand_trigger_logger)

        # 視覺化設定: bbox 區塊裡有幾個不是 plot_bbox 的參數, 先拆出來
        vis_cfg = CFG['visualization']
        self.bbox_cfg = dict(vis_cfg['bbox'])
        self.only_detecting_step = self.bbox_cfg.pop('only_detecting_step', False)
        self.plot_step_roi = self.bbox_cfg.pop('plot_step_roi', False)
        self.roi_color = tuple(self.bbox_cfg.pop('roi_color', (0, 255, 255)))
        self.divider_cfg = dict(vis_cfg.get('zone_divider') or {})
        self.draw_divider = self.divider_cfg.pop('enable', False)

        # 重要標籤
        self.wash_labels = self.tracker_left.step_labels_1d
        self.exit_program = False
        self.capture_cmd = None

        # mqtt callback
        callbacks = {
            'Login': {'left': self.tracker_left.login_callback, 
                      'right': self.tracker_right.login_callback},
            'BnLogin': {'left': self.tracker_left.button_login_callback, 
                        'right': self.tracker_right.button_login_callback},
            'Logout': {'left': self.tracker_left.logout_callback, 
                       'right': self.tracker_right.logout_callback},
            'NextStep': {'left': self.tracker_left.switch_step_callback, 
                         'right': self.tracker_right.switch_step_callback},
            'Capture': self.capture_callback,            
        }
        self.mqtt_manager.add_callbacks(callbacks)
        
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        logger.success('all init succeeded !')

    def run(self):
        try:
            self.camera.start()
            self.streamer.start()
            self.is_running = True

            # timer
            loop_timer = Timer('one complete loop', silent=True)
            read_frame_timer = Timer('read frame', silent=True)
            split_timer = Timer('split detections into two', silent=True)
            ai_timer = Timer('AI forward', silent=True)
            handwash_timer = Timer('handwash detection', silent=True)
            draw_result_timer = Timer('draw result', silent=True)
            frame_copy_timer = Timer('frame copy', silent=True)
            streamer_timer = Timer('push frame to streamer', silent=True)
            video_timer = Timer('write frame to video', silent=True)

            logger.info("Main loop started.")
        except:
            logger.error(f"{traceback.format_exc()}")

        # loop
        while self.is_running:
            try:
                with loop_timer:
                    # read frame
                    with read_frame_timer:
                        ret, (origin, frame) = self.camera.get_latest_frame()
                    if not ret:
                        logger.error(f"cannot read frame, skip this frame !")
                        continue
           
                    # AI Inference
                    with ai_timer:
                        scores, boxes, pred_labels = self.ai_model(frame)

                    # 螢幕內的框進行忽略
                    if self.screen is not None:
                        h, w = frame.shape[:2]
                        screen = self.screen * [w, h, w, h]
                        is_outside = get_boxes_outside(boxes, screen)
                        scores = scores[is_outside]
                        boxes = boxes[is_outside]
                        pred_labels = pred_labels[is_outside]

                    # 把 detections 分至左右區
                    with split_timer:
                        h, w = frame.shape[:2]
                        mid_x = w // 2
                        
                        # detections
                        center_x = boxes[:, 0:3:2].mean(1)
                        is_left = center_x < mid_x
                        left_dets = {
                            'box': boxes[is_left], 
                            'label': pred_labels[is_left], 
                            'score': scores[is_left]
                        }
                        right_dets = {
                            'box': boxes[~is_left], 
                            'label': pred_labels[~is_left], 
                            'score': scores[~is_left]
                        }

                    # 複製一份 frame
                    with frame_copy_timer:
                        frame_copy = frame.copy()

                    # now
                    now = time()

                    # 截圖
                    if self.capture_cmd is not None:
                        args = (self.capture_cmd['path'], now, origin, frame)
                        Thread(target=self.do_capture, args=args, daemon=True).start()
                        self.capture_cmd = None

                    # 洗手檢測
                    with handwash_timer:
                        res_l = self.tracker_left.update(
                            left_dets, 
                            frame_copy,
                            now
                        )
                        if res_l: 
                            self.csv_manager.write_record(res_l)

                        res_r = self.tracker_right.update(
                            right_dets, 
                            frame_copy,
                            now
                        )
                        if res_r:
                            self.csv_manager.write_record(res_r)

                    # visualization
                    with draw_result_timer:
                        # 畫螢幕
                        if self.screen is not None:
                            screen = screen.astype(int)
                            cv2.rectangle(frame_copy, screen[:2], screen[2:4], (0, 255, 0), 3)

                        # 畫左右區分隔線 (畫在 bbox 底下)
                        if self.draw_divider:
                            draw_zone_divider(frame_copy, **self.divider_cfg)

                        # 畫 detections
                        if self.only_detecting_step:
                            keep = self._keep_detecting_step(pred_labels, is_left)
                            boxes_d, labels_d, scores_d = boxes[keep], pred_labels[keep], scores[keep]
                        else:
                            boxes_d, labels_d, scores_d = boxes, pred_labels, scores
                        plot_bbox(frame_copy, 
                                  boxes_d,
                                  labels_d, 
                                  scores_d, 
                                  self.ai_model.classes, 
                                  **self.bbox_cfg)

                        # 畫 step 1、2 的 ROI (只畫 login 中且 == detecting step 的)
                        if self.plot_step_roi:
                            for tracker in (self.tracker_left, self.tracker_right):
                                draw_step_roi(frame_copy, tracker, self.roi_color,
                                              self.bbox_cfg['bbox_thickness'],
                                              self.bbox_cfg['font_scale'],
                                              self.bbox_cfg['font_thickness'])
                        
                        # 畫 debug
                        draw_debug_panel(frame_copy, self.tracker_left, self.tracker_right)

                        # 畫時間戳
                        now_str = get_now_str(now, utc=False)
                        offset = get_utc_offset()
                        offset = f'+{offset}' if offset > 0 else str(offset) 
                        draw_timestamp(frame_copy, f'{now_str} ({offset})', **CFG['visualization']['timestamp'])

                    # push to streamer
                    with streamer_timer:
                        self.streamer.push_frame(frame_copy) 

                    # write frame into video
                    with video_timer:
                        self.origin_video.write_frame(frame)
                        self.result_video.write_frame(frame_copy)
                        if np.isin(pred_labels, self.wash_labels).any():
                            self.predict_video.write_frame(frame)

                        # 錄影
                        if self.tracker_left.is_login:
                            self.tracker_left.origin_clip.write_frame(frame, now)
                            self.tracker_left.result_clip.write_frame(frame_copy, now)

                        if self.tracker_right.is_login:
                            self.tracker_right.origin_clip.write_frame(frame, now)
                            self.tracker_right.result_clip.write_frame(frame_copy, now)

                # log time elapsed
                MY_LOGGER.log(f'[{loop_timer.name}] {loop_timer.elapsed:.6f} (s)', 'INFO', reset=False)
                MY_LOGGER.log(f'[{read_frame_timer.name}] {read_frame_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{split_timer.name}] {split_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{ai_timer.name}] {ai_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{handwash_timer.name}] {handwash_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{draw_result_timer.name}] {draw_result_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{frame_copy_timer.name}] {frame_copy_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{streamer_timer.name}] {streamer_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{video_timer.name}] {video_timer.elapsed:.6f} (s)', 'DEBUG', reset=True)

            except:
                logger.error(f"{traceback.format_exc()}")

    def _keep_detecting_step(self, labels, is_left):
        """
        回傳 bool mask: login 中的區域只保留「屬於 detecting step」的框, 沒 login 的區域全部保留。
        (omit_classes 之後由 plot_bbox 照常處理)
        """
        keep = np.ones(len(labels), dtype=bool)
        for tracker, in_zone in ((self.tracker_left, is_left), (self.tracker_right, ~is_left)):
            if not tracker.is_login:
                continue
            is_step = np.isin(labels, tracker.step_labels[tracker.detecting_step])
            keep &= ~in_zone | is_step
        return keep

    def handle_exit(self, signum, frame):
        if signum == signal.SIGTERM:
            logger.warning('received "SIGTERM", system will shut down or reboot !')
        elif signum == signal.SIGINT:
            logger.warning('received "SIGINT", program will stop !')
        else:
            logger.warning(f'received {signum} signal !')
        self.exit_program = True
        self.stop()

    def stop(self):
        # CSV
        res_l = self.tracker_left.stop(self.exit_program)
        res_r = self.tracker_right.stop(self.exit_program)
        if res_l:
            self.csv_manager.write_record(res_l)
        if res_r:
            self.csv_manager.write_record(res_r)

        # stop others
        if not self.is_running:
            return
        self.is_running = False
        
        self.camera.stop()
        self.streamer.stop()
        self.origin_video.stop()
        self.result_video.stop()
        self.predict_video.stop()
        self.mqtt_manager.disconnect()
        logger.info(f'CSV path: {self.csv_manager.file_path}')
        logger.success("release all sources !")

    def capture_callback(self, cmd):
        self.capture_cmd = cmd

    def do_capture(self, folder, now, origin, frame):
        now = datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M%S.%f')[:-3]
        dst_origin = f'{folder}/{now}_origin.jpg'
        dst_frame = f'{folder}/{now}_frame.jpg'
        os.makedirs(dirname(dst_origin), exist_ok=True)
        cv2.imwrite(dst_origin, origin)
        cv2.imwrite(dst_frame, frame)
        logger.success(f'saved captures in the "{folder}"')



if __name__ == "__main__":
    try:
        device_code = socket.gethostname().split('-')[-1]
        setup_logger(**CFG['log'], suffix=device_code)
        app = App_HandWash()
        app.run()
    except:
        logger.error(traceback.format_exc())
    finally:
        app.exit_program = True
        app.stop()
        logger.success('Application terminated !')
