import signal
import sys
import yaml
import traceback
import socket
from loguru import logger
from tqdm import tqdm
from time import time
from datetime import datetime
from pathlib import Path as p
import re
import cv2
import numpy as np
import socket
from utils import (Mjpeg_Streamer, 
                   RTMDet_ONNX, 
                   Camera,
                   Video,
                   plot_bbox, 
                   setup_logger, MY_LOGGER,
                   Csv_Manager,
                   HandWashTracker,
                   draw_timestamp, draw_status_overlay, draw_debug_panel,
                   Device,
                   Result,
                   Timer,
                   MQTT,
                   Visualization,
                   CFG, SYS_CFG)

# 1. 搓洗秒數不累計


VIDEO_PATH = None


class App_HandWash:
    def __init__(self, device_code):
        setup_logger(**CFG['log'], suffix=device_code)

        self.is_running = False

        CFG['camera']['video_path'] = VIDEO_PATH  # 要讀取的影片
        CFG['video']['origin']['output_path'] = VIDEO_PATH  # 利用檔名產生輸出檔案
        CFG['csv']['output_path'] = VIDEO_PATH.with_suffix('.csv')  # 利用檔名產生輸出檔案

        self.camera = Camera(**CFG['camera'])
        self.ai_model = RTMDet_ONNX(**CFG['AI']['handwash'])
        self.streamer = Mjpeg_Streamer(**CFG['streamer'])
        self.origin_video = Video(**CFG['video']['origin'])
        self.result_video = Video(**CFG['video']['result'])
        self.csv_manager = Csv_Manager(**CFG['csv'])
        self.mqtt_manager = MQTT(**CFG['mqtt'])
        self.draw_manager = Visualization(CFG)
        #self.result_drawer = Result(**CFG['visualization']['result'])
        #self.device = Device(**CFG['device'], device_code=device_code, ai_class=self.ai_model.classes)
        self.device = None
        self.is_alarm = False

        # 檢測洗手
        #self.tracker_left = HandWashTracker(zone_name="Left", devices=self.device.left_data,
        #                                    ai_class=self.ai_model.classes, logic_cfg=CFG['logic'])
        #self.tracker_right = HandWashTracker(zone_name="Right", devices=self.device.right_data, 
        #                                     ai_class=self.ai_model.classes, logic_cfg=CFG['logic'])
        self.tracker_left = HandWashTracker("Left", CFG['logic'], SYS_CFG, self.ai_model.classes, None, 
                                            mqtt=self.mqtt_manager, pub_freq=CFG['mqtt']['pub_freq'])
        self.tracker_right = HandWashTracker("Right", CFG['logic'], SYS_CFG, self.ai_model.classes, None, 
                                             mqtt=self.mqtt_manager, pub_freq=CFG['mqtt']['pub_freq'])

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
            alarm_timer = Timer('find hands and alarm', silent=True)

            logger.info("Main loop started.")

            # 進度條
            pbar = tqdm(unit='frame', desc='Processing')
        except:
            logger.error(f"{traceback.format_exc()}")

        # loop
        while self.is_running:
            try:
                with loop_timer:
                    # read frame
                    with read_frame_timer:
                        ret, frame = self.camera.get_latest_frame()

                    pbar.update(1)

                    if ret is None:
                        #logger.warning(f"{VIDEO_PATH} cannot read frame at {pbar.n}, skip this frame !")
                        continue
                    elif ret is False:
                        logger.error(f"{VIDEO_PATH} stop at {pbar.n} frame !")
                        break
                                            
                    # AI Inference
                    with ai_timer:
                        scores, boxes, pred_labels = self.ai_model(frame)

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

                    # 6 支以上的手出現, 發出警告
                    with alarm_timer:
                        hand_lbs = self.tracker_left.label_bare_hand + self.tracker_left.label_gloved_hand
                        mask = np.isin(pred_labels, hand_lbs)
                        hands = boxes[mask].copy()
                        hands[:, 2:4] -= hands[:, 0:2]
                        hand_scores = scores[mask]
                        ids = cv2.dnn.NMSBoxes(hands, hand_scores, 0., 0.7)
                        hand_classes = [self.ai_model.classes[i] for i in pred_labels[mask][ids]]
                        if len(ids) >= 6 and not self.is_alarm:
                            logger.warning(f'found {len(ids)} hands, detail: {hand_classes} !')
                            self.tracker_left._publish_status(self.mqtt_manager.pub_topics['system'], 'Alarm')
                            self.is_alarm = True
                        elif len(ids) < 6 and self.is_alarm:
                            self.tracker_left._publish_status(self.mqtt_manager.pub_topics['system'], 'AlarmCancel')
                            self.is_alarm = False

                    # 洗手檢測
                    with handwash_timer:
                        if not self.is_alarm:
                            now, res_l = self.tracker_left.update(left_dets, frame_copy)
                            if res_l: 
                                self.csv_manager.write_record(res_l)

                            now, res_r = self.tracker_right.update(right_dets, frame_copy)
                            if res_r: 
                                self.csv_manager.write_record(res_r)

                    # visualization
                    with draw_result_timer:
                        #current_steps = [f'step {self.tracker_left.current_step}, {self.tracker_left.buffer_count}', 
                        #                 f'step {self.tracker_right.current_step}, {self.tracker_right.buffer_count}']
                        #self.result_drawer.draw_step(frame_copy, current_steps)
                        #self.result_drawer.draw_region(frame_copy, np.asarray([d['box'] for d in left_dets]), 'L')
                        #self.result_drawer.draw_region(frame_copy, np.asarray([d['box'] for d in right_dets]), 'R')

                        # 畫 detections
                        plot_bbox(frame_copy, 
                                  boxes,
                                  pred_labels, 
                                  scores, 
                                  self.ai_model.classes, 
                                  **CFG['visualization']['bbox'])
                        
                        ## 畫左 devices
                        #plot_bbox(frame_copy, 
                        #          self.device.left_bboxes,
                        #          self.device.left_labels, 
                        #          ([1.] * len(self.device.left_labels)), 
                        #          self.ai_model.classes, 
                        #          **CFG['visualization']['bbox'])
#
                        ## 畫右 devices
                        #plot_bbox(frame_copy, 
                        #          self.device.right_bboxes,
                        #          self.device.right_labels, 
                        #          [1.] * len(self.device.right_labels), 
                        #          self.ai_model.classes, 
                        #          **CFG['visualization']['bbox'])


                        # 畫 debug
                        draw_debug_panel(frame_copy, self.tracker_left, self.tracker_right)

                        # 畫時間戳
                        now_str = now.strftime('%Y%m%d %H%M%S.%f')[:-3]
                        draw_timestamp(frame_copy, now_str, **CFG['visualization']['timestamp'])

                    # push to streamer
                    with streamer_timer:
                        self.streamer.push_frame(frame_copy) 

                    # write frame into video
                    with video_timer:
                        self.origin_video.write_frame(frame)
                        self.result_video.write_frame(frame_copy)
                        #self.draw_manager.put(frame_copy, now, scores, boxes, pred_labels)

                # log time elapsed
                MY_LOGGER.log(f'[{loop_timer.name}] {loop_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{read_frame_timer.name}] {read_frame_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{split_timer.name}] {split_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{alarm_timer.name}] {alarm_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{ai_timer.name}] {ai_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{handwash_timer.name}] {handwash_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{draw_result_timer.name}] {draw_result_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{frame_copy_timer.name}] {frame_copy_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{streamer_timer.name}] {streamer_timer.elapsed:.6f} (s)', 'DEBUG', reset=False)
                MY_LOGGER.log(f'[{video_timer.name}] {video_timer.elapsed:.6f} (s)', 'DEBUG', reset=True)

            except:
                logger.error(f"{traceback.format_exc()}")

        pbar.close()

    def handle_exit(self, signum, frame):
        if signum == signal.SIGTERM:
            logger.warning('received "SIGTERM", system will shut down or reboot !')
        elif signum == signal.SIGINT:
            logger.warning('received "SIGINT", program will stop !')
        else:
            logger.warning(f'received {signum} signal !')
        self.stop()

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        
        self.camera.stop()
        self.streamer.stop()
        self.origin_video.stop()
        self.result_video.stop()
        #self.draw_manager.stop()
        self.mqtt_manager.disconnect()

        logger.success("release all sources !")


def get_sort_key(path_obj):
    # 1. 資料夾權重 (處理日期資料夾 zza/0692/20260415)
    folder_weights = []
    for part in path_obj.parts[:-1]:
        # 嘗試把 '20260415' 轉成數字，確保 20260416 > 20260415
        folder_weights.append(int(part) if part.isdigit() else part)

    # 2. 處理檔名 (例如: 20260415_8_p2)
    stem = path_obj.stem
    parts = stem.split('_') # 分割結果可能是 ['20260415', '8', 'p2'] 或 ['20260415', '3']

    main_num = 0
    p_value = -1

    if len(parts) >= 2:
        last_part = parts[-1]
        
        if last_part.startswith('p'):
            # 情況：xxx_8_p2 -> last_part 是 'p2'
            try:
                p_value = int(last_part[1:]) # 拿掉 'p' 轉數字
                main_num = int(parts[-2])    # 拿掉 'p2' 後，最後一個就是主編號
            except (ValueError, IndexError):
                pass
        else:
            # 情況：xxx_3 -> last_part 是 '3'
            try:
                main_num = int(last_part)
                p_value = -1 # 沒有 p
            except ValueError:
                pass

    return (*folder_weights, main_num, p_value)



if __name__ == "__main__":
    try:
        paths = sorted(p('video').glob('**/*.mp4'), key=get_sort_key)
        for path in paths:
            if path.name != '20260410_1.mp4':
                continue

            VIDEO_PATH = path
            device_code = socket.gethostname().split('-')[-1]
            app = App_HandWash(device_code)
            app.run()
    except:
        logger.error(traceback.format_exc())
    finally:
        logger.success('Application terminated !')
