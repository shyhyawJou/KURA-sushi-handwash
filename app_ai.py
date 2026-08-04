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
import cv2
import numpy as np
import socket
import subprocess
from utils import (Mjpeg_Streamer, 
                   RTMDet_ONNX, 
                   Camera,
                   Video,
                   setup_logger, MY_LOGGER,
                   Csv_Manager,
                   HandWashTracker,
                   draw_timestamp, draw_debug_panel, plot_bbox,
                   Timer,
                   MQTT,
                   get_now_str, get_boxes_outside,
                   CFG, SYS_CFG)


VIDEO_PATH = None
VIDEO_FOLDER = None


class App_HandWash:
    def __init__(self, device_code):
        setup_logger(**CFG['log'], suffix=device_code)

        self.is_running = False

        date = VIDEO_FOLDER.name.split('_')[0]
        CFG['video']['result']['output_path'] = VIDEO_FOLDER.with_name(date)  # 利用檔名產生輸出檔案
        CFG['video']['predict']['output_path'] = VIDEO_FOLDER.with_name(date)  # 利用檔名產生輸出檔案
        CFG['csv']['output_path'] = VIDEO_FOLDER.with_name(f'{date}.csv')  # 利用檔名產生輸出檔案

        self.camera = Camera(**CFG['camera'])
        self.ai_model = RTMDet_ONNX(**CFG['AI']['handwash'])
        self.streamer = Mjpeg_Streamer(**CFG['streamer'])
        self.origin_video = Video(**CFG['video']['origin'])
        self.result_video = Video(**CFG['video']['result'])
        self.predict_video = Video(**CFG['video']['predict'])
        self.csv_manager = Csv_Manager(**CFG['csv'])
        self.mqtt_manager = MQTT(**CFG['mqtt'])

        # 檢測洗手
        self.tracker_left = HandWashTracker("Left", CFG['logic'], SYS_CFG, self.ai_model.classes, 
                                            self.mqtt_manager, CFG['mqtt']['pub_freq'])
        self.tracker_right = HandWashTracker("Right", CFG['logic'], SYS_CFG, self.ai_model.classes, 
                                             self.mqtt_manager, CFG['mqtt']['pub_freq'])
        self.is_left_login = False
        self.is_right_login = False
        self.is_ai_login = True

        # 重要標籤
        self.wash_labels = [self.tracker_left.step_labels[i] for i in range(1, 12)]
        self.exit_program = False

        # mqtt callback
        callbacks = {
            'Login': {'left': self.tracker_left.login_callback, 
                      'right': self.tracker_right.login_callback},
            'Logout': {'left': self.tracker_left.logout_callback, 
                       'right': self.tracker_right.logout_callback},
            'NextStep': {'left': self.tracker_left.switch_step_callback, 
                         'right': self.tracker_right.switch_step_callback},
            'Trigger': {'left': self.tracker_left.switch_login_mode_callback, 
                        'right': self.tracker_right.switch_login_mode_callback},
        }
        self.mqtt_manager.add_callbacks(callbacks)
        
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

        if not self.is_left_login:
            logger.warning('[Left] current state is logout !')
        if not self.is_right_login:
            logger.warning('[Right] current state is logout !')

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

        # 進度條
        pbar = tqdm(unit='frame', desc='Processing')

        # loop
        while self.is_running:
            try:
                is_login = self.tracker_left.is_login or self.tracker_right.is_login
                login_mode = self.tracker_left.login_mode
                if login_mode == 'scanner' and not is_login:
                    sleep(0.05)
                    continue

                with loop_timer:
                    # read frame
                    with read_frame_timer:
                        ret, frame = self.camera.get_latest_frame()
                    
                    pbar.update(1)

                    if ret is None:
                        continue
                    elif ret is False:
                        logger.error(f"{VIDEO_PATH} stop at {pbar.n} frame !")
                        break
                    
                    # AI Inference
                    with ai_timer:
                        scores, boxes, pred_labels = self.ai_model(frame)

                    # 螢幕位置
                    h, w = frame.shape[:2]
                    screen = np.array([0.33240997, 0.08704062, 0.70637119, 0.2901354])
                    screen = np.int32(screen * [w, h, w, h])
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

                    # 洗手檢測
                    now = time()

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
                        cv2.rectangle(frame_copy, screen[:2], screen[2:4], (0, 255, 0), 3)

                        # 畫 detections
                        plot_bbox(frame_copy, 
                                  boxes,
                                  pred_labels, 
                                  scores, 
                                  self.ai_model.classes, 
                                  **CFG['visualization']['bbox'])
                        
                        # 畫 debug
                        draw_debug_panel(frame_copy, self.tracker_left, self.tracker_right)

                        # 畫時間戳
                        now_str = get_now_str(now, utc=False)
                        draw_timestamp(frame_copy, now_str, **CFG['visualization']['timestamp'])

                    # push to streamer
                    with streamer_timer:
                        self.streamer.push_frame(frame_copy) 

                    # write frame into video
                    with video_timer:
                        self.origin_video.write_frame(frame)
                        self.result_video.write_frame(frame_copy)
                        if np.isin(pred_labels, self.wash_labels).any():
                            self.predict_video.write_frame(frame)

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

        pbar.close()
        self.is_running = False

    def handle_exit(self, signum, frame):
        if signum == signal.SIGTERM:
            logger.warning('received "SIGTERM", system will shut down or reboot !')
        elif signum == signal.SIGINT:
            logger.warning('received "SIGINT", program will stop !')
        else:
            logger.warning(f'received {signum} signal !')
        self.is_running = False
        self.exit_program = True

    def stop(self):
        # CSV
        res_l = self.tracker_left.stop()
        res_r = self.tracker_right.stop()
        if res_l:
            self.csv_manager.write_record(res_l)
        if res_r:
            self.csv_manager.write_record(res_r)

        # stop others
        self.is_running = False
        
        self.camera.stop()
        self.streamer.stop()
        self.origin_video.stop()
        self.result_video.stop()
        self.predict_video.stop()
        self.mqtt_manager.disconnect()
        logger.info(f'CSV path: {self.csv_manager.file_path}')
        logger.success("release all sources !")

    def _login_callback(self, msg):
        try:
            cmd = msg['cmd']
            side = msg['side'].lower()
            if cmd == 'Login':
                if side == 'left':
                    self.is_left_login = True
                    logger.info('[Left] Login, Begin detection !')
                elif side == 'right':
                    self.is_right_login = True
                    logger.info('[Right] Login, Begin detection !')
                else:
                    logger.error(f'Unknown Login message: {msg}')
        except:
            logger.error(traceback.format_exc())


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
    FOLDER = 'video'
    FOLDER = 'videos_for_report'

    for folder in sorted(p(FOLDER).glob('*')):
        if not folder.is_dir():
            continue
        #if folder.name != '20260410':
        #    continue

        # flag
        exit_program = False

        # 重要變數
        VIDEO_FOLDER = folder
            
        # 啟動模擬器（非阻塞）
        try:
            sim_proc = subprocess.Popen(['python', 'simulate_ui_kura.py'])
        except OSError:
            logger.error(f'啟動 simulate_ui_kura.py 失敗: {traceback.format_exc()} ! 影片目錄: {folder}')
            raise

        # 給它一點時間起來，確認沒有立刻掛掉
        sleep(0.5)
        if sim_proc.poll() is not None:
            logger.error(f'simulate_ui_kura.py 啟動後立即結束，returncode={sim_proc.returncode} !')
            raise RuntimeError(f'simulate_ui_kura.py exited immediately with returncode {sim_proc.returncode}')

        # 初始化主程式物件
        device_code = socket.gethostname().split('-')[-1]
        app = App_HandWash(device_code)

        # 影片
        paths = sorted(p(folder).glob('**/*.mp4'), key=get_sort_key)
        for path in paths:
            try:
                if 'result' in path.stem:
                    continue

                # 更新待讀取的影片
                VIDEO_PATH = path
                CFG['camera']['video_path'] = str(VIDEO_PATH)  # 要讀取的影片
                app.camera = Camera(**CFG['camera'])
                app.run()

            except:
                logger.error(f'{path} 發生錯誤: {traceback.format_exc()} !')
            finally:
                logger.success('Application terminated !')

                # 退出程式
                if app:
                    app.camera.stop()
                    app.streamer.stop()
                    
                    if app.exit_program:
                        exit_program = True
                        break

        # 結束 app
        app.stop()
        
        # 結束模擬器
        sim_proc.terminate()
        try:
            sim_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning('simulate_ui_kura.py 未在時限內結束，強制 kill !')
            sim_proc.kill()
            sim_proc.wait()

        if sim_proc.returncode not in (0, -15, -9):  # -15=SIGTERM, -9=SIGKILL 都算正常收尾
            logger.error(f'simulate_ui_kura.py 異常結束，returncode={sim_proc.returncode} !')

        # 退出程式
        if exit_program:
            break
