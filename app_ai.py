import signal
import sys
import yaml
import traceback
import socket
from loguru import logger
from tqdm import tqdm
from datetime import datetime
from pathlib import Path as p
import re
from utils import (Mjpeg_Streamer, 
                   RTMDet_ONNX, 
                   Camera,
                   Timer,
                   Video,
                   plot_bbox, 
                   setup_logger,
                   Csv_Manager,
                   HandWashTracker,
                   draw_timestamp)


VIDEO_PATH = None


with open('utils/config.yaml') as f:
    CFG = yaml.safe_load(f)
    logger.success(f'config: {CFG}')


class App_HandWash:
    def __init__(self):
        self.is_running = False

        CFG['camera']['video_path'] = VIDEO_PATH  # 要讀取的影片
        CFG['video']['output_path'] = VIDEO_PATH  # 利用檔名產生輸出檔案
        CFG['csv']['output_path'] = VIDEO_PATH.with_suffix('.csv')  # 利用檔名產生輸出檔案

        self.camera = Camera(**CFG['camera'])
        self.streamer = Mjpeg_Streamer(**CFG['streamer'])
        self.ai_model = RTMDet_ONNX(**CFG['AI']['handwash'])
        self.video = Video(**CFG['video'])
        self.csv_manager = Csv_Manager(**CFG['csv'])

        # 檢測洗手
        self.tracker_left = HandWashTracker(zone_name="Left", logic_cfg=CFG['logic'])
        self.tracker_right = HandWashTracker(zone_name="Right", logic_cfg=CFG['logic'])

        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def run(self):
        try:
            self.camera.start()
            self.streamer.start()
            self.is_running = True
            
            logger.info("Main loop started.")

            # 進度條
            pbar = tqdm(unit='frame', desc='Processing')

            while self.is_running:
                with Timer('one complete loop', silent=True):
                    ret, frame = self.camera.get_latest_frame()
                    
                    pbar.update(1)

                    if ret is None:
                        logger.warning(f"{VIDEO_PATH} cannot read frame at {pbar.n}, skip this frame !")
                        continue
                    elif ret is False:
                        logger.error(f"{VIDEO_PATH} stop at {pbar.n} frame !")
                        break

                    # AI Inference
                    with Timer("AI detect", silent=True):
                        scores, boxes, pred_labels = self.ai_model(frame)

                    h, w = frame.shape[:2]
                    mid_x = w // 2

                    # 分類偵測結果至左右區
                    left_dets = []
                    right_dets = []
                    for score, box, label_idx in zip(scores, boxes, pred_labels):
                        label = self.ai_model.classes[label_idx]
                        det = {'box': box, 'label': label, 'score': score}  # label 是類別
                        center_x = (box[0] + box[2]) / 2
                        if center_x < mid_x:
                            left_dets.append(det)
                        else:
                            right_dets.append(det)

                    # 更新邏輯
                    now, res_l = self.tracker_left.update(left_dets)
                    if res_l: 
                        self.csv_manager.write_record(res_l)

                    now, res_r = self.tracker_right.update(right_dets)
                    if res_r: 
                        self.csv_manager.write_record(res_r)

                    # visualization
                    with Timer('copy frame', silent=True):
                        frame_copy = frame.copy()
                        plot_bbox(frame_copy, 
                                  boxes,
                                  pred_labels, 
                                  scores, 
                                  self.ai_model.classes, 
                                  **CFG['visualization']['bbox'])
                        
                        now_str = now.strftime('%Y%m%d %H%M%S.%f')[:-3]
                        draw_timestamp(frame_copy, now_str, **CFG['visualization']['timestamp'])

                    # Push to Streamer
                    self.streamer.push_frame(frame_copy) 
                    self.video.write_frame(frame_copy)

            pbar.close()

        except SystemExit:
            logger.success('System exit !')
        except:
            logger.error(f"Main process crashed: {traceback.format_exc()}")
        finally:
            self.stop()

    def handle_exit(self, signum, frame):
        logger.warning(f"Received signal {signum}, shutting down...")
        self.stop()

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        
        self.streamer.stop()
        self.camera.stop() # 統一透過物件釋放
        self.video.stop()

        logger.success("Program exited safely.")
        sys.exit(0)


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
    paths = sorted(p('video').glob('**/*.mp4'), key=get_sort_key)
    for path in paths:
        try:
            VIDEO_PATH = path
            setup_logger(**CFG['log'], suffix=socket.gethostname().split('-')[-1])
            app = App_HandWash()
            app.run()
        except SystemExit:
            logger.success('Application terminated.')
        except:
            logger.error(traceback.format_exc())