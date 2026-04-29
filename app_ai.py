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
import numpy as np
import socket
from utils import (Mjpeg_Streamer, 
                   RTMDet_DLA, 
                   Camera,
                   Video,
                   plot_bbox, 
                   setup_logger, Throttled_Logger,
                   Csv_Manager,
                   HandWashTracker,
                   draw_timestamp,
                   Device,
                   Result,
                   Timer)


VIDEO_PATH = None


with open('utils/config.yaml') as f:
    CFG = yaml.safe_load(f)
    logger.info(f'config: {CFG}')


class App_HandWash:
    def __init__(self, device_code):
        setup_logger(**CFG['log'], suffix=device_code)

        self.throttled_logger = Throttled_Logger(**CFG['throttled_logger'])
        self.is_running = False

        self.camera = Camera(**CFG['camera'])
        self.streamer = Mjpeg_Streamer(**CFG['streamer'])
        self.ai_model = RTMDet_DLA(**CFG['AI']['handwash'])
        self.device = Device(**CFG['device'], device_code=device_code)
        self.video = Video(**CFG['video'])
        self.csv_manager = Csv_Manager(**CFG['csv'])
        self.result_drawer = Result(**CFG['visualization']['result'])

        # 檢測洗手
        self.tracker_left = HandWashTracker(zone_name="Left", logic_cfg=CFG['logic'])
        self.tracker_right = HandWashTracker(zone_name="Right", logic_cfg=CFG['logic'])

        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

        logger.success('all init succeeded !')

    def run(self):
        try:
            self.camera.start()
            self.streamer.start()
            self.is_running = True
            loop_timer = Timer('one complete loop', silent=True)

            logger.info("Main loop started.")

            while self.is_running:
                with loop_timer:
                    ret, frame = self.camera.get_latest_frame()
                    if not ret:
                        logger.error(f"cannot read frame, skip this frame !")
                        continue
                                            
                    # AI Inference
                    scores, boxes, pred_labels = self.ai_model(frame)

                    # 裝置 (水龍頭、水槽等物件)
                    dev_names, dev_bboxes = self.device.named_bboxes
                    scores = np.concatenate([scores, [1.] * len(dev_names)])
                    boxes = np.concatenate([boxes, dev_bboxes], 0)
                    pred_labels = np.concatenate([pred_labels, [self.ai_model.classes.index(name) for name in dev_names]])

                    h, w = frame.shape[:2]
                    mid_x = w // 2

                    # 分類偵測結果至左右區
                    left_dets = []
                    right_dets = []
                    for score, box, label_idx in zip(scores, boxes, pred_labels):
                        label = self.ai_model.classes[label_idx]
                        det = {'box': box, 'label': label, 'score': score}  # label 是類別
                        center_x = (box[0] + box[2]) / 2
                        if label == 'sink':
                            left_dets.append(det)
                            right_dets.append(det)
                        elif center_x < mid_x:
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
                    frame_copy = frame.copy()

                    current_steps = [f'step {self.tracker_left.current_step}, {self.tracker_left.buffer_count}', 
                                     f'step {self.tracker_right.current_step}, {self.tracker_right.buffer_count}']
                    self.result_drawer.draw_step(frame_copy, current_steps)
                    #self.result_drawer.draw_region(frame_copy, np.asarray([d['box'] for d in left_dets]), 'L')
                    #self.result_drawer.draw_region(frame_copy, np.asarray([d['box'] for d in right_dets]), 'R')

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
                    self.video.write_frame(frame)

                # log
                self.throttled_logger.log(f'[{loop_timer.name}] {loop_timer.elapsed:.3f} (s)', 'DEBUG')

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
        
        self.camera.stop()
        self.streamer.stop()
        self.video.stop()

        logger.success("Program exited safely.")
        sys.exit(0)



if __name__ == "__main__":
    try:
        device_code = socket.gethostname().split('-')[-1]
        app = App_HandWash(device_code)
        app.run()
    except SystemExit:
        logger.success('Application terminated.')
    except:
        logger.error(traceback.format_exc())
