import signal
import sys
import yaml
import traceback
import socket
from loguru import logger
from utils import (Mjpeg_Streamer, 
                   RTMDet_DLA, 
                   Camera,
                   Timer,
                   Video,
                   plot_bbox, 
                   plot_class,
                   setup_logger)



with open('utils/config.yaml') as f:
    CFG = yaml.safe_load(f)
    logger.success(f'config: {CFG}')


class App_HandWash:
    def __init__(self):
        self.is_running = False

        self.camera = Camera(**CFG['camera'])
        self.streamer = Mjpeg_Streamer(**CFG['streamer'])
        self.ai_model = RTMDet_DLA(**CFG['AI']['handwash'])
        self.video = Video(**CFG['video'])
        
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def run(self):
        try:
            self.camera.start()
            self.streamer.start()
            self.is_running = True
            
            logger.info("Main loop started.")
            
            while self.is_running:
                with Timer('one complete loop'):
                    ret, frame = self.camera.get_latest_frame()
                    if not ret:
                        continue

                    # AI Inference
                    #with Timer("AI detect"):
                    scores, boxes, pred_labels = self.ai_model(frame)

                    # visualization
                    #with Timer('copy frame'):
                    frame_copy = frame.copy()
                    plot_bbox(frame_copy, boxes, pred_labels, 2)
                    plot_class(frame_copy, scores, self.ai_model.classes, pred_labels, boxes, 2)
                    
                    # Push to Streamer
                    self.streamer.push_frame(frame_copy) 
                    self.video.write_frame(frame)

        except SystemExit:
            pass
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



if __name__ == "__main__":
    try:
        setup_logger(**CFG['log'], suffix=socket.gethostname().split('-')[-1])
        app = App_HandWash()
        app.run()
    except SystemExit:
        logger.success('Application terminated.')
    except:
        logger.error(traceback.format_exc())