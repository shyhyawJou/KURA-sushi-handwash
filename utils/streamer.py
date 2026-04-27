import cv2
import threading
import uvicorn
import queue
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from time import sleep, time
import numpy as np
from loguru import logger
from .image import resize_keep_scale
from .timer import Timer

class Mjpeg_Streamer:
    def __init__(self, 
                 host="0.0.0.0", 
                 port=9527, 
                 route="/meal", 
                 size=(640, 480), 
                 quality=60, 
                 enable=True, 
                 fps=None, 
                 force_stop=True):
        
        self.host = host
        self.port = port
        self.route = route
        self.stream_size = size
        self.fps = fps
        self.quality = quality
        
        # 使用 Queue 來解耦主程序與處理程序
        self.frame_queue = queue.Queue(maxsize=2)
        self.processed_bytes = None  # 儲存處理完後的 JPEG bytes
        
        self.is_enable = enable
        self.is_running = False
        self.force_stop = force_stop
        
        if not self.is_enable: 
            logger.info(f'Streamer is disabled.')
            return
            
        self.app = FastAPI()
        self._setup_routes()
        self.server_thread = None
        self.worker_thread = None
        self.server = None

    def _setup_routes(self):
        @self.app.get(self.route)
        async def video_feed():
            return StreamingResponse(self._generate(), 
                                     media_type="multipart/x-mixed-replace; boundary=frame")

    def _worker(self):
        """背景處理執行緒：負責消耗 Queue 中的影像並進行編碼"""
        while self.is_running:
            try:
                # 取得原始影格，設定 timeout 避免死鎖
                frame = self.frame_queue.get(timeout=1)
                
                with Timer('process streamer', silent=True):
                    # 在背景執行緒處理耗時的 resize 與編碼
                    processed_frame = resize_keep_scale(frame, self.stream_size, 'center')
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                    
                    if ret:
                        # 封裝成標準的 MJPEG 影格格式
                        self.processed_bytes = (b'--frame\r\n'
                                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # 完成後標記任務
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker processing error: {e}")

    def _generate(self):
        """發送執行緒：僅負責發送處理好的內容"""
        last = 0
        target_interval = 1. / self.fps if self.fps and self.fps > 0 else 0

        while self.is_running:
            if self.processed_bytes is None:
                sleep(0.01)
                continue

            cur = time()
            if cur - last < target_interval:
                # 精確計算休眠時間
                sleep_time = max(0.001, target_interval - (cur - last))
                sleep(sleep_time)
                continue
            
            last = cur
            yield self.processed_bytes
            
            if target_interval == 0:
                sleep(0.01)

    def start(self):
        """啟動伺服器與背景處理執行緒"""
        if self.is_running:
            print("[!] Streamer is already running.")
            return
        
        if not self.is_enable:
            return

        self.is_running = True
        
        # 啟動處理影像的 Worker
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        # 設定 Uvicorn
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)

        def run_server():
            self.server.run()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        print(f"[*] MJPEG Streamer started at http://{self.host}:{self.port}{self.route}")

    def stop(self):
        """釋放資源"""
        print("[*] Stopping MJPEG Streamer...")
        self.is_running = False 
        
        if self.server:
            self.server.should_exit = True
        
        self.processed_bytes = None
        
        # 清空 Queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if self.server_thread:
            self.server_thread.join(timeout=2)
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
            
        print("[*] MJPEG Streamer resources released.")

    def push_frame(self, frame):
        """更新影像：現在這對主程序來說非常快"""
        if self.is_running and self.is_enable:
            try:
                # 如果 Queue 滿了，取出最舊的幀（丟棄），確保即時性
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # 放入新幀，非阻塞型態
                self.frame_queue.put_nowait(frame)
            except Exception:
                # 避免主程序因為推播錯誤而中斷
                pass