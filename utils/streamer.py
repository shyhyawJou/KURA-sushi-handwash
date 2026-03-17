import cv2
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from time import sleep
import numpy as np
from loguru import logger
from .image import resize_keep_scale
from .timer import Timer



class Mjpeg_Streamer:
    def __init__(self, host="0.0.0.0", port=9527, route="/meal", size=(640, 480), quality=60, enable=True):
        self.host = host
        self.port = port
        self.route = route
        self.stream_size = size
        self.frame_to_stream = None
        self.quality = quality
        self.is_enable = enable
        self.is_running = False  # 控制迴圈的標記
        if not self.is_enable: 
            logger.info(f'streamer is enable??? -> {self.is_enable} !')
            return
        self.app = FastAPI()
        self._setup_routes()
        self.server_thread = None
        self.server = None

    def _setup_routes(self):
        @self.app.get(self.route)
        async def video_feed():
            return StreamingResponse(self._generate(), 
                                     media_type="multipart/x-mixed-replace; boundary=frame")

    def _generate(self):
        # 這裡檢查 is_running，確保停止時產生器也會終止
        while self.is_running:
            if self.frame_to_stream is None:
                sleep(0.01)
                continue
            
            try:
                with Timer('process streamer', silent=False):
                    #frame = cv2.resize(self.frame_to_stream, self.stream_size)
                    frame = resize_keep_scale(self.frame_to_stream, self.stream_size, 'center')
                    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                    if not ret:
                        continue
                
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                sleep(0.01) 
            except Exception as e:
                print(f"Streaming error: {e}")
                break

    def start(self):
        """啟動伺服器"""
        if self.is_running:
            print("[!] Streamer is already running.")
            return
        
        if not self.is_enable:
            return

        self.is_running = True
        
        # 設定 Uvicorn 配置
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)

        def run_server():
            # Uvicorn 的 run 是一個阻塞調用
            self.server.run()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        print(f"[*] MJPEG Streamer started at http://{self.host}:{self.port}{self.route}")

    def stop(self):
        """釋放資源並關閉伺服器"""
        print("[*] Stopping MJPEG Streamer...")
        self.is_running = False  # 停止 _generate 迴圈
        
        if self.server:
            self.server.should_exit = True # 告訴 Uvicorn 停止
        
        self.frame_to_stream = None # 釋放最後一幀影像記憶體
        
        if self.server_thread:
            self.server_thread.join(timeout=2)
        print("[*] MJPEG Streamer resources released.")

    def push_frame(self, frame):
        """更新最新影像"""
        if self.is_running and self.is_enable:
            self.frame_to_stream = frame