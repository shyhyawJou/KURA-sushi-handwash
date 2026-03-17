import cv2
import subprocess
import re
import numpy as np
import threading
import queue
from collections.abc import Iterable
from loguru import logger
from .timer import Timer



class Camera:
    def __init__(self, wh=(640, 480), crop_area=[]):
        """
        :param wh: Tuple (width, height)
        :param crop_area: Tuple (x1, y1, x2, y2) or None
        """
        assert isinstance(crop_area, Iterable), f'crop_area should be python List but get {crop_area}'

        self.width = wh[0]
        self.height = wh[1]
        self.crop_area = np.array(crop_area)
        self.crop_coords = None
        self.device_path = ""
        self.is_first_frame = True
        
        # Threading & Queue
        self.frame_queue = queue.Queue(maxsize=1)
        self._is_running = False
        self._thread = None
        self.capture = None

        logger.info(f'set crop area: {self.crop_area}')

    def start(self):
        """啟動背景取像執行緒"""
        self._open()
        self._is_running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("Camera background thread started.")

    def get_latest_frame(self):
        """外部呼叫此方法取得最新畫面"""
        try:
            return True, self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def stop(self):
        """停止執行緒並釋放"""
        self._is_running = False
        if self._thread:
            self._thread.join()

    def set_crop(self, x1, y1, x2, y2):
        old = self.crop_area
        self.crop_area = np.array([x1, y1, x2, y2])
        self.crop_coords = None  # reset
        logger.success(f'set crop area from {old} to {self.crop_area} !')

    def _find_usb_camera(self):
        result = subprocess.run('v4l2-ctl --list-devices', capture_output=True, text=True, shell=True)
        camera_ids = []
        is_usb_cam = False
        for s in result.stdout.split('\n'):
            if 'USB' in s and 'Camera' in s:
                is_usb_cam = True
            elif is_usb_cam and 'video' in s:
                res = re.findall(r'\d+', s)
                if res:
                    camera_ids.append(int(res[0]))
                else:
                    break
        if not camera_ids:
            raise ValueError('Cannot find any USB camera!')
        return f'/dev/video{camera_ids[0]}'

    def _open(self):
        self.device_path = self._find_usb_camera()
        gst_str = (
            f'v4l2src device={self.device_path} ! '
            f'image/jpeg,width={self.width},height={self.height},framerate=30/1 ! '
            f'jpegdec ! videoconvert ! appsink drop=true max-buffers=1'
        )

        self.capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            raise ValueError(f"Failed to open camera: {self.device_path}")
        
        logger.info(f"Camera initialized: {self.device_path} at {self.width}x{self.height}")

    def _raw_read(self):
        """底層讀取硬體影像並進行裁切"""
        if self.capture is None:
            return False, None
        
        ret, frame = self.capture.read()
        if not ret:
            return False, None

        if len(self.crop_area) > 0:
            if self.crop_coords is None:
                self.crop_coords = self._cal_crop_region(*frame.shape[:2])
            x1, y1, x2, y2 = self.crop_coords
            #with Timer('copy crop region of frame'):
            frame = frame[y1:y2, x1:x2].copy()

        if self.is_first_frame:
            logger.info(f'frame (h, w): {(frame.shape[:2])}')
            self.is_first_frame = False

        return True, frame

    def _update_loop(self):
        """背景執行緒迴圈：確保 queue 永遠存有最新的一張圖"""
        while self._is_running:
            ret, frame = self._raw_read()
            if ret:
                # 若 queue 已滿，移除舊幀放入新幀，確保處理延遲最低
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
        
        # 釋放資源
        self._release()
    
    def _release(self):
        if self.capture:
            self.capture.release()
            logger.success(f"Camera {self.device_path} and camera thread are released.")

    def _cal_crop_region(self, img_h, img_w):
        crop_area = self.crop_area * [img_w, img_h, img_w, img_h]
        return crop_area.astype(int)