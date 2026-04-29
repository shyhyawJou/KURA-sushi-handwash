import cv2
import subprocess
import re
import numpy as np
import threading
import queue
from collections.abc import Iterable
from loguru import logger
from .image import resize_keep_scale



class Camera:
    def __init__(self, video_path, wh=(640, 480), crop_area=[], max_fake_frames=10):
        """
        :param wh: Tuple (width, height)
        :param crop_area: Tuple (x1, y1, x2, y2) or None
        """
        assert isinstance(crop_area, Iterable), f'crop_area should be python List but get {crop_area}'

        self.width = wh[0]
        self.height = wh[1]
        self.crop_area = np.array(crop_area)
        self.crop_coords = None
        self.video_path = video_path
        self.is_first_frame = True
        self.max_fake_frames = max_fake_frames
        self.n_fake_frame = 0
        
        # Threading & Queue
        self.frame_queue = queue.Queue(maxsize=1)
        self._thread = None
        self.capture = None

        logger.info(f'video read: {video_path}')
        logger.info(f'set crop area: {self.crop_area}')

    def start(self):
        """開啟影片"""
        self._open()
        logger.info("VideoCapture started.")

    def get_latest_frame(self):
        """外部呼叫此方法取得最新畫面"""
        try:
            return self._raw_read()
        except queue.Empty:
            return False, None

    def stop(self):
        """停止執行緒並釋放"""
        self._release()

    def set_crop(self, x1, y1, x2, y2):
        old = self.crop_area
        self.crop_area = np.array([x1, y1, x2, y2])
        self.crop_coords = None  # reset
        logger.success(f'set crop area from {old} to {self.crop_area} !')

    def _open(self):
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise ValueError(f"Failed to open camera: {self.video_path}")
        logger.info(f"Video initialized: {self.video_path} at {self.width}x{self.height}")

    def _raw_read(self):
        """底層讀取硬體影像並進行裁切"""
        if self.capture is None:
            return False, None
        
        ret, frame = self.capture.read()
        if not ret:
            self.n_fake_frame += 1
            return False if self.n_fake_frame == self.max_fake_frames else None, None

        # 重置
        self.n_fake_frame = 0

        if len(self.crop_area) > 0:
            if self.crop_coords is None:
                self.crop_coords = self._cal_crop_region(*frame.shape[:2])
            x1, y1, x2, y2 = self.crop_coords
            frame = frame[y1:y2, x1:x2].copy()

        # resize
        frame = resize_keep_scale(frame, (640, 480), 'corner')

        if self.is_first_frame:
            logger.info(f'frame (h, w): {(frame.shape[:2])}')
            self.is_first_frame = False

        return True, frame

    def _release(self):
        if self.capture:
            self.capture.release()
            logger.success(f"Video {self.video_path} and camera thread are released.")

    def _cal_crop_region(self, img_h, img_w):
        crop_area = self.crop_area * [img_w, img_h, img_w, img_h]
        return crop_area.astype(int)