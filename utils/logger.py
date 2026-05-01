import os
from time import time
import sys
from loguru import logger
from .cfg import CFG



class Throttled_Logger:
    def __init__(self, log_interval=0.0):
        self.log_interval = log_interval
        self.last_log_time = 0.0

    def log(self, msg, level, reset=True):
        current = time()
        # check
        if current - self.last_log_time >= self.log_interval:
            logger.opt(depth=1).log(level, f"{msg}")
            if reset:
                self.last_log_time = current


MY_LOGGER = Throttled_Logger(**CFG['throttled_logger'])


def setup_logger(level="INFO", folder="logs", suffix=None):
    # 1. 確保日誌目錄存在
    if not os.path.exists(folder):
        os.makedirs(folder)

    logger.remove() 

    # 使用 {time} 佔位符，Loguru 會在建立檔案時自動填入時間
    if suffix is None:
        log_file_path = os.path.join(folder, "{time:YYYYMMDD}.log")
    else:
        log_file_path = os.path.join(folder, "{time:YYYYMMDD}" + f"-{suffix}.log")

    logger.add(
        log_file_path, 
        rotation="00:00",    # 每天午夜 00:00 自動輪替
        retention="30 days", # 保留最近 30 天的日誌
        level=level,         # 檔案通常存 INFO 以上即可，避免 DEBUG 塞爆硬碟
        encoding="utf-8",    # 確保中文不亂碼
        enqueue=True         # 是否要先寫入 queue
    )

    logger.add(
        sys.stderr, 
        level=level,         
        enqueue=True         # 是否要先寫入 queue
    )

    logger.success(f'log file path: {log_file_path}')