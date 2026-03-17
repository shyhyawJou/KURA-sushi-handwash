import os
import sys
from loguru import logger



def setup_logger(level="INFO", folder="logs", suffix=None):
    # 1. 確保日誌目錄存在
    if not os.path.exists(folder):
        os.makedirs(folder)

    logger.remove() 

    # 2. 輸出到控制台 (保持即時 Debug 用)
    logger.add(
        sys.stderr, 
        level='DEBUG', 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 3. 輸出到檔案 (關鍵修改：檔名與輪替)
    # 使用 {time} 佔位符，Loguru 會在建立檔案時自動填入時間
    if suffix is None:
        log_file_path = os.path.join(folder, "{time:YYYYMMDD}.log")
    else:
        log_file_path = os.path.join(folder, "{time:YYYYMMDD}" + f"-{suffix}.log")

    logger.add(
        log_file_path, 
        rotation="00:00",    # 每天午夜 00:00 自動輪替
        retention="30 days", # 保留最近 30 天的日誌
        level=level,     # 檔案通常存 INFO 以上即可，避免 DEBUG 塞爆硬碟
        encoding="utf-8",    # 確保中文不亂碼
        enqueue=True         # 啟動非同步寫入，避免 Log 影響 AI 推論效能
    )

    logger.success(f'log file path: {log_file_path}')