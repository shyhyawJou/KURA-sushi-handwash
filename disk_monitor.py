import shutil
import time
import os
from pathlib import Path
from loguru import logger



# --- 設定區 ---
PATH_TO_CHECK = "/mnt/reserved"
THRESHOLD_GB = 1
THRESHOLD_BYTES = THRESHOLD_GB * (1024 ** 3)
LOG_FILE = "/mnt/reserved/disk_monitor.log"  # 建議檢查開頭是否有 /
BIG_FILE_DIR = "/mnt/reserved/record/stream/"

logger.add(LOG_FILE, level="INFO")



def get_free_space():
    try:
        usage = shutil.disk_usage(PATH_TO_CHECK)
        return usage.free
    except Exception as e:
        logger.error(f"無法讀取路徑 {PATH_TO_CHECK}: {e}")
        return None


def custom_cleanup_logic():
    """
    修正後的刪除邏輯
    """
    base_path = Path(BIG_FILE_DIR)
    
    if base_path.exists() and base_path.is_dir():
        # 取得所有子目錄並排序 (按名稱排序，通常代表時間)
        # 這裡過濾出資料夾，避免刪到檔案
        folder = sorted([f for f in base_path.iterdir() if f.is_dir()])[0]
        
        try:
            shutil.rmtree(folder)
            logger.info(f"已刪除資料夾: {folder.name}")
            
            # 重新檢查空間
            space = get_free_space()
            if space is not None and space > THRESHOLD_BYTES:
                logger.success(f"空間已足夠 (> {THRESHOLD_GB}GB)，停止刪除。")
            else:
                logger.warning(f"剩餘空間: {space / (1024 ** 3)}, 30 秒後繼續刪除 !")

        except Exception as e:
            logger.error(f"刪除 {folder} 失敗: {e}")
    else:
        logger.warning(f"路徑 {BIG_FILE_DIR} 不存在，無法執行清理。")


def monitor_disk():
    free_space = get_free_space()
    if free_space is not None:
        free_space /= (1024 ** 3)
    logger.info(f"啟動監控任務。目標: {PATH_TO_CHECK} | 門檻: {THRESHOLD_GB}GB | 現在剩 {free_space:.2f} GB")
    
    try:
        while True:
            free_space = get_free_space()
            
            if free_space is not None and free_space <= THRESHOLD_BYTES:
                current_gb = free_space / (1024 ** 3)
                logger.warning(f"硬碟空間告急！當前僅剩: {current_gb:.2f} GB")
                
                custom_cleanup_logic()
                
                new_free_space = get_free_space()
                new_gb = new_free_space / (1024 ** 3)
                logger.info(f"清理後狀態確認。目前剩餘空間: {new_gb:.2f} GB")

            time.sleep(30)  
            
    except KeyboardInterrupt:
        logger.warning("監控程序由使用者手動停止。")



if __name__ == "__main__":
    monitor_disk()