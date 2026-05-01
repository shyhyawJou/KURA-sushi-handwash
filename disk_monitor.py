import shutil
import time
import os
from pathlib import Path
from loguru import logger



# --- 設定區 ---
PATH_TO_CHECK = "/mnt/reserved"
THRESHOLD_GB = 1
THRESHOLD_BYTES = THRESHOLD_GB * (1024 ** 3)
TIME_INTERVAL = 30
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


def get_dir_size(path: Path):
    """計算目錄總大小 (Bytes)"""
    return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())


def custom_cleanup_logic():
    base_path = Path(BIG_FILE_DIR)
    
    if not (base_path.exists() and base_path.is_dir()):
        print(f"Directory {BIG_FILE_DIR} not found.")
        return

    # 1. 取得當前剩餘空間與計算目標缺口
    current_free = get_free_space()
    if current_free is None:
        print("Failed to get disk space.")
        return
        
    # 計算還需要多少空間才達標 (Bytes)
    needed_size = THRESHOLD_BYTES - current_free
    
    if needed_size <= 0:
        print(f"Space already sufficient. Free: {current_free / (1024**3):.2f} GB")
        return

    print(f"Need to free up approximately: {needed_size / (1024**2):.2f} MB")

    # 2. 取得並按大小排序 (從小到大)
    folder_list = []
    for f in base_path.iterdir():
        if f.is_dir():
            # 使用先前定義的 get_dir_size
            folder_list.append((f, get_dir_size(f)))
    
    folder_list.sort(key=lambda x: x[1])

    # 3. 進入刪除迴圈，累加刪除量
    accumulated_deleted_size = 0
    
    for folder, size in folder_list:
        try:
            shutil.rmtree(folder)
            accumulated_deleted_size += size
            print(f"Deleted: {folder.name} ({size / (1024**2):.2f} MB)")
            
            # 判斷累積刪除量是否已達標
            if accumulated_deleted_size >= needed_size:
                print(f"Target reached! Total freed: {accumulated_deleted_size / (1024**2):.2f} MB")
                break
                
        except Exception as e:
            # 發生錯誤（如 Read-only）時，不累加大小並跳過
            print(f"Failed to delete {folder}: {e}")
            if "Read-only file system" in str(e):
                print("Critical: Disk is Read-only. Aborting cleanup.")
                break


def monitor_disk():
    free_space = get_free_space()
    if free_space is not None:
        free_space /= (1024 ** 3)
    logger.info(f"啟動監控任務。目標: {PATH_TO_CHECK} | "
                f"監測間隔: {TIME_INTERVAL} (s) | "
                f"門檻: {THRESHOLD_GB}GB | "
                f"現在剩 {free_space:.2f} GB")
    
    try:
        while True:
            free_space = get_free_space()
            
            if free_space is not None: 
                current_gb = free_space / (1024 ** 3)
                
                if free_space <= THRESHOLD_BYTES:
                    logger.warning(f"硬碟空間告急！當前僅剩: {current_gb:.2f} GB")
                    
                    custom_cleanup_logic()
                    
                    new_free_space = get_free_space()
                    new_gb = new_free_space / (1024 ** 3)
                    logger.info(f"清理後狀態確認。目前剩餘空間: {new_gb:.2f} GB")
                else:
                    logger.info(f'現在剩餘空間: {current_gb:.2f} GB')

            time.sleep(TIME_INTERVAL)  
            
    except KeyboardInterrupt:
        logger.warning("監控程序由使用者手動停止。")



if __name__ == "__main__":
    monitor_disk()