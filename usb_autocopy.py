import os
from os.path import exists, dirname
import shutil
import subprocess
import time
import hashlib
import traceback
from datetime import datetime
from pathlib import Path as p
from loguru import logger
import glob



# --- 設定區 ---
SOURCE_DIR = "/mnt/reserved/record/stream"   # 影片來源資料夾
DEFAULT_MOUNT = "/mnt/usb"                   # 預設指定的掛載點
CURRENT_MOUNT = ''
CHECK_INTERVAL = 30                          # 檢查間隔（秒）
LOG_PATH = "/mnt/reserved/usb_autocopy.log"  # LOG 路徑
DELETE_OLD = True
DELETE_FAILED = True
TARGET_TOTAL_GB = 10                         # USB 總空間必須大於此數值才會被視為可能的 MOUNT POINT
RELOG_TIMEOUT = 3600                         # 重新等待印出連續的錯誤訊息 (秒)
MAX_ERROR = 50                               # 最大連續發生錯誤次數

# 
ERROR_COUNT = 0


def setup_logger():
    logger.add(
        LOG_PATH, 
        level="INFO",          # 儲存等級
        enqueue=False          # 同步寫入
    )
    logger.info(f'begin saving log into {LOG_PATH}')


def get_md5(file_path):
    """計算檔案的 MD5 值以確保完整性"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_device_total_gb(dev_path):
    """透過 lsblk 取得設備的總容量 (GB)，不需掛載即可檢查"""
    try:
        # 使用 lsblk 取得 bytes 單位的大小
        output = subprocess.check_output(
            ['lsblk', '-bno', 'SIZE', dev_path], 
            text=True
        ).strip()
        if output:
            return int(output) / (1024**3)
    except Exception as e:
        logger.error(f"無法讀取設備 {dev_path} 大小: {e}")
    return 0


def fix_readonly_device(dev_path, mount_point):
    """
    處理掛載失敗或唯讀問題：解除掛載、偵測格式、執行強力修復、重新掛載
    """
    logger.warning(f"Attempting to fix device: {dev_path}")
    try:
        # 1. 強制解除掛載 (確保乾淨)
        subprocess.run(["sudo", "umount", "-l", mount_point], check=False)
        subprocess.run(["sudo", "umount", "-l", dev_path], check=False)

        # 2. 偵測檔案系統格式 (重要：避免用錯工具)
        fs_type = ""
        try:
            blkid_out = subprocess.check_output(["sudo", "blkid", "-o", "value", "-s", "TYPE", dev_path], text=True).strip()
            fs_type = blkid_out.lower()
        except:
            logger.error(f"Cannot detect filesystem type for {dev_path}")

        # 3. 根據格式執行修復[cite: 1]
        if "exfat" in fs_type:
            # -y 比 -p 更強力，強制回答 yes 修復所有錯誤[cite: 1]
            repair_cmd = ["sudo", "fsck.exfat", "-y", dev_path]
        elif "ntfs" in fs_type:
            # NTFS 隨身碟常見問題修復工具
            repair_cmd = ["sudo", "ntfsfix", "-d", dev_path]
        else:
            # 通用修復嘗試
            repair_cmd = ["sudo", "fsck", "-y", dev_path]

        logger.info(f"Running repair ({fs_type}) on {dev_path}...")
        result = subprocess.run(repair_cmd, capture_output=True, text=True)
        
        # 輸出具體修復訊息，方便除錯
        if result.stdout: logger.info(f"Repair Info: {result.stdout.strip()}")

        # 4. 重新掛載[cite: 1]
        # 增加 -t 參數明確指定格式有助於解決 "wrong fs type" 報錯
        mount_cmd = ["sudo", "mount", "-o", "rw", dev_path, mount_point]
        if fs_type:
            mount_cmd = ["sudo", "mount", "-t", fs_type, "-o", "rw", dev_path, mount_point]
            
        subprocess.run(mount_cmd, check=True, capture_output=True)
        logger.success(f"Successfully fixed and mounted {dev_path} to {mount_point}[cite: 1]")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Mount/Repair command failed: {e.stderr.decode().strip()}")
        return False
    except Exception as e:
        logger.error(f"Self-healing critical failure: {e}")
        return False


def check_and_mount():
    """
    1. 檢查 /dev 下的 USB 分區數量
    2. 立即過濾總空間符合 TARGET_TOTAL_GB 的設備
    3. 若過濾後唯一，確認其掛載狀態並處理
    4. 回傳掛載路徑或空字串
    """
    # 搜尋所有 sd[a-z] 的分區 (確保包含 sda1, sdb1...)
    all_usb_devices = glob.glob("/dev/sd[a-z][1-9]")

    # 立即檢查大小 ---
    usb_devices = []
    for dev in all_usb_devices:
        total_gb = get_device_total_gb(dev)
        if total_gb >= TARGET_TOTAL_GB:
            usb_devices.append(dev)

    # 邏輯：必須「唯一」才處理
    if len(usb_devices) != 1:
        if len(usb_devices) == 0:
            logger.warning(f"No USB device detected. (>= {TARGET_TOTAL_GB}GB")
        else:
            logger.error(f"Multiple USB devices detected: {usb_devices}. Ambigious target!")
        return ""

    # 取得唯一的 USB 裝置路徑，並轉換為實體路徑 (處理 /dev/sda1 vs /dev/root 等問題)
    target_dev_path = os.path.realpath(usb_devices[0]) 

    current_mount_point = ""
    is_readonly = False
    found_in_mounts = False

    try:
        with open('/proc/mounts', 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                
                # 同樣將 /proc/mounts 裡的裝置名稱轉為實體路徑再比對
                mount_dev_path = os.path.realpath(parts[0])
                
                if mount_dev_path == target_dev_path:
                    found_in_mounts = True
                    current_mount_point = parts[1]
                    # 檢查掛載參數中是否包含 'ro'
                    if 'ro' in parts[3].split(','):
                        is_readonly = True
                    break
    except Exception as e:
        logger.error(f"Failed to read /proc/mounts: {e}")
        return ""

    # 情況 A: 裝置已掛載，但為唯讀
    if found_in_mounts and is_readonly:
        logger.error(f"Device {target_dev_path} is READ-ONLY! Attempting repair...")
        if fix_readonly_device(target_dev_path, current_mount_point):
            return current_mount_point
        else:
            return ""

    # 情況 B: 裝置完全沒掛載 (lsblk 有但 /proc/mounts 沒有)
    if not found_in_mounts:
        logger.info(f"Detected unique USB {target_dev_path} (Not mounted). Attempting to mount to {DEFAULT_MOUNT}...")
        
        if not os.path.exists(DEFAULT_MOUNT):
            os.makedirs(DEFAULT_MOUNT)

        try:
            # 執行掛載指令，明確指定 rw
            subprocess.run(["sudo", "mount", "-o", "rw", target_dev_path, DEFAULT_MOUNT], check=True, capture_output=True)
            logger.success(f"Successfully mounted {target_dev_path} to {DEFAULT_MOUNT}")
            return DEFAULT_MOUNT
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode().strip()
            logger.error(f"First mount attempt failed: {error_msg}")

            # --- 自我修復機制啟動 ---
            # 如果報錯包含 "wrong fs type" 或 "bad superblock"，嘗試修復
            if "wrong fs type" in error_msg or "bad superblock" in error_msg:
                logger.warning(f"Possible filesystem corruption detected on {target_dev_path}. Running fix...")
                
                # fsck.exfat 或 ntfsfix
                if fix_readonly_device(target_dev_path, DEFAULT_MOUNT):
                    return DEFAULT_MOUNT            
            
            return ""

    # 情況 C: 裝置已掛載且狀態正常 (RW)
    return current_mount_point


def sync_files():
    global CURRENT_MOUNT

    """搜尋並複製已完成的影片及其附屬檔案"""
    current_mount = check_and_mount()
    if CURRENT_MOUNT != current_mount:
        logger.info(f'change mount point from {CURRENT_MOUNT} to {current_mount}')
    CURRENT_MOUNT = current_mount

    # 若為空字串或該路徑並非掛載點
    if not CURRENT_MOUNT:
        return
    if not os.path.ismount(CURRENT_MOUNT):
        logger.error(f'{CURRENT_MOUNT} is not a mounted folder !')
        return

    # 找出所有影片檔（假設副檔名為 .mp4，可自行修改）
    for mark_path in list(p(SOURCE_DIR).glob('**/*.txt')):
        # 目標位置
        video_path = mark_path.with_suffix('.mp4')

        # 跳過
        if not exists(video_path):
            continue

        # 執行複製
        all_success = True
        paths = list(mark_path.parent.glob(f'{mark_path.stem}.*'))
        
        for path in paths:
            try:
                logger.info(f"Copying {path} to USB...")
                dst = f'{CURRENT_MOUNT}/{"/".join(path.parts[-2:])}'
                os.makedirs(dirname(dst), exist_ok=True)
                shutil.copy2(path, dst) # copy2 會保留元數據

                # 強制將快取寫入磁碟
                with open(dst, 'rb') as f:
                    os.fsync(f.fileno())       

                # 驗證 MD5 確保檔案正確
                if get_md5(path) == get_md5(dst):
                    logger.success(f"Verification Success: {dst}")                    
                else:
                    all_success = False
                    logger.error(f"Verification Failed: {dst}. Integrity error!")

            except Exception as e:
                all_success = False
                logger.error(f"Error copying {path}: {e}")

        # 複製成功後刪除來源檔, 失敗則刪除殘留在 USB 的檔案
        if all_success:
            for path in paths:
                if DELETE_OLD:
                    os.remove(path)
                    logger.warning(f"Deleted source: {path}")
        elif DELETE_FAILED:
            for path in paths:
                dst = f'{CURRENT_MOUNT}/{"/".join(path.parts[-2:])}'
                if exists(dst):
                    os.remove(dst)
                    logger.info(f"Cleaned up failed dst: {dst}")


def main():
    setup_logger()

    logger.info(f'Monitor started.\n'
                f'\tconfig:\n'
                f'\t\tvideo folder: {SOURCE_DIR}\n'
                f'\t\tcheck interval: {CHECK_INTERVAL} (s)\n'
                f'\t\tlog path: {LOG_PATH}\n'
                f'\t\tdelete files copied into USB: {DELETE_OLD}\n'
                f'\t\tdelete files failed to copy into USB : {DELETE_FAILED}\n'
                f'\t\tvalid USB min size: {TARGET_TOTAL_GB} GB')
    
    try:
        while True:
            try:
                sync_files()
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                logger.warning("Monitor stopped by KeyboardInterrupt")
                break
            except:
                logger.error(traceback.format_exc())
    finally:
        logger.info('program exited !')



if __name__ == "__main__":
    main()