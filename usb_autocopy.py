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
    處理唯讀問題：解除掛載、執行修復、重新掛載
    """
    logger.warning(f"Attempting to fix read-only device: {dev_path}")
    try:
        # 1. 解除掛載
        subprocess.run(["sudo", "umount", "-l", mount_point], check=False)
        
        # 2. 根據檔案系統類型執行修復 (針對 exfat 使用 fsck.exfat)
        # -p: 自動修復, -y: 遇到問題一律回答 yes
        logger.info(f"Running fsck on {dev_path}...")
        repair_cmd = ["sudo", "fsck.exfat", "-p", dev_path]
        result = subprocess.run(repair_cmd, capture_output=True, text=True)
        
        if result.returncode <= 1: # 0 或 1 通常代表成功或已修正
            logger.success(f"Repair successful on {dev_path}")
        else:
            logger.error(f"Repair failed: {result.stderr}")
            return False

        # 3. 重新掛載
        subprocess.run(["sudo", "mount", "-o", "rw", dev_path, mount_point], check=True)
        logger.success(f"Successfully remounted {dev_path} as RW")
        return True
    except Exception as e:
        logger.error(f"Self-healing failed: {e}")
        return False


def check_and_mount():
    """
    1. 檢查 /dev 下的 USB 分區數量
    2. 立即過濾總空間符合 TARGET_TOTAL_GB 的設備
    3. 若過濾後唯一，確認其掛載狀態並處理
    4. 回傳掛載路徑或空字串
    """
    # 搜尋所有 sd[b-z] 的分區 (例如 sdb1, sdc1...)，排除系統碟 sda
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

    target_dev = usb_devices[0] # 唯一的 USB 裝置路徑

    # 檢查該裝置是否已經掛載在任何地方
    current_mount_point = ""
    is_readonly = False

    try:
        with open('/proc/mounts', 'r') as f:
            for line in f:
                parts = line.split()
                # parts[0] 是裝置名, parts[1] 是掛載路徑
                if parts[0] == target_dev:
                    current_mount_point = parts[1]
                    # 檢查掛載參數中是否包含 'ro'
                    if 'ro' in parts[3].split(','):
                        is_readonly = True
                    break
    except Exception as e:
        logger.error(f"Failed to read /proc/mounts: {e}")
        return ""

    # 如果發現唯讀，執行修復機制
    if is_readonly and current_mount_point:
        logger.error(f"Device {target_dev} is READ-ONLY!")
        if fix_readonly_device(target_dev, current_mount_point):
            return current_mount_point
        else:
            return ""

    # 如果沒掛載，嘗試掛載到指定的 DEFAULT_MOUNT
    if not current_mount_point:
        logger.info(f"Detected unique USB {target_dev}. Attempting to mount to {DEFAULT_MOUNT}...")
        
        if not os.path.exists(DEFAULT_MOUNT):
            os.makedirs(DEFAULT_MOUNT)

        try:
            # 執行掛載指令
            subprocess.run(["sudo", "mount", "-o", "rw", target_dev, DEFAULT_MOUNT], check=True, capture_output=True)
            logger.success(f"Successfully mounted {target_dev} to {DEFAULT_MOUNT}")
            return DEFAULT_MOUNT
        except subprocess.CalledProcessError as e:
            logger.error(f"Mount failed for {target_dev}: {e.stderr.decode().strip()}")
            return ""

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
    for mark_path in p(SOURCE_DIR).glob('**/*.txt'):
        # 目標位置
        video_path = mark_path.with_suffix('.mp4')
        video_dst = f'{CURRENT_MOUNT}/{"/".join(video_path.parts[-2:])}'

        # 跳過
        if not exists(video_path) or exists(video_dst):
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