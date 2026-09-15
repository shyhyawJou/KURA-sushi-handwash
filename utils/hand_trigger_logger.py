import os
import csv
from datetime import datetime
from pathlib import Path as p
from .tool import get_utc_offset
from loguru import logger


class HandTriggerLogger:
    """
    專門記錄「手觸發進入/離開」的時間點，跟主要洗手紀錄的 CSV (Csv_Manager) 完全分開，
    不會污染原本的資料。

    典型用途：login_mode = 'scanner' 時，機台其實是靠掃描器登入，
    但客戶想額外知道「手在鏡頭前出現/消失」這件事實際發生的時間，
    這個手觸發並不會真的讓系統登入，只是借用同一套去彈跳規則來計時。

    CSV 只有一列記錄一次完整的「進入 -> 離開」，欄位: Region, Enter Time, Exit Time。
    檔名產生方式、同一天只用同一份檔案的邏輯，都跟 Csv_Manager 一致。
    """
    HEADERS = ["Region", 'UTC Offset', "Enter Time", "Exit Time"]

    def __init__(self, save_dir):
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = self._generate_path()
        self._pending_enter = {}  # region -> enter time 字串, 等對應的離開時間一起寫成一列
        self._init_csv()

    def _generate_path(self):
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self._find_today_csv(now)
        return path

    def _find_today_csv(self, now: str):
        """ 找到與今天相同日期的 csv """
        for path in p(self.save_dir).glob('*.csv'):
            if path.stem[:8] == now[:8]:
                return path
        return f'{self.save_dir}/{now}.csv'

    def _init_csv(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(self.HEADERS)
            logger.info(f"Created new hand-trigger log file: {self.file_path}")

    def log(self, region: str, event: str, time_str: str):
        """
        event: 'Login' (手觸發進入) 或 'Logout' (手觸發離開)
        time_str: 建議用 utils.tool.get_now_str(now, utc=True) 產生的字串,
                  跟 handwash.py 的 login_time 格式一致

        'Login' 只會先記在記憶體裡, 等對應的 'Logout' 進來才會真的寫成一列
        (Region, Enter Time, Exit Time)。
        """
        if event == 'Login':
            self._pending_enter[region] = time_str
            return

        if event != 'Logout':
            return

        # 規則: 檢查是否跨日, 若是則更換檔案 (跟 Csv_Manager.write_record 一致)
        now_date = datetime.now().strftime('%Y%m%d')
        if now_date != self.current_date:
            self.current_date = now_date
            self.file_path = self._generate_path()
            self._init_csv()

        enter_time = self._pending_enter.pop(region, '')
        with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([region, get_utc_offset(), enter_time, time_str])

        logger.info(f"[{region}] hand-trigger {enter_time} -> {time_str}")