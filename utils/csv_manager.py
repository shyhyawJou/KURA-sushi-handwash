import os
import csv
from datetime import datetime
from pathlib import Path as p
from loguru import logger



class Csv_Manager:
    def __init__(self, save_dir='wash_logs', output_path=None):
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.overwrite = output_path is not None
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        if output_path:
            output_path = p(output_path)
            self.file_path = f'{save_dir}/{"/".join(output_path.parts[-3:-1])}/{output_path.stem}_result.csv'
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        else:
            self.file_path = self._generate_path()

        self._init_csv()

    def _generate_path(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.save_dir, f"{timestamp}.csv")

    def _init_csv(self):
        self.headers = ["Store ID", "Start Time", "End Time"]
        # 動態生成 Step1~12 的 flag, time, count 標題
        for suffix in ["flag", "time", "count"]:
            for i in range(1, 13):
                self.headers.append(f"Step{i} {suffix}")

        if not os.path.exists(self.file_path) or self.overwrite:
            with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            logger.info(f"Created new CSV file: {self.file_path}")

    def write_record(self, data_dict):
        # 規則 15: 檢查是否跨日，若是則更換檔案
        now_date = datetime.now().strftime('%Y%m%d')
        if now_date != self.current_date:
            self.current_date = now_date
            self.file_path = self._generate_path()
            self._init_csv()

        with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
            # 確保欄位順序正確
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(data_dict)
        logger.success(f"Successfully exported wash record to {self.file_path}")