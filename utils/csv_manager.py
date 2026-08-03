import os
import csv
from datetime import datetime
from pathlib import Path as p
from loguru import logger



class Csv_Manager:
    def __init__(self, save_dir, output_path=None):
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
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self._find_today_csv(now)
        return path

    def _init_csv(self):
        self.headers = [
            "Store ID", "User ID", "User Name", "UTC Offset", "Login Mode", "Step Sequence", 
            "Start Time", "Action Confirmed Time", "End Time", "Step Count", "Is Detecting Step", 
            'Duration', "Frame", 'Step Length', 'Finish reason', 'Region'
        ]
        for i in range(1, 13):
            self.headers.append(f'Step{i} min count')
            self.headers.append(f'Step{i} min time')

        if not os.path.exists(self.file_path) or self.overwrite:
            with open(self.file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            logger.info(f"Created new CSV file: {self.file_path}")

        logger.info(f"today's record will be written to {self.file_path} !")

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
        
        region = data_dict['Region'].capitalize()
        logger.success(f"[{region}] Successfully exported wash record to {self.file_path}")

    def _find_today_csv(self, now: str):
        """ 找到與今天相同日期的 csv """
        for path in p(self.save_dir).glob('*.csv'):
            if path.stem[:8] == now[:8]:
                return path
        return f'{self.save_dir}/{now}.csv'