import yaml
import json
from collections import defaultdict
import numpy as np
from os.path import exists
from loguru import logger



class Device:
    def __init__(self, path, device_code):
        self.devices = self._load_yaml(path, device_code)
        self.classes = []
        self.bboxes = []
        self._make_bboxes()
        logger.success(f"load the devices' locations from key '{device_code}'!")
        
    @property
    def named_bboxes(self):
        return self.classes, self.bboxes

    def _make_bboxes(self):
        for cls, boxes in self.devices.items():
            for bbox in boxes:
                self.classes.append(cls)
                self.bboxes.append(bbox)

    def _load_yaml(self, path, device_code):
        with open(path) as f:
            devices = yaml.safe_load(f)[device_code]
        return devices
    

def save_device():
    JSON_PATH = 'z1_20260409_this.json'
    SAVE_PATH = 'utils/device.yaml'
    MAC = '0793'

    with open(JSON_PATH) as f:
        data = json.load(f)    

    print(f'read the {JSON_PATH}')

    labels = defaultdict(list)

    for info in data['shapes']:
        pts = info['points']
        x1, y1 = np.min(pts, 0).tolist()
        x2, y2 = np.max(pts, 0).tolist()
        labels[info['label']].append([x1, y1, x2, y2])

    # final
    if exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            final_data = yaml.safe_load(f)
        final_data[MAC] = dict(labels)
        print(f'{SAVE_PATH} 已存在且更新完畢 !')
    else:
        final_data = {MAC: dict(labels)}

    with open(SAVE_PATH, 'w') as f:
        yaml.safe_dump(final_data, f, indent=4)

    print(f'save the {SAVE_PATH}')



if __name__ == '__main__':
    save_device()