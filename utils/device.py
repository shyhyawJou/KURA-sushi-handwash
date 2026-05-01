import yaml
import json
from collections import defaultdict
import numpy as np
from os.path import exists
from loguru import logger



class Device:
    """ 以 sink 的中心鉛直線區分左右裝置 """

    def __init__(self, path, device_code, ai_class):
        self.devices = self._load_yaml(path, device_code)
        self.data = {'left': defaultdict(list), 'right': defaultdict(list)}
        self.boxes = {'left': [], 'right': []}
        self.labels = {'left': [], 'right': []}
        self.class_map = {cls : ai_class.index(cls) for cls in self.devices}
        self._make_data()
        logger.success(f"load the devices' locations from key '{device_code}'!")
        
    @property
    def left_data(self):
        """ #### 格式: {"device name": boxes} """
        return self.data['left']

    @property
    def left_bboxes(self):
        return self.boxes['left']

    @property
    def left_labels(self):
        return self.labels['left']

    @property
    def right_data(self):
        """ #### 格式: {"device name": boxes} """
        return self.data['right']

    @property
    def right_bboxes(self):
        return self.boxes['right']

    @property
    def right_labels(self):
        return self.labels['right']

    def _make_data(self):
        # sink 中心線
        assert len(self.devices['sink']) == 1
        x1, y1, x2, y2 = self.devices['sink'][0]
        mid_x = (x1 + x2) / 2.

        # sink 切半
        left = [x1, y1, mid_x, y2]
        right = [mid_x, y1, x2, y2]
        self.data['left']['sink'].append(left)
        self.boxes['left'].append(left)
        self.labels['left'].append(self.class_map['sink'])
        self.data['right']['sink'].append(right)
        self.boxes['right'].append(right)
        self.labels['right'].append(self.class_map['sink'])

        # 區分左右
        for cls, boxes in self.devices.items():
            if cls == 'sink':
                continue

            for box in boxes:
                if (box[0] + box[2]) / 2. < mid_x:
                    self.data['left'][cls].append(box)
                    self.boxes['left'].append(box)
                    self.labels['left'].append(self.class_map[cls])
                else:
                    self.data['right'][cls].append(box)
                    self.boxes['right'].append(box)
                    self.labels['right'].append(self.class_map[cls])

        for name in ('left', 'right'):
            for cls, boxes in self.data[name].items():
                self.data[name][cls] = np.stack(boxes, axis=0, dtype='float32')
            self.boxes[name] = np.stack(self.boxes[name], axis=0, dtype='float32')
            self.labels[name] = np.array(self.labels[name])

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