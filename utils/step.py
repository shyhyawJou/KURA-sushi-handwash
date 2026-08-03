from collections.abc import MutableMapping
from threading import Lock
from dataclasses import dataclass
from .tool import get_now_str



@dataclass
class Step:
    id: int
    frame: int
    count: int
    start_time: float
    step_confirmed_time: float
    end_time: float
    duration: float
    is_detecting_step: int

    def __repr__(self):
        return (
            f'\n'
            f'id: {self.id}\n'
            f'frame: {self.frame}\n'
            f'count: {self.count}\n'
            f'duration: {self.duration}\n'
            f'start_time: {get_now_str(self.start_time, utc=False)}\n'
            f'step_confirmed_time: {get_now_str(self.step_confirmed_time, utc=False)}\n'
            f'end_time: {get_now_str(self.end_time, utc=False)}\n'
            f'is_detecting_step: {self.is_detecting_step}\n'
        )
    

class Step_History:
    """ 過程中做的所有洗手步驟 """
    def __init__(self):
        self.ids = []
        self.counts = []
        self.start_times = []
        self.end_times = []
        self.step_confirmed_times = []
        self.durations = []
        self.frames = []
        self.is_detecting_steps = []
        self.detected_steps = {i: None for i in range(1, 13)}

    def __getitem__(self, idx):
        data = Step(self.ids[idx], self.frames[idx], self.counts[idx], self.start_times[idx], 
                    self.step_confirmed_times[idx], self.end_times[idx], self.durations[idx],
                    self.is_detecting_steps[idx])
        return data

    def append(self, step_id, count, start_time, end_time, step_confirmed_time, duration,
               frame, is_detecting_step):
        self.ids.append(step_id)
        self.counts.append(count)
        self.start_times.append(start_time)
        self.end_times.append(end_time)
        self.step_confirmed_times.append(step_confirmed_time)
        self.durations.append(duration)
        self.frames.append(frame)
        self.is_detecting_steps.append(int(is_detecting_step))

        # 已做過的步驟的紀錄, 用來畫在 frame debug 用
        if is_detecting_step:
            self.detected_steps[step_id] = Step(step_id, frame, count, start_time, 
                                                step_confirmed_time, end_time, duration,
                                                is_detecting_step)

    def insert(self, step: Step, idx):
        self.ids.insert(idx, step.id)
        self.counts.insert(idx, step.count)
        self.start_times.insert(idx, step.start_time)
        self.end_times.insert(idx, step.end_time)
        self.step_confirmed_times.insert(idx, step.step_confirmed_time)
        self.durations.insert(idx, step.duration)
        self.frames.insert(idx, step.frame)
        self.is_detecting_steps.insert(idx, step.is_detecting_step)
        if step.is_detecting_step:
            self.detected_steps[step.id] = step

    def pop(self):
        last_id = self.ids.pop()
        self.counts.pop()
        self.start_times.pop()
        self.end_times.pop()
        self.step_confirmed_times.pop()
        self.durations.pop()
        self.frames.pop()
        self.is_detecting_steps.pop()
        if last_id in self.detected_steps:
            self.detected_steps.pop(last_id)

    def __len__(self):
        return len(self.ids)
    

class MyDict(MutableMapping):
    """ 線程安全 """
    def __init__(self, data):
        self.data = data
        self.lock = Lock()

    def __getitem__(self, key):
        with self.lock:
            value = self.data[key]
        return value

    def __setitem__(self, key, value):
        with self.lock:
            self.data[key] = value

    def __delitem__(self, key):
        with self.lock:
            del self.data[key]

    def __iter__(self):
        with self.lock:
            data = iter(list(self.data.keys()))
        return data

    def __len__(self):
        with self.lock:
            n = len(self.data)
        return n

    def __repr__(self):
        with self.lock:
            data = dict(self.data)
        return f'{data}'

    def __contains__(self, key):
        with self.lock:
            res = key in self.data
        return res

    def items(self):
        with self.lock:
            items = list(self.data.items())
        return items