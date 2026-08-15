import cv2
from pathlib import Path as p
import numpy as np
from queue import Queue, Full, Empty
from threading import Thread
import subprocess
import os
from os.path import dirname
from time import sleep, time
from datetime import datetime
from loguru import logger
import csv
import traceback
import shutil
from .tool import get_now_str, get_utc_offset



class Clip:
    def __init__(self, root_dir, enable, tag, crf, bitrate, fps):
        self.root_dir = root_dir
        self.tag = tag
        self.crf = crf
        self.bitrate = bitrate
        self.fps = fps
        self.suffix = f'_{tag.lower()}' if tag else ''
        self.is_enable = enable 
        if not self.is_enable:
            logger.warning(f'[{self.tag}] Clip function is disabled !')
        self._reset()

    def write_frame(self, frame, timestamp):
        if not self.is_enable:
            logger.warning(f'[{self.tag}] function is disabled !')
            return

        try:
            self.frame_q.put_nowait((frame, timestamp))
        except Full:
            self.n_discard_frame += 1  # 計算共丟棄了多少幀
    
    def start(self):
        if not self.is_enable:
            logger.warning(f'[{self.tag}] function is disabled !')
            return
        if self.is_running:
            logger.error(f'[{self.tag}] thread is not stopped, ignored to start a new thread !')
            return

        self.is_running = True
        self.frame_thread = Thread(target=self._run, daemon=True)
        self.frame_thread.start()

    def stop(self, save_video, is_interrupted=False):
        if not self.is_enable:
            logger.warning(f'[{self.tag}] function is disabled !')
            return
        self.is_running = False
        if self.frame_thread:
            self.frame_thread.join(1)
            self.frame_thread = None
        if self.n_discard_frame > 0:
            logger.warning(f'[{self.tag}] discarded the {self.n_discard_frame} frames !')

        if self.csv_file:
            self.csv_file.close()

        if self.save_dir:
            self._encode_video(save_video)
            if save_video and is_interrupted:
                self.video_thread.join(5.)

            if is_interrupted:
                for path in self.save_dir.parent.glob('*'):
                    if path.is_dir():
                        shutil.rmtree(path)

        self._reset()

    def _run(self):
        try:
            while self.is_running:
                try:
                    # timestamp: time(), timezone: int, relative to UTC timezone
                    frame, timestamp = self.frame_q.get_nowait()
                except Empty:
                    sleep(0.05)
                    continue

                now_str = get_now_str(timestamp, utc=False)
                time_offset = get_utc_offset()

                # 第一個 frame
                if self.first_time is None:
                    logger.info(f"[{self.tag}] first frame's time information: {now_str} !")
                    date = now_str[:8]
                    self.save_dir = p(f'{self.root_dir}/{date}/{now_str}{self.suffix}')
                    self.csv_path = p(f'{self.root_dir}/{date}/{now_str}{self.suffix}.csv')
                    self.video_path = p(f'{self.root_dir}/{date}/{now_str}{self.suffix}.mp4')
                    os.makedirs(self.save_dir, exist_ok=True)
                    self._write_csv_header()
                    self.first_time = (now_str, time_offset)
                    logger.success(f"[{self.tag}] the Clip thread started, "
                                   f"create clip's saved folder: {self.save_dir} and "
                                   f"csv path is {self.csv_path} !")

                # save frame and time
                dst_frame = f'{self.save_dir}/{now_str}.jpg'
                cv2.imwrite(dst_frame, frame)
                self._write_csv_row(now_str, time_offset)
                self.n_save_frame += 1

        except:
            logger.error(traceback.format_exc())
        finally:
            logger.success(f'[{self.tag}] clip frame queue thread stopped !')
            self.is_running = False

    def _write_csv_header(self):
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(['local time', 'utc offset'])

    def _write_csv_row(self, now_str, time_offset):
        if self.csv is None:
            logger.error(f'[{self.tag}] csv writer is not inited, ignored to write row !')
            return
        self.csv.writerow([now_str, time_offset])

    def _encode_video(self, save_video):
        """在背景執行緒跑 ffmpeg,不阻塞呼叫端。"""
        data = {'save_dir': self.save_dir, 'video_path': self.video_path, 
                'csv_path': self.csv_path, 'save_video': save_video, 'tag': self.tag,
                'fps': self.fps, 'bitrate': self.bitrate}
        self.video_thread = Thread(target=self._run_ffmpeg, args=(data,), daemon=True)
        self.video_thread.start()

    def _run_ffmpeg(self, data):
        try:
            # 變數
            save_dir = data['save_dir']
            video_path = data['video_path']
            csv_path = data['csv_path']
            save_video = data['save_video']
            tag = data['tag']
            fps = data['fps']
            bitrate = data['bitrate']

            # 不運作
            if not save_video:
                return

            frames = sorted(save_dir.glob('*.jpg'))
            if not frames or len(frames) < 2:
                logger.warning(f'[{tag}] no frame found in {save_dir} or number of frame < 2, skip ffmpeg encoding !')
                return

            #fps = self._calc_fps_from_filenames(tag, frames)

            list_path = save_dir / 'frames.txt'
            with open(list_path, 'w', encoding='utf-8') as f:
                for frame in frames:
                    f.write(f"file '{frame.name}'\n")
                    f.write(f"duration {1 / fps}\n")
                f.write(f"file '{frames[-1].name}'\n")

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', str(list_path),
                '-r', str(round(fps, 2)),
                '-vsync', 'cfr',
                '-pix_fmt', 'yuv420p',
                '-c:v', 'h264_v4l2m2m',
                '-b:v', f'{round(bitrate, 1)}M',           # 位元率
                '-maxrate', f'{round(bitrate * 2, 1)}M',
                '-bufsize', f'{round(bitrate * 4, 1)}M',
                '-g', str(int(fps * 2)),         # gop
                '-num_output_buffers', '32',
                '-num_capture_buffers', '32',
                str(video_path),
            ]
            logger.info(f'ffmpeg command: {cmd}')

            logger.info(f'[{tag}] start encoding video: {video_path} (fps={fps:.2f}) ...')
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            t0 = time()
            stdout, stderr = proc.communicate()
            t1 = time()

            if proc.returncode == 0:
                logger.success(f'[{tag}] encoded video successfully: {video_path}, frame: {len(frames)}, fps: {fps:.2f}, {t1 - t0:.3f} (s) !')
            else:
                logger.error(f'[{tag}] encode video failed! returncode={proc.returncode}\n{stderr}, {t1 - t0:.3f} (s)')
                if video_path.exists():
                    video_path.unlink()
                    logger.warning(f'[{tag}] deleted the {video_path} !')
        except:
            logger.error(traceback.format_exc())
        finally:
            if save_dir and save_dir.exists():
                shutil.rmtree(save_dir)
                logger.warning(f'[{tag}] deleted the {save_dir}')
            if not save_video and csv_path.exists():
                csv_path.unlink()
                logger.warning(f'[{tag}] deleted the {csv_path}')

    def _calc_fps_from_filenames(self, tag, frames, default_fps=30.0):
        """根據資料夾內第一張和最後一張圖片的檔名(時間戳)推算平均 fps。"""
        if len(frames) < 2:
            logger.warning(f'[{tag}] only {len(frames)} frame(s), cannot calc fps, '
                           f'fallback to default fps {default_fps} !')
            return default_fps

        fmt = "%Y%m%d %H%M%S.%f"  # 對應 get_now_str(utc=False) 的輸出格式
        try:
            first_dt = datetime.strptime(frames[0].stem, fmt)
            last_dt = datetime.strptime(frames[-1].stem, fmt)
        except ValueError as e:
            logger.error(f'[{tag}] failed to parse frame filename as datetime: {e}, '
                         f'fallback to default fps {default_fps} !')
            return default_fps

        duration = (last_dt - first_dt).total_seconds()
        if duration <= 0:
            logger.warning(f'[{tag}] invalid duration ({duration}s) calculated from filenames, '
                           f'fallback to default fps {default_fps} !')
            return default_fps

        fps = min((len(frames) - 1) / duration, default_fps)
        logger.info(f'[{tag}] calculated fps={fps:.2f} from {len(frames)} frames, '
                    f'duration={duration:.3f}s !')
        return fps

    def _reset(self):
        self.save_dir = None
        self.csv_path = None
        self.csv_file = None
        self.csv = None
        self.video_path = None
        self.frame_q = Queue(maxsize=256)
        self.n_discard_frame = 0
        self.n_save_frame = 0
        self.first_time = None  # 第一個 frame 的時間資訊
        self.frame_thread = None
        self.video_thread = None
        self.is_running = False
        logger.success(f'[{self.tag}] reset !')