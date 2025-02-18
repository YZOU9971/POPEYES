import os
from typing import NamedTuple, List
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from util.io import load_json

cv2.setNumThreads(0)

FS_LABEL_DIR = r'data/ff'


class Task(NamedTuple):
    video_name: str
    video_path: str
    out_path: str
    target_fps: float
    max_height: int


def get_fs_tasks(video_dir: str, out_dir: str, max_height: int) -> List[Task]:
    tasks = []
    for split in ['data']:
        split_file = os.path.join(FS_LABEL_DIR, split + '.json')
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found, skipping...")
            continue
        labels = load_json(split_file)
        for data in labels:
            video_name = data['video']
            video_out_path = os.path.join(out_dir, video_name) if out_dir else None
            video_path = os.path.join(video_dir, video_name + '.mp4')
            if not os.path.exists(video_path):
                print(f"Warning: {video_path} does not exist, skipping...")
                continue
            tasks.append(Task(
                video_name=video_name,
                video_path=video_path,
                out_path=video_out_path,
                target_fps=data['fps'],
                max_height=max_height))
    return tasks


def extract_frames(task: Task):
    target_width, target_height = task.max_height * 16 // 9, task.max_height
    vc = cv2.VideoCapture(task.video_path)

    if not vc.isOpened():
        print(f"Error: Unable to open video {task.video_path}")
        return

    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if task.max_height < h:
        oh = task.max_height
        ow = int(w / h * task.max_height)
    else:
        oh, ow = h, w

    if task.out_path:
        os.makedirs(task.out_path, exist_ok=True)

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
    previous_frame = None

    for i in range(total_frames):
        ret, frame = vc.read()
        if not ret:
            print(f"Warning: Frame {i} could not be read in {task.video_name}. Using previous frame.")
            if previous_frame is not None:
                frame = previous_frame.copy()
            else:
                frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)  # 如果没有上一帧，用空帧填充

        if frame.shape[0] != oh:
            top = (target_height - oh) // 2
            bottom = target_height - oh - top
            left = (target_width - ow) // 2
            right = target_width - ow - left
            frame = cv2.resize(frame, (ow, oh))
            frame = cv2.copyMakeBorder(frame, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=(123.68, 116.779, 103.939))

        frame_path = os.path.join(task.out_path, f'{i:06d}.jpg')
        cv2.imwrite(frame_path, frame)
        previous_frame = frame

    vc.release()



def main(video_dir: str, out_dir: str, max_height: int = 224, parallelism: int = 4):
    tasks = get_fs_tasks(video_dir, out_dir, max_height)
    os.makedirs(out_dir, exist_ok=True)

    with Pool(parallelism) as pool:
        for _ in tqdm(pool.imap_unordered(extract_frames, tasks), total=len(tasks), desc='Extracting'):
            pass
    print('Done!')


if __name__ == '__main__':
    VIDEO_DIR = r'data\ff\ff_videos'
    OUT_DIR = r'data\ff\ff_frames'
    MAX_HEIGHT = 224
    PARALLELISM = os.cpu_count() // 4

    main(VIDEO_DIR, OUT_DIR, MAX_HEIGHT, PARALLELISM)
