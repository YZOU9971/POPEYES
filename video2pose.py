import os
from typing import List
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from absl import logging
logging.set_verbosity(logging.ERROR)
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize MediaPipe Pose
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)


def get_coco_keypoints_optimized(image, pose):
    coco_indices = np.array([
        [0, 0], [2, 1], [5, 2], [7, 3], [8, 4],
        [11, 5], [12, 6], [13, 7], [14, 8],
        [15, 9], [16, 10], [23, 11], [24, 12],
        [25, 13], [26, 14], [27, 15], [28, 16]
    ])
    results = pose.process(image)
    keypoints = np.zeros((17, 3))
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark])
        selected = landmarks[coco_indices[:, 0]]
        left_hip, right_hip = landmarks[23], landmarks[24]
        center_x = (left_hip[0] + right_hip[0]) / 2
        center_y = (left_hip[1] + right_hip[1]) / 2
        selected[:, 0] -= center_x
        selected[:, 1] -= center_y
        keypoints[:len(selected), :] = selected
    return keypoints


def process_video_from_frames(video_path, output_path, batch_size=16):
    all_keypoints = []
    cap = cv2.VideoCapture(video_path)
    batch = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)
        frame_count += 1
        if len(batch) == batch_size:
            keypoints_batch = process_batch(batch, pose)
            all_keypoints.extend(keypoints_batch)
            batch = []
    if batch:
        keypoints_batch = process_batch(batch, pose)
        all_keypoints.extend(keypoints_batch)
    cap.release()
    np.save(output_path, all_keypoints)


def process_batch(batch, pose):
    keypoints_batch = []
    for img in batch:
        keypoints = get_coco_keypoints_optimized(img, pose)
        keypoints_batch.append(keypoints)
    return keypoints_batch


def get_video_tasks(video_dir: str, output_dir: str) -> List[dict]:
    tasks = []
    for filename in os.listdir(video_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
            tasks.append({'video_path': video_path, 'output_path': output_path})
    return tasks


def process_dataset(video_dir: str, output_dir: str, num_workers: int = 4, batch_size: int = 16):
    os.makedirs(output_dir, exist_ok=True)
    tasks = get_video_tasks(video_dir, output_dir)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_video_from_frames, task['video_path'], task['output_path'], batch_size)
            for task in tasks]
        for _ in tqdm(futures, desc="Processing videos"):
            _.result()


if __name__ == '__main__':
    VIDEO_DIR = r'C:\Learn\Codes\POPE\data\ff\ff_videos'
    OUTPUT_DIR = r'C:\Learn\Codes\POPE\data\ff\ff_pose'
    NUM_WORKERS = os.cpu_count() // 2
    BATCH_SIZE = 16

    process_dataset(VIDEO_DIR, OUTPUT_DIR, NUM_WORKERS, BATCH_SIZE)
