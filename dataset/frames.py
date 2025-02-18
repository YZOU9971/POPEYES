# Standard imports
import copy
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import pickle

# Local imports
from util.io import load_json


class FrameReader:

    def __init__(self, frame_dir):
        self._frame_dir = frame_dir

    def read_frame(self, frame_path):
        return torchvision.io.read_image(frame_path)

    def load_paths(self, video_name, clip_len, start, end, stride=1):
        """
        :param video_name: Video name (basename)
        :param clip_len: Clip length (should be)
        :param start: Start frame index
        :param end: End frame index (+1)
        :param stride: Stride
        :return: [base_path, start, frame name digits, clip_len, fact clip_len]
        """
        ndigits = 6
        base_path = os.path.join(self._frame_dir, video_name)
        found_start = -1
        length = (end - start) // stride

        for frame_num in range(start, end, stride):

            frame_idx = frame_num
            frame_filename = f'{(str(frame_idx).zfill(ndigits))}.jpg'
            frame_path = os.path.join(base_path, frame_filename)

            if os.path.exists(frame_path):
                if found_start == -1:
                    found_start = frame_idx
        return [base_path, found_start, ndigits, clip_len, length]

    def get_blank_frame(self, shape=(3, 224, 224), mean=(0.485, 0.456, 0.406)):
        blank = torch.zeros(shape)
        for c in range(shape[0]):
            blank[c, :, :] = mean[c]
        return blank

    def load_frames(self, paths, stride=1):
        """
        load frames from path, if not satisfy clip_len, pad with blank images of ImageNet mean
        :return: torch.Tensor of size (clip_len, c, h, w)
        """
        base_path, start, ndigits, clip_len, length = paths
        frames = []

        for j in range(length):
            frame_num = start + j * stride
            frame_filename = f'{(str(frame_num).zfill(ndigits))}.jpg'
            frame_path = os.path.join(base_path, frame_filename)
            img = self.read_frame(frame_path)
            frames.append(img)
        if not frames:
            raise ValueError(f'No frames loaded from {base_path}')

        if length < clip_len:
            for i in range(clip_len - length):
                blank_frame = torch.zeros_like(frames[0])
                frames.append(blank_frame)

        frames_tensor = torch.stack(frames, dim=0)
        return frames_tensor


class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            pose_dir,
            store_dir,
            store_mode,
            clip_len,
            dataset_len,
            stride=1,
            overlap_ratio=1,
            radi_displacement=0,
            mix_up=False,
            dataset='ff'
    ):
        assert store_mode in ['store', 'load'], 'store_mode must be either "store" or "load"'
        assert clip_len > 0, 'clip_len must be positive'
        assert stride > 0, 'stride must be positive'
        assert dataset_len > 0, 'dataset_len must be positive'
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._pose_dir = pose_dir
        self._split = os.path.splitext(os.path.basename(label_file))[0]
        self._classes_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._dataset = dataset
        self._store_dir = store_dir
        self._store_mode = store_mode
        self._clip_len = clip_len
        self._stride = stride
        self._overlap_len = overlap_ratio * clip_len
        assert 0 <= self._overlap_len < clip_len, 'overlap_len must be in [0, clip_len)'
        self._step_size = clip_len - self._overlap_len
        assert self._step_size > 0, 'step_size must be positive'
        self._dataset_len = dataset_len
        self._radi_displacement = radi_displacement
        self._mix_up = mix_up
        self._frame_reader = FrameReader(frame_dir)

        if self._store_mode == 'store':
            self.store_clips()
        else:
            self.load_clips()
        self._total_len = len(self._frame_paths)            # num of videos

    def store_clips(self):
        self._frame_paths = []
        self._pose_paths = []
        self._labels_store = []
        self._interval = int(self._step_size * self._stride)
        if self._radi_displacement > 0:
            self._labelsD_store = []

        for video in tqdm(self._labels):
            video_len = int(video['num_frames'])
            labels_file = video['events']
            for base_idx in range(0, video_len + 1 - self._interval, self._interval):
                frame_paths = self._frame_reader.load_paths(
                    video['video'],
                    clip_len=self._clip_len,
                    start=base_idx,
                    end=min(base_idx + self._clip_len, video_len),
                    stride=self._stride)
                pose_paths = [os.path.join(self._pose_dir, video['video']) + '.npy', base_idx]
                labels = []
                if self._radi_displacement > 0:
                    labelsD = []
                for event in labels_file:
                    event_frame = event['frame']
                    label_idx = (event_frame - base_idx) // self._stride
                    if label_idx > base_idx + self._clip_len or label_idx < base_idx:
                        continue
                    if self._radi_displacement > 0:
                        label = self._classes_dict[event['label']]
                        start = max(0, label_idx - self._radi_displacement)
                        end = min(self._clip_len, label_idx + self._radi_displacement + 1)
                        for i in range(start, end):
                            labels.append({'label': label, 'label_idx': i})
                            labelsD.append({'displ': i - label_idx, 'label_idx': i})
                    else:
                        label = self._classes_dict[event['label']]
                        start = max(0, label_idx)
                        end = min(self._clip_len, label_idx + 1)
                        for i in range(start, end):
                            labels.append({'label': label, 'label_idx': i})

                self._frame_paths.append(frame_paths)
                self._pose_paths.append(pose_paths)
                self._labels_store.append(labels)
                if self._radi_displacement > 0:
                    self._labelsD_store.append(labelsD)
                if base_idx + self._clip_len > video_len:
                    break
        # save to store
        store_subdir = f'LEN{self._clip_len}_DIS{self._radi_displacement}_SPLIT{self._split}'
        store_path = os.path.join(self._store_dir, store_subdir)

        os.makedirs(store_path, exist_ok=True)
        with open(store_path + r'\frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + r'\pose_paths.pkl', 'wb') as f:
            pickle.dump(self._pose_paths, f)
        with open(store_path + r'\labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        if self._radi_displacement > 0:
            with open(store_path + r'\labelsD.pkl', 'wb') as f:
                pickle.dump(self._labelsD_store, f)
        print(f'Stored clips to {store_path}, {len(self._frame_paths)} clips in total')

    def load_clips(self):
        store_subdir = f'LEN{self._clip_len}_DIS{self._radi_displacement}_SPLIT{self._split}'
        store_path = os.path.join(self._store_dir, store_subdir)

        with open(store_path + r'\frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + r'\pose_paths.pkl', 'rb') as f:
            self._pose_paths = pickle.load(f)
        with open(store_path + r'\labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        if self._radi_displacement > 0:
            with open(store_path + r'\labelsD.pkl', 'rb') as f:
                self._labelsD_store = pickle.load(f)
        print(f'Loaded clips from {store_path}, {len(self._frame_paths)} clips in total')

    def __len__(self):
        return self._dataset_len

    def _get_one(self):
        idx = random.randint(0, self._total_len-1)
        frames_path = self._frame_paths[idx]
        labels_dict = self._labels_store[idx]
        # Load frames
        frames = self._frame_reader.load_frames(frames_path, stride=self._stride)
        # Load Poses
        poses_info = self._pose_paths[idx]
        poses_path = poses_info[0]
        start = poses_info[1]
        end = poses_info[1]+self._clip_len
        poses_whole = np.load(poses_path)
        l, j, c = poses_whole.shape
        if end >= l + 1:
            padding = np.zeros((end - l, j, c), dtype=poses_whole.dtype)
            poses_whole_padded = np.concatenate((poses_whole, padding), axis=0)
            poses = poses_whole_padded[start:end]
        else:
            poses = poses_whole[start:end]

        labels = np.zeros(self._clip_len, np.int64)
        for label in labels_dict:
            labels[label['label_idx']] = label['label']
        if self._radi_displacement > 0:
            labelsD_dict = self._labelsD_store[idx]
            labelsD = np.zeros(self._clip_len, np.int64)
            for label in labelsD_dict:
                labelsD[label['label_idx']] = label['displ']
            return {
                'video_name': os.path.basename(frames_path[0]),
                'start': start,
                'frame': frames,
                'pose': poses,
                'contains_event': int(np.sum(labels) > 0),
                'label': labels,
                'labelD': labelsD
            }
        return {
            'video_name': os.path.basename(frames_path[0]),
            'start': start,
            'frame': frames,
            'pose': poses,
            'contains_event': int(np.sum(labels) > 0),
            'label': labels
        }

    def __getitem__(self, idx):
        ret = self._get_one()
        if self._mix_up:
            mix = self._get_one()
            ret['frame2'] = mix['frame']
            ret['pose2'] = mix['pose']
            ret['contains_event2'] = mix['contains_event']
            ret['label2'] = mix['label']
            if self._radi_displacement > 0:
                ret['labelD2'] = mix['labelD']
        return ret

    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        labels_file = meta['events']
        num_frames = meta['num_frames']
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int64)
        for event in labels_file:
            frame = event['frame']
            if frame < num_frames:
                labels[frame // self._stride] = self._classes_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return labels

    @property
    def videos(self):
        return sorted([(
            v['video'],
            v['num_frames'] // self._stride,
            v['fps'] / self._stride) for v in self._labels
        ])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy['fps'] /= self._stride
                x_copy['num_frames'] /= self._stride

                for e in x_copy['events']:
                    e['frame'] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


def _print_info_helper(src_file, labels):
    num_frames = sum(x['num_frames'] for x in labels)
    num_events = sum(len(x['events']) for x in labels)
    num_videos = len(labels)
    non_bg_percentage = num_events / num_frames * 100 if num_frames > 0 else 0
    print(f"{src_file}: {num_videos} Videos, {num_frames} Frames, {non_bg_percentage:.5f}% Non-bg frames")
