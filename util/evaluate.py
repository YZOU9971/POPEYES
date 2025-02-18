# Standard imports
import json

from tabulate import tabulate
from tqdm import tqdm
import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import copy
from util.io import store_json

# Local imports
from util.score import compute_mAPs

# Constants
TOLERANCES = [1, 2, 4]
WINDOWS = [1, 3]
INFERENCE_BATCH_SIZE = 4


class ErrorStat:
    """
    统计帧级别预测的错误率 / 准确率
    """
    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        # true, pred: shape = [N,], 表示某段 clip 内每帧的真值和预测
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get_error_rate(self):
        return self._err / self._total

    def get_acc(self):
        return 1.0 - self.get_error_rate()


class ForegroundF1:
    """
    统计前景帧（非背景）预测的精确率、召回率，用于计算 F1
    这里对每个类别以及整体 (None) 都记录 TP, FP, FN
    """
    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        # 如果 pred != 0，则说明网络预测该帧是前景（label != 0）
        if pred != 0:
            if true != 0:
                # 整体 +1 个 TP
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            # 按类别统计
            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                    self._fn[true] += 1

        # 如果 pred == 0，但 true != 0，就说明漏检了
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get_f1(self, label):
        denom = self._tp[label] + 0.5 * self._fp[label] + 0.5 * self._fn[label]
        if denom == 0:
            # 当 denom=0 时，说明没有检测到任何该 label，也没有 GT 是该 label
            # 这种情况下 F1 定义为 0
            return 0.0
        return self._tp[label] / denom

    def get(self, label):
        return self.get_f1(label)

    def tp_fp_fn(self, label):
        return self._tp[label], self._fp[label], self._fn[label]


def non_maximum_suppression(
    predictions: List[Dict],
    window_size: int,
    score_threshold: float = 0.05
) -> List[Dict]:
    """
    对每个视频做 NMS，按分数从高到低遍历，移除相邻 (|frame_i - frame_j| <= window_size) 的事件
    """
    suppressed_pred = copy.deepcopy(predictions)
    new_pred = []

    for video_pred in suppressed_pred:
        # 按 label 分组，分别做 NMS
        events_by_label = defaultdict(list)
        for event in video_pred['events']:
            events_by_label[event['label']].append(event)

        filtered_events = []

        for label, events in events_by_label.items():
            sorted_events = sorted(events, key=lambda e: e['score'], reverse=True)
            while sorted_events:
                top_event = sorted_events.pop(0)
                if top_event['score'] < score_threshold:
                    # 因为已经是降序，如果此事件都低于阈值，后面更不需要处理
                    break
                filtered_events.append(top_event)
                # 在 window_size 范围内的，同一 label 都去掉
                sorted_events = [
                    event for event in sorted_events
                    if abs(event['frame'] - top_event['frame']) > window_size
                ]

        filtered_events.sort(key=lambda e: e['frame'])
        video_pred['events'] = filtered_events
        video_pred['num_events'] = len(filtered_events)
        new_pred.append(video_pred)

    return new_pred


def soft_non_maximum_suppression(
    predictions: List[Dict],
    window_size: int,
    score_threshold: float = 0.01
) -> List[Dict]:
    """
    对每个视频做 soft-NMS，相邻事件(在 window_size 内)的分数做衰减
    """
    suppressed_pred = copy.deepcopy(predictions)
    new_pred = []

    for video_pred in suppressed_pred:
        events_by_label = defaultdict(list)
        for event in video_pred["events"]:
            events_by_label[event["label"]].append(event)

        adjusted_events = []

        for label, events in events_by_label.items():
            sorted_events = sorted(events, key=lambda e: e["score"], reverse=True)

            while sorted_events:
                top_event = sorted_events.pop(0)
                if top_event["score"] < score_threshold:
                    break
                adjusted_events.append(top_event)

                for ev in sorted_events:
                    distance = abs(ev["frame"] - top_event["frame"])
                    if distance <= window_size:
                        # 按距离平方比衰减分数 (可自定义衰减策略)
                        ev["score"] *= (distance ** 2) / (window_size ** 2)

        # 再次筛掉低于阈值的
        adjusted_events = [ev for ev in adjusted_events if ev["score"] >= score_threshold]
        adjusted_events.sort(key=lambda x: x["frame"])

        video_pred["events"] = adjusted_events
        video_pred["num_events"] = len(adjusted_events)
        new_pred.append(video_pred)

    return new_pred


def process_frame_predictions(
        dataset,
        classes: Dict[str, int],
        pred_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        high_recall_score_threshold: float = 0.01
):

    classes_inv = {v: k for k, v in classes.items()}                    # {0: 'bg', 1: 'ADDRESS'....}

    error_state = ErrorStat()
    foreground_f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}

    for video, (scores, support) in (pred_dict.items()):
        labels = dataset.get_labels(video)
        support = np.where(support == 0, 1, support)
        assert np.min(support) > 0, (video, support.tolist())

        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)
        error_state.update(labels, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []

        for frame_idx, pred_label in enumerate(pred):
            true_label = labels[frame_idx]
            foreground_f1.update(true_label, pred_label)

            if pred_label != 0:
                events.append( {
                    "label": classes_inv[pred_label],
                    "frame": frame_idx,
                    "score": scores[frame_idx, pred_label].item()}
                )

            for class_id, class_name in classes_inv.items():
                if scores[frame_idx, class_id] >= high_recall_score_threshold:
                    events_high_recall.append({
                        "label": class_name,
                        "frame": frame_idx,
                        "score": scores[frame_idx, class_id].item()}
                    )

        pred_events.append({
            'video': video,
            'events': events
        })
        pred_events_high_recall.append({
            'video': video,
            'events': events_high_recall
        })

    return error_state, foreground_f1, pred_events, pred_events_high_recall, pred_scores


def evaluate(model, dataset, classes, printed=True, test=False, save_pred=None):

    tolerances = TOLERANCES
    windows = WINDOWS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        video: (
            np.zeros((video_len, len(classes) + 1), dtype=np.float32),
            np.zeros(video_len, dtype=np.float32)
        ) for video, video_len, _ in dataset.videos}

    batch_size = INFERENCE_BATCH_SIZE

    dataloader = DataLoader(dataset, num_workers=8, pin_memory=True, batch_size=batch_size)

    for batch in tqdm(dataloader, desc="Evaluating"):
        videos = batch['video_name']
        starts = batch['start']
        frames = batch['frame'].to(device).float()
        poses = batch['pose'].to(device).float()

        _, batch_pred_scores = model.predict(frames, poses)                             # batch_pred_scores [bs, cl, num_classes+1]

        for i in range(frames.shape[0]):
            video = videos[i]
            scores, support = pred_dict[video]
            pred_scores = batch_pred_scores[i]                                          # pred_scores [cl, num_classes]
            start = starts[i].item()
            end = start + pred_scores.shape[0]
            end = min(end, scores.shape[0])
            pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores
            support[start:end] += (pred_scores.sum(axis=1) != 0).astype(np.float32)

    err, f1, pred_events, pred_events_high_recall, pred_scores = process_frame_predictions(
        dataset, classes, pred_dict, high_recall_score_threshold=0.01)

    if not test:
        pred_events_high_recall = non_maximum_suppression(
            pred_events_high_recall,
            window_size=windows[0],
            score_threshold=0.05)
        mAPs, _ = compute_mAPs(
            dataset.labels,
            pred_events_high_recall,
            tolerances=tolerances,
            printed=True)
        avg_mAP = np.mean(mAPs)
        return avg_mAP

    else:
        temp = [(k, [arr.tolist() for arr in v]) for k, v in pred_dict.items()]
        with open(r'save\ops.json', 'w') as f:
            json.dump(temp, f, indent=4)

        print('==== Results on Test (w/o any NMS) ====')
        print(f'Error (frame-level): {err.get_error_rate() * 100:.2f}\n')

        def get_f1_tab_row(label_key: str) -> List:
            label = classes[label_key] if label_key != 'any' else None
            return [
                label_key,
                f"{f1.get(label) * 100:.2f}",
                f1.tp_fp_fn(label)[0],
                f1.tp_fp_fn(label)[1],
                f1.tp_fp_fn(label)[2]]

        rows = [get_f1_tab_row('any')]

        for c in classes:
            rows.append(get_f1_tab_row(c))

        print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'], floatfmt='0.2f'), '\n')
        mAPs, _ = compute_mAPs(
            dataset.labels,
            pred_events_high_recall,
            tolerances=tolerances,
            printed=printed
        )
        avg_mAP = np.mean(mAPs)

        print(f'\n==== Results on Test (w/ NMS{windows[0]}) ====')
        pred_events_high_recall_nms = non_maximum_suppression(
            pred_events_high_recall,
            window_size=windows[0],
            score_threshold=0.01)
        mAPs, _ = compute_mAPs(
            dataset.labels,
            pred_events_high_recall_nms,
            tolerances=tolerances,
            printed=printed)
        avg_mAP_nms = np.mean(mAPs)

        print(f'\n==== Results on Test (w/ SNMS{windows[1]}) ====')
        pred_events_high_recall_snms = soft_non_maximum_suppression(
            pred_events_high_recall,
            window_size=windows[1],
            score_threshold=0.01)
        mAPs, _ = compute_mAPs(
            dataset.labels,
            pred_events_high_recall_snms,
            tolerances=tolerances,
            printed=printed)
        avg_mAP_snms = np.mean(mAPs)

        if avg_mAP_snms > avg_mAP_nms:
            print('Storing predictions with SNMS')
            pred_events_high_recall_store = pred_events_high_recall_snms
        else:
            print('Storing predictions with NMS')
            pred_events_high_recall_store = pred_events_high_recall_nms

        if save_pred:
            dir_path = os.path.dirname(save_pred)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            store_json(f'{save_pred}', pred_events_high_recall_store)

        return avg_mAP_snms