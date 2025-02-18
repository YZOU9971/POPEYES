import sys
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

def parse_ground_truth(truth):

    label_dict = defaultdict(lambda: defaultdict(list))

    for video_data in truth:
        events = video_data['events']
        for event in events:
            frame = event['frame']
            label = event['label']
            label_dict[label][video_data['video']].append(frame)

    for label in label_dict:
        for video in label_dict[label]:
            label_dict[label][video].sort()

    return label_dict

def get_predictions(pred, label=None):

    flat_pred = [
        (video_data['video'], event['frame'], event['score'])
        for video_data in pred
        for event in video_data['events']
        if label is None or event['label'] == label
    ]

    flat_pred.sort(key=lambda x: x[2], reverse=True)
    return flat_pred


def compute_average_precision(
        pred, ground_truth, tolerance=0, min_precision=0,
        plot_ax=None, plot_label=None, plot_raw_pr=True
):
    total_gt_events = sum([len(x) for x in ground_truth.values()])
    if total_gt_events == 0:
        return 0.0

    recalled = {video: [False] * len(ground_truth[video]) for video in ground_truth}

    # The full precision curve has TOTAL number of bins, when recall increases by in increments of one
    precision_curve = []
    true_positives = 0
    prev_score = float('inf')

    for idx, (video, frame, score) in enumerate(pred, 1):
        assert score <= prev_score
        prev_score = score

        gt_frames = ground_truth.get(video, [])
        if not gt_frames:
            continue

        # Find the ground truth frame that is closest to the prediction
        min_distance = float('inf')
        min_idx = None
        for gt_idx, (gt_frame, is_recalled) in enumerate(zip(gt_frames, recalled[video])):
            if is_recalled:
                continue
            distance = abs(gt_frame - frame)
            if distance <= tolerance and distance < min_distance:
                min_distance = distance
                min_idx = gt_idx
                if distance == 0:
                    break

        # Record precision each time a true positive is encountered
        if min_idx is not None:
            recalled[video][min_idx] = True
            true_positives += 1
            precision = true_positives / idx
            precision_curve.append(precision)

            # Early stopping if precision drops below minimum threshold
            if precision < min_precision:
                break


    interpolated_precision = []
    max_precision = 0
    for p in reversed(precision_curve):
        max_precision = max(max_precision, p)
        interpolated_precision.append(max_precision)
    interpolated_precision = list(reversed(interpolated_precision))

    # Plotting the precision-recall curve

    if plot_ax is not None:
        recall_values = np.arange(1, len(precision_curve) + 1) / total_gt_events

        if plot_raw_pr:
            plot_ax.plot(recall_values, precision_curve, label=plot_label+' (Raw)', alpha=0.5)
        plot_ax.plot(recall_values, interpolated_precision, label=plot_label, alpha=0.8)
        plot_ax.set_xlabel('Recall')
        plot_ax.set_ylabel('Precision')
        plot_ax.set_ylim([0, 1.01])
        plot_ax.set_xlim([0, 1.0])
        plot_ax.legend()

    # Compute AUC by integrating up to TOTAL bins
    average_precision = sum(interpolated_precision) / total_gt_events
    return average_precision

def compute_mAPs(ground_truth_data, pred_data, tolerances=None,
                 plot_pr=False, printed=False, class_list=None):
    """
    Computes the mean Average Precision (mAP) across different tolerances.
    :param ground_truth_data: List of dictionaries containing ground truth data.
    :param pred_data: List of dictionaries containing prediction data.
    :param tolerances: List of tolerance values for evaluation. Defaults to [0, 1, 2, 4].
    :param plot_pr: Whether to plot precision-recall curves. Defaults to False.
    :param printed: Whether to print the AP table. Defaults to False.
    :param class_list: If provided, specifies the exact order of classes to display.
    :return: Tuple containing a list of mAP values and the corresponding tolerances.
    """
    if tolerances is None:
        tolerances = [0, 1, 2, 4]
    ground_truth_videos = {video_data['video'] for video_data in ground_truth_data}
    pred_videos = {video_data['video'] for video_data in pred_data}

    assert ground_truth_videos == pred_videos, 'Video set mismatch.'

    ground_truth_by_label = parse_ground_truth(ground_truth_data)

    if class_list is None:
        class_list = list(ground_truth_by_label.keys())

    fig, axes = None, None
    if plot_pr:
        num_labels = len(class_list)
        fig, axes = plt.subplots(
            num_labels, len(tolerances), sharex=True, sharey=True,
            figsize=(4*len(tolerances), 4*num_labels))

    class_aps_for_tolerances = []
    mAPs = []
    for tol_idx, tolerance in enumerate(tolerances):
        class_aps = []
        for label_idx, (label, gt_per_label) in enumerate(ground_truth_by_label.items()):
            predictions_per_label = get_predictions(pred_data, label=label)
            ap = compute_average_precision(
                predictions_per_label,
                gt_per_label,
                tolerance=tolerance,
                plot_ax=axes[label_idx, tol_idx] if axes is not None else None,
                plot_label=f'{label} @ tol={tolerance}'
            )
            class_aps.append((label, ap))

        mean_ap = np.mean([ap for _, ap in class_aps])
        mAPs.append(mean_ap)
        class_aps.append(('mAP', mean_ap))
        class_aps_for_tolerances.append(class_aps)

    header = ['AP @ tol'] + tolerances
    rows = []
    for c, _ in class_aps_for_tolerances[0]:
        row = [c]
        for class_aps in class_aps_for_tolerances:
            for c2, val in class_aps:
                if c2 == c:
                    row.append(val * 100)
        rows.append(row)

    if printed:
        print(tabulate(rows, headers=header, floatfmt='0.2f'))
        print('Avg mAP (across tolerances): {:0.2f}'.format(np.mean(mAPs) * 100))

    if plot_pr:
        for i, tol in enumerate(tolerances):
            for j, label in enumerate(sorted(ground_truth_by_label.keys())):
                ax = axes[j, i]
                ax.set_xlabel('Recall')
                ax.set_xlim(0, 1)
                ax.set_ylabel('Precision')
                ax.set_ylim(0, 1.01)
                ax.set_title(f'{label} @ tol={tol}')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    sys.stdout.flush()
    return mAPs, tolerances
