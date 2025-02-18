# Standard imports
import os

# Local imports
from util.io import load_text
from dataset.frames import ActionSpotDataset

DEFAULT_STRIDE = 1
DEFALT_OVERLAP = 0.5


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}

def get_datasets(args):
    """
    Load datasets
    :param args:
    :return: classes, train_data, val_data, test_data
    """
    data_dir = os.path.join('data', args.dataset)
    class_file = os.path.join(data_dir, 'class.txt')
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    test_json = os.path.join(data_dir, 'test.json')

    classes = load_classes(class_file)
    dataset_len = args.epoch_num_frames // args.clip_len        # clip_num
    stride = DEFAULT_STRIDE
    overlap = DEFALT_OVERLAP

    dataset_kwargs = {
        'stride': stride,
        'overlap_ratio': overlap,
        'radi_displacement': args.radi_displacement,
        'mix_up': args.mix_up,
        'dataset': args.dataset
    }

    print(f'Dataset size: {dataset_len}')

    train_data = ActionSpotDataset(
        classes, train_json,
        args.frame_dir, args.pose_dir, args.store_dir, args.store_mode,
        args.clip_len, dataset_len, **dataset_kwargs)
    train_data.print_info()

    val_data = ActionSpotDataset(
        classes, val_json,
        args.frame_dir, args.pose_dir, args.store_dir, args.store_mode,
        args.clip_len, dataset_len//4, **dataset_kwargs)
    val_data.print_info()

    test_data = ActionSpotDataset(
        classes, test_json,
        args.frame_dir, args.pose_dir, args.store_dir, args.store_mode,
        args.clip_len, dataset_len, **dataset_kwargs)
    test_data.print_info()

    return classes, train_data, val_data, test_data
