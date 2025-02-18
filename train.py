# Standard imports
import argparse
import os
from functools import partial

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import time
import numpy as np
import random
import logging
from typing import Tuple, List, Any
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR

# Local imports
from util.io import load_json, store_json
from dataset.datasets import get_datasets
from model.POPEModel import POPEModel
from util.evaluate import evaluate

# constants
INFERENCE_BATCH_SIZE = 4
EVAL_SPLITS = ['test']
STRIDE = 1

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1)       # Use gradient accumulation
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def update_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    # Update args with config file
    args.frame_dir = config['frame_dir']
    args.pose_dir = config['pose_dir']
    args.save_dir = os.path.join(config['save_dir'], f'{args.model}-{args.seed}')
    args.store_dir = config['store_dir']
    args.store_mode = config['store_mode']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.crop_dim = config['crop_dim']
    args.dataset = config['dataset']
    args.radi_displacement = config['radi_displacement']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.mix_up = config['mix_up']
    args.modality = config['modality']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.start_val_epoch = config['start_val_epoch']
    args.temporal_arch = config['temporal_arch']
    args.n_layers = config['n_layers']
    args.sgp_ks = config['sgp_ks']
    args.sgp_r = config['sgp_r']
    args.only_test = config['only_test']
    args.criterion = config['criterion']
    args.num_workers = config['num_workers']
    return args


def get_lr_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, num_steps_per_epoch: int) -> ChainedScheduler:
    # Create a learning rate scheduler with linear warmup and cosine annealing
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    logger.info(f'Using Linear Warmup ({args.warm_up_epochs}) + Cosine Annealing LR ({cosine_epochs})')
    return ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * cosine_epochs)])


def set_random_seed(seed: int) -> None:
    # Set random seed for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f'Seed set to {seed}')


def load_config(model_name: str, args: argparse.Namespace) -> argparse.Namespace:
    # Load configuration file and update arguments.
    config_path = os.path.join('config', model_name.split('_')[0], f"{model_name}.json")
    config = load_json(config_path)
    args = update_args(args, config)
    logger.info(f'Configuration loaded from {config_path}')
    return args


def init_train_dataloaders(args: argparse.Namespace) -> Tuple[List[Any], DataLoader, DataLoader, Any, Any]:

    classes, train_data, val_data, test_data = get_datasets(args)
    if args.store_mode == 'store':
        logger.info('Datasets have correctly been stored!')
    else:
        logger.info('Datasets have been loaded from previous versions correctly!')

    loader_batch_size = args.batch_size // args.acc_grad_iter

    train_loader = DataLoader(
        train_data, batch_size=loader_batch_size, shuffle=True,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=2)
    val_loader = DataLoader(
        val_data, batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=2)

    """
    test_loader = DataLoader(test_data, batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers, prefetch_factor=2)
    """
    return classes, train_loader, val_loader, val_data, test_data,


def init_model(args: argparse.Namespace) -> Tuple[POPEModel, torch.optim.Optimizer, torch.cuda.amp.GradScaler]:
    # Initialize the model, optimizer and scaler.
    model = POPEModel(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})
    logger.info(f'Model, optimizer and scaler initialized')
    return model, optimizer, scaler


def save_checkpoint(model: POPEModel, args: argparse.Namespace) -> None:
    # Save the model checkpoint.
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', args.model.split('_')[0], args.model, f'checkpoint_best_{args.seed}.pt')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f'Checkpoint saved at {checkpoint_path}')


def infer(model: POPEModel, test_data, args: argparse.Namespace, classes: List[Any]) -> None:
    logger.info('----- Starting Inference -----')
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', args.model.split('_')[0], args.model, f'checkpoint_best_{args.seed}.pt')
    model.load(torch.load(checkpoint_path))

    for split in EVAL_SPLITS:
        pred_file = os.path.join(args.save_dir, f'pred-{split}') if args.save_dir else None
        evaluate(model, dataset=test_data, classes=classes, save_pred=pred_file, printed=True, test=True)

    logger.info('CORRECTLY FINISHED TRAINING AND INFERENCE')


def main(args: argparse.Namespace):

    set_random_seed(args.seed)
    args = load_config(args.model, args)

    model, optimizer, scaler = init_model(args)
    classes, train_loader, val_loader, val_data, test_data = init_train_dataloaders(args)
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

    losses = []
    best_criterion = 0 if args.criterion == 'mAP' else float('inf')
    logger.info(f'----- Start Training for {args.num_epochs} epochs -----')

    if not args.only_test:

        for epoch in range(args.num_epochs):
            time_train0 = time.time()
            train_loss = model.epoch(train_loader, optimizer, lr_scheduler, scaler, acc_grad_iter=args.acc_grad_iter)
            time_train = time.time() - time_train0
            time_val0 = time.time()
            val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)
            time_val = time.time() - time_val0

            better = False
            val_mAP = 0
            if args.criterion == 'loss':
                if val_loss < best_criterion:
                    best_criterion = val_loss
                    better = True
            elif args.criterion == 'mAP':
                if epoch >= args.start_val_epoch:
                    val_mAP = evaluate(model, dataset=val_data, classes=classes, save_pred=None, printed=True, test=False)
                    if val_mAP > best_criterion:
                        best_criterion = val_mAP
                        better = True
            # Print epoch info
            logger.info(f'[Epoch {epoch+1}/{args.num_epochs}] Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}')
            logger.info(f'Training time: {time_train:.2f} | Validation time: {time_val:.2f}')
            if args.criterion == 'mAP' and epoch >= args.start_val_epoch:
                logger.info(f'Val mAP: {val_mAP: 0.5f}')
                if better:
                    logger.info('New best epoch on mAP!')

            losses.append({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mAP': val_mAP
            })

            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'losses.json'), losses, pretty=True)
                logger.info(f"Losses saved to {os.path.join(args.save_dir, 'losses.json')}")

                if better:
                    save_checkpoint(model, args)

    infer(model, test_data, args, classes)


if __name__ == '__main__':
    args = get_args()
    main(args)