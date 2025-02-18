# Standard imports
import argparse
from contextlib import nullcontext
from typing import Optional, Union, Tuple, Dict

import numpy as np
import timm
import torch
import torch.distributions.beta as Beta
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ChainedScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from model.modules import BaseRGBModel, EDSGPMIXERLayers, FCLayers, STGCNModel, step, process_prediction, SoftCE
from model.shift import make_temporal_shift


class POPEModel(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__()

            self._modality = args.modality
            assert self._modality in ['rgb', 'pose', 'both']

            self._temporal_arch = args.temporal_arch
            assert self._temporal_arch == 'ed_sgp_mixer', 'Only ed_sgp_mixer supported for now'

            self._feature_arch = args.feature_arch
            assert 'rny' in self._feature_arch, 'Only RegNetY supported for now'

            self._d = 256
            self._radi_displacement = args.radi_displacement

            # Init Feature Extractor
            if self._feature_arch.startswith(('rny002', 'rny008')):
                feature_extract_model_name = {
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008'
                }[self._feature_arch.rsplit('_', 1)[0]]

                feature = timm.create_model(feature_extract_model_name, pretrained=True)
                self._d = feature.head.fc.in_features  # 368/768
                # Remove final classification layer
                feature.head.fc = nn.Identity()
            else:
                raise NotImplementedError(f'Unsupported feature arch: {self._feature_arch}')

            # Temporal Shift Modules
            if self._feature_arch.endswith('_gsm'):
                mode = 'gsm'
            elif self._feature_arch.endswith('_gsf'):
                mode = 'gsf'
            else:
                raise NotImplementedError(f'Unsupported feature arch: {self._feature_arch}')
            make_temporal_shift(feature, args.clip_len, mode=mode)

            self._feature = feature

            # Positional Encoding
            self._temp_enc = nn.Parameter(torch.normal(mean=0, std=1/args.clip_len, size=(args.clip_len, self._d)))

            if self._modality == 'pose':
                self._d = 256
            elif self._modality == 'both':
                self._d += 256

            # Temporal Arch
            if self._temporal_arch == 'ed_sgp_mixer':
                self._temp_fine = EDSGPMIXERLayers(
                    feat_dim=self._d,
                    clip_len=args.clip_len,
                    num_layers=args.n_layers,
                    ks=args.sgp_ks,
                    k=args.sgp_r,
                    concat=True)
                self._pred_fine = FCLayers(self._d, args.num_classes + 1)
            else:
                raise NotImplementedError(f'Unsupported temporal arch: {self._temporal_arch}')

            # Displacement Prediction
            if self._radi_displacement > 0:
                self._pred_displ = FCLayers(self._d, 1)

            # Augmentation & (Cropping)
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25)])

            # Standardization
            self.standardization = T.Compose([
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))])

            # Augmentation at inference
            self.augmentationI = T.Compose([T.RandomHorizontalFlip(p=1.0)])

            """
            # Cropping
            self._cropping = args.crop_dim
            if self._cropping:
                self.cropT = T.RandomCrop((self._cropping, self._cropping))
                self.cropI = T.CenterCrop((self._cropping, self._cropping))
            else:
                self.cropT = nn.Identity()
                self.cropI = nn.Identity()
            """

        def forward(
                self,
                x: Optional[torch.Tensor] = None,
                x_pose: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                inference: bool = False,
                inference_augmentation: bool = False
        ) -> Union[Tuple[Dict[str, torch.Tensor], torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
            """
            :param x: input tensor for RGB, (batch_size, clip_len, C, H, W)
            :param x_pose: input tensor for pose, (batch_size, clip_len, 256)
            :param y: label, if inference, None
            :param inference: whether to perform inference or not
            :param inference_augmentation: whether augment during inference
            :return: if rd:     ['feat': feat, 'featD': dispfeat], pred_labels
                     else:      feat, pred_labels
            """
            if self._modality == 'pose':
                feat_final = x_pose
            else:
                x = self.normalize(x)
                batch_size, clip_len, C, H, W = x.shape
                x = x.view(-1, C, H, W)
                """
                if self._cropping:
                    H = self._cropping
                    W = self._cropping
                """
                if not inference:
                    # x = self.cropT(x)
                    x = x.view(batch_size, clip_len, C, H, W)
                    x = self.augment(x)
                else:
                    # x = self.cropI(x)
                    x = x.view(batch_size, clip_len, C, H, W)
                    if inference_augmentation:
                        x = self.augmentI(x)
                x = self.standardize(x)

                # Feature Extraction
                im_feat = self._feature(x.view(-1, C, H, W)).reshape(batch_size, clip_len, self._d)             # (bs, cl, 368/768)
                # Positional Encoding
                im_feat = im_feat + self._temp_enc.expand(batch_size, -1, -1)

                # Concatenate tensor
                if self._modality == 'rgb':
                    feat_final = im_feat
                else:
                    feat_final = torch.cat([im_feat, x_pose], dim=-1)


            feat = self._temp_fine(feat_final)
            if self._radi_displacement > 0:
                featD = self._pred_displ(feat_final).squeeze(-1)
                feat = self._pred_fine(feat)
                return {'feat': feat, 'featD': featD}, y
            feat = self._pred_fine(feat_final)
            return feat, y


        def normalize(self, x: torch.Tensor) -> torch.Tensor:
            return x / 255.

        def augment(self, x: torch.Tensor) -> torch.Tensor:
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def augmentI(self, x: torch.Tensor) -> torch.Tensor:
            for i in range(x.shape[0]):
                x[i] = self.augmentationI(x[i])
            return x

        def standardize(self, x: torch.Tensor) -> torch.Tensor:
            for i in range(x.shape[0]):
                x[i] = self.standardization(x[i])
            return x

        def print_param_stats(self):
            print('Model parameters:',
                  sum(p.numel() for p in self.parameters()))
            print('----Feature Extractor:',
                  sum(p.numel() for p in self._feature.parameters()))
            print('----Temporal Architecture:',
                  sum(p.numel() for p in self._temp_fine.parameters()))
            print('----Head:',
                  sum(p.numel() for p in self._pred_fine.parameters()))

    def __init__(self, device='cuda', args=None) -> None:
        self._device = device
        self._args = args
        self._model = POPEModel.Impl(args=args)
        self._model.print_param_stats()
        self._model.to(device)

        self._num_classes = args.num_classes + 1            # +Background

        graph_args = {'layout': 'coco', 'strategy': 'spatial'}
        self._model_pose = STGCNModel(in_channels=3, graph_args=graph_args).to(self._device)

    def epoch(
            self,
            loader: DataLoader,
            optimizer: Optional[Optimizer] = None,
            lr_scheduler: Optional[ChainedScheduler] = None,
            scaler: Optional[GradScaler] = None,
            acc_grad_iter: int = 1,
            fg_weight: Union[int, float] = 25
    ) -> float:
        """
        :param loader: input DataLoader
        :param optimizer: if no optimizer, val
        :param lr_scheduler: lr_scheduler
        :param scaler: scaler
        :param acc_grad_iter: simulater larger batch_size
        :param fg_weight: foreground weight, ease the classes' imbalance
        :return: loss for train/val
        """
        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        if fg_weight != 1:
            ce_weight = torch.FloatTensor([1.0] + [fg_weight] * (self._num_classes - 1)).to(self._device)
        else:
            ce_weight = None
        soft_ce_fn = SoftCE(weight=ce_weight, reduction='mean')
        ce_fn = nn.CrossEntropyLoss(weight=ce_weight)

        epoch_loss = 0.0
        epoch_lossD = 0.0

        context = torch.no_grad() if optimizer is None else nullcontext()

        with context:
            for batch_idx, batch in enumerate(tqdm(loader, desc='Training  ' if optimizer is not None else 'Validating')):
                frame = batch['frame'].to(self._device).float()                 # [bs, cl, c, h, w]
                pose = batch['pose'].to(self._device).float()                   # [bs, cl, 17, 3]
                label = batch['label'].to(self._device)                         # [bs, cl]
                labelD = batch.get('labelD', None)
                if labelD is not None:
                    labelD = labelD.to(self._device).float()

                if 'frame2' in batch:
                    frame2 = batch['frame2'].to(self._device).float()           # [bs, cl, c, h, w]
                    pose2 = batch['pose2'].to(self._device).float()             # [bs, cl, 17, 3]
                    label2 = batch['label2'].to(self._device)                   # [bs, cl]
                    labelD2 = batch.get('labelD2', None)
                    if labelD2 is not None:
                        labelD2 = labelD2.to(self._device).float()              # [bs, cl]
                        labelD2 = labelD2.long()

                    beta_dist = Beta.Beta(torch.tensor([0.2]).to(self._device), torch.tensor([0.2]).to(self._device))
                    l = beta_dist.sample([frame2.shape[0]]).to(self._device)                    # [bs]
                    l_frame = l.view(-1, 1, 1, 1, 1)                                            # [bs, 1, 1, 1, 1]
                    l_pose = l.view(-1, 1, 1, 1)                                                # [bs, 1, 1, 1]
                    l_labels = l.view(-1, 1, 1)                                                 # [bs, 1, 1]

                    frame = l_frame * frame + (1 - l_frame) * frame2                            # [bs, cl, c, h, w]
                    pose = l_pose * pose + (1 - l_pose) * pose2                                 # [bs, cl, 17, 3]
                    label_onehot = F.one_hot(label, self._num_classes).float()                  # [bs, cl, num_classes]
                    label2_onehot = F.one_hot(label2, self._num_classes).float()                # [bs, cl, num_classes]
                    label = l_labels * label_onehot + (1 - l_labels) * label2_onehot
                    if labelD2 is not None:
                        l_labelsD = l.view(-1, 1)
                        labelD = l_labelsD * labelD + (1 - l_labelsD) * labelD2


                is_soft = (label.dim() == 3)

                if is_soft:
                    label = label.view(-1, self._num_classes)                                   # [bs*cl, num_classes]
                else:
                    label = label.flatten()                                                     # [bs*cl]

                with torch.amp.autocast('cuda'):
                    pred, _ = self._model(x=frame, x_pose=self._model_pose(pose), y=label, inference=inference)
                    if labelD is not None:
                        predD = pred['featD']
                        pred = pred['feat']
                    # pred.shape = (bs, cl, num_classes), predD.shape = (bs, cl)

                    predictions = pred.reshape(-1, self._num_classes)

                    if is_soft:
                        loss = soft_ce_fn(predictions, label)
                    else:
                        loss = ce_fn(predictions, label)

                    if labelD is not None:
                        lossD = F.mse_loss(predD, labelD, reduction='none').mean()
                        loss += lossD

                if optimizer is not None:
                    step(optimizer, scaler, loss/acc_grad_iter,
                        lr_scheduler=lr_scheduler, backward_only=(batch_idx+1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()
                if labelD is not None:
                    epoch_lossD += lossD.detach().item()
                """
                del frame, pose, label, predictions, pred
                if labelD is not None:
                    del labelD, predD
                torch.cuda.empty_cache()
                """
        return epoch_loss / len(loader)

    def predict(
            self,
            frames: Optional[torch.Tensor] = None,
            poses: Optional[torch.Tensor] = None,
            inference_augment: bool = False,
            use_amp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:

        self._model.eval()
        with torch.no_grad():
            context = torch.amp.autocast('cuda') if use_amp else nullcontext()
            with context:
                pred, _ = self._model(x=frames, x_pose=self._model_pose(poses), y=None, inference=True, inference_augmentation=inference_augment)
                if isinstance(pred, dict):
                    predD = pred['featD']
                    pred = pred['feat']
                    pred = process_prediction(pred, predD)
                    pred = torch.softmax(pred, dim=2)
                    pred_cls = torch.argmax(pred, axis=2)
                    return pred_cls.cpu().numpy(), pred.cpu().numpy()
                pred = torch.softmax(pred, axis=2)
                pred_cls = torch.argmax(pred, axis=2)
                return pred_cls.cpu().numpy(), pred.cpu().numpy()

