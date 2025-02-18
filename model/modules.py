# Standard imports
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Local imports
from model.impl.graph import Graph


class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()


class BaseRGBModel(ABCModel):

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._get_params(), **opt_args), torch.amp.GradScaler('cuda') if self._device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)


class EDSGPMIXERLayers(nn.Module):

    def __init__(self, feat_dim, clip_len, num_layers=1, ks=3, k=2, k_factor=2, concat=True):
        """
        :param feat_dim: feature dimension
        :param clip_len: clip length
        :param num_layers: number of layers
        :param ks: kernel size
        :param k:
        :param k_factor: scale factor
        :param concat: whether to concatenate feature vectors
        """
        super().__init__()

        self._num_layers = num_layers
        self._total_layers = num_layers * 2 + 1

        self._sgp = nn.ModuleList([
            SGPBlock(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1)
            for _ in range(self._total_layers)
        ])

        self._pooling = nn.ModuleList([
            nn.AdaptiveMaxPool1d(output_size=math.ceil(clip_len / (k_factor ** (i + 1))))
            for i in range(num_layers)
        ])

        self._sgpMixer = nn.ModuleList([
            SGPMixer(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1, t_size=math.ceil(clip_len / (k_factor ** i)), concat=concat)
            for i in range(num_layers)])

    def forward(self, x):
        store_x = [] # Store the intermediate outputs
        x = x.permute(0, 2, 1).contiguous()
        # (batch_size, clip_length, feat_dim) -> (batch_size, feat_dim, clip_length)

        # Downsample
        for i in range(self._num_layers):
            x = self._sgp[i](x)
            store_x.append(x)
            x = self._pooling[i](x)

        # Intermediate
        x = self._sgp[self._num_layers](x)

        # Upsample
        for i in range(self._num_layers):
            x = self._sgpMixer[- (i + 1)](x = x, z = store_x[- (i + 1)])
            x = self._sgp[self._num_layers + i + 1](x)

        x = x.permute(0, 2, 1).contiguous()
        return x


class SGPBlock(nn.Module):

    def __init__(self, n_embd, kernel_size=3, k=1.5, group=1, n_out=None, n_hidden=None, act_layer=nn.GELU, init_conv_vars=0.1, mode='normal'):
        """
        :param n_embd: dimension of the input features
        :param kernel_size: convolution kernel size
        :param k:
        :param group: group for cnn
        :param n_out: output dimension, if None set n_embd
        :param n_hidden: hidden dimension for mlp
        :param act_layer: nonlinear activation used after conv, default ReLU
        :param init_conv_vars: initial gaussian variance for the weights
        :param mode: mode
        """
        super().__init__()

        self.kernel_size = kernel_size
        n_out = n_embd if n_out is None else n_out
        self.ln = nn.LayerNorm(normalized_shape=[n_embd], eps=1e-5, elementwise_affine=True)
        self.gn = nn.GroupNorm(16, n_embd)
        assert kernel_size % 2 == 1, 'kernel_size must be odd'
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, kernel_size=up_size, stride=1, padding=up_size//2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0, groups=n_embd)

        # two layer mlp
        n_hidden = 4 * n_embd if n_hidden is None else n_hidden

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group))

        self.act = act_layer()
        self.relu = nn.ReLU(inplace=True)
        self.sigm = nn.Sigmoid()
        self.mode = mode
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0.0):
        """
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)
        """
        # Kaiming on convs
        for conv_layer in [self.psi, self.fc, self.convw, self.convkw, self.global_fc]:
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            if conv_layer.bias is not None:
                nn.init.constant_(conv_layer.bias, 0)
        # Xavier on mlp
        for layer in self.mlp:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # X shape: bs, cl, d
        out = self.ln(x.transpose(1, 2)).transpose(1, 2)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = self.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        if self.mode == 'normal':
            out = fc * phi + (convw + convkw) * psi + out #fc * phi instant level / (convw + convkw) * psi window level
        elif self.mode == 'sigm1':
            out = fc*phi + self.sigm(convw+convkw)*psi + out
        elif self.mode == 'sigm2':
            out = fc*self.sigm(phi) + self.sigm(convw+convkw)*psi + out
        elif self.mode == 'sigm3':
            out = self.sigm(fc)*phi + (convw+convkw)*self.sigm(psi) + out
        out = x + out
        out = out + self.mlp(self.gn(out))
        return out


class SGPMixer(nn.Module):

    def __init__(self, n_embd, kernel_size=3, k=1.5, group=1, n_out=None, n_hidden=None, act_layer=nn.GELU, init_conv_vars=0.1, t_size=0, concat=True):
        """
        :param n_embd: dimension of the input features
        :param kernel_size: convolution kernel size
        :param k:
        :param group: group for cnn
        :param n_out: output dimension, if None set n_embd
        :param n_hidden: hidden dimension for mlp
        :param act_layer: nonlinear activation used after conv, default ReLU
        :param init_conv_vars: initial gaussian variance for the weights
        :param t_size:
        :param concat: initial gaussian variance for the weights
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.concat = concat
        n_out = n_embd if n_out is None else n_out
        self.ln1 = nn.LayerNorm(normalized_shape=[n_embd], eps=1e-5, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(normalized_shape=[n_embd], eps=1e-5, elementwise_affine=True)
        self.gn = nn.GroupNorm(16, n_embd)
        assert kernel_size % 2 == 1, 'kernel_size must be odd'
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi1 = nn.Conv1d(n_embd, n_embd, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_embd)
        self.psi2 = nn.Conv1d(n_embd, n_embd, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_embd)
        self.convw1 = nn.Conv1d(n_embd, n_embd, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_embd)
        self.convkw1 = nn.Conv1d(n_embd, n_embd, kernel_size=up_size, stride=1, padding=up_size//2, groups=n_embd)
        self.convw2 = nn.Conv1d(n_embd, n_embd, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=n_embd)
        self.convkw2 = nn.Conv1d(n_embd, n_embd, kernel_size=up_size, stride=1, padding=up_size//2, groups=n_embd)
        self.fc1 = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0, groups=n_embd)
        self.global_fc1 = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0, groups=n_embd)
        self.fc2 = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0, groups=n_embd)
        self.global_fc2 = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0, groups=n_embd)
        self.upsample = nn.Upsample(size=t_size, mode='linear', align_corners=True)

        # two layer mlp
        n_hidden = 4 * n_embd if n_hidden is None else n_hidden
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group))

        if self.concat:
            self.concat_fc = nn.Conv1d(n_embd*6, n_embd, 1, groups=group)

        self.act = act_layer()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        """
        torch.nn.init.normal_(self.psi1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.psi2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc2.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi1.bias, 0)
        torch.nn.init.constant_(self.psi2.bias, 0)
        torch.nn.init.constant_(self.convw1.bias, 0)
        torch.nn.init.constant_(self.convkw1.bias, 0)
        torch.nn.init.constant_(self.convw2.bias, 0)
        torch.nn.init.constant_(self.convkw2.bias, 0)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.constant_(self.global_fc1.bias, 0)
        torch.nn.init.constant_(self.global_fc2.bias, 0)

        if self.concat:
            torch.nn.init.normal_(self.concat_fc.weight, 0, init_conv_vars)
            torch.nn.init.constant_(self.concat_fc.bias, 0)
        """
        # Kaiming on convs
        for conv_layer in [self.psi1, self.psi2, self.convw1, self.convkw1, self.convw2, self.convkw2, self.fc1,
                           self.fc2, self.global_fc1, self.global_fc2]:
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            if conv_layer.bias is not None:
                nn.init.constant_(conv_layer.bias, 0)

        # Xavier on mlp
        for layer in self.mlp:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        if self.concat:
            nn.init.xavier_normal_(self.concat_fc.weight)
            nn.init.constant_(self.concat_fc.bias, 0)

    def forward(self, x, z):
        # X shape: bs, cl, d
        z = self.ln1(z.transpose(1, 2)).transpose(1, 2)
        x = self.ln2(x.transpose(1, 2)).transpose(1, 2)

        # Upsample
        x = self.upsample(x)

        psi1 = self.psi1(z)
        psi2 = self.psi2(x)
        convw1 = self.convw1(z)
        convkw1 = self.convkw1(z)
        convw2 = self.convw2(x)
        convkw2 = self.convkw2(x)

        # Instant level branches
        fc1 = self.fc1(z)
        fc2 = self.fc2(x)
        phi1 = self.relu(self.global_fc1(z.mean(dim=-1, keepdim=True)))
        phi2 = self.relu(self.global_fc2(x.mean(dim=-1, keepdim=True)))

        # Feature fusion
        out1 = (convw1 + convkw1) * psi1
        out2 = (convw2 + convkw2) * psi2
        out3 = fc1 * phi1
        out4 = fc2 * phi2
        if self.concat:
            out = torch.cat((out1, out2, out3, out4, z, x), dim=1)
            out = self.act(self.concat_fc(out))
        else:
            out = out1 + out2 + out3 + out4 + z + x
        # FFN
        out = out + self.mlp(self.gn(out))
        return out


class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        batch_size, clip_len, feat_dim = x.shape
        x = x.view(batch_size, clip_len, feat_dim)
        x = self.fc(x)
        x = x.view(batch_size, clip_len, -1)
        return x


class STGCNModel(nn.Module):

    def __init__(self, in_channels, graph_args, edge_importance_weighting=True, t_kernel_size=9, dtype=torch.float32, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=dtype, requires_grad=False)
        # TODO: dtype=torch.float32 may not be necessary, float16 or int may be enough
        self.register_buffer('A', A)

        # Build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = t_kernel_size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        channels = [in_channels, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        self.st_gcn_networks = nn.ModuleList()
        for i in range(1, len(channels)):
            self.st_gcn_networks.append(
                st_gcn(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    kernel_size=kernel_size,
                    stride=1,
                    residual=(channels[i - 1] == channels[i]),
                    dropout=kwargs.get('dropout', 0) if i != 1 else 0,
                    **kwargs0))

        # Initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(A))
                for _ in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        """
        :param x: (bs, cl, j, c)
            bs = batch_size
            cl = clip_len
            j = joint num
            c = channels (3)
        :return: (bs, cl, c')
        """
        bs, cl, j, c = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs, j*c, cl)
        # (bs, j, c, cl) -> (bs, j*c, cl)
        x = self.data_bn(x)
        x = x.reshape(bs, j, c, cl).permute(0, 2, 3, 1)
        # (bs, j, c, cl) -> (bs, c, cl, j)
        # forward
        for idx, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            A = self.A * importance if isinstance(importance, nn.Parameter) else self.A
            x = gcn(x, A)
            # (bs, c', cl, j)

        # global pooling on joint (j) dimension, but keep the time (cl)
        x = F.avg_pool2d(x, (1, j))
        # (bs, c', cl, 1)
        x = x.squeeze(-1)
        # (bs, c', cl)
        x = x.permute(0, 2, 1)
        # (bs, cl, c')
        return x


class st_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2, 'Kernel size should be (int, int)'
        assert kernel_size[0] % 2 == 1, 'Temporal kernel size should be odd'

        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if residual:
            if in_channels == out_channels and stride == 1:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.residual = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        if self.residual is not None:
            res = self.residual(x)
        else:
            res = 0

        x, _ = self.gcn(x, A)
        x = self.tcn(x)
        if self.residual is not None:
            x = x + res

        return self.relu(x)


class ConvTemporalGraphical(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        """
        :param x: (B, C, T, V)
        :param A: (K, V, W)
        :return: (B, C', T, W)
        """
        B, C, T, V = x.size()                                           # (B, C, T, V)
        # print(f"Input size: B = {B}, C = {C}, T = {T}, V = {V}")
        K = self.kernel_size
        x = self.conv(x)
        # print(f"After conv: B = {B}, CK = {x.size(1)}, T = {T}, V = {V}")
        C_prime = x.size(1) // K

        x = x.view(B, K, C_prime, T, V)                                 # (B, K, C', T, V)
        x = x.permute(0, 3, 2, 1, 4).reshape(B * T, C_prime, K * V)     # (B*T, C', K*V)
        A = A.view(-1, V)                                               # (K*V, W)
        x = torch.matmul(x, A)                                          # (B*T, C', W)
        x = x.view(B, T, C_prime, -1).permute(0, 2, 1, 3)               # (B, C', T, W)
        # print(f"Output size: B = {B}, C' = {C_prime}, T = {T}, V = {V}")
        return x.contiguous(), A.view(K, V, -1)


class SoftCE(nn.Module):
    def __init__(self, weight: Optional[torch.tensor] = None, reduction: str = 'mean'):
        super(SoftCE, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(preds, dim=-1)
        if self.weight is not None:
            soft_targets = soft_targets * self.weight.unsqueeze(0)
        loss = -torch.sum(soft_targets * log_probs, dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def step(optimizer, scaler, loss, lr_scheduler=None, backward_only=False):
    # OK
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if not backward_only:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        if lr_scheduler:
            lr_scheduler.step()

        optimizer.zero_grad()


def process_prediction(pred, predD):

    batch_size, clip_len, num_classes = pred.shape
    aux_pred = torch.zeros_like(pred)

    for bs in range(batch_size):
        for cl in range(clip_len):
            displ = predD[bs, cl].round().int()
            target_t = max(0, min(clip_len - 1, cl - displ))
            aux_pred[bs, target_t] = torch.maximum(aux_pred[bs, target_t], pred[bs, cl])
    """
    sum_probs = aux_pred.sum(dim=2, keepdim=True)
    aux_pred = torch.where(sum_probs > 0, aux_pred / sum_probs, aux_pred)
    """
    return aux_pred
