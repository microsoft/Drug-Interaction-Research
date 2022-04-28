# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from models.common import ModelConfig
from models.igt.impl.net import IGTNet


@dataclass
class IGTOptimizerParams:
    init_lr: float = 0.0001
    lr_reduce_factor: float = 0.5
    lr_schedule_patience: int = 15
    min_lr: float = 1e-6
    weight_decay: float = 1e-5


@dataclass
class IGTNetParams:
    n_layers: int = 10
    n_heads: int = 8
    hidden_dim: int = 512
    out_dim: int = 512

    residual: bool = True
    in_feat_dropout: float = 0.0
    dropout: float = 0.4
    layer_norm: bool = False
    batch_norm: bool = True

    pos_enc_dim: int = 8
    in_dim_node: int = 54 + 34
    in_dim_edge: int = 17 + 4 + 3

    initial_mu: float = 4.0
    initial_dev: float = 1.0
    initial_alpha: float = 0.0
    initial_beta: float = 0.0


@dataclass
class IGTParams:
    optimizer: IGTOptimizerParams = IGTOptimizerParams()
    net: IGTNetParams = IGTNetParams()
    data_funcs: Dict[str, str] = field(
        default_factory=lambda: {'_target_': 'models.igt.data.get_data_funcs'})


@dataclass
class IGTConfig(ModelConfig):
    _target_: str = 'models.igt.model.IGTForDTI'
    args: IGTParams = IGTParams()


class IGTForDTI(nn.Module):
    def __init__(self, args: IGTParams):
        super().__init__()
        self.params = args
        self.net = IGTNet(self.params.net)

    def forward(self, sample):
        batch_whole_graphs = sample['whole_graph']
        batch_receptor_graphs = sample['receptor_graph']
        batch_ligand_graphs = sample['ligand_graph']
        batch_scores = self.net(batch_whole_graphs, batch_ligand_graphs, batch_receptor_graphs)
        pred = torch.sigmoid(batch_scores).squeeze(-1)
        return pred

    def setup_optimizer_and_lr_scheduler(self):
        params = self.params.optimizer
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': params['weight_decay']},
                                      {'params': bias_p, 'weight_decay': 0}], lr=params['init_lr'])

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', factor=params['lr_reduce_factor'],
            patience=params['lr_schedule_patience'], verbose=True
        )
        return optimizer

    def on_epoch_end(self, epoch_val_loss):
        self.scheduler.step(epoch_val_loss)
