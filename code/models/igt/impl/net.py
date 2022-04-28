# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features

"""
from models.igt.impl.layers.graph_transformer_edge_layer import GraphTransformerLayer
from models.igt.impl.layers.mlp_readout_layer import MLPReadout

class InputEmbeddingLayer(nn.Module):
    def __init__(self, net_params):
        super(InputEmbeddingLayer, self).__init__()
        in_dim_node = net_params['in_dim_node']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        initial_mu = net_params['initial_mu']
        initial_dev = net_params['initial_dev']
        pos_enc_dim = net_params['pos_enc_dim']
        self.linear_h = nn.Linear(in_dim_node, hidden_dim)
        self.linear_ec = nn.Linear(in_dim_edge, hidden_dim)
        self.linear_ef = nn.Linear(1, hidden_dim)
        self.linear_e = nn.Linear(2 * hidden_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.ef_mu = nn.Parameter(torch.FloatTensor([initial_mu]))
        self.ef_dev = nn.Parameter(torch.FloatTensor([initial_dev]))
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

    def forward(self, g):
        h = g.ndata['feat']
        et = g.edata['type']
        es = g.edata['sameAA']
        ed = g.edata['distance']
        eb = g.edata['bond_feat']

        h_lap_pos_enc = g.ndata['lap_pos_enc']
        sign_flip = torch.rand(h_lap_pos_enc.size(1), device=h.device)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        h_lap_pos_enc = h_lap_pos_enc * sign_flip.unsqueeze(0)

        h = self.linear_h(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        h = h + h_lap_pos_enc
        h = self.in_feat_dropout(h)

        ef = ed
        ef = torch.exp(-torch.pow(ef-self.ef_mu.expand_as(ef), 2)/self.ef_dev)
        ec = self.linear_ec(torch.cat([et, es, eb], dim=-1).float())
        ef = self.linear_ef(ef)
        e = self.linear_e(torch.cat([ec, ef], dim=-1))
        return h, e

class SubtractionLayer(nn.Module):
    def __init__(self, net_params, last_layer):
        super(SubtractionLayer, self).__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        dropout = net_params['dropout']
        initial_alpha = net_params['initial_alpha']
        initial_beta = net_params['initial_beta']

        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.last_layer = last_layer
        self.alpha = nn.Parameter(torch.FloatTensor([initial_alpha]), requires_grad=True)
        self.beta  = nn.Parameter(torch.FloatTensor([initial_beta]), requires_grad=True)
        if self.last_layer:
            output_dim = out_dim
        else:
            output_dim = hidden_dim
        self.whole_gt = GraphTransformerLayer(hidden_dim, output_dim, num_heads, dropout,
                              self.layer_norm, self.batch_norm, self.residual)
        self.receptor_gt = GraphTransformerLayer(hidden_dim, output_dim, num_heads, dropout,
                                              self.layer_norm, self.batch_norm, self.residual)
        self.ligand_gt = GraphTransformerLayer(hidden_dim, output_dim, num_heads, dropout,
                                              self.layer_norm, self.batch_norm, self.residual)

    def forward(self, whole_g, lig_g, rec_g, whole_h, whole_e, lig_h, lig_e, rec_h, rec_e, whole_batch, lig_batch, rec_batch):

        whole_h, whole_e = self.whole_gt(whole_g, whole_h, whole_e)
        lig_h, lig_e = self.ligand_gt(lig_g, lig_h, lig_e)
        rec_h, rec_e = self.receptor_gt(rec_g, rec_h, rec_e)

        whole_h_list = torch.split(whole_h, tuple(whole_batch.cpu().numpy()), dim=0)
        lig_h_list = torch.split(lig_h, tuple(lig_batch.cpu().numpy()), dim=0)
        rec_h_list = torch.split(rec_h, tuple(rec_batch.cpu().numpy()), dim=0)

        assert len(whole_h_list) == len(lig_h_list) == len(rec_h_list)

        whole_h_list_ = []
        for i in range(len(whole_h_list)):
            assert whole_h_list[i].size(0) == (lig_h_list[i].size(0) + rec_h_list[i].size(0))
            whole_lig_h = torch.cat([self.alpha * lig_h_list[i], torch.zeros_like(rec_h_list[i])], dim=0)
            whole_rec_h = torch.cat([torch.zeros_like(lig_h_list[i]), self.beta * rec_h_list[i]], dim=0)
            whole_h_list_.append(whole_h_list[i] - whole_lig_h - whole_rec_h)

        whole_h = torch.cat(whole_h_list_, dim=0)

        return whole_h, whole_e, lig_h, lig_e, rec_h, rec_e


class IGTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        out_dim = net_params['out_dim']
        n_layers = net_params['n_layers']

        self.layers = nn.ModuleList([SubtractionLayer(net_params, False) for _ in range(n_layers-1) ])
        self.layers.append(SubtractionLayer(net_params, True))
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem
        self.input_embedding_modules = nn.ModuleDict({
            'whole': InputEmbeddingLayer(net_params),
            'receptor': InputEmbeddingLayer(net_params),
            'ligand': InputEmbeddingLayer(net_params),
        })

    def forward(self, whole_g, ligand_g, receptor_g):

        whole_h, whole_e = self.input_embedding_modules['whole'](whole_g)
        receptor_h, receptor_e = self.input_embedding_modules['receptor'](receptor_g)
        ligand_h, ligand_e = self.input_embedding_modules['ligand'](ligand_g)

        whole_batch = whole_g.batch_num_nodes()
        receptor_batch = receptor_g.batch_num_nodes()
        ligand_batch = ligand_g.batch_num_nodes()

        # convnets
        for conv in self.layers:
            whole_h, whole_e, ligand_h, ligand_e, receptor_h, receptor_e = \
                conv(whole_g, ligand_g, receptor_g, whole_h, whole_e, ligand_h, ligand_e, receptor_h, receptor_e, whole_batch, ligand_batch, receptor_batch)
        whole_g.ndata['h'] = whole_h

        hg = dgl.sum_nodes(whole_g, 'h', 'ligand_mask') / dgl.sum_nodes(whole_g, 'ligand_mask')

        return self.MLP_layer(hg)
