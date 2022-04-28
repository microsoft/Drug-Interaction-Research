# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import time
import pickle
import logging
from pprint import pformat
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.conf import ConfigStore
from omegaconf import OmegaConf, MISSING

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from dataset import MolDataset, DistributedSampler

from models.common import ModelConfig, add_model_configs_to_store


@dataclass
class EvaluateConfig:
    batch_size: int = 1  # batch_size

    dataset: str = 'lit-pcba'  # dataset: lit-pcba or muv
    num_workers: int = 4  # number of workers
    data_fpath: str = '/path/to/lit-pcba.pickles'  # file path of the datasets

    # valid_split_ratio: float = 0.3
    data_keys: str = 'keys/lit-pcba.pkl'

    local_rank: Optional[int] = None  # Rank of current process
    local_world_size: Optional[int] = None  # World size (used by the launch script)
    model: ModelConfig = MISSING  # model name

    load_from: str = MISSING  # path to checkpoint
    output_path: str = MISSING  # path to save the result CSV file


configstore = ConfigStore.instance()
configstore.store(name='evaluate', node=EvaluateConfig)
add_model_configs_to_store(configstore)

log = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='evaluate')
def evaluate_main(cfg: EvaluateConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    log.info(s)

    # XXX: workaround ulimit error due to too many open files (with large prefetch factor)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # init distributed process group
    local_rank = int(os.environ.get('LOCAL_RANK', cfg.local_rank))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', cfg.local_world_size))
    utils.init_dist_process_group(local_rank, local_world_size)
    node_is_master = local_rank == 0
    device = torch.device(f'cuda:{local_rank}')

    def gather_distributed_tensor(array):
        result = utils.gather_numpy_arrays(array, local_rank, local_world_size, device)
        if node_is_master:
            return torch.cat(result.unbind(0), dim=0).numpy()
        else:
            return None

    # choose model
    log.info(f'Using model: {cfg.model}')

    # initialize model
    model = hydra.utils.instantiate(cfg.model)
    log.info(f'number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    utils.load_checkpoint(model, cfg.load_from)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, [device], find_unused_parameters=True)

    # load datasets
    datasets = setup_datasets(cfg)

    metric_pprint_width = 200

    loss_fn = torch.nn.BCELoss()

    # define valid step
    def valid_step(i_batch, sample, states):
        sample = utils.move_batch_to_device(sample, device)
        labels = sample['label']

        pred = model(sample)
        loss = loss_fn(pred, labels.float())

        # collect loss, true label and predicted label
        states['losses'].append(loss.detach().cpu().numpy())
        states['true'].append(labels.detach().cpu().numpy())
        states['pred'].append(pred.detach().cpu().numpy())

    # the actual evaluation loop
    model.eval()
    records = []
    dataset, role = datasets
    log.info(f'Dataset: {cfg.dataset}, Role: {role}')
    whole_true = []
    whole_pred = []
    whole_loss = []
    whole_nexample = 0
    whole_npos = 0
    whole_nneg = 0
    for subdataset in dataset.subdatasets:
        data_key = subdataset.data_key
        log.info(f'Evaluating dataset: {data_key}, count = {len(subdataset)}')

        st = time.time()
        nexample, npos, nneg, dataloader = get_dataloader_for_dataset(cfg, subdataset)
        whole_nexample += nexample
        whole_nneg += nneg
        whole_npos += npos

        record = {
            'Dataset': cfg.dataset,
            'Key': str(data_key),
            'Role': role,
            'Count': nexample,
            'NumPositives': npos,
            'NumNegatives': nneg,
        }

        if 0 in [nexample, npos, nneg]:
            log.warning(f'Dataset {data_key} has 0 examples/positives/negatives!')
            log.warning(f'nexample = {nexample}, npos = {npos}, nneg = {nneg}')
            records.append(record)
            continue

        states = {'losses': [], 'true': [], 'pred': []}
        for i_batch, sample in enumerate(dataloader):
            with torch.no_grad():
                valid_step(i_batch, sample, states)
        test_true = gather_distributed_tensor(np.concatenate(states['true'], 0))
        test_pred = gather_distributed_tensor(np.concatenate(states['pred'], 0))
        test_loss = gather_distributed_tensor(np.stack(states['losses']))

        if node_is_master:
            whole_true.append(test_true)
            whole_pred.append(test_pred)
            whole_loss.append(test_loss)
            test_metrics = {'Loss': np.mean(test_loss)}
            if test_true.sum() != len(test_true):
                test_metrics.update(utils.compute_metrics(test_true, test_pred))

            end = time.time()

            record.update(test_metrics)
            records.append(record)

            log.info(f'Key {data_key}: count {len(subdataset)}, loss = {test_metrics["Loss"]}, time = {end - st}')
            log.info('Record:')
            log.info('\n' + pformat(record, width=metric_pprint_width) + '\n')

    # output CSV
    if node_is_master:
        whole_true = np.concatenate(whole_true, 0)
        whole_pred = np.concatenate(whole_pred, 0)
        whole_loss = np.concatenate(whole_loss, 0)
        whole_record = {
            'Dataset': cfg.dataset,
            'Key': "Whole",
            'Role': role,
            'Count': whole_nexample,
            'NumPositives': whole_npos,
            'NumNegatives': whole_nneg,
        }
        whole_metrics = {'Loss': np.mean(whole_loss)}
        if whole_true.sum() != len(whole_true):
            whole_metrics.update(utils.compute_metrics(whole_true, whole_pred))
        whole_record.update(whole_metrics)
        records.append(whole_record)
        log.info(f'Writing CSV to {cfg.output_path}...')
        df = pd.DataFrame.from_records(records)
        df.to_csv(cfg.output_path)
        log.info('Done.')


def setup_datasets(cfg):
    preprocess_fn, _ = hydra.utils.call(cfg.model.args.data_funcs)

    if cfg.dataset == 'muv' or cfg.dataset == 'lit-pcba':
        # test_keys, _, _ = utils.split_train_test(cfg.data_fpath, cfg.valid_split_ratio, cfg.dataset)

        # for reproducibility
        with open(cfg.data_keys, 'rb') as f:
            test_keys = pickle.load(f)
    else:
        NotImplementedError()

    test_dataset = MolDataset(test_keys, cfg.dataset, cfg.data_fpath, preprocess_fn)

    log.info(f'Number of test data: {len(test_dataset)}')
    return test_dataset, 'test'


def get_dataloader_for_dataset(cfg, dataset):
    _, collate_fn = hydra.utils.call(cfg.model.args.data_funcs)

    if cfg.dataset == 'dud-e' or cfg.dataset == 'muv':
        labels = [int('active' in k) for k in dataset.keys()]
    elif cfg.dataset == 'lit-pcba':
        labels = [int(not 'inactive' in k) for k in dataset.keys()]
    else:
        raise NotImplementedError()

    num_chembl = sum(labels)
    num_decoy = len(dataset) - num_chembl

    if 0 in [len(labels), num_chembl, num_decoy]:
        return len(labels), num_chembl, num_decoy, None

    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn,
                            sampler=sampler, drop_last=True, prefetch_factor=8)
    return len(labels), num_chembl, num_decoy, dataloader


if __name__ == '__main__':
    evaluate_main()
