# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, balanced_accuracy_score, roc_curve

import torch
import torch.distributed as dist

import dgl


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint = {
        k.replace('module.', ''): v
        for k, v in checkpoint.items()
    }
    model.load_state_dict(checkpoint)


def tensor_recursion(f, tensor):
    def helper(t):
        if isinstance(t, torch.Tensor):
            return f(t)
        elif isinstance(t, dgl.DGLGraph):
            return f(t)
        elif isinstance(t, dict):
            return {k: helper(v) for k, v in t.items()}
        else:
            print('Non-supported type:', type(t))
            raise NotImplementedError()

    return helper(tensor)


def move_batch_to_device(batch, device):
    return tensor_recursion(lambda t: t.to(device=device), batch)


def init_dist_process_group(local_rank, local_world_size):
    # initialize distributed process group
    if local_rank is None or local_world_size is None:
        raise NotImplementedError('Please run this script in distributed mode!')

    env_dict = {
        key: os.environ[key]
        for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    device_ids = [local_rank]
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = 1, device_ids = {device_ids}"
    )


def gather_numpy_arrays(array, local_rank, local_world_size, device):
    # NOTE: NCCL does not support GATHER, so we will have to simulate that using SUM
    # tensor = torch.tensor(array)
    # tensors = [torch.zeros(tensor.shape) for _ in range(local_world_size)] if local_rank == 0 else None
    # dist.gather(tensor, tensors, 0)
    tensor = torch.tensor(array, device=device)
    tensors = torch.zeros((local_world_size,) + tensor.shape, device=device)
    tensors[local_rank] = tensor
    dist.reduce(tensors, 0, dist.ReduceOp.SUM)
    tensors = tensors.cpu()
    return tensors


def split_train_test(data_dir, ratio, dataset):
    full = []
    if dataset == "dud-e" or dataset == "muv":
        full = [m[:-4] for m in os.listdir(data_dir) if m.endswith(".pkl")]
    elif dataset == "lit-pcba":
        receptors = [rec for rec in os.listdir(data_dir) if not (rec.endswith('seqs') or rec == 'all')]
        for receptor in receptors:
            template_list = [t for t in os.listdir(os.path.join(data_dir, receptor)) if not t.endswith(".pdb")]
            template_list.sort()
            template_list = template_list[:1]  # how many templates to use
            full += [(receptor, m) for m in template_list]
    train, test = train_test_split(full, test_size=ratio, random_state=2021)
    return full, train, test


def compute_metrics(true, pred):
    metrics = [
        ('Roc', roc_auc_score),
        ('AUPRC', pr_auc_score),
        ('Balanced Acc', balanced_accuracy),
        ('ROC Enrich 1', lambda t, p: ROC_enrichment(t, p, 0.01)),
        ('Enrich Factor', enrichment_factor),
        ('LogAUC', LogAUC),
        ('Sensitivity', lambda t, p: sensitivity_specificity(t, p)[0]),
        ('Specificity', lambda t, p: sensitivity_specificity(t, p)[1]),
    ]
    results = {
        name: func(true, pred)
        for name, func in metrics
    }
    return results


def sensitivity_specificity(y_label, y_pred, threshold=0.5):
    y_pred = np.array([int(i > threshold) for i in y_pred])
    tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def pr_auc_score(y_label, y_pred):
    precision, recall, _thresholds = precision_recall_curve(y_label, y_pred)
    return auc(recall, precision)


def balanced_accuracy(y_label, y_pred, threshold=0.5):
    y_pred = np.array([int(i > threshold) for i in y_pred])
    return balanced_accuracy_score(y_label, y_pred)


def ROC_enrichment(y_label, y_pred, given_fpr):
    fpr, tpr, threshold = roc_curve(y_label, y_pred)
    assert fpr[0] == 0
    assert fpr[-1] == 1
    assert np.all(np.diff(fpr) >= 0)
    return np.true_divide(np.interp(given_fpr, fpr, tpr), given_fpr)


def enrichment_factor(y_label, y_pred, top_rate=0.02):
    index = y_pred.argsort()
    index = index[::-1]
    y_label = y_label[index]
    original_rate = np.sum(y_label) / len(y_label)
    topx_number = int(np.ceil(top_rate * len(y_label)))
    enrichment_rate = np.sum(y_label[:topx_number]) / topx_number
    return enrichment_rate / original_rate


def LogAUC(y_label, y_pred, min_fp=0.001, adjusted=True):
    fp, tp, thresholds = roc_curve(y_label, y_pred)

    lam_index = np.searchsorted(fp, min_fp)
    y = np.asarray(tp[lam_index:], dtype=np.double)
    x = np.asarray(fp[lam_index:], dtype=np.double)
    if lam_index != 0:
        y = np.insert(y, 0, tp[lam_index - 1])
        x = np.insert(x, 0, min_fp)

    dy = (y[1:] - y[:-1])
    with np.errstate(divide='ignore'):
        intercept = y[1:] - x[1:] * (dy / (x[1:] - x[:-1]))
        intercept[np.isinf(intercept)] = 0.
    norm = np.log10(1. / float(min_fp))
    areas = ((dy / np.log(10.)) + intercept * np.log10(x[1:] / x[:-1])) / norm
    logauc = np.sum(areas)
    if adjusted:
        logauc -= 0.145  # random curve logAUC
    return logauc
