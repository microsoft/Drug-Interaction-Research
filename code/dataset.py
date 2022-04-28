# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import utils
import random
import pickle
from operator import itemgetter

import numpy as np
from rdkit import Chem

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.sampler import Sampler


class MolDatasetSingleGene(Dataset):
    def __init__(self, data_key, dataset_choice, data_root, preprocess_fn):
        self.data_key = data_key
        self.data_dir = data_root
        self.preprocess_fn = preprocess_fn
        self.dataset_choice = dataset_choice
        if dataset_choice == "muv":
            receptor_path = os.path.join(self.data_dir, f'{data_key}_receptor.pdb')
            ligands_path = os.path.join(self.data_dir, f'{data_key}.pkl')
            if not os.path.isfile(receptor_path) or not os.path.isfile(ligands_path):
                print(f'Warning: Skipping invalid dataset {data_key}!')
                self.data_list = []
            else:
                with open(ligands_path, 'rb') as f:
                    self.data_list = pickle.load(f)
                self.receptor_mol = Chem.MolFromPDBFile(receptor_path, sanitize=False, removeHs=True)
        elif dataset_choice == "lit-pcba":
            receptor, template = self.data_key
            receptor_path = os.path.join(self.data_dir, receptor, f'{template}_protein.pdb')
            if not os.path.isfile(receptor_path):
                print(f'Warning: Skipping invalid dataset {data_key}!')
                self.data_list = []
            else:
                self.receptor_mol = Chem.MolFromPDBFile(receptor_path, sanitize=False, removeHs=True)
                ligand_path = os.path.join(self.data_dir, receptor, template, 'valid.test.small.pkl')
                if not os.path.isfile(ligand_path):
                    print(f'Warning: Skipping invalid "pkl" file {receptor} {template} valid.test.small.pkl!')
                else:
                    with open(ligand_path, 'rb') as f:
                        self.data_list = pickle.load(f)
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ligand_filename, ligand = self.data_list[idx]
        ligand = Chem.RemoveHs(ligand, updateExplicitCount=True, sanitize=False)
        ori_receptor = self.receptor_mol
        ligand_name = ligand_filename[:-4]
        key = f'{self.data_key}_{ligand_name}'

        return self.preprocess_fn(key, ori_receptor, ligand, self.dataset_choice)

    def keys(self):
        for ligand_filename, _ in self.data_list:
            ligand_name = ligand_filename[:-4]
            key = f'{self.data_key}_{ligand_name}'
            yield key


class MolDataset(Dataset):
    def __init__(self, data_keys, dataset_choice, data_dir, preprocess_fn):
        self.data_keys = data_keys
        self.data_dir = data_dir
        self.subdatasets = [MolDatasetSingleGene(g, dataset_choice, data_dir, preprocess_fn) for g in data_keys]
        self.data_list = [(i, j) for i in range(len(data_keys)) for j in range(len(self.subdatasets[i].data_list))]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        gene_idx, ligand_idx = self.data_list[idx]
        sample = self.subdatasets[gene_idx][ligand_idx]
        return sample

    def keys(self):
        for dataset in self.subdatasets:
            for k in dataset.keys():
                yield k


# XXX: From https://github.com/catalyst-team/catalyst
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


# XXX: From https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
            self,
            sampler,
            num_replicas=None,
            rank=None,
            shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DTISampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True, seed=None):
        weights = np.array(weights) / np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed

    def __iter__(self):
        if self.seed is None:
            retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights)
        else:
            status = np.random.get_state()
            np.random.seed(self.seed)
            retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights)
            np.random.set_state(status)

        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples
