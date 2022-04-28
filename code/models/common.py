# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from dataclasses import dataclass

@dataclass
class ModelConfig:
    pass


def add_model_configs_to_store(configstore):
    from models.igt import IGTConfig

    configstore.store(group='model', name='igt', node=IGTConfig)
