

import torch
import os.path as osp
import numpy   as np
from tqdm import tqdm
from dataclasses import dataclass
from matplotlib import pyplot as plt
import pickle

from torch_geometric.data import Data
from torch.utils.data import Dataset

from corenet.utils.utils import generate_plot_grid


corenet_dataset_config = {
    "dataset_folder":   "data/",
    "dataset_files":    [""],
}


class CORENetDataset(Dataset):
    """
    """
    def __init__(
        self,
        name:   str = "corenet",
        config: dict = corenet_dataset_config,
        meta:   dict = {}
    ):
        self.name = name
        self.config = config
        self.meta = meta

        self.process_config()

    def process_config(self):
        self.num_events = 10
        self.gut_test = self.config['gut_test']
        self.gut_true = self.config['gut_true']
        self.weak_test = self.config['weak_test']

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        return {
            'gut_test': torch.Tensor([0, 1, 2, 3, 4]),
            'gut_true': torch.Tensor([0, 1, 2, 3, 4]),
            'weak_test': torch.Tensor([0, 1, 2, 3])
        }

    def save_predictions(
        self,
        model_name,
        predictions,
        indices
    ):
        pass
