"""
Binary accuracy metric class for tpc_ml.
"""
import torch
import torch.nn as nn
import numpy as np

from corenet.metrics import GenericMetric
from corenet.utils.npeet import kldiv


class LatentDistanceMetric(GenericMetric):

    def __init__(
        self,
        name: str = 'latent_distance_metric',
        when_to_compute: str = 'all',
        kl_subsamples: int = 10000,
        cutoff: float = 0.5,
        meta: dict = {}
    ):
        """
        Distance metrics computed on entire datasets
        """
        super(LatentDistanceMetric, self).__init__(
            name,
            when_to_compute,
            meta
        )
        self.kl_subsamples = kl_subsamples
        self.cutoff = cutoff

        self.gut_test = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.gut_test_output = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.gut_true_latent = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.weak_test_latent = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )

    def reset_batch(self):
        self.gut_test = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.gut_test_output = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.gut_true_latent = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.weak_test_latent = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )

    def update(
        self,
        data,
    ):
        # set predictions using cutoff
        self.gut_test = torch.cat(
            (self.gut_test, data['gut_test'].to('cpu')),
            dim=0
        )
        self.gut_test_output = torch.cat(
            (self.gut_test_output, data['gut_test_output'].to('cpu')),
            dim=0
        )
        self.gut_true_latent = torch.cat(
            (self.gut_true_latent, data['gut_true_latent'].to('cpu')),
            dim=0
        )
        self.weak_test_latent = torch.cat(
            (self.weak_test_latent, data['weak_test_latent'].to('cpu')),
            dim=0
        )

    def compute(self):
        return kldiv(
            np.random.shuffle(self.gut_test.numpy())[:self.kl_subsamples],
            np.random.shuffle(self.gut_test_output.numpy())[:self.kl_subsamples]
        )
