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
        num_projections: int = 1000,
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
        self.num_projections = num_projections

        self.gut_test = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.gut_test_output = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.gut_true = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.gut_true_output = torch.empty(
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
        self.gut_true = torch.empty(
            size=(0, 5),
            dtype=torch.float, device='cpu'
        )
        self.gut_true_output = torch.empty(
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
        self.gut_true = torch.cat(
            (self.gut_true, data['gut_true'].to('cpu')),
            dim=0
        )
        self.gut_true_output = torch.cat(
            (self.gut_true_output, data['gut_true_output'].to('cpu')),
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

    def compute(
        self,
        data
    ):
        """
        Get KL Divergence for GUT true/test and 
        between gut_true_latent and weak_test_latent.
        """
        np.random.shuffle(self.gut_test.numpy())
        np.random.shuffle(self.gut_test_output.numpy())
        np.random.shuffle(self.gut_true.numpy())
        np.random.shuffle(self.gut_true_output.numpy())
        np.random.shuffle(self.gut_true_latent.numpy())
        np.random.shuffle(self.weak_test_latent.numpy())
        data['gut_test_kldiv_metric'] = kldiv(
           self.gut_test[:self.kl_subsamples],
           self.gut_test_output[:self.kl_subsamples]
        )
        data['gut_true_kldiv_metric'] = kldiv(
           self.gut_true[:self.kl_subsamples],
           self.gut_true_output[:self.kl_subsamples]
        )
        data['gut_true_weak_test_latent_kldiv_metric'] = kldiv(
           self.gut_true_latent[:self.kl_subsamples],
           self.weak_test_latent[:self.kl_subsamples]
        )
        """Get Wasserstein distance between gut_true_latent and weak_test_latent"""
        # first, generate a random sample on a sphere
        embedding_dimension = self.gut_true_latent.size(1)
        normal_samples = np.random.normal(
            size=(self.num_projections, embedding_dimension)
        )
        normal_samples /= np.sqrt((normal_samples**2).sum())
        projections = torch.tensor(normal_samples).transpose(0, 1).to(self.device)

        # now project the embedded samples onto the sphere
        gut_true_latent_projections = data['gut_true_latent'].matmul(projections.float()).transpose(0, 1).to(self.device)
        weak_test_latent_projections = data['weak_test_latent'].matmul(projections.float()).transpose(0, 1).to(self.device)

        # calculate the distance between the distributions
        wasserstein_distance = (
            torch.sort(gut_true_latent_projections, dim=1)[0] -
            torch.sort(weak_test_latent_projections, dim=1)[0]
        )
        data['gut_true_weak_test_latent_wasserstein_metric'] = (torch.pow(wasserstein_distance, 2)).mean()
        return data
