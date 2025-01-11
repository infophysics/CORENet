"""
Wasserstein loss
"""
import numpy as np
import torch

from corenet.losses import GenericLoss
from corenet.utils.utils import generate_gaussian


class LatentWassersteinLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'latent_wasserstein_loss',
        alpha:          float = 0.0,
        latent_variables: list = [],
        num_projections: int = 1000,
        meta:           dict = {}
    ):
        super(LatentWassersteinLoss, self).__init__(
            name, alpha, meta
        )

        self.latent_variables = latent_variables
        self.num_projections = num_projections
        self.distribution = generate_gaussian(dimension=len(self.latent_variables))

    def _loss(
        self,
        data,
    ):
        """
        We project our distribution onto a sphere and compute the Wasserstein
        distance between the distribution (encoded_samples) and our expected
        distribution (distribution_samples).
        """
        """GUT Test space"""
        distribution_samples = self.distribution[
            torch.randint(
                high=self.distribution.size(0),
                size=(data['gut_test_latent'].size(0),))
        ].to(self.device)
        # first, generate a random sample on a sphere
        embedding_dimension = distribution_samples.size(1)
        normal_samples = np.random.normal(
            size=(self.num_projections, embedding_dimension)
        )
        normal_samples /= np.sqrt((normal_samples**2).sum())
        projections = torch.tensor(normal_samples).transpose(0, 1).to(self.device)

        # now project the embedded samples onto the sphere
        encoded_projections = data['gut_test_latent'].matmul(projections.float()).transpose(0, 1).to(self.device)
        distribution_projections = distribution_samples.float().matmul(projections.float()).transpose(0, 1).to(self.device)

        # calculate the distance between the distributions
        wasserstein_distance = (
            torch.sort(encoded_projections, dim=1)[0] -
            torch.sort(distribution_projections, dim=1)[0]
        )
        data['gut_test_wasserstein_loss'] = (torch.pow(wasserstein_distance, 2)).mean()

        """GUT True space"""
        distribution_samples = self.distribution[
            torch.randint(
                high=self.distribution.size(0),
                size=(data['gut_true_latent'].size(0),))
        ].to(self.device)
        # first, generate a random sample on a sphere
        embedding_dimension = distribution_samples.size(1)
        normal_samples = np.random.normal(
            size=(self.num_projections, embedding_dimension)
        )
        normal_samples /= np.sqrt((normal_samples**2).sum())
        projections = torch.tensor(normal_samples).transpose(0, 1).to(self.device)

        # now project the embedded samples onto the sphere
        encoded_projections = data['gut_true_latent'].matmul(projections.float()).transpose(0, 1).to(self.device)
        distribution_projections = distribution_samples.float().matmul(projections.float()).transpose(0, 1).to(self.device)

        # calculate the distance between the distributions
        wasserstein_distance = (
            torch.sort(encoded_projections, dim=1)[0] -
            torch.sort(distribution_projections, dim=1)[0]
        )
        data['gut_true_wasserstein_loss'] = (torch.pow(wasserstein_distance, 2)).mean()

        # """Weak test space"""
        # distribution_samples = self.distribution[
        #     torch.randint(
        #         high=self.distribution.size(0),
        #         size=(data['weak_test_latent'].size(0),))
        # ].to(self.device)
        # # first, generate a random sample on a sphere
        # embedding_dimension = distribution_samples.size(1)
        # normal_samples = np.random.normal(
        #     size=(self.num_projections, embedding_dimension)
        # )
        # normal_samples /= np.sqrt((normal_samples**2).sum())
        # projections = torch.tensor(normal_samples).transpose(0, 1).to(self.device)

        # # now project the embedded samples onto the sphere
        # encoded_projections = data['weak_test_latent'].matmul(projections.float()).transpose(0, 1).to(self.device)
        # distribution_projections = distribution_samples.float().matmul(projections.float()).transpose(0, 1).to(self.device)

        # # calculate the distance between the distributions
        # wasserstein_distance = (
        #     torch.sort(encoded_projections, dim=1)[0] -
        #     torch.sort(distribution_projections, dim=1)[0]
        # )
        # data['weak_test_wasserstein_loss'] = (torch.pow(wasserstein_distance, 2)).mean()

        data[self.name] = self.alpha * (
            data['gut_test_wasserstein_loss'] + data['gut_true_wasserstein_loss']
        )
        return data
