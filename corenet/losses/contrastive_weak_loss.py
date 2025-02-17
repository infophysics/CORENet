"""
Wrapper for L2 loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from corenet.losses import GenericLoss


class ContrastiveWeakLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'contrastive_weak_loss',
        alpha:          float = 0.0,
        temperature:    float = 0.1,
        meta:           dict = {}
    ):
        super(ContrastiveWeakLoss, self).__init__(
            name, alpha, meta
        )
        self.temperature = temperature
        self.l2_loss = nn.MSELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        # Compute pairwise similarity in latent space
        latent_sim = F.log_softmax(F.cosine_similarity(
            data['weak_test_latent'].unsqueeze(1).to(self.device),
            data['weak_test_latent'].unsqueeze(0).to(self.device),
            dim=2
        ) / self.temperature)

        # Compute pairwise similarity in weak space
        weak_sim = F.cosine_similarity(
            data['weak_test'].unsqueeze(1).to(self.device),
            data['weak_test'].unsqueeze(0).to(self.device),
            dim=2
        )

        # Normalize weak similarity to be between 0 and 1
        weak_sim = (weak_sim + 1) / 2  # Shift cosine similarity from [-1,1] to [0,1]

        # Contrastive loss: push apart dissimilar, pull together similar
        loss_matrix = -weak_sim * latent_sim + (1 - weak_sim) * F.relu(1 - latent_sim)

        # Take mean over pairs
        data[self.name] = self.alpha * loss_matrix.mean()

        return data
