"""
Wrapper for L2 loss
"""
import torch
import torch.nn as nn

from corenet.losses import GenericLoss


class L2GUTLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'l2_gut_loss',
        alpha:          float = 0.0,
        losses:         list = [
            'gut_test_l2_loss',
            'gut_true_l2_loss',
            'weak_test_l2_loss',
            'weak_latent_l2_loss',
            'gut_true_latent_l2_loss'
        ],
        meta:           dict = {}
    ):
        super(L2GUTLoss, self).__init__(
            name, alpha, meta
        )
        self.losses = losses
        self.l2_loss = nn.MSELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        """Computes and returns/saves loss information"""
        data['gut_test_l2_loss'] = self.l2_loss(
            data['gut_test'].to(self.device),
            data['gut_test_output'].to(self.device)
        )
        data['gut_true_l2_loss'] = self.l2_loss(
            data['gut_true'].to(self.device),
            data['gut_true_output'].to(self.device)
        )
        data['weak_test_l2_loss'] = self.l2_loss(
            data['gut_true'].to(self.device),
            data['weak_test_output'].to(self.device)
        )
        data['weak_latent_l2_loss'] = self.l2_loss(
            data['gut_true_latent'].to(self.device),
            data['weak_test_latent'].to(self.device)
        )
        data['gut_true_latent_l2_loss'] = self.l2_loss(
            data['gut_true_latent'].to(self.device),
            data['weak_true_latent'].to(self.device)
        )
        total_loss = 0.0
        for loss in self.losses:
            total_loss += self.alpha * data[loss]
        data[self.name] = total_loss
        return data
