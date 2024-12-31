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
        meta:           dict = {}
    ):
        super(L2GUTLoss, self).__init__(
            name, alpha, meta
        )
        self.l2_loss = nn.MSELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        """Computes and returns/saves loss information"""
        data['gut_test_loss'] = self.alpha * self.l2_loss(
            data['gut_test'].to(self.device),
            data['gut_test_output'].to(self.device)
        )
        data['gut_true_loss'] = self.alpha * self.l2_loss(
            data['gut_true'].to(self.device),
            data['gut_true_output'].to(self.device)
        )
        data['weak_test_loss'] = self.alpha * self.l2_loss(
            data['gut_test'].to(self.device),
            data['gut_true_output'].to(self.device)
        )
        data[self.name] = data['gut_test_loss'] + data['gut_true_loss'] + data['weak_test_loss']
        return data
