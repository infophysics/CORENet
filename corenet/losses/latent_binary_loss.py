"""
Wrapper for bce loss
"""
import torch
import torch.nn as nn

from corenet.losses import GenericLoss


class LatentBinaryLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'bce_gut_loss',
        alpha:          float = 0.0,
        meta:           dict = {}
    ):
        super(LatentBinaryLoss, self).__init__(
            name, alpha, meta
        )
        self.bce_loss = nn.BCELoss(reduction='mean')

    def _loss(
        self,
        data
    ):
        """Computes and returns/saves loss information"""
        data['gut_test_bce_loss'] = self.bce_loss(
            data['gut_test_binary'].to(self.device),
            torch.all(data['gut_test'] == data['gut_true'], dim=1).float().unsqueeze(1).to(self.device)
        )
        data['gut_true_bce_loss'] = self.bce_loss(
            data['gut_true_binary'].to(self.device),
            torch.all(data['gut_true'] == data['gut_true'], dim=1).float().unsqueeze(1).to(self.device)
        )
        data[self.name] = self.alpha * (data['gut_test_bce_loss'] + data['gut_true_bce_loss'])
        return data
