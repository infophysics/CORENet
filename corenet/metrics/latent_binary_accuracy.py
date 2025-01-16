"""
Binary accuracy metric class for tpc_ml.
"""
import torch
import torch.nn as nn

from corenet.metrics import GenericMetric


class LatentBinaryAccuracy(GenericMetric):

    def __init__(
        self,
        name: str = 'latent_binary_accuracy_metric',
        when_to_compute: str = 'all',
        cutoff: float = 0.5,
        meta: dict = {}
    ):
        """
        Binary accuracy metric which essentially computes
        the number of correct guesses defined by a single
        cut along the output dimension.
        """
        super(LatentBinaryAccuracy, self).__init__(
            name,
            when_to_compute,
            meta
        )
        self.cutoff = cutoff
        self.batch_metric = torch.empty(
            size=(0, 1),
            dtype=torch.float, device=self.device
        )

    def reset_batch(self):
        self.batch_metric = torch.empty(
            size=(0, 1),
            dtype=torch.float, device=self.device
        )

    def update(
        self,
        data,
    ):
        # set predictions using cutoff
        predictions = (data['gut_test_binary'] > self.cutoff).unsqueeze(1).to(self.device)
        answers = torch.all(data['gut_test'] == data['gut_true'], dim=1).float().unsqueeze(1).to(self.device)
        accuracy = (predictions == answers).float().mean().to(self.device)
        self.batch_metric = torch.cat(
            (self.batch_metric, torch.tensor([[accuracy]], device=self.device)),
            dim=0
        )

    def compute(self):
        return self.batch_metric.mean()
