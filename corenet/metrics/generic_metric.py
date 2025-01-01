"""
Generic metric class for corenet.
"""
import torch

from corenet.utils.logger import Logger


class GenericMetric:
    """
    Abstract base class for corenet metrics.  The inputs are
        1. name - a unique name for the metric function.
        2. target_type - specifies whether the targets are features/classes/clusters/hits/etc.
        3. when_to_compute - when to compute the metric, 'train', 'validation', 'test', 'train_all', 'inference', 'all'
        4. targets - list of names for the targets.
        5. outputs - list of names of the associated outputs for each target.
        6. augmentations - specified whether augmentations are created for the dataset
        7. meta - meta information from the module.
    """
    def __init__(
        self,
        name:           str = 'generic',
        when_to_compute:    str = 'all',
        meta:           dict = {}
    ):
        self.name = name
        self.logger = Logger(self.name, output="both", file_mode="w")
        self.when_to_compute = when_to_compute
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

    def set_device(
        self,
        device
    ):
        self.device = device

    def _reset_batch(self):
        pass

    def reset_batch(self):
        self._reset_batch()

    def _metric_update(
        self,
        data
    ):
        self.logger.error('"_metric_update" not implemented in Metric!')

    def _metric_compute(self):
        pass

    def compute(self):
        return self._metric_compute()
