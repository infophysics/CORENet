"""
Generic metric class for corenet.
"""
import torch

from corenet.utils.logger import Logger


class GenericMetric:
    """
    Abstract base class for corenet metrics.  The inputs are
        1. name - a unique name for the metric function.
        2. when_to_compute - when to compute the metric, 'train', 'validation', 'test', 'train_all', 'inference', 'all'
        3. meta - meta information from the module.
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

    def reset_batch(self):
        pass

    def update(
        self,
        data
    ):
        self.logger.error('"update" not implemented in Metric!')

    def compute(
        self,
        data
    ):
        self.logger.error('"compute" not implemented in Metric!')
