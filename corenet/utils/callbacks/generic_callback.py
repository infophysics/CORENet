"""
Functions for evaluating and storing training information.
"""
import matplotlib.colors as mcolors
import random

from corenet.losses.loss_handler import LossHandler
from corenet.metrics.metric_handler import MetricHandler


class GenericCallback:
    """
    """
    def __init__(
        self,
        name:               str = 'generic',
        criterion_handler:  LossHandler = None,
        metrics_handler:    MetricHandler = None,
        meta:               dict = {}
    ):
        self.name = name
        self.criterion_handler = criterion_handler
        self.metrics_handler = metrics_handler
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']

        self.epochs = None
        self.num_training_batches = None
        self.num_validation_batches = None
        self.num_test_batches = None
        self.plot_colors = list(mcolors.CSS4_COLORS.values())
        random.shuffle(self.plot_colors)

    def set_device(
        self,
        device
    ):
        self.device = device

    def reset_batch(self):
        pass

    def set_training_info(
        self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:       int,
    ):
        self.epochs = epochs
        self.num_training_batches = num_training_batches
        self.num_validation_batches = num_validation_batches
        self.num_test_batches = num_test_batches

    def evaluate_epoch(
        self,
        train_type='train'
    ):
        pass

    def evaluate_training(self):
        pass

    def evaluate_testing(self):
        pass

    def evaluate_inference(self):
        pass
