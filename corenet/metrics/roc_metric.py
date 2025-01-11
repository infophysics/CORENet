"""
Generic metrics for blip.
"""
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchmetrics.classification import ROC, AUROC

from corenet.utils.utils import fig_to_array
from corenet.metrics.generic_metric import GenericMetric


class CORENetROC(GenericMetric):
    """
    """
    def __init__(
        self,
        name:           str = 'corenet_roc_metric',
        meta:           dict = {}
    ):
        self.name = name
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        self.num_thresholds = 100

        # construct batch metric dictionaries
        self.labels = ['gut_test', 'gut_true']
        self.tasks = {
            'gut_test': 'binary',
            'gut_true': 'binary'
        }
        self.roc = {
            key: ROC(
                task=self.tasks[key],
                num_classes=2,
                thresholds=self.num_thresholds
            ).to(self.device)
            for key in self.labels
        }
        self.auroc = {
            key: AUROC(
                task=self.tasks[key],
                num_classes=2,
                thresholds=self.num_thresholds,
                average=None
            ).to(self.device)
            for key in self.labels
        }

    def reset_batch(self):
        for key in self.roc.keys():
            self.roc[key] = ROC(
                task=self.tasks[key],
                num_classes=2,
                thresholds=self.num_thresholds
            ).to(self.device)
            self.auroc[key] = AUROC(
                task=self.tasks[key],
                num_classes=2,
                thresholds=self.num_thresholds,
                average=None
            ).to(self.device)

    def set_device(
        self,
        device
    ):
        self.device = device
        for key in self.roc.keys():
            self.roc[key].to(self.device)
            self.auroc[key].to(self.device)

    def report_tensorboard(
        self,
        iterations,
        train_type
    ):
        for ii, output in enumerate(self.roc.keys()):
            fpr, tpr, thresholds = self.roc[output].compute()
            aurocs = self.auroc[output].compute()
            if output in ['topology', 'physics']:
                for jj, label in enumerate(self.class_labels[output]):
                    fig, axs = plt.subplots(figsize=(10, 10))
                    axs.set_title(f"{output.capitalize()}:{label.capitalize()} ({train_type.capitalize()})")
                    axs.plot(tpr[jj].cpu(), 1 - fpr[jj].cpu(), linestyle='--', c='k', label=f'AUC = {aurocs[jj].cpu():.2f}')
                    axs.set_xlabel('Signal Acceptance [tpr]')
                    axs.set_ylabel('Background Rejection [1 - fpr]')
                    axs.set_title(f'ROC tpr vs. 1 - fpr: {output}:{label} ({train_type})')
                    axs.legend()
                    fig_array = fig_to_array(fig)
                    self.meta['tensorboard'].add_image(
                        f'{self.name}: {output}:{label} ({train_type})',
                        fig_array,
                        iterations,
                        dataformats='HWC'
                    )
                    plt.close()
            else:
                fig, axs = plt.subplots(figsize=(10, 10))
                axs.set_title(f"{output.capitalize()} ({train_type.capitalize()})")
                axs.plot(tpr.cpu(), 1 - fpr.cpu(), linestyle='--', c='k', label=f'AUC = {aurocs.cpu():.2f}')
                axs.set_xlabel('Signal Acceptance [tpr]')
                axs.set_ylabel('Background Rejection [1 - fpr]')
                axs.set_title(f'ROC tpr vs. 1 - fpr: {output} ({train_type})')
                axs.legend()
                fig_array = fig_to_array(fig)
                self.meta['tensorboard'].add_image(
                    f'{self.name}: {output} ({train_type})',
                    fig_array,
                    iterations,
                    dataformats='HWC'
                )
                plt.close()

    def update(
        self,
        data
    ):
        for ii, output in enumerate(self.roc.keys()):
            if output in ['topology', 'physics']:
                self.roc[output].update(
                    nn.functional.softmax(data['outputs'][output].to(self.device), dim=1, dtype=torch.float),
                    data['labels'].squeeze(0)[:, ii].long().to(self.device)
                )
                self.auroc[output].update(
                    nn.functional.softmax(data['outputs'][output].to(self.device), dim=1, dtype=torch.float),
                    data['labels'].squeeze(0)[:, ii].long().to(self.device)
                )
            else:
                self.roc[output].update(
                    nn.functional.softmax(data['outputs'][output].squeeze(1).to(self.device), dim=0, dtype=torch.float),
                    data['labels'].squeeze(0)[:, ii].long().to(self.device)
                )
                self.auroc[output].update(
                    nn.functional.softmax(data['outputs'][output].squeeze(1).to(self.device), dim=0, dtype=torch.float),
                    data['labels'].squeeze(0)[:, ii].long().to(self.device)
                )

    def compute(
        self,
    ):
        return {
            output: [self.roc[output].compute(), self.auroc[output].compute()]
            for output in self.roc.keys()
        }