"""
Schedulers for corenet.
"""
import numpy as np
import torch.optim as optim


class Scheduler:
    """
    A standard scheduler for pytorch models.
    """
    def __init__(
        self,
        name: str = 'scheduler',
        config: dict = {},
        meta:   dict = {}
    ):
        self.name = name
        self.config = config
        self.meta = meta

        self.parse_config()

    def parse_config(self):
        if self.config['lr_schedule'] == 'cosine':
            if "warmup" not in self.config:
                self.config["warmup"] = 1
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.meta['optimizer'].optimizer,
                self.meta['num_iterations']
            )
        elif self.config['lr_schedule'] == '1cycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.meta['optimizer'].optimizer,
                max_lr=self.config["max_lr"],
                steps_per_epoch=len(self.meta["loader"].train_loader),
                epochs=self.config["num_epochs"],
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4
            )
        else:
            self.scheduler = None

    def step(self):
        return self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def find_optimum_lr(
        self,
        iteration_loss,
        iteration_learning_rate
    ):
        min_loss_idx = np.argmin(iteration_loss)
        optimal_lr = iteration_learning_rate[min_loss_idx]
        self.meta["max_lr"] = optimal_lr

    def report_tensorboard(self, iterations, train_type):
        """Report learning rate to tensorboard"""
        self.meta['tensorboard'].add_scalar(
            f'learning rate ({train_type})',
            self.scheduler.get_last_lr(),
            iterations
        )
