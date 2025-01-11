"""
Class for a generic model trainer.
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from corenet.utils.logger import Logger
from corenet.losses import LossHandler
from corenet.models import ModelHandler
from corenet.metrics import MetricHandler
from corenet.optimizers import Optimizer
from corenet.utils.timing import Timers
from corenet.utils.memory import MemoryTrackers
from corenet.utils.callbacks import CallbackHandler
from corenet.utils.callbacks import TimingCallback, MemoryTrackerCallback

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    This class is an attempt to reduce code rewriting by putting together
    a set of functions that do everything that we could need with
    respect to training.  There are a few objects which must be passed
    to the trainer, which include:
        (a) model     - an object which inherits from nn.Module
        (b) criterion - an object which has a defined function called "loss"
        (c) optimizer - some choice of optimizer, e.g. Adam
        (d) metrics   - (optional) an object which has certain defined functions
        (e) callbacks - (optional) an object which has certain defined functions
    """
    def __init__(
        self,
        name:       str = 'default',
        config:     dict = {},
        meta:   dict = {},
        seed:   int = 0,
    ):
        self.name = name + '_trainer'
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(self.name, output="both", file_mode="w")
        else:
            self.logger = Logger(self.name, level='warning', file_mode="w")
        self.config = config
        # check for devices
        self.gpu = self.meta['gpu']
        self.seed = seed

        self.process_config()

    def process_config(self):
        self.logger.info("constructing model trainer.")
        if "grad_norm" not in self.config.keys():
            self.config["grad_norm"] = False
        self.grad_norm = self.config["grad_norm"]
        self.process_model()
        self.process_directories()
        self.process_criterion()
        self.process_optimizer()
        self.process_metrics()
        self.process_callbacks()
        self.process_consistency_check()

    def process_model(self):
        if "model" not in self.meta:
            self.logger.error('no model specified in meta!')
        if not isinstance(self.meta['model'], ModelHandler):
            self.logger.error(
                f'specified model is of type {type(self.meta["model"])}, but should be of type ModelHandler!'
            )
        self.model = self.meta['model'].model
        self.model.set_device(self.device)

    def process_directories(self):
        # define directories
        if "timing_dir" not in self.config:
            self.config["timing_dir"] = f'{self.meta["run_directory"]}'
        self.meta['timing_dir'] = self.config['timing_dir']
        if not os.path.isdir(self.config['timing_dir']):
            self.logger.info(f"creating timing directory {self.config['timing_dir']}")
            os.makedirs(self.config['timing_dir'])
        if "memory_dir" not in self.config:
            self.config["memory_dir"] = f'{self.meta["run_directory"]}'
        self.meta['memory_dir'] = self.config['memory_dir']
        if not os.path.isdir(self.config['memory_dir']):
            self.logger.info(f"creating timing directory {self.config['memory_dir']}")
            os.makedirs(self.config['memory_dir'])

    def process_criterion(self):
        if "criterion" not in self.meta:
            self.logger.error('no criterion specified in meta!')
        if not isinstance(self.meta['criterion'], LossHandler):
            self.logger.error(
                f'specified criterion is of type {type(self.meta["criterion"])}, but should be of type LossHandler!'
            )
        self.criterion = self.meta['criterion']
        self.criterion.set_device(self.device)

    def process_optimizer(self):
        if "optimizer" not in self.meta:
            self.logger.error('no optimizer specified in meta!')
        if not isinstance(self.meta['optimizer'], Optimizer):
            self.logger.error(
                f'specified optimizer is of type {type(self.meta["optimizer"])}, but should be of type Optimizer!'
            )
        self.optimizer = self.meta['optimizer']

    def process_metrics(self):
        if "metrics" not in self.meta:
            self.logger.error('no metrics specified in meta!')
        if not isinstance(self.meta['metrics'], MetricHandler):
            if not isinstance(self.meta['metrics'], None):
                self.logger.error(
                    f'specified metrics is of type {type(self.meta["metrics"])}, but should be of type MetricHandler or None!'
                )
        self.metrics = self.meta['metrics']
        if self.metrics is not None:
            self.metrics.set_device(self.device)

    def process_callbacks(self):
        if "callbacks" not in self.meta:
            self.logger.error('no callbacks specified in meta!')
        if not isinstance(self.meta['callbacks'], CallbackHandler):
            if not isinstance(self.meta['callbacks'], None):
                self.logger.error(
                    f'specified callbacks is of type {type(self.meta["callbacks"])}, ' +
                    'but should be of type CallbackHandler or None!'
                )
        self.callbacks = self.meta['callbacks']
        if self.callbacks is None:
            # add generic callbacks
            self.callbacks = CallbackHandler(
                name="default"
            )
        # add timing info
        self.timers = Timers(gpu=self.gpu)
        self.timer_callback = TimingCallback(
            output_dir=self.meta['timing_dir'],
            timers=self.timers
        )
        # self.callbacks.add_callback(self.timer_callback)

        # add memory info
        self.memory_trackers = MemoryTrackers(gpu=self.gpu)
        self.memory_callback = MemoryTrackerCallback(
            output_dir=self.meta['memory_dir'],
            memory_trackers=self.memory_trackers
        )
        # self.callbacks.add_callback(self.memory_callback)

    def process_consistency_check(self):
        # run consistency check
        self.logger.info("running consistency check...")

    def save_checkpoint(
        self,
        epoch:  int = 99999
    ):
        if not os.path.exists(f"{self.meta['run_directory']}/.checkpoints/"):
            os.makedirs(f"{self.meta['run_directory']}/.checkpoints/")
        torch.save(
            self.model.state_dict(),
            f"{self.meta['run_directory']}/.checkpoints/checkpoint_{epoch}.ckpt"
        )

    def train(
        self,
        epochs:     int = 100,          # number of epochs to train
        checkpoint: int = 10,           # epochs inbetween weight saving
        progress_bar:   str = 'all',    # progress bar from tqdm
        rewrite_bar:    bool = False,   # wether to leave the bars after each epoch
        save_predictions: bool = True,  # wether to save network outputs for all events to original file
        no_timing:      bool = False,   # wether to keep the bare minimum timing info as a callback
        skip_metrics:   bool = False,   # wether to skip metrics except for testing sets
    ):
        """
        Main training loop.  First, we see if the user wants to omit timing information.
        """
        if (self.model.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.model.device}' are different!")
        if (self.criterion.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.criterion.device}' are different!")

        self.model.save_model(flag='init')
        # setting values in callbacks
        self.callbacks.set_device(self.device)
        self.callbacks.set_training_info(
            epochs,
            self.meta['loader'].num_train_batches,
            self.meta['loader'].num_validation_batches,
            self.meta['loader'].num_test_batches
        )
        if not skip_metrics and self.metrics is None:
            self.logger.error('skip_metrics set to false in config, but no metrics are specified!')
        # Training
        self.logger.info(f"training dataset '{self.meta['dataset'].name}' for {epochs} epochs.")

        # set up tensorboard
        self.meta['tensorboard_dir'] = self.meta['run_directory']
        self.meta['tensorboard'] = SummaryWriter(
            log_dir=self.meta['tensorboard_dir']
        )
        """
        Training usually consists of the following steps:
            (1) Zero-out training/validation/testing losses and metrics
            (2) Loop for N epochs:
                (a) Grab the current batch of (training/validation) data.
                (b) Run the data through the model and calculate losses/metrics.
                (c) Backpropagate the loss (training)
            (3) Evaluate the trained model on testing data.
        """
        train_iteration = 0
        val_iteration = 0
        test_losses = {}
        # iterate over epochs
        for epoch in range(epochs):
            epoch_train_losses = {}
            epoch_val_losses = {}
            """
            Training stage.
            Setup the progress bar for the training loop.
            """
            if (progress_bar == 'all' or progress_bar == 'train'):
                training_loop = tqdm(
                    enumerate(self.meta['loader'].train_loader, 0),
                    total=len(self.meta['loader'].train_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='green'
                )
            else:
                training_loop = enumerate(self.meta['loader'].train_loader, 0)

            # make sure to set model to train() during training!
            self.model.train()
            """
            Setup timing/memory information for epoch.
            """
            self.timers.timers['epoch_training'].start()
            self.memory_trackers.memory_trackers['epoch_training'].start()
            self.timers.timers['training_data'].start()
            self.memory_trackers.memory_trackers['training_data'].start()
            for ii, data in training_loop:
                train_iteration += 1
                self.memory_trackers.memory_trackers['training_data'].end()
                self.timers.timers['training_data'].end()
                # zero the parameter gradients
                """
                There are choices here, either one can do:
                    model.zero_grad() or
                    optimizer.zero_grad() or
                    for param in model.parameters():        <== optimal choice
                        param.grad = None
                """
                self.timers.timers['training_zero_grad'].start()
                self.memory_trackers.memory_trackers['training_zero_grad'].start()
                for param in self.model.parameters():
                    param.grad = None
                self.memory_trackers.memory_trackers['training_zero_grad'].end()
                self.timers.timers['training_zero_grad'].end()
                # get the network output
                """
                The forward call takes in the entire data
                stream, which could have multiple inputs needed.
                It's up to the model to determine what to do with it.
                The forward call of the model could send out
                multiple output tensors, depending on the application
                (such as in an AE where the latent space values are
                important). It's up to the loss function to know what to expect.
                """
                self.timers.timers['training_forward'].start()
                self.memory_trackers.memory_trackers['training_forward'].start()
                data = self.model(data)
                self.memory_trackers.memory_trackers['training_forward'].end()
                self.timers.timers['training_forward'].end()

                # compute loss
                self.timers.timers['training_loss'].start()
                self.memory_trackers.memory_trackers['training_loss'].start()
                data = self.criterion.loss(data, task='training')
                self.memory_trackers.memory_trackers['training_loss'].end()
                self.timers.timers['training_loss'].end()

                # backprop
                self.timers.timers['training_loss_backward'].start()
                self.memory_trackers.memory_trackers['training_loss_backward'].start()
                if self.grad_norm:
                    data['loss'].backward(retain_graph=True)
                    self.criterion.update_task_weights(data['grad_norm_loss'], self.meta['optimizer'])
                else:
                    data['loss'].backward()
                self.memory_trackers.memory_trackers['training_loss_backward'].end()
                self.timers.timers['training_loss_backward'].end()

                # record backprop timing
                self.timers.timers['training_backprop'].start()
                self.memory_trackers.memory_trackers['training_backprop'].start()
                self.optimizer.step()
                self.memory_trackers.memory_trackers['training_backprop'].end()
                self.timers.timers['training_backprop'].end()

                # update progress bar
                self.timers.timers['training_progress'].start()
                self.memory_trackers.memory_trackers['training_progress'].start()
                if (progress_bar == 'all' or progress_bar == 'train'):
                    training_loop.set_description(f"Training: Epoch [{epoch+1}/{epochs}]")
                    training_loop.set_postfix_str(f"loss={data['loss'].item():.2e}")
                self.memory_trackers.memory_trackers['training_progress'].end()
                self.timers.timers['training_progress'].end()

                self.timers.timers['training_data'].start()
                self.memory_trackers.memory_trackers['training_data'].start()

                for key, value in data.items():
                    if ("loss" in key):
                        if key not in epoch_train_losses:
                            epoch_train_losses[key] = []
                        epoch_train_losses[key].append(value.detach().cpu())
                        self.meta['tensorboard'].add_scalar(key + '_train', value, train_iteration)
            for key, value in epoch_train_losses.items():
                self.meta['tensorboard'].add_scalar(key + '_train_avg', np.mean(value), epoch)

            # update timing info
            self.memory_trackers.memory_trackers['epoch_training'].end()
            self.timers.timers['epoch_training'].end()
            if not skip_metrics:
                self.model.eval()
                with torch.no_grad():
                    """
                    Run through a metric loop if there are any metrics
                    defined.
                    """
                    if self.metrics is not None:
                        if (progress_bar == 'all' or progress_bar == 'train'):
                            metrics_training_loop = tqdm(
                                enumerate(self.meta['loader'].train_loader, 0),
                                total=len(self.meta['loader'].train_loader),
                                leave=rewrite_bar,
                                position=0,
                                colour='green'
                            )
                        else:
                            metrics_training_loop = enumerate(self.meta['loader'].train_loader, 0)
                        self.metrics.reset_batch()
                        for ii, data in metrics_training_loop:
                            # update metrics
                            self.timers.timers['training_metrics'].start()
                            self.memory_trackers.memory_trackers['training_metrics'].start()
                            data = self.model(data)
                            self.metrics.update(data, train_type="train")
                            self.memory_trackers.memory_trackers['training_metrics'].end()
                            self.timers.timers['training_metrics'].end()
                            if (progress_bar == 'all' or progress_bar == 'train'):
                                metrics_training_loop.set_description(f"Training Metrics: Epoch [{epoch+1}/{epochs}]")

                        """Get metrics and report to tensorboard"""
                        metrics = self.metrics.compute()
                        for key, value in metrics.items():
                            self.meta['tensorboard'].add_scalar(key + '_train', value, epoch)

            # evaluate callbacks
            self.timers.timers['training_callbacks'].start()
            self.memory_trackers.memory_trackers['training_callbacks'].start()
            self.callbacks.evaluate_epoch(train_type='train')
            self.memory_trackers.memory_trackers['training_callbacks'].end()
            self.timers.timers['training_callbacks'].end()

            """
            Validation stage.
            Setup the progress bar for the validation loop.
            """
            if (progress_bar == 'all' or progress_bar == 'validation'):
                validation_loop = tqdm(
                    enumerate(self.meta['loader'].validation_loader, 0),
                    total=len(self.meta['loader'].validation_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='blue'
                )
            else:
                validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
            # make sure to set model to eval() during validation!
            self.model.eval()
            with torch.no_grad():
                """
                Setup timing information for epoch.
                """
                self.timers.timers['epoch_validation'].start()
                self.memory_trackers.memory_trackers['epoch_validation'].start()
                self.timers.timers['validation_data'].start()
                self.memory_trackers.memory_trackers['validation_data'].start()
                for ii, data in validation_loop:
                    val_iteration += 1
                    self.memory_trackers.memory_trackers['validation_data'].end()
                    self.timers.timers['validation_data'].end()
                    # get the network output
                    self.timers.timers['validation_forward'].start()
                    self.memory_trackers.memory_trackers['validation_forward'].start()
                    data = self.model(data)
                    self.memory_trackers.memory_trackers['validation_forward'].end()
                    self.timers.timers['validation_forward'].end()

                    # compute loss
                    self.timers.timers['validation_loss'].start()
                    self.memory_trackers.memory_trackers['validation_loss'].start()
                    data = self.criterion.loss(data, task='validation')
                    self.memory_trackers.memory_trackers['validation_loss'].end()
                    self.timers.timers['validation_loss'].end()

                    # update progress bar
                    self.timers.timers['validation_progress'].start()
                    self.memory_trackers.memory_trackers['validation_progress'].start()
                    if (progress_bar == 'all' or progress_bar == 'validation'):
                        validation_loop.set_description(f"Validation: Epoch [{epoch+1}/{epochs}]")
                        validation_loop.set_postfix_str(f"loss={data['loss'].item():.2e}")
                    self.memory_trackers.memory_trackers['validation_progress'].end()
                    self.timers.timers['validation_progress'].end()

                    self.timers.timers['validation_data'].start()
                    self.memory_trackers.memory_trackers['validation_data'].start()

                    for key, value in data.items():
                        if ("loss" in key):
                            if key not in epoch_val_losses:
                                epoch_val_losses[key] = []
                            epoch_val_losses[key].append(value.detach().cpu())
                            self.meta['tensorboard'].add_scalar(key + '_val', value, val_iteration)
                for key, value in epoch_val_losses.items():
                    self.meta['tensorboard'].add_scalar(key + '_val_avg', np.mean(value), epoch)

                # update timing info
                self.memory_trackers.memory_trackers['epoch_validation'].end()
                self.timers.timers['epoch_validation'].end()
                """
                Run through a metric loop if there are any metrics
                defined.
                """
                if not skip_metrics:
                    if self.metrics is not None:
                        if (progress_bar == 'all' or progress_bar == 'validation'):
                            metrics_validation_loop = tqdm(
                                enumerate(self.meta['loader'].validation_loader, 0),
                                total=len(self.meta['loader'].validation_loader),
                                leave=rewrite_bar,
                                position=0,
                                colour='blue'
                            )
                        else:
                            metrics_validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
                        self.metrics.reset_batch()
                        for ii, data in metrics_validation_loop:
                            # update metrics
                            self.timers.timers['validation_metrics'].start()
                            self.memory_trackers.memory_trackers['validation_metrics'].start()
                            data = self.model(data)
                            self.metrics.update(data, train_type="validation")
                            self.memory_trackers.memory_trackers['validation_metrics'].end()
                            self.timers.timers['validation_metrics'].end()
                            if (progress_bar == 'all' or progress_bar == 'validation'):
                                metrics_validation_loop.set_description(f"Validation Metrics: Epoch [{epoch+1}/{epochs}]")

                        """Get metrics and report to tensorboard"""
                        metrics = self.metrics.compute()
                        for key, value in metrics.items():
                            self.meta['tensorboard'].add_scalar(key + '_val', value, epoch)

            # evaluate callbacks
            self.timers.timers['validation_callbacks'].start()
            self.memory_trackers.memory_trackers['validation_callbacks'].start()
            self.callbacks.evaluate_epoch(train_type='validation')
            self.memory_trackers.memory_trackers['validation_callbacks'].end()
            self.timers.timers['validation_callbacks'].end()

            # save weights if at checkpoint step
            if epoch % checkpoint == 0:
                self.save_checkpoint(epoch)
            # free up gpu resources
            torch.cuda.empty_cache()
        # evaluate epoch callbacks
        self.callbacks.evaluate_training()
        self.logger.info("training finished.")
        """
        Testing stage.
        Setup the progress bar for the testing loop.
        We do not have timing information for the test
        loop stage, since it is generally quick
        and doesn't need to be optimized for any reason.
        """
        if (progress_bar == 'all' or progress_bar == 'test'):
            test_loop = tqdm(
                enumerate(self.meta['loader'].test_loader, 0),
                total=len(self.meta['loader'].test_loader),
                leave=rewrite_bar,
                position=0,
                colour='red'
            )
        else:
            test_loop = enumerate(self.meta['loader'].test_loader, 0)
        # make sure to set model to eval() during validation!
        self.model.eval()
        if self.metrics is not None:
            self.metrics.reset_batch()
        with torch.no_grad():
            test_iteration = 0
            for ii, data in test_loop:
                test_iteration += 1
                # get the network output
                data = self.model(data)

                # compute loss
                data = self.criterion.loss(data, task='test')

                # update metrics
                if self.metrics is not None:
                    self.metrics.update(data, train_type="test")

                # update progress bar
                if (progress_bar == 'all' or progress_bar == 'test'):
                    test_loop.set_description(f"Testing: Batch [{ii+1}/{self.meta['loader'].num_test_batches}]")
                    test_loop.set_postfix_str(f"loss={data['loss'].item():.2e}")

                for key, value in data.items():
                    if ("loss" in key):
                        if key not in test_losses:
                            test_losses[key] = []
                        test_losses[key].append(value.detach().cpu())
                        self.meta['tensorboard'].add_scalar(key + '_test', value, test_iteration)

            for key, value in test_losses.items():
                self.meta['tensorboard'].add_scalar(key + '_test_avg', torch.mean(torch.tensor(value)), epoch)

            """Get metrics and report to tensorboard"""
            if self.metrics is not None:
                metrics = self.metrics.compute()
                for key, value in metrics.items():
                    self.meta['tensorboard'].add_scalar(key + '_test', value, epoch)

            # evaluate callbacks
            self.callbacks.evaluate_epoch(train_type='test')
        self.callbacks.evaluate_testing()
        # save the final model
        self.model.save_model(flag='trained')

        """Generate plots"""
        self.model.set_device('cpu')
        if (progress_bar == 'all' or progress_bar == 'train'):
            training_loop = tqdm(
                enumerate(self.meta['loader'].train_loader, 0),
                total=len(self.meta['loader'].train_loader),
                leave=rewrite_bar,
                position=0,
                colour='green'
            )
        else:
            training_loop = enumerate(self.meta['loader'].train_loader, 0)
        training_data = {}
        with torch.no_grad():
            for ii, data in training_loop:
                data = self.model(data)
                if ii == 0:
                    for key, value in data.items():
                        training_data[key] = value.cpu()
                else:
                    for key, value in data.items():
                        training_data[key] = torch.cat((training_data[key], value.cpu()))

        self.meta['dataset'].evaluate_outputs(training_data, data_type='training')

        """Now for validation"""
        if (progress_bar == 'all' or progress_bar == 'validation'):
            validation_loop = tqdm(
                enumerate(self.meta['loader'].validation_loader, 0),
                total=len(self.meta['loader'].validation_loader),
                leave=rewrite_bar,
                position=0,
                colour='green'
            )
        else:
            validation_loop = enumerate(self.meta['loader'].train_loader, 0)
        validation_data = {}
        with torch.no_grad():
            for ii, data in validation_loop:
                data = self.model(data)
                if ii == 0:
                    for key, value in data.items():
                        validation_data[key] = value.cpu()
                else:
                    for key, value in data.items():
                        validation_data[key] = torch.cat((validation_data[key], value.cpu()))

        self.meta['dataset'].evaluate_outputs(validation_data, data_type='validation')

        """Then for testing"""
        if (progress_bar == 'all' or progress_bar == 'test'):
            test_loop = tqdm(
                enumerate(self.meta['loader'].test_loader, 0),
                total=len(self.meta['loader'].test_loader),
                leave=rewrite_bar,
                position=0,
                colour='green'
            )
        else:
            test_loop = enumerate(self.meta['loader'].train_loader, 0)
        test_data = {}
        with torch.no_grad():
            for ii, data in test_loop:
                data = self.model(data)
                if ii == 0:
                    for key, value in data.items():
                        test_data[key] = value.cpu()
                else:
                    for key, value in data.items():
                        test_data[key] = torch.cat((test_data[key], value.cpu()))

        self.meta['dataset'].evaluate_outputs(test_data, data_type='test')

        """Now all the datasets together"""
        total_dataset = {}
        for key in training_data.keys():
            total_dataset[key] = torch.cat((training_data[key], validation_data[key], test_data[key]))
        self.meta['dataset'].evaluate_outputs(total_dataset, data_type='all')

        # see if predictions should be saved
        if save_predictions:
            self.logger.info("running inference to save predictions.")
            return self.inference(
                dataset_type='all',
                progress_bar=progress_bar,
                rewrite_bar=rewrite_bar,
                save_predictions=True,
            )

    def inference(
        self,
        dataset_type:   str = 'all',    # which dataset to use for inference
        layers:         list = [],      # which forward views to save
        outputs:        list = [],      # which outputs to save
        save_predictions: bool = True,  # wether to save the predictions
        progress_bar:   bool = True,    # progress bar from tqdm
        rewrite_bar:    bool = True,    # wether to leave the bars after each epoch
        skip_metrics:   bool = False,   # wether to skip metrics except for testing sets
    ):
        """
        Here we just do inference on a particular part
        of the dataset_loader, either 'train', 'validation',
        'test' or 'all'.
        """
        # check that everything is on the same device
        if (self.model.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.model.device}' are different!")
        if (self.criterion.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.criterion.device}' are different!")

        # determine loader
        if dataset_type == 'train':
            inference_loader = self.meta['loader'].train_loader
            num_batches = self.meta['loader'].num_training_batches
            inference_indices = self.meta['loader'].train_indices
        elif dataset_type == 'validation':
            inference_loader = self.meta['loader'].validation_loader
            num_batches = self.meta['loader'].num_validation_batches
            inference_indices = self.meta['loader'].validation_indices
        elif dataset_type == 'test':
            inference_loader = self.meta['loader'].test_loader
            num_batches = self.meta['loader'].num_test_batches
            inference_indices = self.meta['loader'].test_indices
        else:
            inference_loader = self.meta['loader'].all_loader
            num_batches = self.meta['loader'].num_all_batches
            inference_indices = self.meta['loader'].all_indices

        """
        Set up progress bar.
        """
        if (progress_bar is True):
            inference_loop = tqdm(
                enumerate(inference_loader, 0),
                total=len(list(inference_indices)),
                leave=rewrite_bar,
                position=0,
                colour='magenta'
            )
        else:
            inference_loop = enumerate(inference_loader, 0)

        # set up array for predictions
        predictions = {
            layer: []
            for layer in layers
        }
        for output in outputs:
            predictions[output] = []

        self.logger.info(f"running inference on dataset '{self.meta['dataset'].name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            if self.metrics is not None:
                self.metrics.reset_batch()
            for ii, data in inference_loop:
                # get the network output
                data = self.model(data)
                for jj, key in enumerate(data.keys()):
                    if key in predictions.keys():
                        predictions[key].append([data[key].cpu().numpy()])
                for jj, key in enumerate(layers):
                    if key in predictions.keys():
                        predictions[key].append([self.model.forward_views[key].cpu().numpy()])
                # compute loss
                if self.criterion is not None:
                    data = self.criterion.loss(data, task='inference')

                # update metrics
                if not skip_metrics:
                    if self.metrics is not None:
                        self.metrics.update(data, train_type="inference")

                # update progress bar
                if (progress_bar is True):
                    inference_loop.set_description(f"Inference: Batch [{ii+1}/{num_batches}]")
                    inference_loop.set_postfix_str(f"loss={data['loss'].item():.2e}")
        for key in predictions.keys():
            predictions[key] = np.vstack(np.array(predictions[key], dtype=object))
        # save predictions if wanted
        if save_predictions:
            self.meta['dataset'].save_predictions(
                self.model.name + "_predictions",
                predictions,
                np.array(inference_indices, dtype=object)
            )
        self.callbacks.evaluate_inference()
        self.logger.info("returning predictions.")
        return predictions
