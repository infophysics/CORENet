"""
Class for a generic model trainer.
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from corenet.utils.logger import Logger
from corenet.models import ModelHandler


class Trainer:
    """
    This class is an attempt to reduce code rewriting by putting together
    a set of functions that do everything that we could need with
    respect to training.
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
        if "patience" not in self.config.keys():
            self.config["patience"] = 100
        self.patience = self.config["patience"]
        self.process_model()
        self.process_directories()
        self.process_criterion()
        self.process_optimizer()
        self.process_metrics()

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
        self.criterion = self.meta['criterion']
        if self.criterion is not None:
            self.criterion.set_device(self.device)

    def process_optimizer(self):
        self.optimizer = self.meta['optimizer']

    def process_metrics(self):
        self.metrics = self.meta['metrics']
        if self.metrics is not None:
            self.metrics.set_device(self.device)

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
        progress_bar:   bool = True,    # progress bar from tqdm
        rewrite_bar:    bool = False,   # wether to leave the bars after each epoch
        save_predictions: bool = True,  # wether to save network outputs for all events to original file
        prediction_outputs: list = [],
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
        if not skip_metrics and self.metrics is None:
            self.logger.error('skip_metrics set to false in config, but no metrics are specified!')
        # Training
        self.logger.info(f"training dataset '{self.meta['dataset'].name}' for {epochs} epochs.")

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
        torch.autograd.set_detect_anomaly(True)
        early_stopping_loss = 10e9
        patience = 0
        # iterate over epochs
        for epoch in range(epochs):
            epoch_train_losses = {}
            epoch_val_losses = {}
            """
            Training stage.
            Setup the progress bar for the training loop.
            """
            if progress_bar:
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
            for ii, data in training_loop:
                train_iteration += 1
                self.meta['optimizer'].zero_grad()
                data = self.model(data)
                data = self.criterion.loss(data)
                if self.grad_norm:
                    data = self.criterion.grad_norm_loss(data)
                    data['loss'].backward(retain_graph=True)
                    self.criterion.update_task_weights(
                        data['grad_norm_loss'],
                        self.meta['optimizer']
                    )
                else:
                    data['loss'].backward()
                self.optimizer.step()
                if progress_bar:
                    training_loop.set_description(f"Training: Epoch [{epoch+1}/{epochs}]")
                    training_loop.set_postfix_str(f"loss={data['loss'].item():.2e}")
                for key, value in data.items():
                    if ("loss" in key):
                        if key not in epoch_train_losses:
                            epoch_train_losses[key] = []
                        epoch_train_losses[key].append(value.detach().cpu())
                        self.meta['tensorboard'].add_scalar(key + '_train', value, train_iteration)
            for key, value in epoch_train_losses.items():
                self.meta['tensorboard'].add_scalar(key + '_train_avg', np.mean(value), epoch)
            if not skip_metrics:
                self.model.eval()
                with torch.no_grad():
                    if self.metrics is not None:
                        if progress_bar:
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
                            data = self.model(data)
                            self.metrics.update(data, train_type="train")
                            if progress_bar:
                                metrics_training_loop.set_description(f"Training Metrics: Epoch [{epoch+1}/{epochs}]")
                        data = self.metrics.compute(data)
                        for key, value in data.items():
                            if ("metric" in key):
                                self.meta['tensorboard'].add_scalar(key + '_train', value, epoch)
            """
            Validation stage.
            Setup the progress bar for the validation loop.
            """
            if progress_bar:
                validation_loop = tqdm(
                    enumerate(self.meta['loader'].validation_loader, 0),
                    total=len(self.meta['loader'].validation_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='blue'
                )
            else:
                validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
            self.model.eval()
            with torch.no_grad():
                for ii, data in validation_loop:
                    val_iteration += 1
                    data = self.model(data)
                    data = self.criterion.loss(data)
                    if progress_bar:
                        validation_loop.set_description(f"Validation: Epoch [{epoch+1}/{epochs}]")
                        validation_loop.set_postfix_str(f"loss={data['loss'].item():.2e}; patience={patience}/{self.patience}")
                    for key, value in data.items():
                        if ("loss" in key):
                            if key not in epoch_val_losses:
                                epoch_val_losses[key] = []
                            epoch_val_losses[key].append(value.detach().cpu())
                            self.meta['tensorboard'].add_scalar(key + '_val', value, val_iteration)
                for key, value in epoch_val_losses.items():
                    self.meta['tensorboard'].add_scalar(key + '_val_avg', np.mean(value), epoch)
                if not skip_metrics:
                    if self.metrics is not None:
                        if progress_bar:
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
                            data = self.model(data)
                            self.metrics.update(data, train_type="validation")
                            if progress_bar:
                                metrics_validation_loop.set_description(f"Validation Metrics: Epoch [{epoch+1}/{epochs}]")
                        """Get metrics and report to tensorboard"""
                        data = self.metrics.compute(data)
                        for key, value in data.items():
                            if ("metric" in key):
                                self.meta['tensorboard'].add_scalar(key + '_val', value, epoch)
            if epoch % checkpoint == 0:
                self.save_checkpoint(epoch)
            torch.cuda.empty_cache()

            """Check early stopping conditions"""
            if np.mean(epoch_val_losses['loss']) > early_stopping_loss:
                patience += 1
            else:
                patience = 0
                early_stopping_loss = np.mean(epoch_val_losses['loss'])
            if patience > self.patience:
                break
        self.logger.info("training finished.")
        """
        Testing stage.
        """
        if progress_bar:
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
                data = self.model(data)
                data = self.criterion.loss(data)
                if self.metrics is not None:
                    self.metrics.update(data, train_type="test")
                if progress_bar:
                    test_loop.set_description(f"Testing: Batch [{ii+1}/{self.meta['loader'].num_test_batches}]")
                    test_loop.set_postfix_str(f"loss={data['loss'].item():.2e}")
                for key, value in data.items():
                    if ("loss" in key):
                        if key not in test_losses:
                            test_losses[key] = []
                        test_losses[key].append(value.detach().cpu())
                        self.meta['tensorboard'].add_scalar(key + '_test', value, test_iteration)
            for key, value in test_losses.items():
                self.meta['tensorboard'].add_scalar(key + '_test_avg', torch.mean(torch.tensor(value)), 0)

            """Get metrics and report to tensorboard"""
            if self.metrics is not None:
                data = self.metrics.compute(data)
                for key, value in data.items():
                    if ("metric" in key):
                        self.meta['tensorboard'].add_scalar(key + '_test', value, 0)
        # save the final model
        self.model.save_model(flag='trained')

        """Run post training inference"""
        self.post_training_inference(
            progress_bar=progress_bar,
            rewrite_bar=rewrite_bar,
        )

        # see if predictions should be saved
        if save_predictions:
            self.logger.info("running inference to save predictions.")
            return self.inference(
                dataset_type='all',
                progress_bar=progress_bar,
                rewrite_bar=rewrite_bar,
                save_predictions=True,
                prediction_outputs=prediction_outputs,
            )

    def post_training_inference(
        self,
        progress_bar: bool = True,
        rewrite_bar: bool = True
    ):
        """Generate plots and run through mapper"""
        self.model.set_device('cpu')
        if progress_bar:
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

                if progress_bar:
                    training_loop.set_description(
                        f"Inference (training): Batch [{ii+1}/{self.meta['loader'].num_train_batches}]"
                    )
        self.meta['dataset'].evaluate_outputs(training_data, data_type='training')

        """Now for validation"""
        if progress_bar:
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
                if progress_bar:
                    validation_loop.set_description(
                        f"Inference (validation): Batch [{ii+1}/{self.meta['loader'].num_validation_batches}]"
                    )
        self.meta['dataset'].evaluate_outputs(validation_data, data_type='validation')

        """Then for testing"""
        if progress_bar:
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
                if progress_bar:
                    test_loop.set_description(
                        f"Inference (test): Batch [{ii+1}/{self.meta['loader'].num_test_batches}]"
                    )
        self.meta['dataset'].evaluate_outputs(test_data, data_type='test')

        """Now all the datasets together"""
        total_dataset = {}
        for key in training_data.keys():
            total_dataset[key] = torch.cat((training_data[key], validation_data[key], test_data[key]))
        self.meta['dataset'].evaluate_outputs(total_dataset, data_type='all')

    def inference(
        self,
        dataset_type:   str = 'all',    # which dataset to use for inference
        layers:         list = [],      # which forward views to save
        prediction_outputs: list = [],      # which prediction_outputs to save
        save_predictions: bool = True,  # wether to save the predictions
        progress_bar:   bool = True,    # progress bar from tqdm
        rewrite_bar:    bool = True,    # wether to leave the bars after each epoch
    ):
        """
        Here we just do inference on a particular part
        of the dataset_loader, either 'train', 'validation',
        'test' or 'all'.
        """
        # check that everything is on the same device
        self.model.set_device(self.device)

        # determine loader
        if dataset_type == 'train':
            inference_loader = self.meta['loader'].train_loader
            num_batches = self.meta['loader'].num_train_batches
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
        for output in prediction_outputs:
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
                    data = self.criterion.loss(data)

                # update metrics
                if self.metrics is not None:
                    self.metrics.update(data, train_type="inference")

                # update progress bar
                if (progress_bar is True):
                    inference_loop.set_description(f"Inference: Batch [{ii+1}/{num_batches}]")

            if self.metrics is not None:
                data = self.metrics.compute(data)
                for key, value in data.items():
                    if ("metric" in key):
                        self.meta['tensorboard'].add_scalar(key + '_inference', value, 0)

        for key in predictions.keys():
            predictions[key] = np.concatenate(np.array(predictions[key]), axis=1).squeeze()
        # save predictions if wanted
        if save_predictions:
            self.meta['dataset'].save_predictions(
                self.model.name + "_predictions",
                predictions,
                np.array(inference_indices, dtype=object)
            )
        self.logger.info("returning predictions.")
        return predictions
