"""
Generic module
"""
# from corenet.analysis.model_analyzer_handler import ModelAnalyzerHandler
from corenet.models import ModelHandler
from corenet.module.generic_module import GenericModule
from corenet.losses import LossHandler
from corenet.optimizers import Optimizer
from corenet.metrics import MetricHandler
from corenet.trainer import Trainer
from corenet.utils.callbacks import CallbackHandler


class MachineLearningModule(GenericModule):
    """
    The module class helps to organize meta data and objects related to different tasks
    and execute those tasks based on a configuration file.  The spirit of the 'Module' class
    is to mimic some of the functionality of LArSoft, e.g. where you can specify a chain
    of tasks to be completed, the ability to have nested config files where default parameters
    can be overwritten.

    The ML specific module runs in several different modes,
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        mode:   str = '',
        meta:   dict = {}
    ):
        self.name = name
        super(MachineLearningModule, self).__init__(
            self.name, config, mode, meta
        )
        self.consumes = ['dataset', 'loader']
        self.produces = ['predictions']

    def parse_config(self):
        """
        """
        self.check_config()

        self.meta['model'] = None
        self.meta['criterion'] = None
        self.meta['optimizer'] = None
        self.meta['metrics'] = None
        self.meta['callbacks'] = None
        self.meta['trainer'] = None
        self.meta['model_analyzer'] = None

    def check_config(self):
        if "model" not in self.config.keys():
            self.logger.warning('"model" section not specified in config!')

        if (
            self.mode == "training"
        ):
            if "criterion" not in self.config.keys():
                self.logger.error('"criterion" section not specified in config!')
            if "optimizer" not in self.config.keys():
                self.logger.error('"optimizer" section not specified in config!')
            if "metrics" not in self.config.keys():
                self.logger.warn('"metrics" section not specified in config!')
            if "callbacks" not in self.config.keys():
                self.logger.warn('"callbacks" section not specified in config!')
            if "training" not in self.config.keys():
                self.logger.error('"training" section not specified in config!')

        if self.mode == "inference":
            if "inference" not in self.config.keys():
                self.logger.error('"inference" section not specified in config!')

    def parse_model(
        self,
    ):
        """
        """
        if "model" not in self.config.keys():
            self.logger.warn("no model in config file.")
            return
        self.logger.info("configuring model.")
        model_config = self.config["model"]
        self.meta['model'] = ModelHandler(
            self.name,
            model_config,
            meta=self.meta
        )

    def parse_loss(
        self,
    ):
        """
        """
        if "criterion" not in self.config.keys():
            self.logger.warn("no criterion in config file.")
            return
        self.logger.info("configuring criterion.")
        criterion_config = self.config['criterion']
        # add in class weight numbers for loss functions
        self.meta['criterion'] = LossHandler(
            self.name,
            criterion_config,
            meta=self.meta
        )

    def parse_optimizer(
        self,
    ):
        """
        """
        if self.mode == "linear_evaluation":
            return
        if "optimizer" not in self.config.keys():
            self.logger.warn("no optimizer in config file.")
            return
        self.logger.info("configuring optimizer.")
        optimizer_config = self.config['optimizer']
        self.meta['optimizer'] = Optimizer(
            self.name,
            optimizer_config,
            meta=self.meta,
        )

    def parse_metrics(
        self,
    ):
        """
        """
        if "metrics" not in self.config.keys():
            self.logger.warn("no metrics in config file.")
            return
        self.logger.info("configuring metrics.")
        metrics_config = self.config['metrics']
        self.meta['metrics'] = MetricHandler(
            self.name,
            metrics_config,
            meta=self.meta
        )

    def parse_callbacks(
        self,
    ):
        """
        """
        if "callbacks" not in self.config.keys():
            self.logger.warn("no callbacks in config file.")
            return
        self.logger.info("configuring callbacks.")
        callbacks_config = self.config['callbacks']
        if callbacks_config is None:
            self.logger.warn("no callbacks specified.")
        else:
            for callback in callbacks_config.keys():
                if callbacks_config[callback] is None:
                    callbacks_config[callback] = {}
                callbacks_config[callback]['criterion_handler'] = self.meta['criterion']
                callbacks_config[callback]['metrics_handler'] = self.meta['metrics']
        self.meta['callbacks'] = CallbackHandler(
            self.name,
            callbacks_config,
            meta=self.meta
        )

    def parse_training(
        self,
    ):
        """
        """
        if "training" not in self.config.keys():
            self.logger.warn("no training in config file.")
            return
        self.logger.info("configuring training.")
        training_config = self.config['training']
        self.meta['trainer'] = Trainer(
            self.name,
            training_config,
            meta=self.meta,
        )

    def parse_inference(
        self,
    ):
        if "inference" not in self.config.keys():
            self.logger.warn("no inference in config file.")
            return
        self.logger.info("configuring inference.")
        inference_config = self.config['inference']
        self.meta['trainer'] = Trainer(
            self.name,
            inference_config,
            meta=self.meta
        )

        if "layers" in self.config["inference"].keys():
            for layer in self.config["inference"]["layers"]:
                if layer not in self.meta['model'].forward_views().keys():
                    self.logger.error(
                        f"layer '{layer}' not in the model forward views!" +
                        f" possible views: {self.meta['model'].forward_views().keys()}"
                    )
                self.module_data_product[layer] = None
        if "outputs" in self.config["inference"].keys():
            for output in self.config["inference"]["outputs"]:
                self.module_data_product[output] = None

    def run_module(self):
        if self.mode == 'training':
            self.run_training()
        elif self.mode == 'inference':
            self.run_inference()
        else:
            self.logger.warning(f"current mode {self.mode} not an available type!")

    def run_training(self):
        self.parse_model()
        self.parse_loss()
        self.parse_optimizer()
        self.parse_metrics()
        self.parse_callbacks()
        self.parse_training()
        self.module_data_product['predictions'] = self.meta['trainer'].train(
            epochs=self.config['training']['epochs'],
            checkpoint=self.config['training']['checkpoint'],
            progress_bar=self.config['training']['progress_bar'],
            rewrite_bar=self.config['training']['rewrite_bar'],
            save_predictions=self.config['training']['save_predictions'],
            prediction_outputs=self.config['training']['prediction_outputs'],
            skip_metrics=self.config['training']['skip_metrics']
        )
        if self.meta['model_analyzer'] is not None:
            self.meta['model_analyzer'].analyze(self.meta['model'].model)

    def run_inference(self):
        self.parse_model()
        self.parse_inference()
        self.module_data_product['predictions'] = self.meta['trainer'].inference(
            dataset_type=self.config['inference']['dataset_type'],
            layers=self.config['inference']['layers'],
            progress_bar=self.config['inference']['progress_bar'],
            rewrite_bar=self.config['inference']['rewrite_bar'],
            save_predictions=self.config['inference']['save_predictions'],
            prediction_outputs=self.config['inference']['prediction_outputs']
        )
