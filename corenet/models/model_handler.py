"""
Container for models
"""
import os
import torch
import importlib.util
import sys
import inspect

from corenet.utils.logger import Logger
from corenet.models import GenericModel


class ModelHandler:
    """
    """
    def __init__(
        self,
        name:   str,
        config:  dict = {},
        models:  list = [],
        use_sample_weights: bool = False,
        meta:   dict = {}
    ):
        self.name = name + "_model_handler"
        self.use_sample_weights = use_sample_weights
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(self.name)
        else:
            self.logger = Logger(self.name)

        self.model_type = None
        self.single_model_name = ''
        self.models = {}

        if bool(config) and len(models) != 0:
            self.logger.error(
                "handler received both a config and a list of models! " +
                "The user should only provide one or the other!")
        elif bool(config):
            self.set_config(config)
        else:
            if len(models) == 0:
                self.logger.error("handler received neither a config or models!")
            self.models = {
                model.name: model
                for model in models
            }
            if len(models) == 1:
                self.model = list(self.models.values())[0]

    def set_config(self, config):
        self.config = config
        self.process_config()

    def collect_models(self):
        self.available_models = {}
        self.model_files = [
            os.path.dirname(__file__) + '/' + file
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        self.model_files.extend(self.meta['local_corenet_files'])
        for model_file in self.model_files:
            if (
                ("__init__.py" in model_file) or
                ("__pycache__.py" in model_file) or
                ("generic_model.py" in model_file) or
                ("__pycache__" in model_file) or
                (".py" not in model_file)
            ):
                continue
            try:
                self.load_model(model_file)
            except Exception:
                self.logger.warn(f'problem loading model from file: {model_file}')

    def load_model(
        self,
        model_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{model_file.removesuffix(".py")}.name',
            model_file
        )
        custom_model_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{model_file.removesuffix(".py")}.name'] = custom_model_file
        spec.loader.exec_module(custom_model_file)
        for name, obj in inspect.getmembers(sys.modules[f'{model_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_model_file, name)
                if issubclass(custom_class, GenericModel):
                    self.available_models[name] = custom_class

    def load_model_from_config(self):
        model_type = self.config["model_type"]
        model_params = self.config["load_model"]
        checkpoint = torch.load(model_params)
        model_config = checkpoint['model_config']
        if model_type not in self.available_models.keys():
            self.logger.error(
                f"specified model '{model_type}' is not an available type! " +
                f"Available types:\n{self.available_models.keys()}"
            )
        self.model = self.available_models[model_type](
            model_type, model_config, self.meta
        )
        self.model.load_model(checkpoint)
        self.logger.info(f'added model "{model_type}" to ModelHandler.')

    def process_config(self):
        # list of available criterions
        self.collect_models()
        # check config
        if "load_model" in self.config.keys():
            return self.load_model_from_config()
        # process models
        for item in self.config.keys():
            # check that model exists
            if item not in self.available_models.keys():
                self.logger.error(
                    f"specified model '{item}' is not an available type! " +
                    f"Available types:\n{self.available_models.keys()}"
                )
            self.model = self.available_models[item](
                item, self.config[item], self.meta
            )
            self.logger.info(f'added model "{item}" to ModelHandler.')

    def set_device(
        self,
        device
    ):
        self.logger.info(f'setting device to "{device}".')
        self.model.set_device(device)
        self.device = device

    def add_model(
        self,
        model:   GenericModel
    ):
        if issubclass(type(model), GenericModel):
            self.logger.info(f'added model function "{model}" to ModelHandler.')
            self.models[model.name] = model
        else:
            self.logger.error(
                'specified model {model} is not a child of "GenericModel"!' +
                ' only models which inherit from GenericModel can' +
                ' be used by the ModelHandler in corenet.'
            )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def forward_views(self):
        return self.model.forward_views()
