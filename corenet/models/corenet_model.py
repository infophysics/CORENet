"""
Implementation of the corenet model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from corenet.models.common import activations, normalizations
from corenet.models import GenericModel

corenet_config = {
    # dimension of the input variables
    'input_dimension':      5,
    # encoder parameters
    'encoder_dimensions':   [10, 25, 50, 25, 10],
    'encoder_activation':   'leaky_relu',
    'encoder_activation_params':    {'negative_slope': 0.02},
    'encoder_normalization': 'bias',
    # desired dimension of the latent space
    'latent_dimension':     5,
    # decoder parameters
    'decoder_dimensions':   [10, 25, 50, 25, 10],
    'decoder_activation':   'leaky_relu',
    'decoder_activation_params':    {'negative_slope': 0.02},
    'decoder_normalization': 'bias',
    # output activation
    'output_activation':    'linear',
    'output_activation_params':     {},
}


class CORENet(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str,
        config: dict = corenet_config,
        meta:   dict = {}
    ):
        super(CORENet, self).__init__(
            name, config, meta
        )
        self.config = config
        # check config
        self.logger.info(f"checking corenet architecture using config: {self.config}")
        for item in corenet_config.keys():
            if item not in self.config:
                self.logger.error(f"parameter {item} was not specified in config file {self.config}")
                raise AttributeError(f"parameter {item} was not specified in config file {self.config}")
        if self.config['encoder_activation'] not in activations:
            self.logger.error(f"Specified activation {self.config['encoder_activation']} is not an allowed type.")
            raise AttributeError(f"Specified activation {self.config['encoder_activation']} is not an allowed type.")
        if self.config['encoder_normalization'] not in normalizations:
            self.logger.error(f"Specified normalization {self.config['encoder_normalization']} is not an allowed type.")
            raise AttributeError(f"Specified normalization {self.config['encoder_normalization']} is not an allowed type.")
        if self.config['decoder_activation'] not in activations:
            self.logger.error(f"Specified activation {self.config['decoder_activation']} is not an allowed type.")
            raise AttributeError(f"Specified activation {self.config['decoder_activation']} is not an allowed type.")
        if self.config['decoder_normalization'] not in normalizations:
            self.logger.error(f"Specified normalization {self.config['decoder_normalization']} is not an allowed type.")
            raise AttributeError(f"Specified normalization {self.config['decoder_normalization']} is not an allowed type.")
        if self.config['output_activation'] != 'linear':
            if self.config['output_activation'] not in activations:
                self.logger.error(f"Specified activation {self.config['output_activation']} is not an allowed type.")
                raise AttributeError(f"Specified activation {self.config['output_activation']} is not an allowed type.")
        # construct the model
        self.forward_views = {}
        self.forward_view_map = {}
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build corenet architecture using config: {self.config}")
        _encoder_dict = OrderedDict()
        _latent_dict = OrderedDict()
        _decoder_dict = OrderedDict()
        _output_dict = OrderedDict()
        _core_dict = OrderedDict()

        self.input_dimension = self.config['input_dimension']
        input_dimension = self.input_dimension
        # iterate over the encoder
        for ii, dimension in enumerate(self.config['encoder_dimensions']):
            if self.config['encoder_normalization'] == 'bias':
                _encoder_dict[f'encoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _encoder_dict[f'encoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _encoder_dict[f'encoder_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _encoder_dict[f'encoder_{ii}_activation'] = activations[
                self.config['encoder_activation']
            ](**self.config['encoder_activation_params'])
            input_dimension = dimension

        # create the latent space
        _latent_dict['latent_layer'] = nn.Linear(
            in_features=dimension,
            out_features=self.config['latent_dimension'],
            bias=False
        )

        input_dimension = self.config['latent_dimension']
        # iterate over the decoder
        for ii, dimension in enumerate(self.config['decoder_dimensions']):
            if self.config['decoder_normalization'] == 'bias':
                _decoder_dict[f'decoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _decoder_dict[f'decoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _decoder_dict[f'decoder_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _decoder_dict[f'decoder_{ii}_activation'] = activations[
                self.config['decoder_activation']
            ](**self.config['decoder_activation_params'])
            input_dimension = dimension
        # create the output
        _output_dict['output'] = nn.Linear(
            in_features=dimension,
            out_features=self.input_dimension,
            bias=False
        )
        if self.config['output_activation'] != 'linear':
            _output_dict['output_activation'] = activations[
                self.config['output_activation']
            ](**self.config['output_activation_params'])

        # create the core dictionary
        input_dimension = self.config['core_input_dimension']
        for ii, dimension in enumerate(self.config['core_dimensions']):
            if self.config['core_normalization'] == 'bias':
                _core_dict[f'core_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _core_dict[f'core_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _core_dict[f'core_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _core_dict[f'core_{ii}_activation'] = activations[
                self.config['core_activation']
            ](**self.config['core_activation_params'])
            input_dimension = dimension
        # create the dictionaries
        self.encoder_dict = nn.ModuleDict(_encoder_dict)
        self.latent_dict = nn.ModuleDict(_latent_dict)
        self.decoder_dict = nn.ModuleDict(_decoder_dict)
        self.output_dict = nn.ModuleDict(_output_dict)
        self.core_dict = nn.ModuleDict(_core_dict)
        # record the info
        self.logger.info(
            f"constructed corenet with dictionaries:\n{self.encoder_dict}"
            + f"\n{self.latent_dict}\n{self.decoder_dict}\n{self.output_dict}."
        )

    def forward(
        self,
        data
    ):
        """
        Iterate over the model dictionary
        """
        gut_test = data['gut_test'].to(self.device)
        gut_true = data['gut_true'].to(self.device)
        weak_test = data['weak_test'].to(self.device)
        # first the gut test encoder
        for layer in self.encoder_dict.values():
            gut_test = layer(gut_test)
            gut_true = layer(gut_true)
        # then apply the weak layer
        for layer in self.core_dict.values():
            weak_test = layer(weak_test)
        # apply the latent layers
        for layer in self.latent_dict.values():
            gut_test = layer(gut_test)
            gut_true = layer(gut_true)
            weak_test = layer(weak_test)
        data['gut_test_latent'] = gut_test
        data['gut_true_latent'] = gut_true
        data['weak_test_latent'] = weak_test
        # now get the gut outputs
        for layer in self.decoder_dict.values():
            gut_test = layer(gut_test)
            gut_true = layer(gut_true)
            weak_test = layer(weak_test)
        # apply outputs
        for layer in self.output_dict.values():
            gut_test = layer(gut_test)
            gut_true = layer(gut_true)
            weak_test = layer(weak_test)
        data['gut_test_output'] = gut_test
        data['gut_true_output'] = gut_true
        data['weak_test_output'] = weak_test

        return data

    def sample(
        self,
        x
    ):
        """
        Returns an output given a input from the latent space
        """
        x = x.to(self.device)
        for layer in self.decoder_dict.keys():
            x = self.decoder_dict[layer](x)
        for layer in self.output_dict.keys():
            x = self.output_dict[layer](x)
        return x

    def latent(
        self,
        x,
    ):
        """
        Get the latent representation of an input
        """
        x = x[0].to(self.device)
        # first the encoder
        for layer in self.encoder_dict.keys():
            x = self.encoder_dict[layer](x)
        x = self.latent_dict['latent_layer'](x)
        return x
