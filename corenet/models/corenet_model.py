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
        if 'mix_gradients' not in self.config.keys():
            self.config['mix_gradients'] = False
        if 'chuncc' not in self.config.keys():
            self.config['chuncc'] = False

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
        _latent_dict['latent_norm'] = nn.BatchNorm1d(
            num_features=self.config['latent_dimension']
        )
        _latent_dict['latent_binary'] = nn.Linear(
            in_features=dimension,
            out_features=1,
            bias=False
        )
        _latent_dict['latent_binary_activation'] = activations['sigmoid']()

        input_dimension = self.config['latent_dimension']
        if self.config['chuncc']:
            input_dimension += 1

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

        ""
        _core_dict['latent_layer'] = nn.Linear(
            in_features=dimension,
            out_features=self.config['latent_dimension'],
            bias=False
        )
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

        """Apply encoder to gut test and gut true"""
        for layer in self.encoder_dict.values():
            gut_test = layer(gut_test)
            gut_true = layer(gut_true)

        """Apply the latent layers"""
        gut_test_latent = self.latent_dict['latent_layer'](gut_test)
        gut_test_latent = self.latent_dict['latent_norm'](gut_test_latent)
        gut_true_latent = self.latent_dict['latent_layer'](gut_true)
        gut_true_latent = self.latent_dict['latent_norm'](gut_true_latent)

        """Grab a copy of the latent output for gut_test"""
        gut_test_latent_copy = gut_test_latent.clone()

        """Determine whether we want to mix the gradients"""
        if not self.config['mix_gradients']:
            gut_test_latent_copy = gut_test_latent_copy.detach()

        """Form weak input"""
        weak_test_latent = torch.cat(
            (data['weak_test'].to(self.device), gut_test_latent_copy),
            dim=1
        )

        """Apply the CORENet to the weak input"""
        for layer in self.core_dict.values():
            weak_test_latent = layer(weak_test_latent)

        """Apply the latent binary"""
        gut_test_binary = self.latent_dict['latent_binary'](gut_test)
        gut_test_binary = self.latent_dict['latent_binary_activation'](gut_test_binary)
        gut_true_binary = self.latent_dict['latent_binary'](gut_true)
        gut_true_binary = self.latent_dict['latent_binary_activation'](gut_true_binary)
        weak_test_binary = torch.all(gut_true == gut_true, dim=1).int().unsqueeze(1)

        """Save the latent values"""
        data['gut_test_latent'] = gut_test_latent
        data['gut_true_latent'] = gut_true_latent
        data['weak_test_latent'] = weak_test_latent
        data['gut_test_binary'] = gut_test_binary
        data['gut_true_binary'] = gut_true_binary
        data['weak_test_binary'] = weak_test_binary

        if self.config['chuncc']:
            gut_test_latent = torch.cat((gut_test_latent, gut_test_binary), dim=1)
            gut_true_latent = torch.cat((gut_true_latent, gut_true_binary), dim=1)
            weak_test_latent = torch.cat((weak_test_latent, weak_test_binary), dim=1)

        gut_test = gut_test_latent
        gut_true = gut_true_latent
        weak_test = weak_test_latent

        """Pass values through the decoder"""
        for layer in self.decoder_dict.values():
            gut_test = layer(gut_test)
            gut_true = layer(gut_true)
            weak_test = layer(weak_test)

        """Apply the output layers"""
        for layer in self.output_dict.values():
            gut_test = layer(gut_test)
            gut_true = layer(gut_true)
            weak_test = layer(weak_test)

        """Save output values"""
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
