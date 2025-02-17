"""
Implementation of the corenet model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torchviz import make_dot
from PIL import Image
import io

from corenet.models.common import activations, normalizations
from corenet.models import GenericModel

corenet_config = {
    # dimension of the input variables
    'input_dimension':      5,
    # encoder parameters
    'encoder_dimensions':   [25, 100, 25],
    'encoder_activation':   'leaky_relu',
    'encoder_activation_params':    {'negative_slope': 0.02},
    # desired dimension of the latent space
    'latent_dimension':     5,
    # decoder parameters
    'decoder_dimensions':   [25, 100, 25],
    'decoder_activation':   'leaky_relu',
    'decoder_activation_params':    {'negative_slope': 0.02},
    # output activation
    'output_activation':    'linear',
    'output_activation_params':     {},
}


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        activation_params,
        dropout: float = 0.3
    ):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(
            in_features,
            out_features,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_features)
        self.activation = activations[
            activation
        ](**activation_params)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            out_features,
            out_features,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_features)

        # Adjust dimension if in_features != out_features
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(
                    in_features,
                    out_features,
                    bias=False
                ),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += residual  # Skip connection
        out = self.activation(out)
        return out


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
        if 'use_weak_values' not in self.config.keys():
            self.config['use_weak_values'] = False
        if 'chuncc' not in self.config.keys():
            self.config['chuncc'] = False

        self.input_dimension = self.config['input_dimension']

        self.construct_encoder()
        self.construct_latent_space()
        self.construct_core()
        self.construct_decoder()
        self.construct_output()

        self.save_model_architecture()

        # record the info
        self.logger.info("constructed corenet.")

    def construct_encoder(self):
        """
        Construction of encoder layer. This takes GUT -> latent space
        """
        _encoder_dict = OrderedDict()

        input_dimension = self.input_dimension

        # iterate over the encoder
        for ii, dimension in enumerate(self.config['encoder_dimensions']):
            _encoder_dict[f'encoder_{ii}'] = ResidualBlock(
                in_features=input_dimension,
                out_features=dimension,
                activation=self.config['encoder_activation'],
                activation_params=self.config['encoder_activation_params'],
                dropout=self.config['encoder_dropout']
            )
            input_dimension = dimension
        self.encoder_dict = nn.ModuleDict(_encoder_dict)

    def construct_latent_space(self):
        """
        Construction of additional layers for projection to the latent space.
        """
        _latent_dict = OrderedDict()
        dimension = self.config["encoder_dimensions"][-1]

        # create the latent space
        _latent_dict['latent_layer'] = nn.Linear(
            in_features=dimension,
            out_features=self.config['latent_dimension'],
            bias=False
        )
        _latent_dict['latent_norm'] = nn.BatchNorm1d(
            num_features=self.config['latent_dimension']
        )
        self.latent_dict = nn.ModuleDict(_latent_dict)

    def construct_decoder(self):
        """
        Construction of the decoder which goes from latent -> GUT.
        """
        _decoder_dict = OrderedDict()
        input_dimension = self.config['latent_dimension']

        # iterate over the decoder
        for ii, dimension in enumerate(self.config['decoder_dimensions']):
            _decoder_dict[f'decoder_{ii}'] = ResidualBlock(
                in_features=input_dimension,
                out_features=dimension,
                activation=self.config['decoder_activation'],
                activation_params=self.config['decoder_activation_params'],
                dropout=self.config['decoder_dropout']
            )
            input_dimension = dimension
        self.decoder_dict = nn.ModuleDict(_decoder_dict)

    def construct_output(self):
        """
        Construction of final output projection from decoder
        """
        _output_dict = OrderedDict()
        dimension = self.config["decoder_dimensions"][-1]
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
        self.output_dict = nn.ModuleDict(_output_dict)

    def construct_core(self):
        """
        Construction of CORE network which takes weak + latent -> latent.
        """
        _core_dict = OrderedDict()
        # create the core dictionary
        input_dimension = self.config['core_input_dimension']
        for ii, dimension in enumerate(self.config['core_dimensions']):
            _core_dict[f'core_{ii}'] = ResidualBlock(
                in_features=input_dimension,
                out_features=dimension,
                activation=self.config['core_activation'],
                activation_params=self.config['core_activation_params'],
                dropout=self.config['core_dropout']
            )
            input_dimension = dimension

        _core_dict['latent_layer'] = nn.Linear(
            in_features=dimension,
            out_features=self.config['latent_dimension'],
            bias=False
        )
        self.core_dict = nn.ModuleDict(_core_dict)

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
        gut_true_latent_copy = gut_true_latent.clone()

        """Determine whether we want to mix the gradients"""
        if not self.config['mix_gradients']:
            gut_test_latent_copy = gut_test_latent_copy.detach()
            gut_true_latent_copy = gut_true_latent_copy.detach()

        """Form weak input"""
        if self.config["use_weak_values"]:
            weak_test_latent = torch.cat(
                (data['weak_test'].to(self.device), gut_test_latent_copy),
                dim=1
            )
            weak_true_latent = torch.cat(
                (data['weak_test'].to(self.device), gut_true_latent_copy),
                dim=1
            )
        else:
            weak_test_latent = gut_test_latent_copy
            weak_true_latent = gut_true_latent_copy

        """Apply the CORENet to the weak input"""
        for layer in self.core_dict.values():
            weak_test_latent = layer(weak_test_latent)
            weak_true_latent = layer(weak_true_latent)

        """Save the latent values"""
        data['gut_test_latent'] = gut_test_latent
        data['gut_true_latent'] = gut_true_latent
        data['weak_test_latent'] = weak_test_latent
        data['weak_true_latent'] = weak_true_latent

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

    def freeze_swae(self):
        """
        Freezes the weights of the SWAE (encoder, latent space, and decoder)
        while keeping CORENet (core_dict) trainable.
        """
        for param in self.encoder_dict.parameters():
            param.requires_grad = False
        for param in self.latent_dict.parameters():
            param.requires_grad = False
        for param in self.decoder_dict.parameters():
            param.requires_grad = False
        for param in self.output_dict.parameters():
            param.requires_grad = False

    def unfreeze_swae(self):
        for param in self.encoder_dict.parameters():
            param.requires_grad = True
        for param in self.latent_dict.parameters():
            param.requires_grad = True
        for param in self.decoder_dict.parameters():
            param.requires_grad = True
        for param in self.output_dict.parameters():
            param.requires_grad = True

    def freeze_core(self):
        for param in self.core_dict.parameters():
            param.requires_grad = False

    def unfreeze_core(self):
        for param in self.core_dict.parameters():
            param.requires_grad = True

    def save_model_architecture(self):
        """Create computation graph with torchviz"""
        x = {
            'gut_test': torch.randn(10, self.input_dimension).to(self.device),
            'gut_true': torch.randn(10, self.input_dimension).to(self.device),
            'weak_test': torch.randn(
                10, self.config['core_input_dimension'] - self.config['latent_dimension']
            ).to(self.device)
        }
        self.to(self.device)
        output = self.forward(x)
        combined_output = torch.cat([v.flatten() for v in output.values()])
        dot = make_dot(
            combined_output,
            params=dict(self.named_parameters()),
        )

        # Save the graph
        dot.format = 'png'
        dot.render(f'{self.meta["run_directory"]}/model_graph')
        png_bytes = dot.pipe(format='png')
        image = Image.open(io.BytesIO(png_bytes))

        # Convert to tensorboard-friendly format
        image_tensor = torch.tensor(
            np.array(image)
        ).permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize to [0,1]
        self.meta["tensorboard"].add_image('Model Graph', image_tensor[0])
