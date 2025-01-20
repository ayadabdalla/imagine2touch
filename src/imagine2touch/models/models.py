import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from src.imagine2touch.utils.utils import NotAdaptedError


class simpleMLP(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        hidden_dims=[64, 64],
        activation_fn=nn.Tanh,
        output_activation=None,
    ):
        super(simpleMLP, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        layer_dims = [n_input] + hidden_dims + [n_output]
        layers = []

        for d in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[d], layer_dims[d + 1]))
            if d < len(layer_dims) - 2:
                layers.append(activation_fn())

        if output_activation is not None:
            layers.append(output_activation())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class vanilla_model(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        feature_dim=20,
        feat_hidden=[64, 64],
        activation_fn=nn.Tanh,
        feat_activation=None,
        output_hidden=[64, 64],
        output_activation=None,
        pred_Fz=True,
        pred_Fxy=False,
    ):
        super(vanilla_model, self).__init__()
        self.n_input = n_input
        # self.n_output = 2 + int(pred_Fz) + 2*int(pred_Fxy)
        self.n_output = n_output
        self.feature_dim = feature_dim
        self.feat_model = simpleMLP(
            n_input=n_input,
            n_output=feature_dim,
            hidden_dims=feat_hidden,
            activation_fn=activation_fn,
            output_activation=feat_activation,
        )
        self.output_model = simpleMLP(
            feature_dim,
            self.n_output,
            hidden_dims=output_hidden,
            activation_fn=activation_fn,
            output_activation=output_activation,
        )

    def forward(self, sens):
        return self.output_model(self.get_feature(sens))

    def get_feature(self, sens):
        return self.feat_model(sens)

    def get_out_from_feature(self, feature):
        return self.output_model(feature)


# credit to https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
class imagine2touch_model(nn.Module):
    def __init__(
        self,
        tactile_decoder_hidden,
        images_decoder_hidden,
        tactile_embedding_dim=5,
        tactile_input_shape=15,
        cnn_images_encoder=True,
        images_encoder_hidden=0,
        images_input_shape=0,
        images_output_shape=0,
        image_embedding_dim=0,
        var_ae=False,
    ):
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        # define layers
        self.images_encoder_hidden = images_encoder_hidden
        self.tactile_decoder_hidden = tactile_decoder_hidden
        self.images_decoder_hidden = images_decoder_hidden
        self.images_input_shape = images_input_shape
        self.images_output_shape = images_output_shape
        self.cnn_images_encoder = cnn_images_encoder
        self.var_ae = var_ae
        super().__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(8)

        # Max Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Embedding Layer
        self.embedding = nn.Linear(32, image_embedding_dim)
        self.decoder_hidden_layer_tactile = nn.Linear(
            in_features=image_embedding_dim,
            out_features=self.tactile_decoder_hidden,
        )
        self.decoder_output_layer_tactile = nn.Linear(
            in_features=self.tactile_decoder_hidden,
            # in_features=tactile_embedding_dim,
            out_features=tactile_input_shape,
        )
        self.decoder_output_layer_mask = nn.Linear(
            in_features=self.images_decoder_hidden,
            out_features=self.images_output_shape,
        )
        if not self.cnn_images_encoder:
            self.encoder_image_input_layer = nn.Linear(
                in_features=images_input_shape,
                out_features=self.images_encoder_hidden,
            )
            self.encoder_image_hidden_layer = nn.Linear(
                in_features=self.images_encoder_hidden,
                out_features=self.images_encoder_hidden,
            )
            self.encoder_image_output_layer = nn.Linear(
                in_features=self.images_encoder_hidden,
                out_features=image_embedding_dim,
            )
            self.encoder_image_output_layer_var = nn.Linear(
                in_features=self.images_encoder_hidden,
                out_features=image_embedding_dim,
            )
            self.encoder_image_input_layer.apply(init_weights)
            self.encoder_image_hidden_layer.apply(init_weights)
            self.encoder_image_output_layer.apply(init_weights)
        else:
            self.encoder_image_output_layer_var = nn.Linear(
                in_features=32,
                out_features=image_embedding_dim,
            )
        self.decoder_hidden_layer_image = nn.Linear(
            in_features=image_embedding_dim, out_features=self.images_decoder_hidden
        )
        self.decoder_output_layer_image = nn.Linear(
            in_features=self.images_decoder_hidden, out_features=images_input_shape
        )
        self.flatten = nn.Flatten()
        self.decoder_hidden_layer_tactile.apply(init_weights)
        self.decoder_output_layer_tactile.apply(init_weights)
        self.decoder_hidden_layer_image.apply(init_weights)
        self.decoder_output_layer_image.apply(init_weights)
        self.decoder_output_layer_mask.apply(init_weights)

        self.embedding.apply(init_weights)
        self.pool1.apply(init_weights)
        self.pool2.apply(init_weights)
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.conv3.apply(init_weights)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_image(self, image_data):
        if self.cnn_images_encoder:
            x = nn.functional.relu(self.batchnorm1(self.conv1(image_data)))
            x = self.pool1(x)
            x = nn.functional.relu(self.batchnorm2(self.conv2(x)))
            x = self.pool2(x)
            x = nn.functional.relu(self.batchnorm3(self.conv3(x)))
            x = torch.sigmoid(self.flatten(x))
            code_image = self.embedding(x)
        else:
            activation_image = self.encoder_image_input_layer(image_data)
            activation_image = torch.relu(activation_image)
            hidden_layer_image = self.encoder_image_hidden_layer(activation_image)
            hidden_layer_image = torch.relu(hidden_layer_image)
            code_image = self.encoder_image_output_layer(hidden_layer_image)
        if self.var_ae:
            code_logvar_image = self.encoder_image_output_layer_var(hidden_layer_image)
            code_image = self.reparameterize(code_image, code_logvar_image)
        return code_image

    def forward(self, image_data):
        code_image = self.encode_image(image_data)
        activation_image = self.decoder_hidden_layer_image(code_image)
        activation_image = torch.relu(activation_image)
        activation_image = self.decoder_output_layer_image(activation_image)
        image = torch.relu(activation_image)
        activation_tactile = self.decoder_hidden_layer_tactile(code_image)
        activation_tactile = torch.relu(activation_tactile)
        tactile = self.decoder_output_layer_tactile(activation_tactile)
        return tactile, image, code_image
