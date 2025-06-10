# -*- coding: utf-8 -*-

import torch.nn as nn

from ghrp.model_definitions.components.def_encoder_attn import EncoderTransformer
from ghrp.model_definitions.components.def_decoder_attn import DecoderTransformer

from ghrp.model_definitions.components.def_encoder import Encoder
from ghrp.model_definitions.components.def_decoder import Decoder

from ghrp.model_definitions.components.def_perceiver import Perceiver


###############################################################################
# define regular AE
# ##############################################################################


class AE(nn.Module):
    """
    tbd
    """

    def __init__(self, config):
        super(AE, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        z = self.encoder(x)
        return z

    def forward_decoder(self, z):
        y = self.decoder(z)
        return y


###############################################################################
# define Attention AE
# ##############################################################################


class AE_attn(nn.Module):
    """
    tbd
    """

    def __init__(self, config):
        super(AE_attn, self).__init__()

        self.encoder = EncoderTransformer(config)
        self.decoder = DecoderTransformer(config)

    def forward(self, x):
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        z, _ = self.encoder(x)
        return z

    def forward_decoder(self, z):
        y, _ = self.decoder(z)
        return y


###############################################################################
# define Perceiver AE
# ##############################################################################


class AE_perceiver(nn.Module):
    """
    tbd
    """

    def __init__(self, config):
        super(AE_perceiver, self).__init__()

        print(f"##### initialize PERCEIVER AE")
        # prepare dicts
        key_list = [
            "perceiver::num_freq_bands",
            "perceiver::depth",
            "perceiver::max_freq",
            "perceiver::input_channels",
            "perceiver::input_axis",
            "perceiver::num_latents",
            "perceiver::latent_dim",
            "perceiver::cross_heads",
            "perceiver::latent_heads",
            "perceiver::cross_dim_head",
            "perceiver::latent_dim_head",
            "perceiver::attn_dropout",
            "perceiver::ff_dropout",
            "perceiver::weight_tie_layers",
            "perceiver::fourier_encode_data",
            "perceiver::self_per_cross_attn",
            "perceiver::out_num_axis",
            "perceiver::out_num_channels",
        ]
        config_encoder = {}
        config_decoder = {}
        for key in key_list:
            config_encoder[key] = config[key]
            config_decoder[key] = config[key]

        # encoder: latent_dim and num_latents actually match
        # decoder: match with encoder inputs
        config_decoder["perceiver::input_channels"] = config_encoder[
            "perceiver::latent_dim"
        ]  # latent dim is last dimension of x
        config_decoder[
            "perceiver::input_axis"
        ] = 1  # there's just one axis, which is the number of latents
        config_decoder["perceiver::num_latents"] = config["perceiver::out_num_axis"]
        config_decoder["perceiver::latent_dim"] = config[
            "perceiver::decoder_latent_dim"
        ]  # and the other way around
        self.num_latents = config["perceiver::num_latents"]
        self.latent_dim = config["perceiver::latent_dim"]
        self.decoder_out_axis = config["perceiver::out_num_channels"]
        # check position encoding for decoder latents
        if config.get("perceiver::decoder_latent_position_enc", False):
            config_decoder["perceiver::fourier_encode_latents"] = True

        self.encoder = Perceiver(config_encoder)
        self.decoder = Perceiver(config_decoder)

    def forward(self, x):
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        # assume x.shape = [batch_size,number_of_weights] -> add trailing dimension
        x = x.unsqueeze(dim=-1)
        z = self.encoder(x)
        # flatten z to be compatible to downstream tasks and contrastive loss
        z = z.view(z.shape[0], -1)
        return z

    def forward_decoder(self, z):
        # bring z back to sequential shape
        z = z.view(z.shape[0], self.num_latents, self.latent_dim)
        # pass through decoder
        y = self.decoder(z)
        # slice for unnessary depth of outputs.
        slice_axis = self.decoder_out_axis
        y = y[:, :, :slice_axis]
        # assume y.shape = [batch_size,number_of_weights] -> remove trailing dimension
        y = y.squeeze()
        return y

    def forward_encoder(self, x):
        # flatten embeddings for downstream tasks
        # assume x.shape = [batch_size,number_of_weights] -> add trailing dimension
        x = x.unsqueeze(dim=-1)
        z = self.encoder(x)
        return z.view(z.shape[0], -1)
