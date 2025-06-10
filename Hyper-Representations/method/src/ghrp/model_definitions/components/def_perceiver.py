import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

from math import pi, log
from functools import wraps


class Perceiver(nn.Module):

    """
    implements the perceiver architecture published in https://arxiv.org/pdf/2107.14795.pdf
    implmenetation inspired by https://github.com/lucidrains/perceiver-pytorch/
    """

    def __init__(self, config):
        """
        Args taken from lucidrains:

        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True

        The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)
        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
        """
        super().__init__()

        self.num_freq_bands = config.get("perceiver::num_freq_bands")
        self.depth = config.get("perceiver::depth")
        self.max_freq = config.get("perceiver::max_freq")
        self.input_channels = config.get("perceiver::input_channels", 3)
        self.input_axis = config.get("perceiver::input_axis", 2)
        self.num_latents = config.get("perceiver::num_latents", 512)
        self.latent_dim = config.get("perceiver::latent_dim", 512)
        self.cross_heads = config.get("perceiver::cross_heads", 1)
        self.latent_heads = config.get("perceiver::latent_heads", 8)
        self.cross_dim_head = config.get("perceiver::cross_dim_head", 64)
        self.latent_dim_head = config.get("perceiver::latent_dim_head", 64)
        self.attn_dropout = config.get("perceiver::attn_dropout", 0.0)
        self.ff_dropout = config.get("perceiver::ff_dropout", 0.0)
        self.weight_tie_layers = config.get("perceiver::weight_tie_layers", False)
        self.fourier_encode_data = config.get("perceiver::fourier_encode_data", True)
        self.fourier_encode_latents = config.get(
            "perceiver::fourier_encode_latents", False
        )
        self.self_per_cross_attn = config.get("perceiver::self_per_cross_attn", 1)

        # fix fourier channels and input dim
        self.fourier_channels = (
            (self.input_axis * ((self.num_freq_bands * 2) + 1))
            if self.fourier_encode_data
            else 0
        )
        self.input_dim = self.fourier_channels + self.input_channels

        # init learned query / latents
        self.fourier_channels_latent = (
            (1 * ((self.num_freq_bands * 2) + 1)) if self.fourier_encode_latents else 0
        )
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        self.latent_dim = self.latent_dim + self.fourier_channels_latent

        # make module List

        self.layers = nn.ModuleList()

        for idx in range(self.depth):
            # get latent attn modules
            latent_attn_list = nn.ModuleList()
            for jdx in range(self.self_per_cross_attn):
                latent_attn_list.append(
                    LatentAttention(
                        latent_dim=self.latent_dim,
                        latent_heads=self.latent_heads,
                        latent_dim_head=self.latent_dim_head,
                        attn_dropout=self.attn_dropout,
                        ff_dropout=self.ff_dropout,
                    )
                )
            # get cross-attn module
            cross_attn = CrossAttention(
                latent_dim=self.latent_dim,
                input_dim=self.input_dim,
                cross_heads=self.cross_heads,
                cross_dim_head=self.cross_dim_head,
                attn_dropout=self.attn_dropout,
                ff_dropout=self.ff_dropout,
            )

            # append both stacks to module List
            layer_dx = nn.ModuleList([cross_attn, latent_attn_list])
            self.layers.append(layer_dx)

            # if weights between layers are shared, stop after first iteration.
            if self.weight_tie_layers:
                break

    def forward(self, data, mask=None):

        # 1) get data dimensions and assert axis match initialization
        b, *axis, _, device = *data.shape, data.device
        assert (
            len(axis) == self.input_axis
        ), f"input data must have the right number of axis. found {len(axis)} expected {self.input_axis}"

        # 2) fourier encode data
        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(
                map(
                    lambda size: torch.linspace(-1.0, 1.0, steps=size, device=device),
                    axis,
                )
            )
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
            enc_pos = repeat(enc_pos, "... -> b ...", b=b)

            data = torch.cat((data, enc_pos), dim=-1)

        if self.fourier_encode_latents:
            *axis, _ = self.latents.shape
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(
                map(
                    lambda size: torch.linspace(-1.0, 1.0, steps=size, device=device),
                    axis,
                )
            )
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, "...  n d -> ... (n d)")

            # add positions to latents
            # x = torch.cat((self, enc_pos), dim=-1)
            # print(f"self.latents.shape: {self.latents.shape}")
            # print(f"enc_pos.shape: {enc_pos.shape}")
            x = torch.cat((self.latents, enc_pos), dim=-1)
            x = repeat(x, "n ... -> b n ...", b=b)

            # self.latents.shape: torch.Size([2464, 1])
            # enc_pos.shape: torch.Size([2464, 65])
            # data.shape: torch.Size([32, 128, 129])
            # x.shape: torch.Size([32, 2464, 66])
        else:
            # prepare batch of latents
            x = repeat(self.latents, "n ... -> b n ...", b=b)

        # 3) prep data
        # concat channels and flatten axis (keep first axis b and last axis d the same, cat all in the middle )
        # if num axis = 2, add [1] in the middle
        data = rearrange(data, "b ... d -> b (...) d")
        # print(f"data.shape: {data.shape}")
        # print(f"x.shape: {x.shape}")

        if self.weight_tie_layers:
            [cross_attn, latent_attn_list] = self.layers[0]
            for _ in range(self.depth):
                # pass through cross attn
                x = cross_attn(
                    x=x,
                    context=data,
                    mask=mask,
                )
                for lat_attn in latent_attn_list:
                    x = lat_attn(x=x, mask=mask)

        #   if weights are not shared: iterate through entries in self.layers and forward pass through them
        else:
            for [cross_attn, latent_attn_list] in self.layers:
                for _ in range(self.depth):
                    # pass through cross attn
                    x = cross_attn(
                        x=x,
                        context=data,
                        mask=mask,
                    )
                    for lat_attn in latent_attn_list:
                        x = lat_attn(x=x, mask=mask)
        if self.fourier_encode_latents:
            # slice first component
            x = x[:, :, 0]
        return x


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, context_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, context=None):
        x = self.norm(x)

        if exists(self.norm_context) and context is not None:
            normed_context = self.norm_context(context)
            return x, normed_context
        else:
            return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        # get queries
        q = self.to_q(x)

        # get keys and values, either from context (if exists: cross attn, or from x: latent attn)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # number of heads was previously clustered in token dimension, now get's moved to no_tokens
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class CrossAttention(nn.Module):
    """
    Implements the cross attention module mix betw. low-dim latents and high-dim data. Includes the Feedfoward layer in between attention layers
    pass through norm + attn. / norm+ff has residual-like connections
    """

    def __init__(
        self,
        latent_dim,
        input_dim,
        cross_heads,
        cross_dim_head,
        attn_dropout,
        ff_dropout,
    ):

        super().__init__()

        # norm module pre-attention
        # normalize over both latents and inputs (context)
        self.norm_pre_attn = PreNorm(dim=latent_dim, context_dim=input_dim)
        # Attention module
        self.attn = Attention(
            query_dim=latent_dim,
            context_dim=input_dim,
            heads=cross_heads,
            dim_head=cross_dim_head,
            dropout=attn_dropout,
        )
        # norm module pre-ff
        self.norm_pre_ff = PreNorm(dim=latent_dim, context_dim=None)
        # Feedforward
        self.feedforward = FeedForward(dim=latent_dim, dropout=ff_dropout)

    def forward(
        self,
        x,
        context,
        mask,
    ):
        # pass through pre-norm
        x2, context = self.norm_pre_attn(x, context=context)
        # pass through attn
        x = self.attn(x=x2, context=context, mask=mask) + x
        # pass through pre-norm_ff
        x2 = self.norm_pre_ff(x)
        # pass through ff
        x = self.feedforward(x2) + x
        # return overall output
        return x


class LatentAttention(nn.Module):
    """
    Implements the latent attention module on low-dim latents. Includes the Feedfoward layer in between attention layers
    pass through norm + attn. / norm+ff has residual-like connections
    """

    def __init__(
        self, latent_dim, latent_heads, latent_dim_head, attn_dropout, ff_dropout
    ):

        super().__init__()

        # norm module pre-attention
        # normalize over both latents and inputs (context)
        self.norm_pre_attn = PreNorm(dim=latent_dim, context_dim=None)
        # Attention module
        self.attn = Attention(
            query_dim=latent_dim,
            context_dim=None,
            heads=latent_heads,
            dim_head=latent_dim_head,
            dropout=attn_dropout,
        )
        # norm module pre-ff
        self.norm_pre_ff = PreNorm(dim=latent_dim, context_dim=None)
        # Feedforward
        self.feedforward = FeedForward(dim=latent_dim, dropout=ff_dropout)

    def forward(
        self,
        x,
        mask,
    ):
        # pass through pre-norm
        x2 = self.norm_pre_attn(x)
        # pass through attn
        x = self.attn(x=x2, context=None, mask=mask) + x
        # pass through pre-norm_ff
        x2 = self.norm_pre_ff(x)
        # pass through ff
        x = self.feedforward(x2) + x
        # return overall output
        return x
