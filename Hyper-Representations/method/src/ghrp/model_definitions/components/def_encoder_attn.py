import torch
import torch.nn as nn
from ghrp.model_definitions.components.def_transformers import (
    Embedder,
    get_clones,
    Norm,
    EncoderLayer,
    EmbedderNeuronGroup,
    EmbedderNeuronGroup_index,
    EmbedderNeuron,
)
from ghrp.model_definitions.components.def_encoder import Encoder

from ghrp.model_definitions.components.def_attn_embedder import AttnEmbedder

from einops import repeat


class EncoderTransformer(nn.Module):
    """
    Encoder Transformer
    """

    def __init__(self, config):
        super(EncoderTransformer, self).__init__()

        # get config
        self.N = config["model::N_attention_blocks"]
        self.input_dim = config["model::i_dim"]
        self.embed_dim = config["model::dim_attention_embedding"]
        self.normalize = config["model::normalize"]
        self.heads = config["model::N_attention_heads"]
        self.dropout = config["model::dropout"]
        self.d_ff = config["model::attention_hidden_dim"]
        self.latent_dim = config["model::latent_dim"]
        self.device = config["device"]
        self.compression = config.get("model::compression", "linear")
        # catch deprecated stuff.
        compression_token = config.get("model::compression_token", "NA")
        if not compression_token == "NA":
            if compression_token == True:
                self.compression == "token"
            elif compression_token == False:
                self.compression == "linear"

        print(f"init attn encoder")

        ### get token embeddings / config
        if config.get("model::encoding", "weight") == "weight":
            # encode each weight separately
            self.max_seq_len = self.input_dim
            self.token_embeddings = Embedder(self.input_dim, self.embed_dim)
        elif config.get("model::encoding", "weight") == "neuron":
            # encode weights of one neuron together
            if config.get("model::encoder") == "attn":
                # use attn embedder (attn of tokens for individual weights)
                print("## attention encoder -- use index_dict")
                index_dict = config.get("model::index_dict", None)
                d_embed = config.get("model::attn_embedder_dim")
                n_heads = config.get("model::attn_embedder_nheads")
                self.token_embeddings = AttnEmbedder(
                    index_dict,
                    d_model=int(self.embed_dim),
                    d_embed=d_embed,
                    n_heads=n_heads,
                )
                self.max_seq_len = self.token_embeddings.__len__()
            else:
                # encode weights of a neuron linearly
                print("## encoder -- use index_dict")
                index_dict = config.get("model::index_dict", None)
                self.token_embeddings = EmbedderNeuronGroup_index(
                    index_dict, self.embed_dim
                )
                self.max_seq_len = self.token_embeddings.__len__()
        elif config.get("model::encoding", "weight") == "neuron_in_out":
            # embed ingoing + outgoing weights together
            index_dict = config.get("model::index_dict", None)
            self.token_embeddings = EmbedderNeuron(index_dict, self.embed_dim)
            self.max_seq_len = self.token_embeddings.__len__()

        ### set compression token embedding
        if self.compression == "token":
            self.comp_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            # add sequence length of 1
            self.max_seq_len += 1

        #### get learned position embedding
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.embed_dim)

        ### compose transformer layers
        self.transformer_type = config.get("model::transformer_type", "pol")
        if self.transformer_type == "pol":
            self.layers = get_clones(
                EncoderLayer(
                    d_model=self.embed_dim,
                    heads=self.heads,
                    normalize=self.normalize,
                    dropout=self.dropout,
                    d_ff=self.d_ff,
                ),
                self.N,
            )
        elif self.transformer_type == "pytorch":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                activation="relu",
            )
            tra_norm = None
            if self.normalize is not None:
                tra_norm = Norm(d_model=self.embed_dim)
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=self.N, norm=tra_norm
            )

        ### mapping from tranformer output to latent space
        # full, average, or compression token
        bottleneck = config.get("model::bottleneck", "linear")
        if self.compression == "token" or self.compression == "average":
            bottleneck_input = self.embed_dim
        else:  # self.compression=="linear"
            bottleneck_input = self.embed_dim * self.max_seq_len
        # get mapping: linear, linear bounded (with tanh) or mlp
        if bottleneck == "linear":
            self.vec2neck = nn.Sequential(nn.Linear(bottleneck_input, self.latent_dim))
        elif bottleneck == "linear_bounded":
            self.vec2neck = nn.Sequential(
                nn.Linear(bottleneck_input, self.latent_dim), nn.Tanh()
            )
        elif bottleneck == "mlp":
            h_layers_mlp = config.get("model::bottleneck::h_lays", 3)
            config_mlp = {
                "model::res_blocks": 0,
                "model::res_block_lays": 0,
                "model::h_layers": h_layers_mlp,
                "model::i_dim": bottleneck_input,
                "model::latent_dim": self.latent_dim,
                "model::transition": "lin",
                "model::nlin": "leakyrelu",
                "model::dropout": self.dropout,
                "model::init_type": "kaiming_normal",
                "model::normalize_latent": True,
            }
            self.vec2neck = Encoder(config_mlp)

    def forward(self, x, mask=None):
        """
        forward function: get token embeddings, add position encodings, pass through transformer, map to bottleneck
        """
        attn_scores = []  # not yet implemented, to prep interface
        # embedd weights
        x = self.token_embeddings(x)
        # add a compression token to the beginning of each sequence (dim = 1)
        if self.compression == "token":
            b, n, _ = x.shape
            copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=b)
            x = torch.cat((copm_tokens, x), dim=1)
        # embedd positions
        positions = torch.arange(self.max_seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(positions).expand_as(x)

        # pass through encoder
        # x = self.encoder(x, mask)
        if self.transformer_type == "pol":
            for ndx in range(self.N):
                x, scores = self.layers[ndx](x, mask)
                attn_scores.append(scores)
        elif self.transformer_type == "pytorch":
            x = self.transformer(x)

        # compress to bottleneck
        if self.compression == "token":
            # take only first part of the sequence / token
            x = x[:, 0, :]
        elif self.compression == "average":
            # take only first part of the sequence / token
            x = torch.mean(x, dim=1)
        else:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2])

        x = self.vec2neck(x)
        #
        return x, attn_scores
