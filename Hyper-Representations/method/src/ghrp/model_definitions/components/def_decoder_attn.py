import torch
import torch.nn as nn
from ghrp.model_definitions.components.def_transformers import (
    Debedder,
    PositionalEncoder,
    get_clones,
    Norm,
    EncoderLayer,
    Neck2Seq,
    DebedderNeuronGroup,
    DebedderNeuronGroup_index,
    DebedderNeuron,
)
from ghrp.model_definitions.components.def_decoder import Decoder

from einops import repeat


class DecoderTransformer(nn.Module):
    """
    Decoder Transformer
    """

    def __init__(self, config):
        super(DecoderTransformer, self).__init__()

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
        self.decompression = config.get("model::decompression", "linear")

        ### get token embeddings / config
        if config.get("model::encoding", "weight") == "weight":
            # encode each weight separately
            self.max_seq_len = self.input_dim
            self.token_debeddings = Debedder(self.input_dim, self.embed_dim)
        elif config.get("model::encoding", "weight") == "neuron":
            # encode weights of one neuron togetehr
            index_dict = config.get("model::index_dict", None)
            debedder_layers = config.get("model::debedder_layers", 1)
            self.token_debeddings = DebedderNeuronGroup_index(
                index_dict=index_dict,
                d_model=self.embed_dim,
                layers=debedder_layers,
                dropout=self.dropout,
            )
            self.max_seq_len = self.token_debeddings.__len__()
        elif config.get("model::encoding", "weight") == "neuron_in_out":
            # embed ingoing + outgoing weights together
            index_dict = config.get("model::index_dict", None)
            self.token_debeddings = DebedderNeuron(index_dict, self.embed_dim)
            self.max_seq_len = self.token_debeddings.__len__()

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

        ### mapping from latent space to sequence
        # determine whether to map from latent space to one or all tokens
        if self.decompression == "repeat":
            vec_dim = self.embed_dim
        else:  # self.decompression=="linear"
            vec_dim = self.embed_dim * self.max_seq_len
        # get bottleneck type
        bottleneck = config.get("model::bottleneck", "linear")
        if bottleneck == "linear" or bottleneck == "linear_bounded":
            # fc to map latent space to embedding
            self.nec2vec = nn.Linear(self.latent_dim, vec_dim, bias=True)
        elif bottleneck == "mlp":
            h_layers_mlp = config.get("model::bottleneck::h_lays", 3)
            config_mlp = {
                "model::res_blocks": 0,
                "model::res_block_lays": 0,
                "model::h_layers": h_layers_mlp,
                "model::i_dim": self.embed_dim * self.max_seq_len,
                "model::latent_dim": self.latent_dim,
                "model::transition": "lin",
                "model::nlin": "leakyrelu",
                "model::dropout": self.dropout,
                "model::init_type": "kaiming_normal",
            }
            self.nec2vec = Decoder(config_mlp)

        # add a learnable decoder bias, so that latent representation only embeds variance
        if config.get("model::decoder_bias", None):
            # initialize with small values close to 0
            self.decoder_bias = nn.Parameter(torch.randn(self.latent_dim) / 1000)
        else:
            self.decoder_bias = torch.zeros(self.latent_dim)

    def forward(self, z, mask=None):
        """
        forward function: map to token sequence, add position encodings, pass through transformer, map back to weights
        """

        attn_scores = []  # not yet implemented, to prep interface

        # add decoder bias (could be learned or zeros)
        b_ = repeat(self.decoder_bias, "d -> b d", b=z.shape[0]).to(z.device)
        z = z + b_  # repeat bias for # tokens

        # decompress
        y = self.nec2vec(z)
        if self.decompression == "repeat":
            b, d = y.shape
            y = repeat(y, "b d -> b n d", n=self.max_seq_len)
        else:
            y = y.view(z.shape[0], self.max_seq_len, self.embed_dim)

        # embedd positions
        positions = torch.arange(self.max_seq_len, device=y.device).unsqueeze(0)
        y = y + self.position_embeddings(positions).expand_as(y)
        
        # apply attention
        if self.transformer_type == "pol":
            for ndx in range(self.N):
                y, scores = self.layers[ndx](y, mask)
                attn_scores.append(scores)
        elif self.transformer_type == "pytorch":
            y = self.transformer(y)

        # map back to original space.
        y = self.token_debeddings(y)
        
        return y, attn_scores
