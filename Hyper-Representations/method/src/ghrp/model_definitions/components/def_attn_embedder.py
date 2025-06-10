import torch
import torch.nn as nn
from einops import repeat


# # Tranformer Encoder
class AttnEmbedder(nn.Module):
    def __init__(self, index_dict, d_model, n_heads, d_embed, seed=22):
        super().__init__()

        self.index_dict = index_dict

        self.get_kernel_slices()

        assert d_model % d_embed == 0, "d_model and d_embed need to be divisible"
        self.output_dim = d_model
        self.embed_dim = d_embed
        self.heads = n_heads
        self.d_ff = int(1.5 * self.embed_dim)
        self.dropout = 0.1
        self.N = 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation="relu",
        )
        tra_norm = None
        # if self.normalize is not None:
        # tra_norm = Norm(d_model=self.embed_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.N, norm=tra_norm
        )

        # position encoding
        self.max_seq_len = self.get_max_weights_per_neuron()
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.embed_dim)
        # weights-to-seqenence embedder
        self.comp_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def get_max_weights_per_neuron(
        self,
    ):
        weights_list = []
        for idx, layer in enumerate(self.index_dict["layer"]):
            weights_size = (
                self.index_dict["kernel_size"][idx]
                * self.index_dict["channels_in"][idx]
            )
            weights_list.append(int(weights_size) + 1)
        # get max number of weights
        max_no_weights = max(weights_list)
        # print(max_no_weights)
        return max_no_weights

    def get_kernel_slices(
        self,
    ):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                kernel_start = (
                    self.index_dict["idx_start"][idx]
                    + kernel_dx
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                kernel_end = (
                    kernel_start
                    + self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                bias = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                    + kernel_dx
                )
                index_kernel = list(range(kernel_start, kernel_end))
                index_kernel.append(bias)

                kernel_slice_lst.append(index_kernel)
                # print(index_kernel)
            slice_lst.append(kernel_slice_lst)
        self.slice_lst = slice_lst

    def __len__(
        self,
    ):
        counter = 0
        for layer_embeddings in self.slice_lst:
            counter += len(layer_embeddings)
        return counter

    def forward(self, x):
        y_lst = []
        # loop over layers
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # slice weights
                w_tmp = x[:, kernel_index]
                # unsqueeze -> last dimension to one
                w_tmp = w_tmp.unsqueeze(dim=-1)
                # repeat weights multiple times
                w_tmp = repeat(w_tmp, "b n () -> b n d", d=self.embed_dim)
                # apply position encoding
                # embedd positions
                b, n, d = w_tmp.shape
                # print(w_tmp.shape)
                positions = torch.arange(
                    # self.max_seq_len, device=w_tmp.device
                    n,
                    device=w_tmp.device,
                ).unsqueeze(0)
                w_tmp = w_tmp + self.position_embeddings(positions).expand_as(w_tmp)
                # compression token
                b, n, _ = w_tmp.shape
                copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=b)
                w_tmp = torch.cat((copm_tokens, w_tmp), dim=1)
                # pass through attn
                y_tmp = self.transformer(w_tmp)
                # get compression tokens
                y_tmp = y_tmp[:, 0, :]
                y_lst.append(y_tmp)
        y = torch.stack(y_lst, dim=1)
        # repeat to get output dimensions
        # print(y.shape)
        repeat_factor = int(self.output_dim / self.embed_dim)
        y = y.repeat([1, 1, repeat_factor])
        # print(y.shape)
        return y
