import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


# # Transformer Shared Layers


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)

    return output, scores


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores, sc = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output, sc


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, normalize=True, dropout=0.1, d_ff=2048):
        super().__init__()
        self.normalize = normalize
        if normalize:
            self.norm_1 = Norm(d_model)
            self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        if self.normalize:
            x2 = self.norm_1(x)
        else:
            x2 = x.clone()
        res, sc = self.attn(x2, x2, x2, mask)
        # x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x = x + self.dropout_1(res)
        if self.normalize:
            x2 = self.norm_2(x)
        else:
            x2 = x.clone()
        x = x + self.dropout_2(self.ff(x2))
        # return x
        return x, sc


class Encoder(nn.Module):
    def __init__(
        self, input_dim, d_model, N, heads, max_seq_len, dropout, d_ff, device
    ):
        super().__init__()
        self.device = device
        self.N = N
        self.embed = Embedder(input_dim, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len, device)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout, d_ff), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        scores = []
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x, sc = self.layers[i](x, mask)
            scores.append(sc)
        return self.norm(x), scores


class Embedder(nn.Module):
    def __init__(self, input_dim, embed_dim, seed=22):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(1, embed_dim)

    def forward(self, x):
        y = []
        # use the same embedder to embedd all weights
        for idx in range(self.input_dim):
            # embedd single input / feature dimension
            tmp = self.embed(x[:, idx].unsqueeze(dim=1))
            y.append(tmp)
        # stack along dimension 1
        y = torch.stack(y, dim=1)
        return y


class Debedder(nn.Module):
    def __init__(self, input_dim, d_model, seed=22):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.weight_debedder = nn.Linear(d_model, 1)

    def forward(self, x):
        y = self.weight_debedder(x)
        y = y.squeeze()
        return y


# # Tranformer Encoder
class EmbedderNeuron(nn.Module):
    # collects all weights connected to one neuron / kernel
    def __init__(self, index_dict, d_model, seed=22):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict
        self.get_kernel_slices()

        for idx, kernel_lst in enumerate(self.slice_lst):
            i_dim = len(kernel_lst[0])
            # check sanity of slices
            for slice in kernel_lst:
                assert (
                    len(slice) == i_dim
                ), f"layer-wise slices are not of the same lenght: {i_dim} vs {len(slice)}"
            # get layers
            self.layer_lst.append(nn.Linear(i_dim, d_model))
            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")

    def get_kernel_slices(
        self,
    ):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                # get current kernel index
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

                # get next_layers connected weights
                if idx < len(self.index_dict["layer"]) - 1:
                    # -> find corresponding indices
                    # -> get offset to beginning of next layer
                    for kernel_dx_next in range(self.index_dict["kernel_no"][idx + 1]):
                        kernel_next_start = (
                            # get start of next layer
                            self.index_dict["idx_start"][idx + 1]
                            # offset by current kernel*dim of kernel_size (columns)
                            + kernel_dx * self.index_dict["kernel_size"][idx + 1]
                            # offset by rows: overall parameters per channel out
                            + kernel_dx_next
                            * self.index_dict["channels_in"][idx + 1]
                            * self.index_dict["kernel_size"][idx + 1]
                        )
                        kernel_next_end = (
                            kernel_next_start + self.index_dict["kernel_size"][idx + 1]
                        )

                        # extend
                        kernel_next_idx = list(
                            range(kernel_next_start, kernel_next_end)
                        )
                        index_kernel.extend(kernel_next_idx)

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
                # print(index_kernel)
                y_tmp = self.layer_lst[idx](x[:, kernel_index])
                y_lst.append(y_tmp)
        y = torch.stack(y_lst, dim=1)
        return y


class DebedderNeuron(nn.Module):
    def __init__(self, index_dict, d_model, seed=22, layers=1, dropout=0.1):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict
        self.get_kernel_slices()

        for idx, kernel_lst in enumerate(self.slice_lst):
            i_dim = len(kernel_lst[0])
            # check sanity of slices
            for slice in kernel_lst:
                assert (
                    len(slice) == i_dim
                ), f"layer-wise slices are not of the same lenght: {i_dim} vs {len(slice)}"
            # get layers
            if layers == 1:
                self.layer_lst.append(nn.Linear(d_model, i_dim))
            else:
                from model_definitions.def_net import MLP

                layertmp = MLP(
                    i_dim=d_model,
                    h_dim=[d_model for _ in range(layers - 2)],
                    o_dim=i_dim,
                    nlin="leakyrelu",
                    dropout=dropout,
                    init_type="kaiming_normal",
                    use_bias=True,
                )
                self.layer_lst.append(layertmp)
            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")
            # self.layer_lst.append(nn.Linear(d_model, i_dim))

    def get_kernel_slices(
        self,
    ):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                # get current kernel index
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

                # get next_layers connected weights
                if idx < len(self.index_dict["layer"]) - 1:
                    # -> find corresponding indices
                    # -> get offset to beginning of next layer
                    for kernel_dx_next in range(self.index_dict["kernel_no"][idx + 1]):
                        kernel_next_start = (
                            # get start of next layer
                            self.index_dict["idx_start"][idx + 1]
                            # offset by current kernel*dim of kernel_size (columns)
                            + kernel_dx * self.index_dict["kernel_size"][idx + 1]
                            # offset by rows: overall parameters per channel out
                            + kernel_dx_next
                            * self.index_dict["channels_in"][idx + 1]
                            * self.index_dict["kernel_size"][idx + 1]
                        )
                        kernel_next_end = (
                            kernel_next_start + self.index_dict["kernel_size"][idx + 1]
                        )

                        # extend
                        kernel_next_idx = list(
                            range(kernel_next_start, kernel_next_end)
                        )
                        index_kernel.extend(kernel_next_idx)

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
        device = x.device
        # get last value of last layer last kernel last index - zero based -> +1
        i_dim = self.slice_lst[-1][-1][-1] + 1
        y = torch.zeros((x.shape[0], i_dim)).to(device)

        # loop over layers
        embed_dx = 0
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # print(index_kernel)
                # get values for this embedding
                y_tmp = self.layer_lst[idx](x[:, embed_dx])
                # !!add!! values in right places
                y[:, kernel_index] += y_tmp
                # raise counter
                embed_dx += 1

        # first layer and last layer get only embedded once,
        # while all middle layers overlap.
        # -> get index list for beginning of second and ending of second to last layer
        # -> devide embedded values by 2
        if len(self.index_dict["idx_start"]) > 2:
            index_start = self.index_dict["idx_start"][1]
            index_end = self.index_dict["idx_start"][-1]
            idx = list(range(index_start, index_end))
            # create tensor of same shape with 0.5 values put it on device
            factor = torch.ones(y[:, idx].shape) * 0.5
            factor = factor.to(y.device)
            # multiply with 0.5
            y[:, idx] = y[:, idx] * factor
        return y


# # Tranformer Encoder
class EmbedderNeuronGroup_index(nn.Module):
    def __init__(self, index_dict, d_model, seed=22, split_kernels_threshold=0):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict

        self.split_kernels_threshold = split_kernels_threshold

        for idx, layer in enumerate(index_dict["layer"]):
            i_dim = index_dict["kernel_size"][idx] * index_dict["channels_in"][idx] + 1
            if (self.split_kernels_threshold != 0) and (
                i_dim > self.split_kernels_threshold
            ):
                i_dim = self.split_kernels_threshold
            self.layer_lst.append(nn.Linear(i_dim, d_model))
            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")

        self.get_kernel_slices()

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
                # print(index_kernel)
                if (self.split_kernels_threshold != 0) and (
                    len(kernel_index) > self.split_kernels_threshold
                ):
                    from math import ceil

                    no_tokens = ceil(len(kernel_index) / self.split_kernels_threshold)
                    for idx in range(no_tokens):
                        idx_token_start = idx * self.split_kernels_threshold
                        idx_token_end = idx_token_start + self.split_kernels_threshold
                        kernel_tmp = kernel_index[idx_token_start:idx_token_end]
                        if idx == no_tokens - 1:  # last
                            x_tmp = torch.zeros(
                                size=[x.shape[0], self.split_kernels_threshold]
                            )  # pad
                            x_tmp[:, : len(kernel_index)] = x[:, kernel_tmp]
                        else:
                            x_tmp = x[:, kernel_tmp]
                        y_tmp = self.layer_lst[idx](x_tmp)
                        y_lst.append(y_tmp)
                else:
                    y_tmp = self.layer_lst[idx](x[:, kernel_index])
                    y_lst.append(y_tmp)
        y = torch.stack(y_lst, dim=1)
        return y


class DebedderNeuronGroup_index(nn.Module):
    def __init__(self, index_dict, d_model, seed=22, layers=1, dropout=0.1):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict

        for idx, layer in enumerate(index_dict["layer"]):
            i_dim = index_dict["kernel_size"][idx] * index_dict["channels_in"][idx] + 1
            # get layers
            if layers == 1:
                self.layer_lst.append(nn.Linear(d_model, i_dim))
            else:
                from model_definitions.def_net import MLP

                layertmp = MLP(
                    i_dim=d_model,
                    h_dim=[d_model for _ in range(layers - 2)],
                    o_dim=i_dim,
                    nlin="leakyrelu",
                    dropout=dropout,
                    init_type="kaiming_normal",
                    use_bias=True,
                )
                self.layer_lst.append(layertmp)

            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")

        self.get_kernel_slices()

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
        device = x.device
        dtype = x.dtype
        # get last value of last layer last kernel last index - zero based -> +1
        i_dim = self.slice_lst[-1][-1][-1] + 1
        y = torch.zeros((x.shape[0], i_dim), dtype=dtype).to(device)

        # loop over layers
        embed_dx = 0
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # print(index_kernel)
                # get values for this embedding
                y_tmp = self.layer_lst[idx](x[:, embed_dx])
                # match data types for mixed precision
                if not y_tmp.dtype == y.dtype:
                    y = y.to(y_tmp.dtype)
                # put values in right places
                y[:, kernel_index] = y_tmp
                # raise counter
                embed_dx += 1

        return y


class EmbedderNeuronGroup(nn.Module):
    def __init__(self, d_model, seed=22):
        super().__init__()

        self.neuron_l1 = nn.Linear(16, d_model)
        self.neuron_l2 = nn.Linear(5, d_model)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        # Hardcoded position for easy-fast integration
        l = []
        # l1
        for ndx in range(5):
            idx_start = ndx * 16
            idx_end = idx_start + 16
            l.append(self.neuron_l1(v[:, idx_start:idx_end]))
        # l2
        for ndx in range(4):
            idx_start = 5 * 16 + ndx * 5
            idx_end = idx_start + 5
            l.append(self.neuron_l2(v[:, idx_start:idx_end]))

        final = torch.stack(l, dim=1)

        # print(final.shape)
        return final


class EncoderNeuronGroup(nn.Module):
    def __init__(self, d_model, N, heads, max_seq_len, dropout, d_ff):
        super().__init__()
        self.N = N
        self.embed = EmbedderNeuronGroup(d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout, d_ff), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        scores = []
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x, sc = self.layers[i](x, mask)
            scores.append(sc)
        return self.norm(x), scores


class DebedderNeuronGroup(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.neuron_l1 = nn.Linear(d_model, 16)
        self.neuron_l2 = nn.Linear(d_model, 5)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        l = []
        for ndx in range(5):
            l.append(self.neuron_l1(v[:, ndx]))
        for ndx in range(5, 9):
            l.append(self.neuron_l2(v[:, ndx]))

        final = torch.cat(l, dim=1)

        # print(final.shape)
        return final


# # Custom Tranformer Decoder


class Neck2Seq(nn.Module):
    def __init__(self, d_model, neck):
        super().__init__()

        self.neuron11 = nn.Linear(neck, d_model)
        self.neuron12 = nn.Linear(neck, d_model)
        self.neuron13 = nn.Linear(neck, d_model)
        self.neuron14 = nn.Linear(neck, d_model)
        self.neuron15 = nn.Linear(neck, d_model)
        self.neuron21 = nn.Linear(neck, d_model)
        self.neuron22 = nn.Linear(neck, d_model)
        self.neuron23 = nn.Linear(neck, d_model)
        self.neuron24 = nn.Linear(neck, d_model)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        # print("V shape: ", v.shape)
        l = []
        l.append(self.neuron11(v))
        l.append(self.neuron12(v))
        l.append(self.neuron13(v))
        l.append(self.neuron14(v))
        l.append(self.neuron15(v))
        l.append(self.neuron21(v))
        l.append(self.neuron22(v))
        l.append(self.neuron23(v))
        l.append(self.neuron24(v))
        final = torch.stack(l, dim=1)

        # print(final.shape)
        return final


class Seq2Vec(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.neuron11 = nn.Linear(d_model, 16)
        self.neuron12 = nn.Linear(d_model, 16)
        self.neuron13 = nn.Linear(d_model, 16)
        self.neuron14 = nn.Linear(d_model, 16)
        self.neuron15 = nn.Linear(d_model, 16)
        self.neuron21 = nn.Linear(d_model, 5)
        self.neuron22 = nn.Linear(d_model, 5)
        self.neuron23 = nn.Linear(d_model, 5)
        self.neuron24 = nn.Linear(d_model, 5)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        l = []
        l.append(self.neuron11(v[:, 0]))
        l.append(self.neuron12(v[:, 1]))
        l.append(self.neuron13(v[:, 2]))
        l.append(self.neuron14(v[:, 3]))
        l.append(self.neuron15(v[:, 4]))
        l.append(self.neuron21(v[:, 5]))
        l.append(self.neuron22(v[:, 6]))
        l.append(self.neuron23(v[:, 7]))
        l.append(self.neuron24(v[:, 8]))
        final = torch.cat(l, dim=1)

        # print(final.shape)
        return final


class DecoderNeuronGroup(nn.Module):
    def __init__(self, d_model, N, heads, max_seq_len, dropout, d_ff, neck):
        super().__init__()
        self.N = N
        self.embed = Neck2Seq(d_model, neck)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout, d_ff), N)
        self.norm = Norm(d_model)

        self.lay = Seq2Vec(d_ff=d_ff)

    def forward(self, src, mask=None):
        scores = []
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x, sc = self.layers[i](x, mask)
            scores.append(sc)
        return self.lay(self.norm(x)), scores


# # AutoEncoder


class TransformerAE(nn.Module):
    def __init__(
        self,
        max_seq_len=9,
        N=1,
        heads=1,
        d_model=100,
        d_ff=100,
        neck=20,
        dropout=0.0,
        **kwargs,
    ):

        super().__init__()

        self.enc = EncoderNeuronGroup(d_model, N, heads, max_seq_len, dropout, d_ff)
        self.dec = DecoderNeuronGroup(
            d_model, N, heads, max_seq_len, dropout, d_ff, neck
        )

        # Addition Approach
        print("Addition Approach!")
        # self.vec2neck = nn.Linear(d_ff, neck)

        # Stacking Approach
        print("Stack Approach!")
        self.vec2neck = nn.Linear(d_ff * max_seq_len, neck)

        self.tanh = nn.Tanh()

        # Xavier Uniform Initialitzation
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inp):

        # First Approach
        out, scEnc = self.enc(inp)

        # Addition
        # neck = self.tanh(self.vec2neck(torch.sum(out, dim=1, keepdim=False)))

        # Stacking
        out = out.view(out.shape[0], out.shape[1] * out.shape[2])
        neck = self.tanh(self.vec2neck(out))

        out, scDec = self.dec(neck)

        return out, neck, scEnc, scDec

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def numParams(self):
        encNumParams = self.count_parameters(self.enc)
        neckNumParams = self.count_parameters(self.vec2neck)
        decNumParams = self.count_parameters(self.dec)
        modelParams = self.count_parameters(self)

        return (
            "EncParams: {}, NeckParams: {}, DecParams: {}, || ModelParams: {} ".format(
                encNumParams, neckNumParams, decNumParams, modelParams
            )
        )


"""
mod = TransformerAE(max_seq_len=9, 
                    N=1, 
                    heads=1, 
                    d_model=100, 
                    d_ff=100,
                    neck=20, 
                    dropout=0.0)

print(mod.numParams())

vec = torch.rand(10,100).cuda()
mod.cuda()
out = mod(vec)
print("Output Shape: ", out[0].shape)

# Training Params
#batchSize = 1000
#criterion = torch.nn.MSELoss(reduction='mean')
#optimizer = torch.optim.Adam(mod.parameters(), lr=1e-3, weight_decay=1e-9)

"""
