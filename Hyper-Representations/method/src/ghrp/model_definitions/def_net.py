# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
from pathlib import Path

import timeit

"""
define net
##############################################################################
"""


class MLP(nn.Module):
    def __init__(
        self,
        i_dim=14,
        h_dim=[30, 15],
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
        use_bias=True,
    ):
        super().__init__()
        self.use_bias = use_bias
        # init module list
        self.module_list = nn.ModuleList()

        # get hidden layer's list
        # wrap h_dim in list of it's not already
        if not isinstance(h_dim, list):
            try:
                h_dim = [h_dim]
            except Exception as e:
                print(e)
        # add i_dim to h_dim
        h_dim.insert(0, i_dim)

        # get if bias should be used or not
        for k in range(len(h_dim) - 1):
            # add linear layer
            self.module_list.append(
                nn.Linear(h_dim[k], h_dim[k + 1], bias=self.use_bias)
            )
            # add nonlinearity
            if nlin == "elu":
                self.module_list.append(nn.ELU())
            if nlin == "celu":
                self.module_list.append(nn.CELU())
            if nlin == "gelu":
                self.module_list.append(nn.GELU())
            if nlin == "leakyrelu":
                self.module_list.append(nn.LeakyReLU())
            if nlin == "relu":
                self.module_list.append(nn.ReLU())
            if nlin == "tanh":
                self.module_list.append(nn.Tanh())
            if nlin == "sigmoid":
                self.module_list.append(nn.Sigmoid())
            if nlin == "silu":
                self.module_list.append(nn.SiLU())
            if dropout > 0:
                self.module_list.append(nn.Dropout(dropout))
        # init output layer
        self.module_list.append(nn.Linear(h_dim[-1], o_dim, bias=self.use_bias))
        # normalize outputs between 0 and 1
        # self.module_list.append(nn.Sigmoid())

        # initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                if self.use_bias:
                    m.bias.data.fill_(0.01)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            #     print(f"layer {layer}")
            #     print(f"input shape:: {x.shape}")
            x = layer(x)
            # print(f"output shape:: {x.shape}")
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            activations.append(x)
        return x, activations


###############################################################################
# define net
# ##############################################################################
def compute_outdim(i_dim, stride, kernel, padding, dilation):
    o_dim = (i_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return o_dim


class CNN(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


class CNN2(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(6, 9, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(9, 6, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 6, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)
        return x, activations


class CNN3(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 32x32 image size
        ## chn_in * 32 * 32
        ## compose layer 0
        self.module_list.append(nn.Conv2d(channels_in, 16, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 16 * 15 * 15
        ## compose layer 1
        self.module_list.append(nn.Conv2d(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 32 * 7 * 7 // 32 * 6 * 6
        ## compose layer 2
        self.module_list.append(nn.Conv2d(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 15 * 2 * 2
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(15 * 2 * 2, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


################################################################################################
class ResCNN(nn.Module):
    """
    extension of the CNN class.
    Added residual connections via max-pooling and 1d convs to the conv layers
    Added a function to load state-dicts from the cnns without res cons
    """

    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()

        if dropout > 0.0:
            raise NotImplementedError(
                "dropout is not yet impemented for the residual connections.|"
            )
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        # input shape bx1x28x28

        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 8, 12, 12]
        ## residual connection stack 1
        self.res1_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res1_conv = nn.Conv2d(
            in_channels=channels_in, out_channels=8, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 6, 4, 4]
        self.res2_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res2_conv = nn.Conv2d(
            in_channels=8, out_channels=6, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## output [15, 4, 3, 3]
        self.res3_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.res3_conv = nn.Conv2d(
            in_channels=6, out_channels=4, kernel_size=1, stride=1, padding=0
        )

        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        # print("initialze model")
        for m in self.module_list:
            m = self.init_single(init_type, m)

        init_type = "kaiming_uniform"  # init the residual blocks with kaiming always
        self.res1_conv = self.init_single(init_type, self.res1_conv)
        self.res2_conv = self.init_single(init_type, self.res2_conv)
        self.res3_conv = self.init_single(init_type, self.res3_conv)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            m.bias.data.fill_(0.01)
        return m

    def get_nonlin(self, nlin):
        """
        gets nn class object for keyword
        """
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def load_weights_from_cnn_checkpoint(self, check):
        """
        takes weights and biases from CNN architecture without residual connections
        assumes this particular data structure, will not fail gracefully otherwise
        """
        self.module_list[0].weight.data = check["module_list.0.weight"]
        self.module_list[0].bias.data = check["module_list.0.bias"]
        self.module_list[3].weight.data = check["module_list.3.weight"]
        self.module_list[3].bias.data = check["module_list.3.bias"]
        self.module_list[6].weight.data = check["module_list.6.weight"]
        self.module_list[6].bias.data = check["module_list.6.bias"]
        self.module_list[9].weight.data = check["module_list.9.weight"]
        self.module_list[9].bias.data = check["module_list.9.bias"]
        self.module_list[11].weight.data = check["module_list.11.weight"]
        self.module_list[11].bias.data = check["module_list.11.bias"]

    def forward(self, x):

        # layer 1
        x_ = x.clone()
        for idx in range(0, 3):
            x_ = self.module_list[idx](x_)
        x = self.res1_pool(x)
        x = self.res1_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 2
        x_ = x.clone()
        for idx in range(3, 6):
            x_ = self.module_list[idx](x_)
        x = self.res2_pool(x)
        x = self.res2_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 3
        x_ = x.clone()
        for idx in range(6, 8):
            x_ = self.module_list[idx](x_)
        x = self.res3_pool(x)
        x = self.res3_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # flatten
        # fc1
        # fc2
        for m in self.module_list[8:]:
            x = m(x)

        return x


################################################################################
class CNN_more_layers(nn.Module):
    def __init__(self, init_type,channels_in=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, 8, 5),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 4, 5, bias=False),
            nn.Tanh(),
            nn.Conv2d(4, 6, 4, bias=False),
            nn.Tanh(),
            nn.Conv2d(6, 4, 3, bias=False),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(36, 18),
            nn.Tanh(),
            nn.Linear(18, 10),
        )

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        # print("initialze model")
        for m in self.layers:
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def forward(self, x):
        return self.layers(x)


###############################################################################
class CNN_residual(nn.Module):
    def __init__(self, init_type,channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 6, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(6, 4, 2, bias=False)
        self.act3 = nn.Tanh()

        self.identity = nn.Conv2d(8, 4, 1, bias=False)

        self.flatten = nn.Flatten()

        self.fc4 = nn.Linear(36, 20, bias=False)
        self.act4 = nn.Tanh()

        self.fc5 = nn.Linear(20, 10)

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        # print("initialze model")
        self.conv1 = self.init_single(init_type, self.conv1)
        self.conv2 = self.init_single(init_type, self.conv2)
        self.conv3 = self.init_single(init_type, self.conv3)
        self.fc4 = self.init_single(init_type, self.fc4)
        self.fc5 = self.init_single(init_type, self.fc5)

        init_type = "kaiming_uniform"
        self.identity = self.init_single(init_type, self.identity)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def forward(self, x):
        x = self.conv1(x)
        #         print(x.shape)

        x = self.act1(self.pool1(x))
        #         print(x.shape)

        x_identity = self.identity(x)
        # print("x_identity", x_identity.shape)

        x = self.act2(self.pool2(self.conv2(x)))
        #         print(x.shape)

        x = self.act3(self.conv3(x))
        # print(x.shape)

        x = x + x_identity[:, :, ::4, ::4]

        # print("skip connection applied")

        x = self.flatten(x)

        x = self.act4(self.fc4(x))
        #         print(x.shape)

        x = self.fc5(x)
        #         print(x.shape)

        return x


###############################################################################
class CNN_more_layers_residual(nn.Module):
    def __init__(self, init_type,channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 4, 5, bias=False)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(4, 6, 4, bias=False)
        self.act3 = nn.Tanh()

        self.conv4 = nn.Conv2d(6, 4, 3, bias=False)
        self.act4 = nn.Tanh()

        self.flatten = nn.Flatten()

        self.fc5 = nn.Linear(36, 18)
        self.act5 = nn.Tanh()

        self.fc6 = nn.Linear(18, 10)

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        # print("initialze model")
        self.conv1 = self.init_single(init_type, self.conv1)
        self.conv2 = self.init_single(init_type, self.conv2)
        self.conv3 = self.init_single(init_type, self.conv3)
        self.conv4 = self.init_single(init_type, self.conv4)
        self.fc5 = self.init_single(init_type, self.fc5)
        self.fc6 = self.init_single(init_type, self.fc6)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass

        return m

    def forward(self, x):
        x = self.conv1(x)
        #         print(x.shape)

        x = self.act1(self.pool1(x))
        #         print(x.shape)

        x = self.act2(self.conv2(x))
        # print(x.shape)

        x_identity = x.clone()

        x = self.act3(self.conv3(x))
        #         print(x.shape)

        x = self.act4(self.conv4(x))
        # print(x.shape)

        x = x + x_identity[:, :, 1:-1, 1:-1][:, :, ::2, ::2]

        # print("skip connection applied")

        x = self.flatten(x)

        x = self.act5(self.fc5(x))
        #         print(x.shape)

        x = self.fc6(x)
        #         print(x.shape)

        return x


import torchvision

from torchvision.models.resnet import ResNet, BasicBlock


class ResNet18(ResNet):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
    ):
        # call init from parent class
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=out_dim)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m


###############################################################################
# define FNNmodule
# ##############################################################################
class NNmodule(nn.Module):
    def __init__(self, config, cuda=False, seed=42, verbosity=0):
        super(NNmodule, self).__init__()

        # set verbosity
        self.verbosity = verbosity

        if cuda and torch.cuda.is_available():
            self.cuda = True
            if self.verbosity > 0:
                print("cuda availabe:: send model to GPU")
        else:
            self.cuda = False
            if self.verbosity > 0:
                print("cuda unavailable:: train model on cpu")

        # setting seeds for reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # construct model
        if config["model::type"] == "MLP":

            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model MLP")
            i_dim = config["model::i_dim"]
            h_dim = config["model::h_dim"]
            o_dim = config["model::o_dim"]
            nlin = config["model::nlin"]
            dropout = config["model::dropout"]
            init_type = config["model::init_type"]
            use_bias = config["model::use_bias"]
            model = MLP(i_dim, h_dim, o_dim, nlin, dropout, init_type, use_bias)

        elif config["model::type"] == "CNN":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "CNN2":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN2(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "CNN3":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN3(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "ResCNN":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = ResCNN(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "CNN_more_layers":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN_more_layers(
                init_type=config["model::init_type"],
                channels_in=config["model::channels_in"],
            )
        elif config["model::type"] == "CNN_residual":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN_residual(
                init_type=config["model::init_type"],
                channels_in=config["model::channels_in"],
            )
        elif config["model::type"] == "CNN_more_layers_residual":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN_more_layers_residual(
                init_type=config["model::init_type"],
                channels_in=config["model::channels_in"],
            )
        elif config["model::type"] == "Resnet18":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating Resnet18")
            model = ResNet18(
                channels_in=config["model::channels_in"],
                out_dim=config["model::o_dim"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        else:
            raise NotImplementedError("error: model type unkown")

        if self.cuda:
            model = model.cuda()

        self.model = model

        # define loss function (criterion) and optimizer
        # set loss
        self.task = config.get("training::task", "classification")
        self.logsoftmax = False
        if self.task == "classification":
            if config.get("training::loss", "nll"):
                self.criterion = nn.NLLLoss()
                self.logsoftmax = True
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif self.task == "regression":
            self.criterion = nn.MSELoss(reduction="mean")
        if self.cuda:
            self.criterion.cuda()

        # set opimizer
        self.set_optimizer(config)

        self.set_scheduler(config)

        self.best_epoch = None
        self.loss_best = None

    # module forward function
    def forward(self, x):
        # compute model prediction
        y = self.model(x)
        if self.logsoftmax:
            y = F.log_softmax(y, dim=1)
        return y

    # set optimizer function - maybe we'll only use one of them anyways..
    def set_optimizer(self, config):
        if config["optim::optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config["optim::lr"],
                momentum=config["optim::momentum"],
                weight_decay=config["optim::wd"],
                nesterov=config.get("optim::nesterov", False),
            )
        if config["optim::optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
            )
        if config["optim::optimizer"] == "rms_prop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
                momentum=config["optim::momentum"],
            )

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == "OneCycleLR":
            print("use onecycleLR scheduler")
            max_lr = config["optim::lr"]
            steps_per_epoch = config["scheduler::steps_per_epoch"]
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_lr,
                epochs=config["training::epochs_train"],
                steps_per_epoch=steps_per_epoch,
            )

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            # print(fname)
            perf_dict["state_dict"] = self.model.state_dict()
            torch.save(perf_dict, fname)
        return None

    def compute_mean_loss(self, dataloader):
        # step 1: compute data mean
        # get output data
        if self.verbosity > 5:
            print(f"len(dataloader): {len(dataloader)}")

        # get shape of data
        for idx, (_, target) in enumerate(dataloader):
            # unsqueeze scalar targets for compatibility
            if len(target.shape) == 1:
                target = target.unsqueeze(dim=1)
            x_mean = torch.zeros(target.shape[1])
            break

        if self.verbosity > 5:
            print(f"x_mean.shape: {x_mean.shape}")
        n_data = 0
        # collect mean
        for idx, (_, target) in enumerate(dataloader):
            # unsqueeze scalar targets for compatibility
            if len(target.shape) == 1:
                target = target.unsqueeze(dim=1)
            # compute mean weighted with batch size
            n_data += target.shape[0]
            x_mean += target.mean(dim=0) * target.shape[0]

        # scale x_mean back
        x_mean /= n_data
        if self.verbosity > 5:
            print(f"x_mean = {x_mean}")
        n_data = 0
        loss_mean = 0
        # collect loss
        for idx, (_, target) in enumerate(dataloader):
            # unsqueeze scalar targets for compatibility
            if len(target.shape) == 1:
                target = target.unsqueeze(dim=1)
            # compute mean weighted with batch size
            n_data += target.shape[0]
            # broadcast x_mean to target shape
            target_mean = torch.zeros(target.shape).add(x_mean)
            # commpute loss
            loss_batch = self.criterion(target, target_mean)
            # add and weight
            loss_mean += loss_batch.item() * target.shape[0]
        # scale back
        loss_mean /= n_data

        # compute mean
        self.loss_mean = loss_mean
        if self.verbosity > 5:
            print(f" mean loss: {self.loss_mean}")

        return self.loss_mean

    # one training step / batch
    def train_step(self, input, target):
        # zero grads before training steps
        self.optimizer.zero_grad()
        # compute pde residual
        output = self.forward(input)
        # realign target dimensions
        if self.task == "regression":
            target = target.view(output.shape)
            # compute loss
        loss = self.criterion(output, target)
        # prop loss backwards to
        loss.backward()
        # update parameters
        self.optimizer.step()
        # scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        # compute correct
        correct = 0
        if self.task == "classification":
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
        return loss.item(), correct

    # one training epoch
    def train_epoch(self, trainloader, epoch, idx_out=10):
        if self.verbosity > 2:
            print(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()

        if self.verbosity > 2:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        # init accumulated loss, accuracy
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        #
        if self.verbosity > 4:
            start = timeit.default_timer()

        # enter loop over batches
        for idx, data in enumerate(trainloader):
            input, target = data
            # send to cuda
            if self.cuda:
                input, target = input.cuda(), target.cuda()

            # take one training step
            if self.verbosity > 2:
                printProgressBar(
                    idx + 1,
                    len(trainloader),
                    prefix="Batch Progress:",
                    suffix="Complete",
                    length=50,
                )
            loss, correct = self.train_step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
            # logging
            if idx > 0 and idx % idx_out == 0:
                loss_running = loss_acc / n_data
                if self.task == "classification":
                    accuracy = correct_acc / n_data
                elif self.task == "regression":
                    # use r2
                    accuracy = 1 - loss_running / self.loss_mean

                if self.verbosity > 1:
                    print(
                        f"epoch {epoch} -batch {idx}/{len(trainloader)} --- running ::: loss: {loss_running}; accuracy: {accuracy} "
                    )

        if self.verbosity > 4:
            end = timeit.default_timer()
            print(f"training time for epoch {epoch}: {end-start} seconds")

        self.model.eval()
        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.task == "classification":
            accuracy = correct_acc / n_data
        elif self.task == "regression":
            # use r2
            accuracy = 1 - loss_running / self.loss_mean
        return loss_running, accuracy

    # test batch
    def test_step(self, input, target):
        with torch.no_grad():
            # forward pass: prediction
            output = self.forward(input)
            # realign target dimensions
            if self.task == "regression":
                target = target.view(output.shape)
            # compute loss
            loss = self.criterion(output, target)
            correct = 0
            if self.task == "classification":
                # compute correct
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()
            return loss.item(), correct

    # test epoch
    def test_epoch(self, testloader, epoch):
        if self.verbosity > 1:
            print(f"validate at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # initilize counters
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        for idx, data in enumerate(testloader):
            input, target = data
            # send to cuda
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            # take one training step
            loss, correct = self.test_step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
        # logging
        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.task == "classification":
            accuracy = correct_acc / n_data
        elif self.task == "regression":
            # use r2
            accuracy = 1 - loss_running / self.loss_mean
        if self.verbosity > 1:
            print(f"test ::: loss: {loss_running}; accuracy: {accuracy}")

        return loss_running, accuracy
    
    @torch.no_grad()
    def accs_and_incorrect_indices(self, testloader):
        self.model.eval()
        correct_acc = 0
        n_data = 0
        incorrectness = []
        
        for _, data in enumerate(testloader):
            input, target = data
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            output = self.forward(input)
            _, predicted = torch.max(output.data, 1)
            incorrectness.extend((predicted != target).cpu().numpy().tolist())
            correct_acc += (predicted == target).sum().item()
            n_data += len(target)
        
        accuracy = correct_acc / n_data
        return accuracy, incorrectness

    # test batch
    def _step(self, input, target):
        # forward pass: prediction
        output = self.forward(input)
        # realign target dimensions
        if self.task == "regression":
            target = target.view(output.shape)
        # compute loss
        loss = self.criterion(output, target)
        correct = 0
        if self.task == "classification":
            # compute correct
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
        return loss, correct

    # test epoch
    def _eval(self, testloader):
        if self.verbosity > 1:
            print(f"validate at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # initilize counters
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        for idx, data in enumerate(testloader):
            input, target = data
            # send to cuda
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            # take one training step
            loss, correct = self._step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
        # logging
        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.task == "classification":
            accuracy = correct_acc / n_data
        elif self.task == "regression":
            # use r2
            accuracy = 1 - loss_running / self.loss_mean
        if self.verbosity > 1:
            print(f"test ::: loss: {loss_running}; accuracy: {accuracy}")

        return loss_running, accuracy

    def compute_confusion_matrix(self, testloader, nb_classes):
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(testloader):
                if self.cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                output = self.model(inputs)
                _, preds = torch.max(output.data, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        # print(confusion_matrix)

        return confusion_matrix

    # training loop over all epochs
    def train_loop(self, config, tune=False):
        if self.verbosity > 0:
            print("##### enter training loop ####")

        # unpack training_config
        batchsize = config["training::batchsize"]
        epochs_train = config["training::epochs_train"]
        start_epoch = config["training::start_epoch"]
        output_epoch = config["training::output_epoch"]
        val_epochs = config["training::val_epochs"]
        idx_out = config["training::idx_out"]
        checkpoint_dir = config["training::checkpoint_dir"]

        trainloader = config["training::trainloader"]
        testloader = config["training::testloader"]
        # dataloader

        if self.task == "regression":
            self.compute_mean_loss(testloader)

        perf_dict = {
            "train_loss": 1e15,
            "train_accuracy": 0.0,
            "test_loss": 1e15,
            "test_accuracy": 0.0,
        }
        self.save_model(epoch=0, perf_dict=perf_dict, path=checkpoint_dir)
        self.best_epoch = 0
        self.loss_best = 1e15

        # initialize the epochs list
        epoch_iter = range(start_epoch, start_epoch + epochs_train)
        # enter training loop
        for epoch in epoch_iter:

            # enter training loop over all batches
            loss, accuracy = self.train_epoch(trainloader, epoch, idx_out=idx_out)

            if epoch % val_epochs == 0:
                loss_test, accuracy_test = self.test_epoch(testloader, epoch)

                if loss_test < self.loss_best:
                    self.best_epoch = epoch
                    self.loss_best = loss_test
                    perf_dict["epoch"] = epoch
                    perf_dict["train_loss"] = loss
                    perf_dict["train_accuracy"] = accuracy
                    perf_dict["test_loss"] = loss_test
                    perf_dict["test_accuracy"] = accuracy_test
                    self.save_model(
                        epoch="best", perf_dict=perf_dict, path=checkpoint_dir
                    )
                if self.verbosity > 1:
                    print(f"best loss: {self.loss_best} at epoch {self.best_epoch}")

            if epoch % output_epoch == 0:
                perf_dict["train_loss"] = loss
                perf_dict["train_accuracy"] = accuracy
                perf_dict["test_loss"] = loss_test
                perf_dict["test_accuracy"] = accuracy_test
                self.save_model(epoch=epoch, perf_dict=perf_dict, path=checkpoint_dir)

        if self.verbosity > 0:
            print(f"best loss: {self.loss_best} at epoch {self.best_epoch}")
        return self.loss_best


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
