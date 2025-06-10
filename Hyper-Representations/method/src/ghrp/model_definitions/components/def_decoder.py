import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, dim, nlayers, nlin, dropout):
        super().__init__()

        self.resblockList = nn.ModuleList()

        for ldx in range(nlayers - 1):
            self.resblockList.append(nn.Linear(dim, dim, bias=True))
            # add nonlinearity
            if nlin == "elu":
                self.resblockList.append(nn.ELU())
            if nlin == "celu":
                self.resblockList.append(nn.CELU())
            if nlin == "gelu":
                self.resblockList.append(nn.GELU())
            if nlin == "leakyrelu":
                self.resblockList.append(nn.LeakyReLU())
            if nlin == "relu":
                self.resblockList.append(nn.ReLU())
            if nlin == "tanh":
                self.resblockList.append(nn.Tanh())
            if nlin == "sigmoid":
                self.resblockList.append(nn.Sigmoid())
            if dropout > 0:
                self.resblockList.append(nn.Dropout(dropout))
        # init output layer
        self.resblockList.append(nn.Linear(dim, dim, bias=True))
        # add output nonlinearity
        if nlin == "elu":
            self.nonlin_out = nn.ELU()
        if nlin == "celu":
            self.nonlin_out = nn.CELU()
        if nlin == "gelu":
            self.nonlin_out = nn.GELU()
        if nlin == "leakyrelu":
            self.nonlin_out = nn.LeakyReLU()
        if nlin == "tanh":
            self.nonlin_out = nn.Tanh()
        if nlin == "sigmoid":
            self.nonlin_out = nn.Sigmoid()
        else:  # relu
            self.nonlin_out = nn.ReLU()

    def forward(self, x):
        # clone input
        x_inp = x.clone()
        # forward prop through res block
        for m in self.resblockList:
            x = m(x)
        # add input and new x together
        y = self.nonlin_out(x + x_inp)
        return y


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        # load config
        res_blocks = config.get("model::res_blocks", 0)
        res_block_lays = config.get("model::res_block_lays", 0)
        h_layers = config.get("model::h_layers", 1)
        i_dim = config.get("model::i_dim", (14 * 14) * 10 + 10 * 10)
        latent_dim = config.get("model::latent_dim", 10)
        transition = config.get("model::transition", "lin")
        nlin = config.get("model::nlin", "leakyrelu")
        dropout = config.get("model::dropout", 0.2)
        init_type = config.get("model::init_type", "uniform")
        self.init_type = init_type

        # set flag for residual blocks
        self.res = False
        if res_blocks > 0 and res_block_lays > 0:
            self.res = True

        if self.res:
            # start with encoder resblock
            # decoder resblock - just the same
            self.resDecoder = nn.ModuleList()
            for _ in range(res_blocks):
                self.resDecoder.append(
                    ResBlock(
                        dim=i_dim, nlayers=res_block_lays, nlin=nlin, dropout=dropout
                    )
                )

        # get array of dimensions (encoder, decoder is reverse)
        if transition == "lin":
            dimensions = np.linspace(i_dim, latent_dim, h_layers + 2).astype("int")
        else:
            raise NotImplementedError

        # decoder
        self.decoder = nn.ModuleList()
        # start with last entry and go up to second
        for idx, _ in enumerate(dimensions[2:]):
            self.decoder.append(nn.Linear(dimensions[-idx - 1], dimensions[-idx - 2]))
            # add nonlinearity
            if nlin == "elu":
                self.decoder.append(nn.ELU())
            if nlin == "celu":
                self.decoder.append(nn.CELU())
            if nlin == "gelu":
                self.decoder.append(nn.GELU())
            if nlin == "leakyrelu":
                self.decoder.append(nn.LeakyReLU())
            if nlin == "relu":
                self.decoder.append(nn.ReLU())
            if nlin == "tanh":
                self.decoder.append(nn.Tanh())
            if nlin == "sigmoid":
                self.decoder.append(nn.Sigmoid())
            if dropout > 0:
                self.decoder.append(nn.Dropout(dropout))
        self.decoder.append(nn.Linear(dimensions[1], dimensions[0]))

        # initialize weights with se methods
        print("initialze decoder")
        self.decoder = self.initialize_weights(self.decoder)
        if self.res:
            self.resDecoder = self.initialize_weights(self.resDecoder)

    def initialize_weights(self, module_list):
        for m in module_list:
            if type(m) == nn.Linear:
                if self.init_type == "xavier_uniform":
                    nn.init.xavier_uniform(m.weight)
                if self.init_type == "xavier_normal":
                    nn.init.xavier_normal(m.weight)
                if self.init_type == "uniform":
                    nn.init.uniform(m.weight)
                if self.init_type == "normal":
                    nn.init.normal(m.weight)
                if self.init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight)
                if self.init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)
        return module_list

    def forward(self, x):
        # forward prop through module_list
        for layer in self.decoder:
            x = layer(x)
        # forward prop through resDecoder
        if self.res:
            for resblock in self.resDecoder:
                x = resblock(x)
        return x
