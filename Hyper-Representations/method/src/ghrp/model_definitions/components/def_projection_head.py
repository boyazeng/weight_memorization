import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, config):

        super(ProjectionHead, self).__init__()

        # get configuration
        self.no_layers = config.get("model::projection_head_layers", None)
        if not self.no_layers:
            print(
                "Warning: creating projection head w/o information on layers. using default..."
            )
            self.no_layers = 2
        self.batchnorm = config.get("model::projection_head_batchnorm", -999)
        if self.batchnorm == -999:
            print(
                "Warning: creating projection head w/o information on batchnorm. using default..."
            )
            self.batchnorm = True

        # self.idim = config.get("model::projection_head_idim", None)
        self.idim = config.get("model::latent_dim")
        if not self.idim:
            print(
                "Warning: creating projection head w/o information on layers. this won't work..."
            )
            raise NotImplementedError
        self.hdim = config.get("model::projection_head_hdim", None)
        if not self.hdim:
            print(
                "Warning: creating projection head w/o information on hidden dimension. using default..."
            )
            self.hdim = self.idim
        self.odim = config.get("model::projection_head_odim", None)
        if not self.odim:
            print(
                "Warning: creating projection head w/o information on layers. using default..."
            )
            self.odim = self.idim // 2
        self.init_type = config.get("model::init_type", "kaiming_normal")

        # create module list
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(nn.Linear(in_features=self.idim, out_features=self.hdim))
        if self.batchnorm:
            self.layers.append(nn.BatchNorm1d(num_features=self.hdim))

        self.layers.append(nn.ReLU())
        # hidden layers
        for _ in range(self.no_layers - 1):
            self.layers.append(nn.Linear(in_features=self.hdim, out_features=self.hdim))
            if self.batchnorm:
                self.layers.append(nn.BatchNorm1d(num_features=self.hdim))
            self.layers.append(nn.ReLU())

        # last layer
        self.layers.append(nn.Linear(in_features=self.hdim, out_features=self.odim))
        self.layers.append(nn.BatchNorm1d(num_features=self.odim))

        # initialize weights with se methods
        print("initialze projection head")
        self.layers = self.initialize_weights(self.layers)

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
        # forward prop through projection head
        for layer in self.layers:
            x = layer(x)
        return x
