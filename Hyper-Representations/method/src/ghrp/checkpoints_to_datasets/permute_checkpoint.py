import torch


def permute_checkpoint(checkpoint, layer_lst, permute_layers, permutation_idxs_dct):
    """
    Permutes kernels / neurons in model given as torch checkpoint.
    All layers to be considered have to be indicated in layer_lst, with entries as tuple (index, "type"), e.g. (3,"conv2d")
    Layers to be permuted have to be indicated in permute_layers, a list with layer index.
    The new order/index per layer is given in permutation_idx_dct, with one entry per layer
    """
    ## compute kernel size
    layer_kernels = []
    for kdx in permute_layers:
        layer_type = [y for (x, y) in layer_lst if x == kdx][0]
        if layer_type == "conv2d":
            weights = checkpoint.get(f"module_list.{kdx}.weight", torch.empty(0))
            kernels = weights.shape[0]
            layer_kernels.append(kernels)
        elif layer_type == "fc":
            weights = checkpoint.get(f"module_list.{kdx}.weight", torch.empty(0))
            kernels = weights.shape[0]
            layer_kernels.append(kernels)
        else:
            print(
                f"permutations for layers of type {layer_type} are not yet implemented"
            )
            raise NotImplementedError

    # apply permutation
    for ldx, (layer, type) in enumerate(layer_lst):
        # load input data and label from self.data
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", None)

        # check if permutations are applied data
        if layer in permute_layers:
            # get type of permuted layer
            # layer_type = [y for (x, y) in layer_lst if x == layer]
            kdx = permute_layers.index(layer)
            # create index 0,...,n-1
            index_old = list(range(layer_kernels[kdx]))

            # get type and data of following layer
            try:
                (layer_next, layer_next_type) = layer_lst[ldx + 1]
            except Exception as e:
                print(e)
                print(
                    f"permuting layer {ldx}, there was en error loading the following layer. your probably trying to permute the last layer, which doesn't work."
                )
                continue
            # load next layers weights
            weight_next = checkpoint.get(
                f"module_list.{layer_next}.weight", torch.empty(0)
            )

            ## permute current layer
            # get new index of layer and get permutation
            index_new = permutation_idxs_dct[f"layer_{layer}"]
            if type == "conv2d":
                # create new input cnn_layer
                weight_new = torch.zeros_like(weight)
                weight_new[index_old, :, :, :] = weight[index_new, :, :, :]
                if bias is not None:
                    bias_new = torch.zeros_like(bias)
                    bias_new = bias[index_new]
                # create new output cnn_layer
            elif type == "fc":
                # permute by first axis
                # input
                weight_new = torch.zeros_like(weight)
                weight_new = weight[index_new]
                if bias is not None:
                    bias_new = torch.zeros_like(bias)
                    bias_new = bias[index_new]
                # output
            else:
                raise NotImplementedError
                break

            # permute following layer with transposed
            # permute followup layer also by input channels. (weights only, bias only affects output channels...)
            if layer_next_type == "conv2d":
                # permute followup layer 2nd axis
                # input
                weight_next_new = torch.zeros_like(weight_next)
                weight_next_new[:, index_old, :, :] = weight_next[:, index_new, :, :]
            elif layer_next_type == "fc":
                # fc input dimensions correspond to channels
                if weight_next.shape[1] == layer_kernels[kdx]:
                    weight_next_new = torch.zeros_like(weight_next)
                    # permute by second axis
                    weight_next_new[:, index_old] = weight_next[:, index_new]

                else:
                    weight_next_new = torch.zeros_like(weight_next)
                    # assume: output of conv2d is flattened.
                    # flatting happens within channels first.
                    # channel outputs must be devider of fc dim[1]
                    assert (
                        weight_next.shape[1] % layer_kernels[kdx] == 0
                    ), "devider must be of type integer, dimensions don't add up"

                    fc_block_length = int(
                        int(weight_next.shape[1]) / int(layer_kernels[kdx])
                    )
                    # iterate over blocks and change indices accordingly
                    for idx_old, idx_new in zip(index_old, index_new):
                        for fcdx in range(fc_block_length):
                            offset_old = idx_old * fc_block_length + fcdx
                            offset_new = idx_new * fc_block_length + fcdx
                            weight_next_new[:, offset_old] = weight_next[:, offset_new]

                # input
            else:
                raise NotImplementedError
                break

            # update weights in checkpoint
            checkpoint[f"module_list.{layer}.weight"] = weight_new
            if bias is not None:
                checkpoint[f"module_list.{layer}.bias"] = bias_new
            # overwrite next layers weights
            checkpoint[f"module_list.{layer_next}.weight"] = weight_next_new

    return checkpoint
