"""
# Defines Function to map checkpoints to unique subspace of parameter space
"""

import torch
import numpy as np

from ghrp.checkpoints_to_datasets.permute_checkpoint import permute_checkpoint


def sort_layers_checkpoint(
    checkpoint,
    layer_lst,
    permute_layers,
    permutation_idxs_dct=None,
    mode="discover",
    metric="absolute",
):

    if mode == "discover":
        # discover unique mapping, i.e. of initial checkpoint
        permutation_idxs_dct = {}
        # sort by ascending weights
        # get layer kernels
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
        # iterate over layers
        for ldx, (layer, type) in enumerate(layer_lst):
            # load input data and label from self.data
            weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
            bias = checkpoint.get(f"module_list.{layer}.bias", None)
            if layer in permute_layers:
                # get type of permuted layer
                # layer_type = [y for (x, y) in layer_lst if x == layer]
                kdx = permute_layers.index(layer)
                # create index 0,...,n-1
                index_old = list(range(layer_kernels[kdx]))
                if metric == "absolute":
                    if type == "fc":
                        # change over to numpy for sorting
                        W_np = weight.numpy()
                        if bias is not None:
                            b_np = bias.numpy()
                        index_np = np.array(index_old)
                        # add biases and index as column to end of Weight matrix
                        if bias is not None:
                            W_np = np.insert(W_np, W_np.shape[1], b_np, axis=1)
                        W_np = np.insert(W_np, W_np.shape[1], index_np, axis=1)
                        # Sorting numpy arrays requires
                        # get list / str of types in b | this goes columnwise
                        typelist = [W_np.dtype.name for _ in range(W_np.shape[1])]
                        # make str of typelist
                        type_str = str(typelist[0])
                        for tdx in range(1, len(typelist)):
                            type_str += ","
                            type_str += typelist[tdx]
                        # get field names | this goes columnwise
                        # let's for now assume they are all f{running number}
                        order_lst = [f"f{wdx}" for wdx in range(W_np.shape[1])]
                        # do actual ordering
                        W_np_ord = np.sort(
                            W_np.view(type_str), order=order_lst[:-1], axis=0
                        )
                        # get new indices, last column of newly ordered matrix
                        index_new = torch.tensor(W_np_ord[order_lst[-1]].astype("int"))
                        index_new_lst = [
                            int(index_new[ldx]) for ldx in range(index_new.shape[0])
                        ]
                    elif type == "conv2d":
                        # change over to numpy for sorting
                        # we'll ignore the biases for now. shapes are unclear.
                        # TODO: include biases
                        # pack everything in rows
                        temp = weight.view(weight.shape[0], -1)
                        W_list = []
                        for idx in range(temp.shape[0]):
                            line = []
                            for jdx in range(temp.shape[1]):
                                line.append(temp[idx, jdx].item())
                            # append index
                            line.append(idx)
                            W_list.append(tuple(line))
                        # get last value of each tuple in sorted W_list, which is the index
                        index_new_lst = [x[-1] for x in sorted(W_list)]
                        # print(f"layer {layer}")
                        # print("index_old")
                        # print(index_old)
                        # print("index_new")
                        # print(index_new_lst)

                    else:
                        raise NotImplementedError
                elif metric == "l2":
                    # concat all relevant values for each kernel
                    if bias is None:
                        temp = weight.view(weight.shape[0], -1)
                    else:
                        temp = torch.cat(
                            [weight.view(weight.shape[0], -1), bias.unsqueeze(dim=1)],
                            dim=1,
                        )
                    # compute norm of each kernel
                    l2_norms = torch.norm(temp, dim=1, p=2)
                    # iterate over outgoing kernels
                    norms = []
                    for idx in range(l2_norms.shape[0]):
                        # add index
                        norms.append((l2_norms[idx], idx))
                    # sort by norm, retrieve only index - sort reverse :)
                    index_new_lst = [x[-1] for x in sorted(norms, reverse=True)]
                # save new index order in permutation_idxs_dct
                permutation_idxs_dct[f"layer_{layer}"] = index_new_lst
                # apply permutation to that layer, before figuring out the order for the next one...
                # therefore, permute_layer = [layer]
                checkpoint = permute_checkpoint(
                    checkpoint.copy(), layer_lst, [layer], permutation_idxs_dct
                )
        chkpt_sorted = checkpoint
    else:
        # apply given transformation to other checkpoints
        # trafo is indicated in permutation_idxs_dct
        chkpt_sorted = permute_checkpoint(
            checkpoint.copy(), layer_lst, permute_layers, permutation_idxs_dct
        )

    return chkpt_sorted, permutation_idxs_dct
