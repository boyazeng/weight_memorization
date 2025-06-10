import torch
from torch.utils.data import Dataset

from pathlib import Path
import random
import copy

import itertools
from math import factorial


from ghrp.checkpoints_to_datasets.dataset_base import ModelDatasetBase
from ghrp.checkpoints_to_datasets.permute_checkpoint import permute_checkpoint
from ghrp.checkpoints_to_datasets.map_to_canonical import sort_layers_checkpoint
from ghrp.checkpoints_to_datasets.dataset_auxiliaries import (
    get_net_epoch_lst_from_label,
    get_net_epoch_from_label,
    vectorize_checkpoint,
    add_noise_to_checkpoint,
    printProgressBar,
    vector_to_checkpoint,
)
from ghrp.checkpoints_to_datasets.random_erasing import RandomErasingVector
import ray
from ghrp.checkpoints_to_datasets.progress_bar import ProgressBar

#####################################################################
# Define Dataset class
# as other permute datste
# has function: to_tensor(self) that returns the tensors
#####################################################################
class SimCLRDataset(ModelDatasetBase):
    """
    This class inherits from the base ModelDatasetBase class.
    It extends it by permutations of the dataset in the init function.
    """

    # init
    def __init__(
        self,
        root,  # path from which to load the dataset
        layer_lst=[
            (0, "conv2d"),
            (3, "conv2d"),
            (6, "fc"),
        ],  # details on model composition, depends on model checkpoints
        epoch_lst=10,  # list of epochs to load
        mode="vectorize",  # "vector", "vectorize", "checkpoint" | vector: models are vectorized at init. vectorize: models are vectorized at __getitem__
        permutations_number=10, # number of precomputed permutations
        permute_layers=[0, 2], # layers to permute
        permutation_mode="complete",  # "random" # permute mode. caution: 'complete' becomes intractable quickly...
        view_1_canonical=False, # maps view1 to 'canoncial representation'
        view_2_canonical=False, # maps view2 to 'canoncial representation'
        map_2_canon_metric="absolute", # metric for canoncial represnetation
        add_noise_input=[False],  # set level of noise augmentatino 0.15
        add_noise_output=[False],  # 0.15
        erase_augment=None,  # set erasing augmentation parameters {"p": 0.5,"scale":(0.02,0.33),"value":0,"mode":"block"}
        erase_input=None,  # use this to catch wrong parameters
        use_bias=False, # set whether to load  model biases as well
        train_val_test="train", # set dataset split
        ds_split=[0.7, 0.3], # sets ration between [train, test] or [train, val, test]
        weight_threshold=float("inf"), # set weight threshold. samples are filtered out if one weight has higher absolute value
        max_samples=None,  # limit the number of models to integer number (full model trajectory, all epochs)
        filter_function=None,  # gets sample path as argument and returns True if model needs to be filtered out
        property_keys=None, # keys of properties to load
        num_threads=4,
        verbosity=0,
    ):
        # call init of base class
        super().__init__(
            root=root,
            layer_lst=layer_lst,
            epoch_lst=epoch_lst,
            mode="checkpoint",
            task="reconstruction",
            use_bias=use_bias,
            train_val_test=train_val_test,
            ds_split=ds_split,
            weight_threshold=weight_threshold,
            max_samples=max_samples,
            filter_function=filter_function,
            property_keys=property_keys,
            num_threads=num_threads,
            verbosity=verbosity,
        )
        self.mode = mode
        self.permutations_number = permutations_number
        self.permute_layers = permute_layers
        self.permutation_mode = permutation_mode
        self.add_noise_input = add_noise_input
        self.add_noise_output = add_noise_output
        self.num_threads = num_threads

        self.view_1_canonical = view_1_canonical
        self.view_2_canonical = view_2_canonical
        self.map_2_canon_metric = map_2_canon_metric
        if view_1_canonical and view_2_canonical:
            print(
                f"both view 1 and view 2 are set to canonical. number of permutations is set to 0"
            )
            self.permutations_number = 0

        # set erase augmnet
        self.set_erase(erase_augment)
        if erase_input is not None:
            self.set_erase(erase_input)

        ### initialize permutations ##########################################################################################################################################
        # list of permutations (list of list with indexes)
        if self.permutations_number > 0:
            print("init permutations")
            self.init_permutations()
            print("compute all possible permutation")
            self.get_permutation_map()
            print("prepare permutation dicts")
            self.prepare_permutations_dct_list()

        if self.view_1_canonical or self.view_1_canonical:
            print("prepare canonical form")
            self.map_data_to_canonical()

        if self.mode == "vector":
            print("vectorize data")
            self.vectorize_data()
            if self.permutations_number > 0:
                print("precompute full permutation indices")
                self.precompute_permutation_index()

        ######################################################################################################################################################################

    ## getitem ####################################################################################################################################################################
    def __getitem__(self, index):
        """function call differs depending on self.mode"""
        # get permutation index -> pick random number from available perms
        if self.permutations_number > 0:
            perm_idx, perm_jdx = random.choices(
                list(range(self.permutations_number)), k=2
            )

        ## mode "vector": data_in is alrady a tensor. Augmentations are pre-computed and can be cheaply applied
        if self.mode == "vector":
            # get raw data
            ddx_in = self.data_in[index]
            ddx_in = copy.deepcopy(ddx_in)
            label_in = self.labels_in[index]
            label_in = copy.deepcopy(label_in)

            # permutation
            if self.view_1_canonical:
                ddx_in_idx = self.data_canon[index]
                ddx_in_idx = copy.deepcopy(ddx_in_idx)
                label_in_idx = f"{label_in}#_#canon"
            elif self.permutations_number > 0:
                index_p_idx = self.permutation_index_list[perm_idx]
                ddx_in_idx = ddx_in[index_p_idx]
                label_in_idx = f"{label_in}#_#per_{perm_idx}"
            else:
                ddx_in_idx = copy.deepcopy(ddx_in)
                label_in_idx = copy.deepcopy(label_in)

            # permute data jdx
            if self.view_2_canonical:
                ddx_in_jdx = self.data_canon[index]
                ddx_in_jdx = copy.deepcopy(ddx_in_jdx)
                label_in_jdx = f"{label_in}#_#canon"
            elif self.permutations_number > 0:
                index_p_jdx = self.permutation_index_list[perm_jdx]
                ddx_in_jdx = ddx_in[index_p_jdx]
                label_in_jdx = f"{label_in}#_#per_{perm_jdx}"
            else:
                ddx_in_jdx = copy.deepcopy(ddx_in)
                label_in_jdx = copy.deepcopy(label_in)

            # noise
            if not self.add_noise_input == False:
                # check sigma is number
                assert isinstance(self.add_noise_input, float)
                # add noise to input
                # check sigma is larger than 0
                if self.add_noise_input > 0:
                    if self.use_multiplicative_noise:
                        # noise idx
                        noise = 1.0 + self.add_noise_input * torch.randn(ddx_in.shape)
                        ddx_in_idx *= noise
                        # noise jdx
                        noise = 1.0 + self.add_noise_input * torch.randn(ddx_in.shape)
                        ddx_in_jdx *= noise
                    else:
                        # noise idx
                        noise = self.add_noise_input * torch.randn(ddx_in.shape)
                        ddx_in_idx += noise
                        # noise jdx
                        noise = self.add_noise_input * torch.randn(ddx_in.shape)
                        ddx_in_jdx += noise

            # erase_input/output augmentation
            if self.erase_augment is not None:
                ddx_in_idx = self.erase_augment(ddx_in_idx)
                ddx_in_jdx = self.erase_augment(ddx_in_jdx)

            return ddx_in_idx, label_in_idx, ddx_in_jdx, label_in_jdx
        ### end mode=="vector"

        ### else: either mode vectorize, or mode checkpoint. Both cases: augmentations are applied on checkpoint

        # view 1
        if self.view_1_canonical:
            chkpoint_idx_in = self.data_canon[index]
            chkpoint_idx_in = copy.deepcopy(chkpoint_idx_in)
            label_idx_in = f"{self.labels_in[index]}#_#canon"
        else:
            (
                chkpoint_idx_in,
                label_idx_in,
                chkpoint_idx_out,
                label_idx_out,
            ) = self.permute_single_sample(
                chkpt_in=copy.deepcopy(self.data_in[index]),
                lab_in=self.labels_in[index],
                chkpt_out=copy.deepcopy(self.data_out[index]),
                lab_out=self.labels_out[index],
                pdx=perm_idx,
            )

        # view 2
        if self.view_1_canonical:
            chkpoint_jdx_in = self.data_canon[index]
            chkpoint_jdx_in = copy.deepcopy(chkpoint_jdx_in)
            label_jdx_in = f"{self.labels_in[index]}#_#canon"
        else:
            (
                chkpoint_jdx_in,
                label_jdx_in,
                chkpoint_jdx_out,
                label_jdx_out,
            ) = self.permute_single_sample(
                chkpt_in=copy.deepcopy(self.data_in[index]),
                lab_in=self.labels_in[index],
                chkpt_out=copy.deepcopy(self.data_out[index]),
                lab_out=self.labels_out[index],
                pdx=perm_jdx,
            )

        # add noise to input
        # if "input" in self.add_noise:
        if not self.add_noise_input == False:
            if self.use_multiplicative_noise is not False:
                raise NotImplementedError(
                    "Multiplicative noise (Adrian) only implemented for mode 'vector'!"
                )
            # check sigma is number
            assert isinstance(self.add_noise_input, float)
            # add noise to input
            # check sigma is larger than 0
            if self.add_noise_input > 0:
                chkpoint_idx_in = add_noise_to_checkpoint(
                    copy.deepcopy(chkpoint_idx_in),
                    layer_lst=self.layer_lst,
                    use_bias=self.use_bias,
                    sigma=self.add_noise_input,
                )
                chkpoint_jdx_in = add_noise_to_checkpoint(
                    copy.deepcopy(chkpoint_jdx_in),
                    layer_lst=self.layer_lst,
                    use_bias=self.use_bias,
                    sigma=self.add_noise_input,
                )
        if not self.add_noise_output == False:
            if self.use_multiplicative_noise is not False:
                raise NotImplementedError(
                    "Multiplicative noise (Adrian) only implemented for mode 'vector'!"
                )
            # check sigma is number
            assert isinstance(self.add_noise_output, float)
            # add noise to input
            # check sigma is larger than 0
            if self.add_noise_output > 0:
                chkpoint_idx_out = add_noise_to_checkpoint(
                    copy.deepcopy(chkpoint_idx_out),
                    layer_lst=self.layer_lst,
                    use_bias=self.use_bias,
                    sigma=self.add_noise_output,
                )
                chkpoint_jdx_out = add_noise_to_checkpoint(
                    copy.deepcopy(chkpoint_jdx_out),
                    layer_lst=self.layer_lst,
                    use_bias=self.use_bias,
                    sigma=self.add_noise_output,
                )

        ## append layers
        if self.mode == "vectorize_full":
            ddx_idx_in = vectorize_checkpoint(
                chkpoint_idx_in, self.layer_lst, self.use_bias
            )
            ddx_idx_out = vectorize_checkpoint(
                chkpoint_idx_out, self.layer_lst, self.use_bias
            )
            ddx_jdx_in = vectorize_checkpoint(
                chkpoint_jdx_in, self.layer_lst, self.use_bias
            )
            ddx_jdx_out = vectorize_checkpoint(
                chkpoint_jdx_out, self.layer_lst, self.use_bias
            )
            # check erase_input/output augmentation
            if self.erase_augment is not None:
                ddx_idx_in = self.erase_augment(ddx_idx_in)
                ddx_idx_out = self.erase_augment(ddx_idx_out)
                ddx_jdx_in = self.erase_augment(ddx_jdx_in)
                ddx_jdx_out = self.erase_augment(ddx_jdx_out)

            return (
                ddx_idx_in,
                label_idx_in,
                ddx_idx_out,
                label_idx_out,
                ddx_jdx_in,
                label_jdx_in,
                ddx_jdx_out,
                label_jdx_out,
            )
        elif self.mode == "vectorize":
            ddx_idx_in = vectorize_checkpoint(
                chkpoint_idx_in, self.layer_lst, self.use_bias
            )
            ddx_jdx_in = vectorize_checkpoint(
                chkpoint_jdx_in, self.layer_lst, self.use_bias
            )
            # check erase_input/output augmentation
            if self.erase_augment is not None:
                ddx_idx_in = self.erase_augment(ddx_idx_in)
                ddx_jdx_in = self.erase_augment(ddx_jdx_in)
            return (
                ddx_idx_in,
                label_idx_in,
                ddx_jdx_in,
                label_jdx_in,
            )

        elif self.mode == "image":
            raise NotImplementedError
        elif self.mode == "checkpoint":
            return (
                chkpoint_idx_in,
                label_idx_in,
                chkpoint_idx_out,
                label_idx_out,
                chkpoint_jdx_in,
                label_jdx_in,
                chkpoint_jdx_out,
                label_jdx_out,
            )
        else:
            raise NotImplementedError

    ### len ##################################################################################################################################################################
    def __len__(self):
        return len(self.data_in)

    ## init permutations #####################################################################################################################################################
    def init_permutations(self):
        """
        This function creates self.permutations_dct, a dictionary with mappings for all permutations.
        it contains keys for all layers, with lists as values. the lists contain one mapping per permutation.
        """
        # dict of list for every layer, with lists of index permutations
        self.permutations_dct = {}

        # check # of kernels for first data entry
        self.layer_kernels = []
        for kdx in self.permute_layers:
            layer_type = [y for (x, y) in self.layer_lst if x == kdx][0]
            if layer_type == "conv2d":
                weights = self.data_in[0].get(
                    f"module_list.{kdx}.weight", torch.empty(0)
                )
                kernels = weights.shape[0]
                self.layer_kernels.append(kernels)
            elif layer_type == "fc":
                weights = self.data_in[0].get(
                    f"module_list.{kdx}.weight", torch.empty(0)
                )
                kernels = weights.shape[0]
                self.layer_kernels.append(kernels)
            else:
                print(
                    f"permutations for layers of type {layer_type} are not yet implemented"
                )
                raise NotImplementedError

        # add all possible permutations for all permutable layers
        for kdx, layer in enumerate(self.permute_layers):
            index_old = list(range(self.layer_kernels[kdx]))
            # initialize empty list
            self.permutations_dct[f"layer_{layer}"] = []
            # Mode 1: precompute all permutations
            if self.permutation_mode == "complete":
                # iterate over all complete combinations of index_old
                for index_new in itertools.permutations(index_old, len(index_old)):
                    # append list of new index to list per layer
                    self.permutations_dct[f"layer_{layer}"].append(list(index_new))
            elif self.permutation_mode == "random":
                # figure out layer size
                theoretical_permutations = factorial(len(index_old))
                no_perms_this_layer = min(
                    theoretical_permutations, self.permutations_number
                ) // len(self.permute_layers)
                print(
                    f"compute {no_perms_this_layer} random permutations for layer {kdx} - {layer}"
                )
                for pdx in range(no_perms_this_layer):
                    if no_perms_this_layer > 1000:
                        printProgressBar(iteration=pdx, total=no_perms_this_layer)
                    index_new = copy.deepcopy(index_old)
                    random.shuffle(index_new)
                    # append list of new index to list per layer
                    self.permutations_dct[f"layer_{layer}"].append(list(index_new))

    ### get permutation map #########################################################################################################################################################
    def get_permutation_map(self):
        """
        pre-computes permutations for all layers by iterating over the layers.
        """
        # Mode 1: precompute all permutations
        if self.permutation_mode == "complete":
            combination_lst = []
            # get #of permutations per layer
            for kdx, layer in enumerate(self.permute_layers):
                n_perms = len(self.permutations_dct[f"layer_{layer}"])
                index_kdx = list(range(n_perms))
                combination_lst.append(index_kdx)
            # get all combinations of permutation indices
            combinations = list(itertools.product(*combination_lst))
            # random shuffle combinations around
            random.shuffle(combinations)
            self.permutations_index = combinations
        # pick only random permutations from indices prepared
        elif self.permutation_mode == "random":
            combinations = []
            for pdx in range(self.permutations_number):
                # random pick index to permutation in perm_dict for each layer
                combination_single = []
                for kdx, layer in enumerate(self.permute_layers):
                    # pick random index list for that layer
                    n_perms = len(self.permutations_dct[f"layer_{layer}"])
                    index_kdx = random.choice(list(range(n_perms)))
                    combination_single.append(index_kdx)
                # append tuple to list
                combinations.append(tuple(combination_single))
            self.permutations_index = combinations
        print(f"prepared {len(combinations)} permutations")

    ### prep list of permutation dicts #########################################################################################################################################################
    def prepare_permutations_dct_list(self):
        """
        re-order the index in one stand-alone dict per permutation, so that the dicts don't have to be put together at runtime.
        the list get's an index and returns a dict with all necessary indices.
        """
        permutations_dct_lst = []
        # compute one dict for the number of wanted permutations
        for pdx in range(self.permutations_number):
            prmt_dct = {}
            for kdx, layer in enumerate(self.permute_layers):
                # get permutation index for permutation pdx and layer kdx
                permutation_idx = self.permutations_index[pdx][kdx]
                prmt_dct[f"layer_{layer}"] = self.permutations_dct[f"layer_{layer}"][
                    permutation_idx
                ]
            permutations_dct_lst.append(copy.deepcopy(prmt_dct))

        self.permutations_dct_lst = permutations_dct_lst

    ### permute_single_sample #########################################################################################################################################################
    def permute_single_sample(self, chkpt_in, lab_in, chkpt_out, lab_out, pdx):
        """
        perform permutation on one checkpoint
        """
        ## perform actual permutation ################
        # adapt label with permutation index
        lab_in_p = f"{lab_in}#_#per_{pdx}"
        lab_out_p = f"{lab_out}#_#per_{pdx}"

        # get perm dict
        prmt_dct = self.permutations_dct_lst[pdx]

        # apply permutation on input data
        chkpt_in_p = permute_checkpoint(
            copy.deepcopy(chkpt_in),
            self.layer_lst,
            self.permute_layers,
            prmt_dct,
        )
        # apply permutation on output data
        chkpt_out_p = permute_checkpoint(
            copy.deepcopy(chkpt_out),
            self.layer_lst,
            self.permute_layers,
            prmt_dct,
        )
        # append data to permuted list
        return chkpt_in_p, lab_in_p, chkpt_out_p, lab_out_p

    ### set erase ############################################################
    def set_erase(self, erase=None):
        """
        helper function to set the erase augmentation
        """
        if erase is not None:
            assert (
                self.mode == "vectorize" or self.mode == "vector"
            ), "erasing is only for vectorized mode implemenetd"
            erase = RandomErasingVector(
                p=erase["p"],
                scale=erase["scale"],
                value=erase["value"],
                mode=erase["mode"],
            )
        else:
            erase = None
        self.erase_augment = erase

    ### vectorize_data #########################################################################################################################################################
    def vectorize_data(self):
        """
        helper function to vectorize weight of all checkpoints in data_in
        """
        # save base checkpoint
        self.checkpoint_base = self.data_in[0]
        # iterate over length of dataset
        for idx in range(self.__len__()):
            checkpoint_in = copy.deepcopy(self.data_in[idx])
            checkpoint_out = copy.deepcopy(self.data_out[idx])
            ddx_in = vectorize_checkpoint(checkpoint_in, self.layer_lst, self.use_bias)
            ddx_out = vectorize_checkpoint(
                checkpoint_out, self.layer_lst, self.use_bias
            )
            self.data_in[idx] = ddx_in
            self.data_out[idx] = ddx_out

        if self.view_1_canonical or self.view_2_canonical:
            print("vectorize canonical forms")
            for idx in range(self.__len__()):
                checkpoint_canon = copy.deepcopy(self.data_canon[idx])
                ddx_canon = vectorize_checkpoint(
                    checkpoint_canon, self.layer_lst, self.use_bias
                )
                self.data_canon[idx] = ddx_canon

    ### precompute_permutation_index #########################################################################################################################################################
    def precompute_permutation_index(self):
        """
        main function to pre-compute permutations. 
        Idea: fill checkpoint with 0-based indices (index_vector)
        Apply all permutations on that checkpoint.
        vectorize 'weights'/indices to get permuted indices
        """
        # ASSUMES THAT DATA IS ALREADY VECTORIZED
        permutation_index_list = []
        # create index vector
        # print(f"vector shape: {self.data_in[0].shape}")
        index_vector = torch.tensor(list(range(self.data_in[0].shape[0])))
        # cast index vector to double
        index_vector = index_vector.double()
        # print(f"index vector: {index_vector}")
        # reference checkpoint
        reference_checkpoint = copy.deepcopy(self.checkpoint_base)
        # cast index vector to checkpoint
        index_checkpoint = vector_to_checkpoint(
            checkpoint=copy.deepcopy(reference_checkpoint),
            vector=copy.deepcopy(index_vector),
            layer_lst=self.layer_lst,
            use_bias=self.use_bias,
        )

        ## init multiprocessing environment ############
        ray.init(num_cpus=self.num_threads)

        ### gather data #############################################################################################
        print(f"preparing permutation indices from {self.root}")
        pb = ProgressBar(total=self.permutations_number)
        pb_actor = pb.actor

        # loop over all permutations in self.permutations_number
        for pdx in range(self.permutations_number):
            # get perm dict
            prmt_dct = self.permutations_dct_lst[pdx]
            #
            index_p = compute_single_index_vector_remote.remote(
                index_checkpoint=copy.deepcopy(index_checkpoint),
                prmt_dct=prmt_dct,
                layer_lst=self.layer_lst,
                permute_layers=self.permute_layers,
                use_bias=self.use_bias,
                pba=pb_actor,
            )
            # append to permutation_index_list
            permutation_index_list.append(index_p)

        # update progress bar
        pb.print_until_done()

        # collect actual data
        permutation_index_list = ray.get(permutation_index_list)

        ray.shutdown()

        self.permutation_index_list = permutation_index_list

    ### map data to canoncial #############################################################################################
    def map_data_to_canonical(self):
        """
        map data to canonical representation
        """
        data_canon = []

        ## init multiprocessing environment ############
        ray.init(num_cpus=self.num_threads)

        ### gather data #############################################################################################
        print(f"preparing computing canon form...")
        pb = ProgressBar(total=len(self.data_in))
        pb_actor = pb.actor

        for idx, checkpoint in enumerate(self.data_in):
            checkpoint_canon = compute_single_canon_form.remote(
                checkpoint=checkpoint,
                layer_lst=self.layer_lst,
                permute_layers=self.permute_layers,
                map_2_canon_metric=self.map_2_canon_metric,
                pba=pb_actor,
            )
            data_canon.append(checkpoint_canon)

        data_canon = ray.get(data_canon)
        ray.shutdown()
        #
        self.data_canon = data_canon


### helper parallel function #############################################################################################
@ray.remote(num_returns=1)
def compute_single_canon_form(
    checkpoint, layer_lst, permute_layers, map_2_canon_metric, pba
):
    """
    helper function to compute a single canonical form
    """
    checkpoint_canon, _ = sort_layers_checkpoint(
        checkpoint,
        layer_lst=layer_lst,
        permute_layers=permute_layers,
        permutation_idxs_dct=None,
        mode="discover",
        metric=map_2_canon_metric,
    )
    # update counter
    pba.update.remote(1)
    # return list
    return checkpoint_canon


### helper parallel function #############################################################################################
@ray.remote(num_returns=1)
def compute_single_index_vector_remote(
    index_checkpoint, prmt_dct, layer_lst, permute_layers, use_bias, pba
):
    """
    ray helper function for parallelization
    """
    # apply permutation on copy of unit checkpoint
    chkpt_p = permute_checkpoint(
        index_checkpoint,
        layer_lst,
        permute_layers,
        prmt_dct,
    )

    # cast back to vector
    vector_p = vectorize_checkpoint(copy.deepcopy(chkpt_p), layer_lst, use_bias)
    # cast vector back to int
    vector_p = vector_p.int()
    # we specifically don't check for uniqueness of indices. we'd rather let this run into index errors to catch the issue
    index_p = copy.deepcopy(vector_p.tolist())
    # update counter
    pba.update.remote(1)
    # return list
    return index_p
