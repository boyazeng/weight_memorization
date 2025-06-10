from ray.tune import Trainable
from ray.tune.utils import wait_for_gpu
import torch
import sys

import json

# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

import ghrp.checkpoints_to_datasets

# import model_definitions
from ghrp.model_definitions.def_simclr_ae_module import SimCLRAEModule

from ghrp.checkpoints_to_datasets.dataset_simclr import SimCLRDataset

from torch.utils.data import DataLoader

from ghrp.model_definitions.downstream_tasks.def_downstream_module import (
    DownstreamTaskLearner,
)

###############################################################################
# define Tune Trainable
###############################################################################
class SimCLR_AE_tune_trainable(Trainable):
    """
    This is the trainer class for hyper-representations. It relies on ray.tune and implements
    - setup: load data, model, initialize training setup
    - step: take one training step
    - save_checkpoint
    - load_checkpoint

    The model itself is implemented in 'def_simclr_ae_module.py'
    """

    def setup(self, config, data=None):
        # test sys_path
        # set trainable properties
        self.config = config
        self.seed = config["seed"]
        self.device = config["device"]

        # set loss weights before anything else
        self.set_loss_weights()

        #### GPU Resources
        #  figure out how much of the GPU to wait for
        resources = config.get("resources", None)
        if resources is not None:
            gpu_resource_share = resources["gpu"]
            # more than at least one gpu
            if gpu_resource_share > 1.0 - 1e-5:
                target_util = 0.01
            else:
                # set target util maximum full load minus share - buffer
                target_util = 1.0 - gpu_resource_share - 0.01
        else:
            target_util = 0.01
        # wait for gpu memory to be available
        if self.device == torch.device("cuda"):
            print("cuda detected: wait for gpu memory to be available")
            wait_for_gpu(gpu_id=None, target_util=target_util, retry=20, delay_s=5)

        ### get nidek cibfug
        # load config if restore from previous checkpoint
        if config.get("model::checkpoint_path", None):
            config_path = config.get("model::checkpoint_path", None).joinpath(
                "..", "params.json"
            )
            print(
                f"restore model from previous checkpoint. load config from {config_path}"
            )
            config_old = json.load(config_path.open("r"))
            # transfer all 'model' keys to
            for key in config_old.keys():
                if "model::" in key:
                    self.config[key] = config_old[key]

        #### MODEL
        if config.get("model::type", None) is None:
            # infer model type
            if self.config.get("model::encoding", None) is not None:
                print(
                    f"instanciate transformer encoder with {self.config.get('model::encoding')} encoding"
                )
                self.config["model::type"] = "transformer"
            else:
                print("instanciate vanilla encoder --- encoding is None")
                self.config["model::type"] = "vanilla"
                lr_factor = self.config.get("optim::vanilla_lr_factor", 1)
                self.config["optim::lr"] = self.config["optim::lr"] * lr_factor

        ## get latent_dim (if not set explicitly)
        if not self.config.get("model::latent_dim", None):
            # infer latent dim
            self.config["model::latent_dim"] = (
                self.config["perceiver::num_latents"]
                * self.config["perceiver::latent_dim"]
            )
        if self.config.get("model::attn_dim_ratio", None):
            self.config["model::attention_hidden_dim"] = (
                self.config["model::attn_dim_ratio"]
                * config["model::dim_attention_embedding"]
            )

        # init model
        self.SimCLR = SimCLRAEModule(self.config)

        # load checkpoint if restart
        if config.get("model::checkpoint_path", None):
            print(
                f'restore model state from {config.get("model::checkpoint_path",None)}'
            )
            # load all state dicts
            self.load_checkpoint(config.get("model::checkpoint_path", None))
            # reset optimizer
            self.SimCLR.set_optimizer(config)

        #### DATA
        # get dataset, either as argument or from file
        if data is not None:
            dataset = data
        else:
            dataset = torch.load(self.config["dataset::dump"])

        self.trainset = dataset["trainset"]
        self.testset = dataset["testset"]
        self.valset = dataset.get("valset", None)

        # set noise augmentation
        self.trainset.add_noise_input = self.config.get(
            "trainset::add_noise_input", 0.0
        )
        self.trainset.add_noise_output = self.config.get(
            "trainset::add_noise_output", 0.0
        )
        self.testset.add_noise_input = self.config.get("testset::add_noise_input", 0.0)
        self.testset.add_noise_output = self.config.get(
            "testset::add_noise_output", 0.0
        )
        # set permutation augmentation
        self.trainset.permutations_number = self.config["trainset::permutations_number"]
        self.testset.permutations_number = self.config["testset::permutations_number"]

        # set erase augmentation
        self.trainset.set_erase(self.config.get("trainset::erase_augment", None))
        self.testset.set_erase(self.config.get("testset::erase_augment", None))

        # dataloaders
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.config["trainset::batchsize"],
            shuffle=True,
            drop_last=True,  # important: we need equal batch sizes
            num_workers=self.config.get("trainloader::workers", 2),
        )

        # dataloaders
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.config["trainset::batchsize"],
            shuffle=False,
            drop_last=True,  # important: we need equal batch sizes
            num_workers=self.config.get("testloader::workers", 2),
        )
        if self.valset is not None:
            # dataloaders
            self.valloader = torch.utils.data.DataLoader(
                self.valset,
                batch_size=self.config["trainset::batchsize"],
                shuffle=False,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("testloader::workers", 2),
            )

        # set loss normalization (layer-wise loss norm init)
        if self.config.get("training::normalize_loss", False):
            weights_train = self.trainset.__get_weights__()
            index_dict = self.config["model::index_dict"]
            self.SimCLR.set_normalization(weights_train, index_dict)

        # set feature normalization (end to end)
        if self.config.get("model::feature_normalization", False):
            weights_train = self.trainset.__get_weights__()
            index_dict = self.config["model::index_dict"]
            self.SimCLR.set_feature_normalization(weights_train, index_dict)

        # compute loss_mean (for R^2 computation)
        weights_train = self.trainset.__get_weights__()
        self.SimCLR.criterion.set_mean_loss(weights_train)

        # save initial checkpoint
        # self.save() #breaks since ray.__version__ = 1.8.1

        # run first test epoch and log results, this logs epoch=0, state before training
        self._iteration = -1

        #### Initialize linear probes for downstream tasks
        if self.trainset.properties is not None:
            print(
                "Found properties in dataset - downstream tasks are going to be evaluated at test time."
            )
            self.dstk = DownstreamTaskLearner()
        else:
            print("No properties found in dataset - skip downstream tasks.")
            self.dstk = None

        ### Save Model Summary for later / parameter count comparison
        try:
            print(f"generate model summary")
            import pytorch_model_summary as pms

            inpts = torch.randn(20, config["model::i_dim"]).to(config["device"])
            summary = pms.summary(
                self.SimCLR,
                inpts,
                show_input=False,
                show_hierarchical=True,
                print_summary=False,
                max_depth=5,
                show_parent_layers=False,
            )
            fname_summary = Path(self.logdir).joinpath("model_summary.txt")
            with fname_summary.open("w") as f:
                f.write(summary)
        except Exception as e:
            print(e)

    # step ####
    def step(self):
        """
        The step function performs n training epochs, one test epoch, one val epoch and computes the downstreamtask performance
        """
        # set model to eval mode as default
        self.SimCLR.model.eval()

        # run several training epochs before one test epoch
        if self._iteration < 0:
            print("test first validation mode")
            perf_train = self.SimCLR.test_epoch(
                self.trainloader,
                0,
                writer=None,
                tf_out=10,
            )

        else:
            for _ in range(self.config["training::test_epochs"]):
                # set model to training mode
                self.SimCLR.model.train()
                # run one training epoch
                perf_train = self.SimCLR.train_epoch(
                    self.trainloader, 0, writer=None, tf_out=10
                )
                # set model to training mode
                self.SimCLR.model.eval()
        # run one test epoch
        perf_test = self.SimCLR.test_epoch(
            self.testloader,
            0,
            writer=None,
            tf_out=10,
        )
        result_dict = {}
        for key in perf_test.keys():
            result_dict[f"{key}_test"] = perf_test[key]

        for key in perf_train.keys():
            result_dict[f"{key}_train"] = perf_train[key]

        if self.valset is not None:
            # run one test epoch
            perf_val = self.SimCLR.test_epoch(
                self.valloader,
                0,
                writer=None,
                tf_out=10,
            )
            for key in perf_val.keys():
                result_dict[f"{key}_val"] = perf_val[key]

        # if DownstreamTaskLearner exist. apply downstream task
        if self.dstk is not None:
            performance = self.dstk.eval_dstasks(
                # model=self.SimCLR.model,
                model=self.SimCLR,
                trainset=self.trainset,
                testset=self.testset,
                valset=self.valset,
                batch_size=self.config["trainset::batchsize"],
            )
            # append performance values to result_dict
            for key in performance.keys():
                result_dict[key] = performance[key]
        return result_dict

    def set_loss_weights(
        self,
    ):
        """
        helper function to set weights for loss per layers
        """
        index_dict = self.config["model::index_dict"]
        weights = []
        for idx, layer in enumerate(index_dict["layer"]):
            w = self.config.get(f"training::loss_weight_layer_{idx+1}", 1.0)
            weights.append(w)
        self.config["model::index_dict"]["loss_weight"] = weights
        print("#### weighting the loss per layer as follows:")
        for idx, layer in enumerate(index_dict["layer"]):
            print(
                f'layer {layer} - weight {self.config["model::index_dict"]["loss_weight"][idx]}'
            )

    # make save_checkpoint instead
    def save_checkpoint(self, experiment_dir):
        # define checkpoint path
        path = Path(experiment_dir).joinpath("checkpoints")
        torch.save(self.SimCLR.model.state_dict(), path)
        # projection_head state_dict
        if self.config.get("model::projection_head_layers", None) > 0:
            path = Path(experiment_dir).joinpath("projection_head")
            torch.save(
                self.SimCLR.criterion.loss_contrast.projection_head.state_dict(), path
            )
        # optimizer state_dict
        path = Path(experiment_dir).joinpath("optimizer")
        torch.save(self.SimCLR.optimizer.state_dict(), path)

        # tune apparently expects to return the directory
        return experiment_dir

    # make load_checkpoint instead
    def load_checkpoint(self, experiment_dir):
        # load model checkpoint
        path = Path(experiment_dir).joinpath("checkpoints")
        checkpoint = torch.load(path)
        self.SimCLR.model.load_state_dict(checkpoint)
        # load projection_head state_dict
        if self.config.get("model::projection_head_layers", None) > 0:
            path = Path(experiment_dir).joinpath("projection_head")
            proj_head = torch.load(path)
            self.SimCLR.criterion.loss_contrast.projection_head.load_state_dict(
                proj_head
            )
        # optimizer state_dict
        path = Path(experiment_dir).joinpath("optimizer")
        opt = torch.load(path)
        self.SimCLR.optimizer.load_state_dict(opt)

        # tune apparently expects to return the directory
        return experiment_dir

    def reset_config(self, new_config):
        success = False
        try:
            print(
                "### warning: reuse actors / reset_config only if the dataset remains exactly the same. Getitem functions can be altered \n ### only dataloader and model are reconfiugred"
            )
            self.config = new_config
            self.seed = self.config["seed"]
            self.device = self.config["device"]

            # init model
            self.SimCLR = SimCLRAEModule(self.config)
            ## init dataloaders

            # set noise
            self.trainset.add_noise_input = self.config.get(
                "trainset::add_noise_input", 0.0
            )
            self.trainset.add_noise_output = self.config.get(
                "trainset::add_noise_output", 0.0
            )
            self.testset.add_noise_input = self.config.get(
                "testset::add_noise_input", 0.0
            )
            self.testset.add_noise_output = self.config.get(
                "testset::add_noise_output", 0.0
            )
            # set permutations
            self.trainset.permutations_number = self.config[
                "trainset::permutations_number"
            ]
            self.testset.permutations_number = self.config[
                "testset::permutations_number"
            ]
            # set map_2_canon
            self.trainset.view_1_canoncial = self.config.get(
                "trainset::view_1_canonical", False
            )
            self.trainset.view_2_canoncial = self.config.get(
                "trainset::view_2_canonical", False
            )
            self.testset.view_1_canoncial = self.config.get(
                "testset::view_1_canonical", False
            )
            self.testset.view_2_canoncial = self.config.get(
                "testset::view_2_canonical", False
            )
            # if norm dosn't match.. recompute canon form and vectorize again
            self.testset.map_2_canon_metric = self.trainset.map_2_canon_metric
            if self.trainset.map_2_canon_metric != self.config.get(
                "trainset::map_2_canon_metric", "absolute"
            ):
                print(f"canon metrics don't match: recompute canon forms and vectorize")
                self.trainset.map_2_canon_metric = self.config.get(
                    "trainset::map_2_canon_metric", "absolute"
                )
                self.testset.map_2_canon_metric = self.config.get(
                    "trainset::map_2_canon_metric", "absolute"
                )
                print("prepare canonical form")
                self.trainset.map_data_to_canonical()
                self.testset.map_data_to_canonical()
                #
                if self.trainset.mode == "vector":
                    print("vectorize data")
                    self.trainset.vectorize_data()
                if self.testset.mode == "vector":
                    print("vectorize data")
                    self.testset.vectorize_data()

            # set erase
            self.trainset.set_erase(self.config.get("trainset::erase_augment", None))
            self.testset.set_erase(self.config.get("testset::erase_augment", None))

            # get full dataset in tensors
            self.trainloader = DataLoader(
                self.trainset,
                batch_size=self.config["trainset::batchsize"],
                shuffle=True,
                drop_last=True,  # important: we need equal batch sizes
            )

            # testset
            # get full dataset in tensors
            self.testloader = torch.utils.data.DataLoader(
                self.testset,
                batch_size=self.config["trainset::batchsize"],
                shuffle=False,
                drop_last=True,  # important: we need equal batch sizes
            )
            # compute loss_mean
            self.SimCLR.loss_mean = self.SimCLR.criterion.compute_mean_loss(
                self.testloader
            )

            # save initial checkpoint
            self.save()

            # if we got to this point:
            success = True

        except Exception as e:
            print(e)

        return success
