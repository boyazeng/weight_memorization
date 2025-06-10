import copy
import os
import sys

os.chdir("../method")
# Using it to make pyrender work on clusters
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.append("../method")
sys.path.append("siren")

import hydra
from omegaconf import DictConfig
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import wandb

from dataset import WeightDataset
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from hd_utils import Config, get_mlp, generate_mlp_from_weights
from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder
from transformer import Transformer


class HyperDiffusion(pl.LightningModule):
    def __init__(
        self, model, train_dt, val_dt, test_dt, mlp_kwargs, image_shape, method, cfg
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.method = method
        self.mlp_kwargs = mlp_kwargs
        self.val_dt = val_dt
        self.train_dt = train_dt
        self.test_dt = test_dt
        fake_data = torch.randn(*image_shape)

        encoded_outs = fake_data
        print("encoded_outs.shape", encoded_outs.shape)
        timesteps = Config.config["timesteps"]
        betas = torch.tensor(np.linspace(1e-4, 2e-2, timesteps))
        self.image_size = encoded_outs[:1].shape

        # Initialize diffusion utiities
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[cfg.diff_config.params.model_mean_type],
            model_var_type=ModelVarType[cfg.diff_config.params.model_var_type],
            loss_type=LossType[cfg.diff_config.params.loss_type],
            diff_pl_module=self,
        )

    def weights_to_meshes(self, x_0s, folder_name="meshes", info="0", res=64, level=0):
        x_0s = x_0s.view(len(x_0s), -1)
        curr_weights = Config.get("curr_weights")
        x_0s = x_0s[:, :curr_weights]
        meshes = []
        sdfs = []
        for i, weights in enumerate(x_0s):
            mlp = generate_mlp_from_weights(weights, self.mlp_kwargs)
            sdf_decoder = SDFDecoder(
                self.mlp_kwargs.model_type,
                None,
                "nerf" if self.mlp_kwargs.model_type == "nerf" else "mlp",
                self.mlp_kwargs,
            )
            sdf_decoder.model = mlp.cuda().eval()
            with torch.no_grad():
                effective_file_name = (
                    f"{folder_name}/mesh_epoch_{self.current_epoch}_{i}_{info}"
                    if folder_name is not None
                    else None
                )
                v, f, sdf = sdf_meshing.create_mesh(
                    sdf_decoder,
                    effective_file_name,
                    N=res,
                    level=level
                    if self.mlp_kwargs.output_type in ["occ", "logits"]
                    else 0,
                )
                if (
                    "occ" in self.mlp_kwargs.output_type
                    or "logits" in self.mlp_kwargs.output_type
                ):
                    tmp = copy.deepcopy(f[:, 1])
                    f[:, 1] = f[:, 2]
                    f[:, 2] = tmp
                sdfs.append(sdf)
                mesh = trimesh.Trimesh(v, f)
                meshes.append(mesh)
        sdfs = torch.stack(sdfs)
        return meshes, sdfs

    def save_point_clouds(self, split):
        dataset_path = os.path.join(
            Config.config["dataset_dir"],
            Config.config["dataset"] + f"_{self.cfg.val.num_points}_pc",
        )
        object_names = np.genfromtxt(
            os.path.join(dataset_path, f"{split}_split.lst"), dtype="str"
        )
        object_names = sorted(object_names)
        blacklist = set(np.genfromtxt(self.cfg.filter_bad_path, dtype=str))
        object_names = [name for name in object_names if name.split(".")[0] not in blacklist]

        pcs = []
        for obj_name in object_names:
            pc = np.load(os.path.join(dataset_path, obj_name + ".npy"))
            pc = pc[:, :3]
            pc = torch.tensor(pc).float()
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
            pc = (pc - shift) / scale
            pcs.append(pc)
        ref_pcs = torch.stack(pcs)
        torch.save(ref_pcs, f"../data/{split}_point_clouds.pth")

    def save_train_weights(self):
        training_weights = torch.stack([self.train_dt[i][0] for i in range(len(self.train_dt))]).cuda()
        torch.save(training_weights, f"../data/training_weights.pth")

    def save_train_meshes(self):
        meshes_save_dir = f"../data/train_meshes"
        os.makedirs(meshes_save_dir, exist_ok=True)
        for i in tqdm(range(len(self.train_dt)), desc="Saving training meshes"):
            weight = self.train_dt[i][0]
            mesh, _ = self.weights_to_meshes(
                weight.unsqueeze(0) / self.cfg.normalization_factor,
                None,
                res=356,
                level=1.386,
            )
            mesh = mesh[0]
            if len(mesh.vertices) > 0:
                mesh.export(os.path.join(meshes_save_dir, f"mesh_{i}.obj"))
            else:
                print(f"Mesh {i} is empty")

    def generate(self):
        meshes_save_dir = f"../data/gen_meshes"
        os.makedirs(meshes_save_dir, exist_ok=True)
        number_of_samples_to_generate = int(606 * self.cfg.test_sample_mult)
        sample_mlp_weights = []
        test_batch_size = 100

        for _ in tqdm(range(number_of_samples_to_generate // test_batch_size), desc="Generating MLP weights"):
            sample_mlp_weights.append(
                self.diff.ddim_sample_loop(
                    self.model, (test_batch_size, *self.image_size[1:])
                )
            )
        if number_of_samples_to_generate % test_batch_size != 0:
            sample_mlp_weights.append(
                self.diff.ddim_sample_loop(
                    self.model,
                    (
                        number_of_samples_to_generate % test_batch_size,
                        *self.image_size[1:],
                    ),
                )
            )
        sample_mlp_weights = torch.vstack(sample_mlp_weights)
        torch.save(sample_mlp_weights, f"../data/gen_weights.pth")

        sample_batch = []
        for i, sample_mlp_weight in tqdm(enumerate(sample_mlp_weights), desc="Saving generated meshes"):
            """
            (1) Generate meshes from MLP weights.
            (2) Sample points from the meshes.
            (3) Normalize the points (0 mean & unit std).
            """
            mesh, _ = self.weights_to_meshes(
                sample_mlp_weight.unsqueeze(0) / self.cfg.normalization_factor,
                None,
                res=356,
                level=1.386,
            )
            mesh = mesh[0]
            mesh.export(os.path.join(meshes_save_dir, f"{i}.obj"))

            pc = torch.tensor(mesh.sample(self.cfg.val.num_points))
            pc = pc * 2
            pc = pc.float()
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
            pc = (pc - shift) / scale
            sample_batch.append(pc)

        sample_pcs = torch.stack(sample_batch)
        torch.save(sample_pcs, f"../data/gen_point_clouds.pth")

    def noise(self, noise_level):
        sample_batch = []
        selected_indices = np.random.choice(len(self.train_dt), int(606 * self.cfg.test_sample_mult), replace=False)
        for selected_index in tqdm(selected_indices, desc="Saving point clouds from training MLPs with noise added"):
            noised_weight = self.train_dt[selected_index][0]
            noised_weight = noised_weight + noise_level * torch.randn_like(noised_weight)
            mesh, _ = self.weights_to_meshes(
                noised_weight.unsqueeze(0) / self.cfg.normalization_factor,
                None,
                res=356,
                level=1.386,
            )
            mesh = mesh[0]

            pc = torch.tensor(mesh.sample(self.cfg.val.num_points))
            pc = pc * 2
            pc = pc.float()
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
            pc = (pc - shift) / scale
            sample_batch.append(pc)

        sample_pcs = torch.stack(sample_batch)
        torch.save(sample_pcs, f"../data/noise{noise_level}_point_clouds.pth")

    def generate_and_evaluate(self):
        os.makedirs("../data", exist_ok=True)
        self.save_point_clouds("train")
        self.save_point_clouds("test")
        self.save_train_weights()
        self.save_train_meshes()
        self.generate()
        self.noise(0.02)
        self.noise(0.04)


@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs",
    config_name="train_plane",
)
def main(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method")
    mlp_kwargs = Config.config["mlp_config"]["params"]

    wandb.init(
        project="hyperdiffusion",
        dir=config["tensorboard_log_dir"],
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        tags=[Config.get("mode")],
        mode="disabled",
        config=dict(config),
    )

    wandb_logger = WandbLogger()
    wandb_logger.log_text("config", ["config"], [[str(config)]])
    print("wandb", wandb.run.name, wandb.run.id)

    train_dt = val_dt = test_dt = None

    # Although it says train, it includes all the shapes but we only extract training ones in WeightDataset
    mlps_folder_train = Config.get("mlps_folder_train")

    mlp = get_mlp(mlp_kwargs)
    state_dict = mlp.state_dict()
    layers = []
    layer_names = []
    for l in state_dict:
        shape = state_dict[l].shape
        layers.append(np.prod(shape))
        layer_names.append(l)
    model = Transformer(
        layers, layer_names, **Config.config["transformer_config"]["params"]
    ).cuda()

    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])
    train_object_names = np.genfromtxt(
        os.path.join(dataset_path, "train_split.lst"), dtype="str"
    )
    train_object_names = set([str.split(".")[0] for str in train_object_names])
    # Check if dataset folder already has train,test,val split; create otherwise.
    mlps_folder_all = mlps_folder_train
    all_object_names = np.array(
        [obj for obj in os.listdir(dataset_path) if ".lst" not in obj]
    )
    total_size = len(all_object_names)
    val_size = int(total_size * 0.05)
    test_size = int(total_size * 0.15)
    train_size = total_size - val_size - test_size
    if not os.path.exists(os.path.join(dataset_path, "train_split.lst")):
        train_idx = np.random.choice(
            total_size, train_size + val_size, replace=False
        )
        test_idx = set(range(total_size)).difference(train_idx)
        val_idx = set(np.random.choice(train_idx, val_size, replace=False))
        train_idx = set(train_idx).difference(val_idx)
        print(
            "Generating new partition",
            len(train_idx),
            train_size,
            len(val_idx),
            val_size,
            len(test_idx),
            test_size,
        )

        # Sanity checking the train, val and test splits
        assert len(train_idx.intersection(val_idx.intersection(test_idx))) == 0
        assert len(train_idx.union(val_idx.union(test_idx))) == total_size
        assert (
            len(train_idx) == train_size
            and len(val_idx) == val_size
            and len(test_idx) == test_size
        )

        np.savetxt(
            os.path.join(dataset_path, "train_split.lst"),
            all_object_names[list(train_idx)],
            delimiter=" ",
            fmt="%s",
        )
        np.savetxt(
            os.path.join(dataset_path, "val_split.lst"),
            all_object_names[list(val_idx)],
            delimiter=" ",
            fmt="%s",
        )
        np.savetxt(
            os.path.join(dataset_path, "test_split.lst"),
            all_object_names[list(test_idx)],
            delimiter=" ",
            fmt="%s",
        )

    val_object_names = np.genfromtxt(
        os.path.join(dataset_path, "val_split.lst"), dtype="str"
    )
    val_object_names = set([str.split(".")[0] for str in val_object_names])
    test_object_names = np.genfromtxt(
        os.path.join(dataset_path, "test_split.lst"), dtype="str"
    )
    test_object_names = set([str.split(".")[0] for str in test_object_names])

    train_dt = WeightDataset(
        mlps_folder_train,
        wandb_logger,
        model.dims,
        mlp_kwargs,
        cfg,
        train_object_names,
    )
    train_dl = DataLoader(
        train_dt,
        batch_size=Config.get("batch_size"),
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_dt = WeightDataset(
        mlps_folder_train,
        wandb_logger,
        model.dims,
        mlp_kwargs,
        cfg,
        val_object_names,
    )
    test_dt = WeightDataset(
        mlps_folder_train,
        wandb_logger,
        model.dims,
        mlp_kwargs,
        cfg,
        test_object_names,
    )

    print(
        "Train dataset length: {} Val dataset length: {} Test dataset length".format(
            len(train_dt), len(val_dt), len(test_dt)
        )
    )
    input_data = next(iter(train_dl))[0]
    print(
        "Input data shape, min, max:",
        input_data.shape,
        input_data.min(),
        input_data.max(),
    )

    best_model_save_path = Config.get("best_model_save_path")

    # Initialize HyperDiffusion
    diffuser = HyperDiffusion.load_from_checkpoint(
        best_model_save_path,
        model=model,
        train_dt=train_dt,
        val_dt=val_dt,
        test_dt=test_dt,
        mlp_kwargs=mlp_kwargs,
        image_shape=input_data.shape,
        method=method,
        cfg=cfg
    )

    diffuser.generate_and_evaluate()
    wandb_logger.finalize("Success")


if __name__ == "__main__":
    main()
