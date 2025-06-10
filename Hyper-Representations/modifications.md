## Modifications to Hyper-Representations Source Code

We document the minimal changes made to the evaluated method's source code (under [method](method)):
1. [setup.py](method/setup.py): since we removed the ``.git`` from ``method``, we manually set a fixed version number for the package, instead of relying on git metadata.

```diff
-     setup(use_scm_version={"version_scheme": "no-guess-dev"})
+     setup(version="1.0.0")
```

2. [data/download_data.sh](method/data/download_data.sh): Zenodo uses HTTP redirections for file downloads, which ``curl`` does not follow by default. Therefore, we add the ``-L`` option to ensure the request reaches the final download URL.

```diff
echo Downloading Hyper-Representations from https://zenodo.org/record/7529960/files/hyper_reps.zip?download=1

- curl -# -o "hyper_reps.zip" "https://zenodo.org/record/7529960/files/hyper_reps.zip?download=1"
+ curl -# -L -o "hyper_reps.zip" "https://zenodo.org/record/7529960/files/hyper_reps.zip?download=1"

echo Unzipping hyper_reps

unzip hyper_reps.zip

echo Downloading pre-packaged zoos from https://zenodo.org/record/7529960/files/zoos.zip?download=1

- curl -# -o "zoos.zip" "https://zenodo.org/record/7529960/files/zoos.zip?download=1"
+ curl -# -L -o "zoos.zip" "https://zenodo.org/record/7529960/files/zoos.zip?download=1"
```

3. [src/ghrp/model_definitions/def_simclr_ae_module.py](method/src/ghrp/model_definitions/def_simclr_ae_module.py): the ``verbose`` argument is no longer supported in PyTorch, so we remove it.

```diff
             self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                 self.optimizer,
                 mode=mode,
                 factor=factor,
                 patience=patience,
                 threshold=threshold,
                 threshold_mode=threshold_mode,
                 cooldown=cooldown,
                 min_lr=min_lr,
                 eps=eps,
-                verbose=False,
             )
```