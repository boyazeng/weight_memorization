## Modifications to HyperDiffusion Source Code

We document the minimal changes made to the evaluated method's source code (under [method](method)):
1. [dataset.py](method/dataset.py): sort the ``mlp_files`` in ``WeightDataset`` so that the weights and meshes can correspond even if they're saved from separate evaluation runs.

```diff
         if object_names is None:
-            self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
+            self.mlp_files = [file for file in list(sorted(os.listdir(mlps_folder)))]
         else:
             self.mlp_files = []
-            for file in list(os.listdir(mlps_folder)):
+            for file in list(sorted(os.listdir(mlps_folder))):
                 # Excluding black listed shapes
                 if cfg.filter_bad and file.split("_")[1] in blacklist:
                     continue
                 # Check if file is in corresponding split (train, test, val)
                 # In fact, only train split is important here because we don't use test or val MLP weights
                 if ("_" in file and (file.split("_")[1] in object_names or (
                         file.split("_")[1] + "_" + file.split("_")[2]) in object_names)) or (file in object_names):
                     self.mlp_files.append(file)
```