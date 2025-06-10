## HyperDiffusion

We evaluate the pre-trained HyperDiffusion checkpoint for the airplane shape category.

### Install environment
We use the environment from the original codebase (under [method](method)):
```
cd method
conda env create --file hyperdiffusion_env.yaml
conda activate hyper-diffusion
pip install gdown==4.7.3
```

### Download MLP weights, point clouds, and model checkpoint

```
# MLP weights
gdown https://drive.google.com/uc?id=1D3ILUpA1EHloF0mSi5YUuwFdQvtfISqF
unzip 3d_128_plane_multires_4_manifoldplus_slower_no_clipgrad.zip
mv 3d_128_plane_multires_4_manifoldplus_slower_no_clipgrad mlp_weights/
# point clouds
gdown https://drive.google.com/uc?id=1syo9MdJ9BGfT7CbcFr87ylOk7j-Br8Ww
unzip 02691156_2048_pc.zip
mv 02691156_2048_pc data/
# checkpoint
gdown https://drive.google.com/uc?id=1wR_SIAiAlX1Yix1dyoUjrflxcAV0qVSE
```

### Generate and evaluate weights

```
cd ../evaluation
python generate_and_evaluate.py --config-name=train_plane mode=test
```

### Generate plots

```
cd ..
python heatmap.py
python min_distance_to_training.py
python render_shapes.py
python performance_vs_novelty.py
```