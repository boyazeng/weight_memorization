## G.pt

We evaluate the pre-trained G.pt checkpoint trained on MNIST classification model weights.

### Install environment
We use the environment from the original codebase (under [method](method)):
```
cd method
conda env create -f environment.yml
conda activate G.pt
pip install -e .
pip install matplotlib scikit-learn scipy
```

### Download the pre-trained checkpoint and training weights

```
python Gpt/download.py
```

### Generate and evaluate weights

```
cd ../evaluation
python generate_and_evaluate.py --config-path configs/test --config-name mnist_loss.yaml num_gpus=1 wandb.mode=disabled
```

### Generate plots

```
cd ..
python heatmap.py
python min_distance_to_training.py
python decision_boundary.py
python performance_vs_novelty.py
```