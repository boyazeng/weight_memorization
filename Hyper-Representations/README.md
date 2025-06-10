## Hyper-Representations

We evaluate the pre-trained Hyper-Representations checkpoint trained on SVHN classification model weights.

### Install environment
We use the environment from the original codebase (under [method](method)):
```
cd method
conda create -n hyperrep python=3.9
conda activate hyperrep
pip3 install -e .
pip install "ray[tune]==2.37.0" tensorboard
```

### Download the pre-trained checkpoint and training weights

```
cd data
bash download_data.sh
```

### Generate model weights

```
cd ../../evaluation
python reconstruct.py
python sample.py
python noise_baseline.py
```

### Evaluate reconstructed, generated, and noised weights

```
python evaluate.py --type reconstructed
python evaluate.py --type generated
python evaluate.py --type noise --noise_amplitude 0.02
python evaluate.py --type noise --noise_amplitude 0.04
```

### Generate plots

```
cd ..
python heatmap.py
python min_distance_to_training.py
python decision_boundary.py
python performance_vs_novelty.py
```