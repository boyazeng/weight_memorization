## P-diff

P-diff does not provide pre-trained checkpoints. We train it from scratch and evaluate it in its primary experimental setting, where it generates the last two normalization layers of a ResNet-18 model for CIFAR-100 classification.

### Install environment
We use the environment from the original codebase (under [method](method)):
```
cd method
conda create -n pdiff python=3.11
conda activate pdiff
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install scikit-learn scipy
```

Then, run ``accelerate config`` and choose "No distributed training".

### Create training checkpoints and train p-diff

```
cd workspace
bash run_all.sh main cifar100_resnet18 0
```

### Run baseline methods for generating new weights

```
cd ../../evaluation
python baseline.py --type noise --noise_amplitude 0.06
python baseline.py --type noise --noise_amplitude 0.12
python baseline.py --type average
python baseline.py --type gaussian
```

### Evaluate training, generated, and baseline weights

```
python evaluate.py --type training
python evaluate.py --type generated
python evaluate.py --type noise --noise_amplitude 0.06
python evaluate.py --type noise --noise_amplitude 0.12
python evaluate.py --type average
python evaluate.py --type gaussian
```

### Generate plots

```
cd ..
python heatmap.py
python min_distance_to_training.py
python decision_boundary.py
python performance_vs_novelty.py

python param_value_distribution.py
python performance_vs_novelty_interpolate.py
python tsne_interpolate.py
```