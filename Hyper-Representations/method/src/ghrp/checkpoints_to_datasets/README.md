# Checkpoints to Datasets
This module contains functions to load pytorch checkpoints as datasets and augment model checkpoints, as well as necessary auxiliary functions.

The base class to load checkpoints into a dataset is contained in 'dataset_base.py'. 
The dataset class used for hyper-representation training can be found in 'dataet_simclr.py', it inherits from datsaet_base and implements the data augmentation.
The augmentations are implemented in 'permute_checkpoint.py' and 'random_erasing.py'.
'dataset_properties.py' contains a lightweight dataset class that only loads model properties, not their checkpoints.

Some auxiliary functions that are (sometimes) needed outside of datasets are collected in 'dataset_auxiliaries.py'.
