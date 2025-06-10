from multiprocessing import Pool
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

def best_assignment(mtx):
    """
    Find the permutation of training model's hidden neurons that minimizes the distance to the target weights.

    Args:
        mtx (ndarray): [#hidden neurons in training MLP, #hidden neurons in target MLP] = [10, 10]
            mtx[i, j] is the distance between the parameters associated with the i-th hidden neuron in the training MLP
            and the parameters associated with the j-th hidden neuron in the target MLP.

    Returns:
        ndarray[#hidden neurons]: the optimal permutation of the training model's hidden neurons.
        float: the distance between the training model and the target model under this permutation.
    """
    mtx = mtx.T
    # since mtx is square, row_indices is always numpy.arange(mtx.shape[0])
    row_indices, col_indices = linear_sum_assignment(mtx, maximize=False)
    min_sum = mtx[row_indices, col_indices].sum()
    return col_indices, min_sum

@torch.no_grad()
def group_by_hidden_neuron(module):
    """
    Group the parameters of the input MLP by hidden neurons.

    Returns:
        Tensor[#hidden neurons, #params per hidden neuron] = [10, 805].
    """
    return torch.stack([torch.cat([
        module.fc1.weight[hidden_neuron_id].flatten(),
        module.fc1.bias[hidden_neuron_id].flatten(),
        module.fc2.weight[:,hidden_neuron_id].flatten(),
    ]) for hidden_neuron_id in range(10)])

@torch.no_grad()
def permute(module, permutation):
    """Reorder hidden neurons of the input MLP according to the input permutation."""
    module.fc1.weight = torch.nn.Parameter(module.fc1.weight[permutation].cuda())
    module.fc1.bias = torch.nn.Parameter(module.fc1.bias[permutation].cuda())
    module.fc2.weight = torch.nn.Parameter(module.fc2.weight[:,permutation].cuda())
    module.fc2.bias = torch.nn.Parameter(module.fc2.bias.cuda())
    return module

def process_single_pair(args):
    i, j, pairwise_distance = args
    best_training_permutation, min_dist = best_assignment(pairwise_distance)
    return i, j, best_training_permutation, min_dist

@torch.no_grad()
def calculate_min_distance_and_best_permutation(training_ckpts, target_ckpts, batch_size=256):
    pool = Pool(32)
    training_grouped = torch.stack([group_by_hidden_neuron(it) for it in training_ckpts]) # shape = [num_training, num_hidden_neuron, num_param_per_neuron]
    target_grouped = torch.stack([group_by_hidden_neuron(it) for it in target_ckpts]) # shape = [num_generated, num_hidden_neuron, num_param_per_neuron]
    training_grouped_expanded = training_grouped.unsqueeze(1) # shape = [num_training, 1, num_hidden_neuron, num_param_per_neuron]
    target_grouped_expanded = target_grouped.unsqueeze(0) # shape = [1, num_target, num_hidden_neuron, num_param_per_neuron]
    num_training = training_grouped.size(0)
    num_target = target_grouped.size(0)
    full_min_dist = np.zeros((num_training, num_target))
    best_training_permutation = np.zeros((num_training, num_target, 10))

    for training_idx_start in tqdm(range(0, num_training, batch_size), desc="Processing training weights in batches"):
        training_idx_end = min(training_idx_start + batch_size, num_training)
        training_batch = training_grouped_expanded[training_idx_start:training_idx_end] # shape = [batch_size, 1, num_hidden_neuron, num_param_per_neuron]
        # for every pair of (training, target) checkpoint, we compute the pairwise distances between the grouped training parameters and the grouped target parameters
        pairwise_distance = torch.cdist(training_batch, target_grouped_expanded, p=2) ** 2 # shape = [batch_size, num_target, num_hidden_neuron, num_hidden_neuron]
        pairwise_distance = pairwise_distance.cpu().numpy()
        args = [(i, j, pairwise_distance[i, j]) for i in range(pairwise_distance.shape[0]) for j in range(pairwise_distance.shape[1])]
        results = pool.map(process_single_pair, args)
        for i, j, perm, min_dist in results:
            full_min_dist[training_idx_start + i, j] = min_dist
            best_training_permutation[training_idx_start + i, j] = perm

    training_outside_group = torch.stack([module.fc2.bias.flatten() for module in training_ckpts])
    target_outside_group = torch.stack([module.fc2.bias.flatten() for module in target_ckpts])
    pairwise_distance_outside_group = torch.cdist(training_outside_group, target_outside_group, p=2) ** 2 # shape = [num_training, num_target]
    pairwise_distance_outside_group = pairwise_distance_outside_group.cpu().numpy()
    full_min_dist = full_min_dist + pairwise_distance_outside_group

    full_min_dist = np.sqrt(full_min_dist)
    if training_ckpts is target_ckpts:
        np.fill_diagonal(full_min_dist, np.inf)
    return full_min_dist, best_training_permutation