import os; os.chdir("../method")
import sys; sys.path.append("../method")
from copy import deepcopy
import hydra
import omegaconf
import torch
import torch.utils.data
from tqdm import tqdm

from Gpt.data.dataset_lmdb import ParameterDataset
from Gpt.diffusion import create_diffusion
from Gpt.download import find_model
from Gpt.models.transformer import Gpt
from Gpt.tasks import TASK_METADATA
from Gpt.utils import setup_env, requires_grad
from Gpt.vis import moduleify, scalar_expand, synth


class GptEvaluator:
    def __init__(
            self,
            dataset_name,            # string name of dataset
            task_net_dict,           # dictionary mapping a string to a (N, D) tensor of network that G.pt will process
            unnormalize_fn,          # function that "unnormalizes" sampled neural network parameters
            net_mb_size=1,           # minibatch size at which input networks are processed
            thresholding="none",     # "none" or "static"; thresholding alg. to use after each diffusion sampling step
            param_range=None,        # A tuple of the (min, max) values that unnormalized network parameters can have
    ):
        self.metadata = TASK_METADATA[dataset_name]
        self.inp_data = self.metadata['data_fn']()  # Instantiates data (MNIST, CIFAR-10) or RL environments (IsaacGym)
        self.unnormalize_fn = unnormalize_fn
        self.create_synth_fn(thresholding, param_range)
        self.task_net_dict = task_net_dict
        self.net_mb_size = net_mb_size
        self.dataset_name = dataset_name

    def create_synth_fn(self, thresholding="none", param_range=None):
        """
        Creates a function to sample new parameters from G.pt. This is mainly for convenience (so we don't have to
        pass in the thresholding and param_range arguments later on below).
        """
        self.synth_fn = lambda *args: synth(*args, param_range=param_range, thresholding=thresholding)

    def test_function(self, pred):
        modulified = moduleify(pred, self.metadata['constructor'], self.unnormalize_fn).to('cuda')
        val = self.metadata['task_test_fn'](*self.inp_data, modulified)
        if isinstance(val, (tuple, list)):
            val = val[0]
        return val

    def test_accs_and_incorrect_indices(self, pred):
        modulified = moduleify(pred, self.metadata['constructor'], self.unnormalize_fn).to('cuda')
        accs, incorrect_indices = self.metadata['test_accs_and_incorrect_indices'](*self.inp_data, modulified)
        return accs, incorrect_indices

    def batch_test_fn(self, net_batch, check_dims=True, device='cuda'):
        if check_dims:
            assert net_batch.dim() == 2
        result = [self.test_function(net.unsqueeze(0)) for net in net_batch]
        result = torch.tensor(result, dtype=torch.float, device=device)
        return result

    def batch_test_accs_and_incorrect_indices(self, net_batch, check_dims=True, device='cuda'):
        if check_dims:
            assert net_batch.dim() == 2
        futures = [torch.jit.fork(self.test_accs_and_incorrect_indices, net.unsqueeze(0)) for net in net_batch]
        result = [torch.jit.wait(fut) for fut in futures]
        accs, incorrect_indices = [it[0] for it in result], [it[1] for it in result]
        accs = torch.tensor(accs, dtype=torch.float, device=device)
        incorrect_indices = torch.stack(incorrect_indices, dim=0)
        return accs, incorrect_indices

    @torch.inference_mode()
    def one_step_prompting(self, task_net, diffusion, G):
        """Optimizes a batch of input neural networks with one update sampled from G.pt."""
        num_nets = task_net.size(0)
        current_losses = self.batch_test_fn(task_net)  # (N,)
        desired_losses = scalar_expand(self.metadata['best_prompt'], num_nets)  # (N,)
        if not self.metadata['minimize']:
            desired_losses = desired_losses.flip(1,)

        desired_losses_in = desired_losses.flatten()  # (N * 1,)
        evolved_nets = self.synth_fn(
            diffusion, G, desired_losses_in, current_losses, task_net
        )
        evolved_nets = tqdm(evolved_nets, leave=False, desc='one step prompting eval', total=evolved_nets.size(0))
        accs, incorrect_indices = self.batch_test_accs_and_incorrect_indices(evolved_nets, check_dims=False)
        return accs, incorrect_indices, [item for item in evolved_nets]

    @torch.inference_mode()
    def generate_and_evaluate(self, diffusion, model):
        model.eval()
        save_root = f"../data"
        os.makedirs(save_root, exist_ok=True)
        for task_key in ["test_set", "training_set"]:
            task_nets = self.task_net_dict[task_key]
            task_net_batches = torch.split(task_nets, self.net_mb_size, dim=0)

            full_accs, full_incorrect_indices, full_gen_weights = [], [], []
            for task_net_batch in tqdm(task_net_batches, desc=f"Evaluate {task_key} checkpoints"):
                if task_key == "test_set":
                    accs, incorrect_indices, gen_weights = self.one_step_prompting(
                        task_net_batch, diffusion, model
                    )
                    full_gen_weights += gen_weights
                else:
                    accs, incorrect_indices = self.batch_test_accs_and_incorrect_indices(task_net_batch, check_dims=False)
                full_accs.append(accs); full_incorrect_indices.append(incorrect_indices)
            full_accs = torch.cat(full_accs, 0)
            full_incorrect_indices = torch.cat(full_incorrect_indices, 0)
            
            if task_key == "test_set":
                torch.save(full_accs, os.path.join(save_root, 'generated_accs.pt'))
                torch.save(full_incorrect_indices, os.path.join(save_root, 'generated_incorrect_indices.pt'))
                full_gen_weights = torch.stack(full_gen_weights)
                torch.save(full_gen_weights, os.path.join(save_root, 'generated_weights.pt'))
            else:
                torch.save(full_accs, os.path.join(save_root, 'training_accs.pt'))
                torch.save(full_incorrect_indices, os.path.join(save_root, 'training_incorrect_indices.pt'))
                torch.save(task_nets, os.path.join(save_root, 'training_weights.pt'))

        random_indices = torch.randperm(len(self.task_net_dict["training_set"]))[:100]
        for noise in [0.05, 0.1]:
            noised_weights = [it + torch.normal(mean=0, std=noise, size=it.size()).cuda() for it in self.task_net_dict["training_set"][random_indices]]
            noised_weights = torch.stack(noised_weights)
            task_net_batches = torch.split(noised_weights, self.net_mb_size, dim=0)
            full_accs, full_incorrect_indices = [], []
            for task_net_batch in tqdm(task_net_batches, desc=f"Evaluate training checkpoints with noise {noise}"):
                accs, incorrect_indices = self.batch_test_accs_and_incorrect_indices(task_net_batch, check_dims=False)
                full_accs.append(accs); full_incorrect_indices.append(incorrect_indices)
            full_accs = torch.cat(full_accs, 0)
            full_incorrect_indices = torch.cat(full_incorrect_indices, 0)
            torch.save(full_accs, os.path.join(save_root, f'noise{noise}_accs.pt'))
            torch.save(full_incorrect_indices, os.path.join(save_root, f'noise{noise}_incorrect_indices.pt'))

@hydra.main(config_path="configs/train", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig):
    seed = setup_env(cfg)

    gpt_evaluator = GptEvaluator(
        cfg.dataset.name,
        None,
        None,
        net_mb_size=cfg.vis.net_mb_size_per_gpu,
        thresholding=cfg.sampling.thresholding,
        param_range=None
    )

    train_dataset = ParameterDataset(
        dataset_dir=cfg.dataset.path,
        dataset_name=cfg.dataset.name,
        num_test_runs=cfg.dataset.num_test_runs,
        normalizer_name=cfg.dataset.normalizer,
        openai_coeff=cfg.dataset.openai_coeff,
        split="train",
        train_metric=cfg.dataset.train_metric,
        permute_augment=cfg.dataset.augment,
        target_epoch_size=cfg.dataset.target_epoch_size,
        single_run_debug=cfg.debug_mode,
        max_train_runs=cfg.dataset.max_train_runs
    )

    test_dataset = ParameterDataset(
        dataset_dir=cfg.dataset.path,
        dataset_name=cfg.dataset.name,
        num_test_runs=cfg.dataset.num_test_runs,
        normalizer_name=cfg.dataset.normalizer,
        openai_coeff=cfg.dataset.openai_coeff,
        split="test",
        train_metric=cfg.dataset.train_metric,
        permute_augment=False,
        target_epoch_size=cfg.dataset.target_epoch_size,
        min_val=train_dataset.min_val,
        max_val=train_dataset.max_val,
        single_run_debug=cfg.debug_mode
    )

    model = Gpt(
        parameter_sizes=train_dataset.parameter_sizes,
        parameter_names=train_dataset.parameter_names,
        predict_xstart=cfg.transformer.predict_xstart,
        absolute_loss_conditioning=cfg.transformer.absolute_loss_conditioning,
        chunk_size=cfg.transformer.chunk_size,
        split_policy=cfg.transformer.split_policy,
        max_freq_log2=cfg.transformer.max_freq_log2,
        num_frequencies=cfg.transformer.num_frequencies,
        n_embd=cfg.transformer.n_embd,
        encoder_depth=cfg.transformer.encoder_depth,
        decoder_depth=cfg.transformer.decoder_depth,
        n_layer=cfg.transformer.n_layer,
        n_head=cfg.transformer.n_head,
        attn_pdrop=cfg.transformer.dropout_prob,
        resid_pdrop=cfg.transformer.dropout_prob,
        embd_pdrop=cfg.transformer.dropout_prob
    )

    ema = deepcopy(model)
    requires_grad(ema, False)

    diffusion = create_diffusion(
        learn_sigma=False, predict_xstart=cfg.transformer.predict_xstart,
        noise_schedule='linear', steps=1000
    )

    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    ema = ema.cuda(device=cur_device)

    resume_checkpoint = find_model(cfg.resume_path)
    model.load_state_dict(resume_checkpoint['G'])
    ema.load_state_dict(resume_checkpoint['G_ema'])
    try:
        start_epoch = int(os.path.basename(cfg.resume_path).split('.')[0])
    except ValueError:
        start_epoch = 0
    print(f'Resumed G.pt from checkpoint {cfg.resume_path}, using start_epoch={start_epoch}')

    task_net_dict = {
        'training_set': torch.stack([train_dataset.get_run_network(i, iter=-1) for i in range(len(train_dataset.run_jsons))]).cuda(),
        'test_set':     torch.stack([test_dataset.get_run_network(i)  for i in range(128)]).cuda()
    }

    gpt_evaluator.task_net_dict = task_net_dict
    gpt_evaluator.unnormalize_fn = train_dataset.unnormalize
    gpt_evaluator.create_synth_fn(
        thresholding=cfg.sampling.thresholding,
        param_range=train_dataset.get_range(normalize=True)
    )

    gpt_evaluator.generate_and_evaluate(diffusion, ema)
    return


if __name__ == "__main__":
    main()
