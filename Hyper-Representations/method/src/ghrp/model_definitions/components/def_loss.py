# -*- coding: utf-8 -*-
################################################################################
# code originally take from https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py
##########################

import torch
import torch.nn as nn

from ghrp.model_definitions.components.def_projection_head import ProjectionHead

from einops import repeat

import warnings

################################################################################################
# contrastive loss
################################################################################################
class NT_Xent(nn.Module):
    def __init__(
        self, batch_size, temperature, device, projection_head=False, config=None
    ):
        """
        Implements normalized temperature-controlled Cross Entropy Loss
        """
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        if projection_head and config is not None:
            self.projection_head = ProjectionHead(config)
        else:
            self.projection_head = None

    def mask_correlated_samples(self, batch_size):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # forward pass through projection_head
        if self.projection_head is not None:
            z_i = self.projection_head(z_i)
            z_j = self.projection_head(z_j)
        # dimension of similarity matrix
        N = 2 * self.batch_size
        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).long()
        # the logits are NxN (N+1?) predictions for imaginary classes.
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class NT_Xent_pos(nn.Module):
    def __init__(
        self, batch_size, temperature, device, projection_head=False, config=None
    ):
        """
        Implements postive contrast version of normalized temperature-controlled Cross Entropy Loss
        """
        super(NT_Xent_pos, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion = nn.MSELoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        # projection_head
        if projection_head and config is not None:
            self.projection_head = ProjectionHead(config)
        else:
            self.projection_head = None

    def mask_correlated_samples(self, batch_size):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # forward pass through projection_head
        if self.projection_head is not None:
            z_i = self.projection_head(z_i)
            z_j = self.projection_head(z_j)
        # dimension of similarity matrix
        N = 2 * self.batch_size
        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        # negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).unsqueeze(dim=1)
        # just minimize the distance of positive samples to zero
        loss = self.criterion(positive_samples, labels)
        loss /= N
        return loss


################################################################################################
# reconstruction loss
################################################################################################
class ReconLoss(nn.Module):
    """
    Regular MSE w/ normalization
    """

    def __init__(self, reduction, normalization_var=None):
        super(ReconLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.normalization_var = normalization_var
        self.loss_mean = None

    def forward(self, output, target):
        assert (
            output.shape == target.shape
        ), f"MSE loss error: prediction and target don't have the same shape. output {output.shape} vs target {target.shape}"
        if self.normalization_var is not None:
            output /= self.normalization_var
            target /= self.normalization_var
        loss = self.criterion(output, target)
        # init rsq
        rsq = -999
        if self.loss_mean:
            rsq = torch.tensor(1 - loss.item() / self.loss_mean)
        return {"loss_recon": loss, "rsq": rsq}

    def set_normalization(self, reference_weights, index_dict):
        """set normalization koefficinet at init"""
        # compute variance of the weights __per layer__
        variances = []
        for start, length in zip(index_dict["idx_start"], index_dict["idx_length"]):
            idx_start = start
            idx_end = start + length
            sliced = reference_weights[:, idx_start:idx_end]
            tmp = sliced.flatten()
            var_tmp = torch.var(tmp)
            var = torch.ones(sliced.shape[1]) * var_tmp
            variances.append(var)
        variances = torch.cat(variances, dim=0)
        # set norm in recon loss
        self.normalization_var = variances

    def set_mean_loss(self, weights: torch.Tensor):
        # check that weights are tensor..
        assert isinstance(weights, torch.Tensor)
        w_mean = weights.mean(dim=0)  # compute over samples (dim0)
        # scale up to same size as weights
        weights_mean = repeat(w_mean, "d -> n d", n=weights.shape[0])
        out_mean = self.forward(weights_mean, weights)

        # compute mean
        print(f" mean loss: {out_mean['loss_recon']}")

        self.loss_mean = out_mean["loss_recon"]


class MSELossClipped(nn.Module):
    """
    implementation of MSE with error clipping
    thresholds the error term of MSE loss at value threshold.
    Idea: limit maximum influence of data points with large error to prevent them from dominating the entire error term
    """

    def __init__(self, reduction, threshold):

        super(MSELossClipped, self).__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, x, y):
        # compure raw error
        error = self.mse(x, y)
        # clip values
        if self.threshold:
            error = -torch.nn.functional.threshold(
                -error, -self.threshold, -self.threshold
            )
        error = torch.sum(error)
        if self.reduction == "mean":
            nsamples = torch.numel(error)
            error /= nsamples
        return error


class LayerWiseReconLoss(nn.Module):
    """
    MSE w/ layer-wise normalization
    """

    def __init__(self, reduction, index_dict, normalization_koeff=None, threshold=None):
        super(LayerWiseReconLoss, self).__init__()
        self.threshold = threshold
        if self.threshold:
            self.criterion = MSELossClipped(reduction="sum", threshold=self.threshold)
        else:
            self.criterion = nn.MSELoss(reduction="sum")
        self.reduction = reduction
        self.normalization_koeff = normalization_koeff
        self.get_index_idx(index_dict)
        self.loss_mean = None

    def forward(self, output, target):
        # check validity
        assert (
            output.shape == target.shape
        ), f"MSE loss error: prediction and target don't have the same shape. output {output.shape} vs target {target.shape}"
        # normalize outputs and targets
        if self.normalization_koeff is not None:
            dev = output.device
            self.normalization_koeff = self.normalization_koeff.to(dev)
            output = torch.clone(output) / self.normalization_koeff
            target = torch.clone(target) / self.normalization_koeff

        # compute layer-wise loss / rsq
        out = {}
        loss = torch.tensor(0.0, device=output.device).float()
        # iterate over layers
        for layer, idx_start, idx_end, loss_weight_idx in self.layer_index:
            # slice weight vector
            out_tmp = output[:, idx_start:idx_end]
            tar_tmp = target[:, idx_start:idx_end]
            # compute loss
            loss_tmp = self.criterion(out_tmp, tar_tmp)
            # reduction
            if self.reduction == "global_mean":
                # scale with overall number of paramters. each weight has the same contribution to loss
                loss_tmp /= output.shape[0] * output.shape[1]
            elif self.reduction == "layer_mean":
                # scale with layer number of paramters. each layer has the same contribution to loss
                loss_tmp /= output.shape[0] * out_tmp.shape[1]
            else:
                raise NotImplementedError
            # reweight with # of weights in this layer
            loss += loss_weight_idx * loss_tmp
            out[f"loss_recon_l{layer[0]}"] = loss_tmp.detach()
            # if loss_mean exists: compute rsq for this layer
            if self.loss_mean:
                out[f"rsq_l{layer[0]}"] = torch.tensor(
                    1
                    - loss_tmp.item() / self.loss_mean[f"loss_recon_l{layer[0]}"].item()
                )
        # pass loss_recon to output
        out["loss_recon"] = loss
        # of loss_mean exists: compute overall rsq
        if self.loss_mean:
            out["rsq"] = torch.tensor(
                1 - loss.item() / self.loss_mean["loss_recon"].item()
            )
        return out

    def get_index_idx(self, index_dict):
        """
        Helper function that creates a list of indeces per layer
        """
        # compute variance of the weights __per layer__
        index = []
        layer_lst = index_dict["layer"]
        start_lst = index_dict["idx_start"]
        length_lst = index_dict["idx_length"]
        loss_weight_lst = index_dict.get(
            "loss_weight", [1.0 for _ in start_lst]
        )  # get list of loss_weights, defaults to 1
        for idx in range(len(layer_lst)):
            idx_start = start_lst[idx]
            idx_end = start_lst[idx] + length_lst[idx]
            layer = layer_lst[idx]
            loss_weight = loss_weight_lst[idx]
            index.append((layer, idx_start, idx_end, loss_weight))
        self.layer_index = index

    def set_normalization(self, reference_weights, index_dict):
        """
        Function to set the normaliztion coefficient per layer
        """
        # compute std of the weights __per layer__
        norm_std = []
        for start, length in zip(index_dict["idx_start"], index_dict["idx_length"]):
            idx_start = start
            idx_end = start + length
            sliced = reference_weights[:, idx_start:idx_end]
            tmp = sliced.flatten()
            std_tmp = torch.std(tmp)
            # apply thresholding to prevent division by zero
            epsilon = 1e-4
            if std_tmp.item() < epsilon:
                std_tmp = torch.tensor(1) * epsilon
            std = torch.ones(sliced.shape[1]) * std_tmp
            norm_std.append(std)
        norm_std = torch.cat(norm_std, dim=0)
        assert (
            norm_std.shape[0] == reference_weights.shape[1]
        ), "normalization tensor and weights shape don't match."
        # set norm in recon loss
        self.normalization_koeff = norm_std

    def set_mean_loss(self, weights: torch.Tensor):
        """
        Function to compute the loss w.r.t the mean weights for rsq computation
        """
        # check that weights are tensor..
        assert isinstance(weights, torch.Tensor)
        w_mean = weights.mean(dim=0)  # compute over samples (dim0)
        # scale up to same size as weights
        weights_mean = repeat(w_mean, "d -> n d", n=weights.shape[0])
        loss_mean = self.forward(weights_mean, weights)

        # compute mean
        print(f" mean loss: {loss_mean}")

        self.loss_mean = loss_mean


################################################################################################
# KLD Loss for VAE
################################################################################################


class KldLoss(nn.Module):
    """
    Implementation adapted from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
    """

    def __init__(self, mu_ref=None, std_ref=None, normalization=False):
        super(KldLoss, self).__init__()
        self.mu_ref = mu_ref
        self.std_ref = std_ref

        # set to standard values
        if self.mu_ref == None:
            self.mu_ref = 0.0
        if self.std_ref == None:
            self.std_ref = 1.0

        self.normalization = normalization

    def forward(self, mu, log_var, norm_fact):
        """
        mu / logvar -> probability distro to sample from

        mu_ref and std_ref are the reference distribution values. expect equivariate gaussian

        """
        mu_ref = self.mu_ref
        std_ref = self.std_ref

        if mu_ref - 0.0 < 1e-5 and std_ref - 1.0 < 1e-5:
            # assume mu_ref = 0 and std ref = 1
            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
            )
        else:
            # general KL two Gaussians using logvar and std_ref
            u2 = torch.ones_like(mu) * mu_ref
            s2 = torch.ones_like(log_var) * std_ref
            kld_loss = self.kld_gauss_tensor(u1=mu, logvar1=log_var, u2=u2, s2=s2)

        if self.normalization and norm_fact:
            kld_loss /= norm_fact

        return kld_loss

    def kld_gauss_tensor(self, u1, logvar1, u2, s2):
        # general KL two Gaussians
        # u2, s2 often N(0,1)
        # https://stats.stackexchange.com/questions/7440/ +
        # kl-divergence-between-two-univariate-gaussians
        # log(s2/s1) + [( s1^2 + (u1-u2)^2 ) / 2*s2^2] - 0.5
        # var = s**2
        # given logvar_1
        # kl = 1/2* [ logvar2 - logvar1 + (var1+(u1-u2)**2)/var2 ]
        # var1 = logvar1.exp()
        var2 = s2 * s2
        # logvar2 = torch.log(var2)
        kl = 0.5 * (torch.log(var2) - logvar1 + (logvar1.exp() + (u1 - u2) ** 2) / var2)
        return kl


################################################################################################
# contrastive + recon loss combination
################################################################################################
class GammaContrastReconLoss(nn.Module):
    """
    Combines NTXent Loss with reconstruction loss.
    L = gamma*NTXentLoss + (1-gamma)*ReconstructionLoss
    """

    def __init__(
        self,
        gamma: float,
        reduction: str,
        batch_size: int,
        temperature: float,
        device: str,
        contrast="simclr",
        projection_head=False,
        threshold: float = None,
        z_var_penalty: float = 0.0,
        z_norm_penalty: float = 0.0,
        config=None,
    ) -> None:
        super(GammaContrastReconLoss, self).__init__()
        # test for allowable gamma values
        assert 0 <= gamma <= 1
        self.gamma = gamma

        self.projection_head = projection_head
        self.config = config

        # threshold for error clipping
        self.threshold = threshold

        # z_var penalty: l2 penalty on variance of embeddings
        self.z_var_penalty = z_var_penalty
        # z_norm penalty: l2 penalty on norm of embeddings
        self.z_norm_penalty = z_norm_penalty

        # set contrast
        if contrast == "simclr":
            print("model: use simclr NT_Xent loss")
            self.loss_contrast = NT_Xent(
                batch_size, temperature, device, self.projection_head, self.config
            )
        elif contrast == "positive":
            print("model: use only positive contrast loss")
            self.loss_contrast = NT_Xent_pos(
                batch_size, temperature, device, self.projection_head, self.config
            )
        else:
            print("unrecognized contrast - use reconstruction only")

        # set reconstruction loss
        index_dict = config["model::index_dict"]
        self.loss_recon = LayerWiseReconLoss(
            reduction=reduction,
            index_dict=index_dict,
            normalization_koeff=None,
            threshold=threshold,
        )

        # init loss_mean
        self.loss_mean = None

    # set mean loss for rsq computation
    def set_mean_loss(self, weights: torch.Tensor):
        # call function of recon loss
        self.loss_recon.set_mean_loss(weights)

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        z_i, z_j are the two different views of the same batch encoded in the representation space. dim: batch_size x representation space
        y: reconstruction. dim: batch_sizexinput_size
        t: target dim: batch_sizexinput_size
        """
        # case 1: only reconstruction
        if self.gamma < 1e-10:
            out_recon = self.loss_recon(y, t)
            out = {
                "loss": out_recon["loss_recon"],
                "loss_contrast": torch.tensor(0.0),
                "loss_recon": out_recon["loss_recon"],
            }
            for key in out_recon.keys():
                if key not in out:
                    out[key] = out_recon[key]
        # case 2: only contrast
        elif abs(1.0 - self.gamma) < 1e-10:
            loss_contrast = self.loss_contrast(z_i, z_j)
            out = {
                "loss": loss_contrast,
                "loss_contrast": loss_contrast,
                "loss_recon": torch.tensor(0.0),
            }
        # case 3: contrast + reconstruction
        else:
            # combine loss components
            loss_contrast = self.loss_contrast(z_i, z_j)
            out_recon = self.loss_recon(y, t)
            loss = (
                self.gamma * loss_contrast + (1 - self.gamma) * out_recon["loss_recon"]
            )
            out = {
                "loss": loss,
                "loss_contrast": loss_contrast,
                "loss_recon": out_recon["loss_recon"],
            }
            for key in out_recon.keys():
                if key not in out:
                    out[key] = out_recon[key]

        # compute embedding properties
        z_norm = torch.linalg.norm(z_i, ord=2, dim=1).mean()
        z_var = torch.mean(torch.var(z_i, dim=0))
        out["z_norm"] = z_norm
        out["z_var"] = z_var
        # add var/norm penalty terms
        out["loss"] = out["loss"] + self.z_var_penalty * z_var
        out["loss"] = out["loss"] + self.z_norm_penalty * z_norm

        return out

    def compute_mean_loss(self, dataloader):
        warnings.warn(
            "This computation of the mean loss is deprecated and will soon be replaced.",
            DeprecationWarning,
            stacklevel=2,
        )
        # step 1: compute data mean
        # get output data
        print(f"compute mean loss")
        print(f"len(dataloader): {len(dataloader)}")

        # get shape of data
        data_1, _, _, _ = next(iter(dataloader))
        print(f"compute x_mean")
        x_mean = torch.zeros(data_1.shape[1])
        print(f"x_mean.shape: {x_mean.shape}")
        n_data = 0
        # collect mean
        for idx, (data_1, _, _, _) in enumerate(dataloader):
            # compute mean weighted with batch size
            n_data += data_1.shape[0]
            x_mean += data_1.mean(dim=0) * data_1.shape[0]
        # scale x_mean back
        x_mean /= n_data
        n_data = 0
        loss_mean = 0
        # collect loss
        for idx, (data_1, _, _, _) in enumerate(dataloader):
            # compute mean weighted with batch size
            n_data += data_1.shape[0]
            # broadcast x_mean to target shape
            data_mean = torch.zeros(data_1.shape).add(x_mean)
            # commpute reconstruction loss
            loss_batch = self.loss_recon(data_1, data_mean)
            # add and weight
            loss_mean += loss_batch.item() * data_1.shape[0]
        # scale back
        loss_mean /= n_data

        # compute mean
        print(f" mean loss: {loss_mean}")

        return loss_mean
