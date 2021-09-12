
# statistics utility

import torch
import torch.nn.functional as F


def get_gumbels(num_gunbels):
    # F^-1(u) = -log(-log(u))
    # u ~ Uniform(0, 1)
    return -torch.log(-torch.log(torch.rand(num_gunbels)))


def get_perturbed_log_probs(parent_perturbed_log_prob, child_log_probs):
    gumbels = get_gumbels(len(child_log_probs))
    perturbed_log_probs = child_log_probs + gumbels

    assert perturbed_log_probs.dim() == 1
    max_child_perturbed_log_prob, _ = perturbed_log_probs.max(dim=0)

    T = parent_perturbed_log_prob
    Z = max_child_perturbed_log_prob

    u = T - perturbed_log_probs + torch.log1p(- torch.exp(perturbed_log_probs - Z))
    perturbed_log_probs = T - F.relu(u) - torch.log1p(torch.exp(-u.abs()))

    return perturbed_log_probs, gumbels
