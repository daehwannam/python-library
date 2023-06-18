
# statistics utility

import torch
import torch.nn.functional as F

from ..iteration import distinct_pairs


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


def get_measure(name, higher_better: bool):
    return dict(name=name, higher_better=higher_better)


def get_performance(*args, **kwargs):
    '''Pairs of measure names and values'''
    return dict(distinct_pairs(*args, **kwargs))


def is_better_value(value1, value2, higher_better):
    if higher_better:
        return value1 > value2
    else:
        return value1 < value2


def is_better_performance(performance1, performance2, measures):
    assert len(performance1) == len(performance2) == len(measures)
    for measure in measures:
        score1 = performance1[measure['name']]
        score2 = performance2[measure['name']]
        higher_better = measure['higher_better']
        if is_better_value(score1, score2, higher_better):
            return True
        elif is_better_value(score2, score1, higher_better):
            return False
    return False
