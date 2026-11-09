"""Multi-positive crop references with only explicit face-cut negatives."""
import torch
from torch.nn import functional as F

def acceptable_set_loss(scores,positive,negative):
    if scores.shape!=positive.shape or scores.shape!=negative.shape:raise ValueError('matching candidate masks required')
    if bool((positive&negative).any()):raise ValueError('positive and explicit-negative masks overlap')
    valid=positive.any(-1)&negative.any(-1)
    # Finite masking keeps empty rows differentiable with exactly zero gradient.
    positive_mass=torch.logsumexp(scores.masked_fill(~positive,-1e30),-1)
    negative_mass=torch.logsumexp(scores.masked_fill(~negative,-1e30),-1)
    return F.softplus(negative_mass-positive_mass)*valid,valid

