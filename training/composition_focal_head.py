"""Research preference head over a frozen 62-value FocalNet descriptor."""
import torch
from torch import nn


class FocalPreferenceHead(nn.Module):
    def __init__(self, state, prior_logit):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(62), nn.Linear(62, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head.load_state_dict(state, strict=True)
        self.condition = nn.Linear(2, 64, bias=False)
        self.accept = nn.Linear(32, 1)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.accept.weight)
        nn.init.constant_(self.accept.bias, prior_logit)

    def forward(self, descriptors, conditions):
        hidden = self.head[1](self.head[0](descriptors)) + self.condition(conditions)
        hidden = self.head[4](self.head[3](self.head[2](hidden)))
        return torch.cat((self.head[5](hidden), self.accept(hidden)), dim=-1)
