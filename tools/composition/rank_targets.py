"""Relative local-score targets; uninformative queries remain unknown."""
import numpy as np


def targets(value, selected):
    mask = np.zeros(128, bool)
    mask[selected] = True
    deviation = float(value[selected].std()) if selected else 0.
    available = len(selected) >= 2 and deviation >= 1e-4
    target = np.zeros(128, np.float32)
    if available:
        target[selected] = (value[selected]-value[selected].mean())/deviation
    ordered = sorted(selected, key=lambda j: (-float(value[j]), j))
    options = [(a, b) for a in ordered for b in reversed(ordered) if available and target[a]-target[b] >= .25]
    if options:
        options = [options[j] for j in np.linspace(0, len(options)-1, min(32, len(options)), dtype=int)]
    pair = np.zeros((32, 2), np.int64)
    pair[:len(options)] = np.asarray(options, np.int64).reshape(-1, 2)
    return mask, available, target, pair, np.arange(32) < len(options)
