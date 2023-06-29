import math
import torch
import numpy as np

# compute time
def time_since(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

# transform prob to label
def get_labels(probs: torch.Tensor):
    results = torch.max(probs, 2) 
    return results[1].tolist()

# logits adjustment
def compute_adjustment(targets_loader: dict, tro: float, USE_CUDA: bool):
    """compute the base probabilities"""

    label_freq = {}

    targets = targets_loader['labels']
    for target in targets:
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    if USE_CUDA:
        adjustments = adjustments.cuda()
    return adjustments