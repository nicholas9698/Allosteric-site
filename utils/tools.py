import math
import torch


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