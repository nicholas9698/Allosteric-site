import torch
import random
import numpy as np
from transformers import BertModel, EncoderDecoderModel

USE_CUDA = torch.cuda.is_available()

batch_size = 32
test_batch_size = 32
learning_rate = 5e-5
weight_decay = 1e-5
n_epoch = 80
seed = 42

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(seed)
