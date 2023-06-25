import torch
import random
import numpy as np
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from models.ResidueEncoderDecoderModel import ResidueEncoderDecoderModel

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

model = ResidueEncoderDecoderModel.from_encoder_decoder_pretrained('models/residue-roberta', 'models/residue-roberta', tie_encoder_decoder=True, encoder_config='models/residue-roberta/config.json')
tokenizer = BertTokenizer.from_pretrained('models/residue-roberta')

# set model's config
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.num_beams = 5
model.config.max_new_tokens = 2048
model.config.early_stopping = True

# model size
size = 0    
for n, p in model.named_parameters():
    size += p.nelement()
print('Total parameters: {}'.format(size))

# model.save_pretrained('test/')