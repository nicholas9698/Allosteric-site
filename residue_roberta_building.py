import os
import torch
import random
import numpy as np
from transformers import RobertaConfig, BertTokenizer
from models.ResidueRobertaModel import ResidueRobertaModel

USE_CUDA = torch.cuda.is_available()

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)
# Initializing a RoBERTa configuration
configuration = RobertaConfig(vocab_size=100, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', 
                              hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=2048, type_vocab_size=2, initializer_range=0.02, 
                              layer_norm_eps=1e-12, pad_token_id=0, bos_token_id=1, eos_token_id=2, position_embedding_type='absoulte', use_cache=True, 
                              classifier_dropout=None, is_decoder=False)
model = ResidueRobertaModel(configuration)

if USE_CUDA:
    model.cuda()

tokenizer = BertTokenizer.from_pretrained('models/tokenizer/residue')

if not os.path.exists('models/residue-roberta'):
    os.mkdir('models/residue-roberta')

model.save_pretrained('models/residue-roberta')
tokenizer.save_pretrained('models/residue-roberta')