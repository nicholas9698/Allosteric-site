import os
import torch
import time
import random
import numpy as np
from utils.tools import time_since, mask_tokens
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from utils.pre_data import (
    load_data,
    prepare_train_batch_pretrain,
    pad_sequence_pretrain
)
from models.ResidueRobertaModel import ResidueRobertaForMaskedLM

USE_CUDA = torch.cuda.is_available()

batch_size = 8
accumulation_steps = 32
learning_rate = 5e-5
weight_decay = 1e-5
n_epoch = 40
seed = 42
train_file_dir = "data/pretrain_rcsb_inputs/"
temp_dir = "models/pretraing/"
output_dir = "models/residue-roberta"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
log_file = temp_dir+'pretraing-'+time.strftime("%Y-%m-%d_%H-%M", time.localtime())+'.log'
with open(log_file, 'w') as f:
    f.write("Strat time: "+time.asctime()+'\n')

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(seed)

model = ResidueRobertaForMaskedLM.from_pretrained("models/residue-roberta")
tokenizer = BertTokenizer.from_pretrained("models/residue-roberta")

# model size
size = 0
for n, p in model.named_parameters():
    size += p.nelement()
print("Total parameters: {}".format(size))

train_file_list = os.listdir(train_file_dir)
train_pair = []
for item in train_file_list:
    temp = load_data(train_file_dir+item)
    train_pair.extend(temp)


if USE_CUDA:
    model.cuda()

no_decay = ["bias", "LayerNorm.weight"]
optimizer_ground_paramters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": weight_decay,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_ground_paramters, lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0.1 * n_epoch, num_training_steps=n_epoch
)

model.zero_grad()

for epoch in range(n_epoch):
    loss_total = 0
    print("epoch", epoch + 1)
    start_time = time.time()

    data_batches = prepare_train_batch_pretrain(train_pair, batch_size)
    print("Finish prepare batches")

    model.train()

    for idx in range(len(data_batches)):
        inputs = pad_sequence_pretrain(data_batches[idx], tokenizer, USE_CUDA) 
        inputs["input_ids"], labels = mask_tokens(inputs["input_ids"], tokenizer)
        inputs["labels"] = labels
        loss = model(**inputs).loss
        loss_total += (loss.item() / len(data_batches))
        loss = loss / accumulation_steps
        loss.backward()
        
        if (idx+1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

    print("loss:", loss_total)
    print("training time", time_since(time.time() - start_time))
    print("--------------------------------")
    scheduler.step()


    model.eval()
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    torch.save(optimizer.state_dict(), temp_dir+'optimizer.pt')
    torch.save(scheduler.state_dict(), temp_dir+'scheduler.pt')
    with open(log_file, 'a') as f:
        f.write('-'*120+'\n')
        f.write("epoch: "+str(epoch+1)+'\n')
        f.write("loss: "+str(loss_total)+'\n')


model.eval()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
