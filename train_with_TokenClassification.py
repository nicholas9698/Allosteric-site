import os
import sys
import torch
import time
import random
import operator
import numpy as np
from utils.tools import time_since, get_labels
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from utils.pre_data import (
    load_data_target,
    prepare_train_batch,
    pad_sequence_category,
    prepare_test_batch,
)
from models.ResidueRobertaModel import ResidueRobertaForTokenClassification

USE_CUDA = torch.cuda.is_available()

batch_size = 2
learning_rate = 5e-5
weight_decay = 1e-2
n_epoch = 80
seed = 42
train_file = "data/allosteric_site/data_train.json"
test_file = "data/allosteric_site/data_test.json"
output_dir = "test/"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


set_seed(seed)

model = ResidueRobertaForTokenClassification.from_pretrained("models/residue-roberta", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("models/residue-roberta")
new_tokens = ["0", "1"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# model size
size = 0
for n, p in model.named_parameters():
    size += p.nelement()
print("Total parameters: {}".format(size))

train_pair = load_data_target(train_file, tokenizer)
test_pair = load_data_target(test_file, tokenizer)
test_batches, test_targets = prepare_test_batch(test_pair, batch_size)

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
optimizer = AdamW(optimizer_ground_paramters, lr=learning_rate, eps=1e-8, betas=(0.9, 0.98))
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0.1 * n_epoch, num_training_steps=n_epoch
)

model.zero_grad()

for epoch in range(n_epoch):
    loss_total = 0
    print("epoch", epoch + 1)
    data_batches, target_batches = prepare_train_batch(train_pair, batch_size)
    start_time = time.time()
    model.train()
    for idx in range(len(data_batches)):
        inputs = pad_sequence_category(
            data_batches[idx], target_batches[idx], tokenizer, USE_CUDA
        )
        # for item in inputs['labels']:
        #     print(item)
        # sys.exit()
        loss = model(**inputs).loss
        loss.backward()
        loss_total += loss.item()

        optimizer.step()
        model.zero_grad()

    print("loss:", loss_total / len(data_batches))
    print("training time", time_since(time.time() - start_time))
    print("--------------------------------")
    scheduler.step()

    if (epoch + 1) % 1 == 0 or n_epoch - epoch < 6:
        start_time = time.time()
        model.eval()
        total = 0
        ac = 0

        for idx, item in enumerate(test_batches):
            test_batch = pad_sequence_category(item, test_targets[idx], tokenizer, USE_CUDA)
            test_output = model(
                input_ids=test_batch["input_ids"],
                xyz_position=test_batch["xyz_position"],
                attention_mask=test_batch["attention_mask"],
                labels=None
            )
            test_output = get_labels(test_output['logits'])

            for i, target in enumerate(test_targets[idx]):
                if operator.eq(test_output[i][:len(target)], target):
                    ac += 1
                total += 1
        print(ac, total)
        print("test_acc", float(ac) / total)
        print("testing time", time_since(time.time() - start_time))
        print("------------------------------------------------------")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
