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

batch_size = 4
seed = 42
train_file = "data/allosteric_site/data_train.json"
test_file = "data/allosteric_site/data_test.json"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


set_seed(seed)

model = ResidueRobertaForTokenClassification.from_pretrained("test")
tokenizer = BertTokenizer.from_pretrained("test")

# model size
size = 0
for n, p in model.named_parameters():
    size += p.nelement()
print("Total parameters: {}".format(size))

test_pair = load_data_target(test_file, tokenizer)
test_batches, test_targets = prepare_test_batch(test_pair, batch_size)

if USE_CUDA:
    model.cuda()

model.zero_grad()

start_time = time.time()
model.eval()

sequence_acc = 0
sequence_total = 0
allosteric_total = 0
allosteric_ac = 0
ac = 0
total = 0

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
        temp = test_output[i][:len(target)]
        if temp == target:
            sequence_acc += 1
        sequence_total += 1
        for l in range(len(temp)):
            if temp[l] == target[l] and target[l] == 2:
                allosteric_ac += 1
                allosteric_total += 1
                ac += 1
            elif temp[l] == target[l]:
                ac += 1
            elif target[l] == 2:
                allosteric_total += 1

            total += 1
print("All residue site", ac, total)
print("Sequence", sequence_acc, sequence_total)
print("Allosteric site", allosteric_ac, allosteric_total)
print("residue_acc", float(ac) / total)
print("residue_recall", float(allosteric_ac) / allosteric_total)
print("sequence_acc", float(sequence_acc) / float(sequence_total))
print("testing time", time_since(time.time() - start_time))
print("------------------------------------------------------")