import torch
import time
import random
import numpy as np
from utils.tools import time_since, get_labels
from transformers import BertTokenizer
from utils.pre_data import (
    load_data_target,
    pad_sequence_category,
    prepare_test_batch,
)
from models.ResidueRobertaModel import ResidueRobertaForTokenClassification

USE_CUDA = torch.cuda.is_available()

batch_size = 2
seed = 42
test_file = "data/allosteric_site/data_test.json"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


set_seed(seed)

model = ResidueRobertaForTokenClassification.from_pretrained("models/residue-roberta")
tokenizer = BertTokenizer.from_pretrained("models/residue-roberta")

# model size
size = 0
for n, p in model.named_parameters():
    size += p.nelement()
print("Total parameters: {}".format(size))

test_pair = load_data_target(test_file)
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
fp = 0
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
        try:
            target_len = target.index(0)
        except:
            target_len = len(target)
        temp = test_output[i][:target_len]
        if temp == target[:target_len]:
            sequence_acc += 1
        sequence_total += 1
        for l in range(len(temp)):
            total += 1
            if target[l] == 2:
                allosteric_total += 1
                if temp[l] == target[l]:
                    allosteric_ac += 1
                    ac += 1
            elif target[l] == 1:
                if temp[l] == target[l]:
                    ac += 1
                elif temp[l] == 2:
                    fp += 1

print("All residue site", ac, total)
print("Allosteric site", allosteric_ac, allosteric_total)
print("residue_acc", float(ac) / total)
if allosteric_ac + fp != 0:
    precision = float(allosteric_ac) / (allosteric_ac + fp)
    print("residue_precision", precision)
recall = float(allosteric_ac) / allosteric_total
print("residue_recall", recall)
if precision + recall != 0:
    print("residue_f1", (2 * precision * recall) / (precision + recall)) 
print("Sequence", sequence_acc, sequence_total)
print("sequence_acc", float(sequence_acc) / float(sequence_total))
print("testing time", time_since(time.time() - start_time))
print("-" * 100)