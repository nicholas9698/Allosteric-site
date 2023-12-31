import os
import torch
import time
import random
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

batch_size = 8
learning_rate = 5e-5
weight_decay = 1e-5
n_epoch = 80
seed = 42
train_file = "data/allosteric_site/data_train_orthosite.json"
test_file = "data/allosteric_site/data_test_orthosite.json"
output_dir = "models/fine-tuned/token_classification_CE/"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


set_seed(seed)

model = ResidueRobertaForTokenClassification.from_pretrained("models/pretrained_model", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("models/pretrained_model")

# model size
size = 0
for n, p in model.named_parameters():
    size += p.nelement()
print("Total parameters: {}".format(size))

train_pair = load_data_target(train_file)
test_pair = load_data_target(test_file)
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
optimizer = AdamW(optimizer_ground_paramters, lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0.1 * n_epoch, num_training_steps=n_epoch
)

model.zero_grad()

for epoch in range(n_epoch):
    loss_total = 0
    print("epoch", epoch + 1)
    start_time = time.time()

    data_batches, target_batches = prepare_train_batch(train_pair, batch_size)
    
    model.train()
    for idx in range(len(data_batches)):
        inputs = pad_sequence_category(
            data_batches[idx], target_batches[idx], tokenizer, USE_CUDA
        )

        loss = model(**inputs).loss
        loss.backward()
        loss_total += (loss.item() / len(data_batches))

        optimizer.step()
        model.zero_grad()

    print("loss:", loss_total)
    print("training time", time_since(time.time() - start_time))
    print("--------------------------------")
    scheduler.step()

    if (epoch + 1) % 5 == 0 or n_epoch - epoch < 6:
        start_time = time.time()
        model.eval()
        sequence_acc = 0
        sequence_total = 0
        allosteric_total = 0
        allosteric_ac = 0
        fp = 0
        ac = 0
        total = 0

        for idx, item in enumerate(test_batches):
            test_batch = pad_sequence_category(item, test_targets[idx], tokenizer, USE_CUDA)
            test_output = model(
                input_ids=test_batch["input_ids"],
                xyz_position=test_batch["xyz_position"],
                orthosteric_position=test_batch["orthosteric_position"],
                attention_mask=test_batch["attention_mask"],
                labels=None,
                adjustment=None
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

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
