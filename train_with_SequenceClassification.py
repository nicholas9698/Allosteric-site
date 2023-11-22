import os
import torch
import time
import random
import numpy as np
from utils.tools import time_since, get_labels, compute_adjustment
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from utils.pre_data import (
    load_data_target,
    prepare_train_batch,
    pad_sequence_seq,
    prepare_test_batch,
)
from models.ResidueRobertaModel import ResidueRobertaForSequenceClassification

USE_CUDA = torch.cuda.is_available()

batch_size = 8
learning_rate = 5e-5
weight_decay = 1e-5
n_epoch = 80
seed = 42
train_file = "data/allosteric_site/data_train.json"
test_file = "data/allosteric_site/data_test.json"
output_dir = "models/fine-tuned/sequence_classification/"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


set_seed(seed)

model = ResidueRobertaForSequenceClassification.from_pretrained("models/residue-roberta", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("models/residue-roberta")

# model size
size = 0
for n, p in model.named_parameters():
    size += p.nelement()
print("Total parameters: {}".format(size))

train_pair = load_data_target(train_file, True)
test_pair = load_data_target(test_file, True)
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

target_loader = {"labels": torch.LongTensor([_[3] for _ in train_pair])}
adjustment = compute_adjustment(target_loader, tro=1.0, USE_CUDA=USE_CUDA)

for epoch in range(n_epoch):
    loss_total = 0
    print("epoch", epoch + 1)
    start_time = time.time()

    data_batches, target_batches = prepare_train_batch(train_pair, batch_size)
    
    model.train()
    for idx in range(len(data_batches)):
        inputs = pad_sequence_seq(
            data_batches[idx], target_batches[idx], tokenizer, USE_CUDA
        )
        inputs["adjustment"] = adjustment

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
        allosteric_total = 0
        allosteric_ac = 0
        fp = 0
        ac = 0
        total = 0

        for idx, item in enumerate(test_batches):
            test_batch = pad_sequence_seq(item, test_targets[idx], tokenizer, USE_CUDA)
            test_output = model(
                input_ids=test_batch["input_ids"],
                xyz_position=test_batch["xyz_position"],
                pocket_position=None,
                token_type_ids=test_batch["token_type_ids"],
                attention_mask=test_batch["attention_mask"],
                labels=None,
                adjustment=None
            )
            test_output = get_labels(test_output['logits'], pocket_classification=True)

            for i, target in enumerate(test_targets[idx]):
                total += 1
                if target == 1:
                    allosteric_total += 1
                    if target == test_output[i]:
                        allosteric_ac += 1
                        ac += 1
                elif target == 0:
                    if target == test_output[i]:
                        ac += 1
                    else:
                        fp += 1

        print("All residue pocket", ac, total)
        print("Allosteric pocket", allosteric_ac, allosteric_total)
        print("pocket_acc", float(ac) / total)
        if allosteric_ac + fp != 0:
            precision = float(allosteric_ac) / (allosteric_ac + fp)
            print("pocket_precision", precision)
        recall = float(allosteric_ac) / allosteric_total
        print("pocket_recall", recall)
        if precision + recall != 0:
            print("pocket_f1", (2 * precision * recall) / (precision + recall)) 
        print("testing time", time_since(time.time() - start_time))
        print("-" * 100)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
