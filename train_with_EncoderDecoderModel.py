import os
import torch
import time
import random
import operator
import numpy as np
from utils.tools import time_since
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from utils.pre_data import (
    load_data_target,
    prepare_train_batch,
    pad_sequence,
    prepare_test_batch,
)
from models.ResidueEncoderDecoderModel import ResidueEncoderDecoderModel

USE_CUDA = torch.cuda.is_available()

batch_size = 1
learning_rate = 5e-5
weight_decay = 1e-5
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

model = ResidueEncoderDecoderModel.from_encoder_decoder_pretrained(
    "models/residue-roberta", "models/residue-roberta", tie_encoder_decoder=True
)
tokenizer = BertTokenizer.from_pretrained("models/residue-roberta")
new_tokens = ["0", "1"]
tokenizer.add_tokens(new_tokens)
model.encoder.resize_token_embeddings(len(tokenizer))
model.decoder.resize_token_embeddings(len(tokenizer))

# set model's config
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.num_beams = 5
model.config.max_new_tokens = 1024
model.config.early_stopping = True

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
optimizer = AdamW(optimizer_ground_paramters, lr=learning_rate, eps=1e-8)
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
        inputs = pad_sequence(
            data_batches[idx], target_batches[idx], tokenizer, USE_CUDA
        )
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
            test_batch = pad_sequence(item, test_targets[idx], tokenizer, USE_CUDA)
            test_output = model.generate(
                input_ids=test_batch["input_ids"],
                xyz_position=test_batch["xyz_position"],
                attention_mask=test_batch["attention_mask"],
                max_new_tokens=1024,
                num_beams=5,
                num_return_sequences=1,
            )
            test_output = tokenizer.batch_decode(test_output, skip_special_tokens=True)

            for i, target in enumerate(test_targets[idx]):
                if operator.eq(test_output[i].strip().split(), target):
                    ac += 1
                total += 1

        print(ac, total)
        print("test_acc", float(ac) / total)
        print("testing time", time_since(time.time() - start_time))
        print("------------------------------------------------------")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model, output_dir + "model.pth")
        tokenizer.save_pretrained(output_dir)
