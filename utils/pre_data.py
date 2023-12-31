import os
import copy
import random
import torch
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

'''
    Universal function
'''
def load_data_target(train_file: str, pocket_classification: bool = False):
    if pocket_classification:
        with open(train_file, "r") as f:
            train_set = json.load(f)

        train_pair = []

        print("Processing data...")
        for item in tqdm(train_set):
            input = item["input"]["sequence"]
            indecies = [i for i,j in enumerate(item["input"]["pocket_input"]) if j == 2]
            # The [CLS] and [EOS] are added for sequence classification
            if (len(input)+len(indecies)) > 1024:
                continue
            x_s = item["input"]["x_s"]
            y_s = item["input"]["y_s"]
            z_s = item["input"]["z_s"]
            positions = []
            for i in range(len(x_s)):
                positions.append([x_s[i], y_s[i], z_s[i]])

            target = item["target"]
            pocket_position = item["input"]["pocket_input"]
            if "orthosite" in item["input"].keys():
                orthosite = item["input"]["orthosite"]
                train_pair.append((input, positions, pocket_position, orthosite, target))
            else:
                train_pair.append((input, positions, pocket_position, target))

        print("Load data with (sequence+pocket) <= 1024: ", len(train_pair))

        return train_pair
    else:
        with open(train_file, "r") as f:
            train_set = json.load(f)

        train_pair = []

        print("Processing data...")
        for item in tqdm(train_set):
            input = item["input"]["sequence"]
            if len(input) > 1024:
                continue
            x_s = item["input"]["x_s"]
            y_s = item["input"]["y_s"]
            z_s = item["input"]["z_s"]
            positions = []
            for i in range(len(x_s)):
                positions.append([x_s[i], y_s[i], z_s[i]])

            target = item["target"]

            if "orthosite" in item["input"].keys():
                orthosite = item["input"]["orthosite"]
                train_pair.append((input, positions, target, orthosite))
            else:
                train_pair.append((input, positions, target))

        print("Load data with sequence <= 1024: ", len(train_pair))

        return train_pair
    
def prepare_test_batch(data_train: list, batch_size: int):
    train_pair = data_train
    batches = []
    pos = 0

    while pos + batch_size < len(train_pair):
        batches.append(train_pair[pos : pos + batch_size])
        pos += batch_size
    batches.append(train_pair[pos:])

    train_inputs = []
    train_targets = []
    
    count_input = len(batches[0][0])
    if count_input == 4:
        for batch in batches:
            train_input = []
            train_target = []
            for item in batch:
                train_input.append((item[0], item[1], item[2]))
                train_target.append(item[3])
            train_inputs.append(train_input)
            train_targets.append(train_target)
    elif count_input == 3:
        for batch in batches:
            train_input = []
            train_target = []
            for item in batch:
                train_input.append((item[0], item[1]))
                train_target.append(item[2])
            train_inputs.append(train_input)
            train_targets.append(train_target)

    return train_inputs, train_targets

def prepare_train_batch(data_train: list, batch_size: int):
    train_pair = copy.deepcopy(data_train)
    random.shuffle(train_pair)
    batches = []
    pos = 0

    while pos + batch_size < len(train_pair):
        batches.append(train_pair[pos : pos + batch_size])
        pos += batch_size
    batches.append(train_pair[pos:])

    train_inputs = []
    train_targets = []
    
    count_input = len(batches[0][0])
    if count_input == 4:
        for batch in batches:
            train_input = []
            train_target = []
            for item in batch:
                train_input.append((item[0], item[1], item[2]))
                train_target.append(item[3])
            train_inputs.append(train_input)
            train_targets.append(train_target)
    elif count_input == 3:
        for batch in batches:
            train_input = []
            train_target = []
            for item in batch:
                train_input.append((item[0], item[1]))
                train_target.append(item[2])
            train_inputs.append(train_input)
            train_targets.append(train_target)        

    return train_inputs, train_targets


'''
    Functions of the ResidueRobertaMLM pre-training
'''

# Load the pre-training data from file
def load_data(train_file: str):
    with open(train_file, "r") as f:
        train_set = json.load(f)

    train_pair = []

    print("Processing data (Tips: Sequence with len(sequence)>1024 will be drop)...")
    for item in tqdm(train_set):
        input = item["sequence"]
        if len(input) > 1024:
            continue
        x_s = item["x_s"]
        y_s = item["y_s"]
        z_s = item["z_s"]
        positions = []
        for i in range(len(x_s)):
            positions.append([x_s[i], y_s[i], z_s[i]])

        train_pair.append((input, positions))
    return train_pair

# Split training data by batch-size, and shuffle data in each epoch
def prepare_train_batch_pretrain(data_train: list, batch_size: int):
    random.shuffle(data_train)
    train_pair = copy.deepcopy(data_train)
    batches = []
    pos = 0

    while pos + batch_size < len(train_pair):
        batches.append(train_pair[pos : pos + batch_size])
        pos += batch_size
    batches.append(train_pair[pos:])

    train_inputs = []
    for batch in batches:
        train_input = []
        for item in batch:
            train_input.append((item[0], item[1]))
        train_inputs.append(train_input)

    return train_inputs

#  Pad the batch with tokenizer.pad_token_id to get the same length inputs.
def pad_sequence_pretrain(input_ls: list, tokenizer: BertTokenizer, USE_CUDA: bool):
    max_length = 0
    xyz_positions = []
    sequences = []
    input_s = []
    for item in input_ls:
        sequences.append(item[0])
        xyz_positions.append(item[1])
        if max_length < len(item[0]):
            max_length = len(item[0])
    for i in range(len(xyz_positions)):
        current_len = len(xyz_positions[i])
        xyz_positions[i].extend(
            [[0.0, 0.0, 0.0] for _ in range(max_length - current_len)]
        )

    input_s = tokenizer(sequences, return_tensors="pt", add_special_tokens=False, padding=True, is_split_into_words=True)
    xyz_positions = torch.FloatTensor(xyz_positions)

    if USE_CUDA:
        inputs = {
            "input_ids": input_s.input_ids.cuda(),
            "xyz_position": xyz_positions.cuda(),
            "attention_mask": input_s.attention_mask.cuda(),
        }
    else:
        inputs = {
            "input_ids": input_s.input_ids,
            "xyz_position": xyz_positions,
            "attention_mask": input_s.attention_mask,
        }

    return inputs


'''
    Functions of ResidueRoberta to achieve TokenClassification
'''
def pad_sequence_category(
    input_ls: list, target_ls: list, tokenizer: BertTokenizer, USE_CUDA: bool
):
    max_length = 0
    xyz_positions = []
    sequences = []
    target_s = []
    input_s = []
    orthosite_s = []

    count_input = len(input_ls[0])
    if count_input == 2:
        for item in input_ls:
            sequences.append(item[0])
            xyz_positions.append(item[1])

            if max_length < len(item[0]):
                max_length = len(item[0])
        for i in range(len(xyz_positions)):
            current_len = len(xyz_positions[i])
            xyz_positions[i].extend(
                [[0.0, 0.0, 0.0] for _ in range(max_length - current_len)]
            )
            target_ls[i].extend([0 for _ in range(max_length - current_len)])
            target_s.append(target_ls[i])
        
        input_s = tokenizer(
            sequences,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            is_split_into_words=True,
        )
        target_s = torch.LongTensor(target_s)
        xyz_positions = torch.FloatTensor(xyz_positions)

        if USE_CUDA:
            inputs = {
                "input_ids": input_s.input_ids.cuda(),
                "xyz_position": xyz_positions.cuda(),
                "attention_mask": input_s.attention_mask.cuda(),
                "labels": target_s.cuda(),
            }
        else:
            inputs = {
                "input_ids": input_s.input_ids,
                "xyz_position": xyz_positions,
                "attention_mask": input_s.attention_mask,
                "labels": target_s,
            }
    elif count_input == 3:
        for item in input_ls:
            sequences.append(item[0])
            xyz_positions.append(item[1])
            orthosite_s.append(item[2])
            
            if max_length < len(item[0]):
                max_length = len(item[0])
        for i in range(len(xyz_positions)):
            current_len = len(xyz_positions[i])
            xyz_positions[i].extend(
                [[0.0, 0.0, 0.0] for _ in range(max_length - current_len)]
            )
            orthosite_s[i].extend([0 for _ in range(max_length - current_len)])
            target_ls[i].extend([0 for _ in range(max_length - current_len)])
            target_s.append(target_ls[i]) 

        input_s = tokenizer(
            sequences,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            is_split_into_words=True,
        )
        target_s = torch.LongTensor(target_s)
        orthosite_s = torch.LongTensor(orthosite_s)
        xyz_positions = torch.FloatTensor(xyz_positions)

        if USE_CUDA:
            inputs = {
                "input_ids": input_s.input_ids.cuda(),
                "xyz_position": xyz_positions.cuda(),
                "attention_mask": input_s.attention_mask.cuda(),
                "orthosteric_position": orthosite_s.cuda(),
                "labels": target_s.cuda(),
            }
        else:
            inputs = {
                "input_ids": input_s.input_ids,
                "xyz_position": xyz_positions,
                "attention_mask": input_s.attention_mask,
                "orthosteric_position": orthosite_s,
                "labels": target_s,
            }

    return inputs

# Apply logits adjustment to balance Indicates the influence of unbalanced labels
def prepare_train_data(data_train: list):
    train_inputs = []
    train_targets = []
    for item in data_train:
        train_inputs.append((item[0], item[1]))
        train_targets.append(item[2])

    return train_inputs, train_targets

def prepare_train_batch_adjust(inputs: list, batch_size: int):
    train_pair = copy.deepcopy(inputs)
    random.shuffle(train_pair)
    batches = []
    pos = 0

    while pos + batch_size < len(train_pair):
        batches.append(train_pair[pos : pos + batch_size])
        pos += batch_size
    batches.append(train_pair[pos:])

    final_inputs = []
    for batch in batches:
        input_ids = []
        xyz_position = []
        attention_mask = []
        labels = []
        for item in batch:
            input_ids.append(item[0])
            xyz_position.append(item[1])
            attention_mask.append(item[2])
            labels.append(item[3])
        final_inputs.append(
            {
                "input_ids": torch.LongTensor(input_ids),
                "xyz_position": torch.FloatTensor(xyz_position),
                "attention_mask": torch.FloatTensor(attention_mask),
                "labels": torch.LongTensor(labels),
            }
        )
    return final_inputs

def inputs_to_list(inputs: dict):
    input_ids = inputs["input_ids"].tolist()
    xyz_position = inputs["xyz_position"].tolist()
    attention_mask = inputs["attention_mask"].tolist()
    labels = inputs["labels"].tolist()

    total_inputs = []
    for i in range(len(input_ids)):
        total_inputs.append(
            (input_ids[i], xyz_position[i], attention_mask[i], labels[i])
        )

    return total_inputs


'''
    Function for SequenceClassification
'''
def pad_sequence_seq(input_ls: list, target_ls: list, tokenizer: BertTokenizer, USE_CUDA: bool, deepcopy: bool = False):
    if deepcopy:
        input_ls = copy.deepcopy(input_ls)
    max_length = 0
    xyz_positions = []
    sequences = []
    target_s = []
    pocket_s = []
    input_s = []
    token_type_ids = []
    for item in input_ls:
        sequences.append(item[0])
        token_type_ids.append([0 for _ in range(len(item[0]))])
        xyz_positions.append(item[1])
        # pocket_s.append(item[2])
        pocket_s.append([i for i,j in enumerate(item[2]) if j == 2])
        if max_length < (len(item[0]) + len(pocket_s[-1])):
            max_length = len(item[0]) + len(pocket_s[-1])
    for i in range(len(xyz_positions)):
        # xyz_positions[i].insert(0, [0.0, 0.0, 0.0])
        # xyz_positions[i].append([0.0, 0.0, 0.0])
        # pocket_s[i].insert(0, 0)
        # pocket_s[i].append(0)
        # current_len = len(xyz_positions[i]) - 2 

        for idx in pocket_s[i]:
            sequences[i].append(sequences[i][idx])
            xyz_positions[i].append(xyz_positions[i][idx])
            token_type_ids[i].append(1)
        current_len = len(xyz_positions[i])
        xyz_positions[i].extend([[0.0, 0.0, 0.0] for _ in range(max_length - current_len)])
        # pocket_s[i].extend([0 for _ in range(max_length - current_len)])
        token_type_ids[i].extend([1 for _ in range(max_length - current_len)])

    input_s = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        is_split_into_words=True,
    )
    target_s = torch.LongTensor(target_ls)
    xyz_positions = torch.FloatTensor(xyz_positions)
    token_type_ids = torch.LongTensor(token_type_ids)

    if USE_CUDA:
        inputs = {
            "input_ids": input_s.input_ids.cuda(),
            "xyz_position": xyz_positions.cuda(),
            "pocket_position": None,
            "token_type_ids": token_type_ids.cuda(),
            "attention_mask": input_s.attention_mask.cuda(),
            "labels": target_s.cuda(),
        }
    else:
        inputs = {
            "input_ids": input_s.input_ids,
            "xyz_position": xyz_positions,
            "pocket_position": None,
            "token_type_ids": token_type_ids,
            "attention_mask": input_s.attention_mask,
            "labels": target_s,
        }

    return inputs


'''
    Functions for EncoderDecoderModel to generate sequence-structured predictions
'''
def pad_sequence(
    input_ls: list, target_ls: list, tokenizer: BertTokenizer, USE_CUDA: bool
):
    max_length = 0
    xyz_positions = []
    sequences = []
    for item in input_ls:
        sequences.append(item[0])
        xyz_positions.append(item[1])
        if max_length < len(item[0]):
            max_length = len(item[0])
    for i in range(len(xyz_positions)):
        while len(xyz_positions[i]) < max_length:
            xyz_positions[i].append([0.0, 0.0, 0.0])

    input_ls = tokenizer(
        sequences,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        is_split_into_words=True,
    )
    target_ls = tokenizer(
        target_ls,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        is_split_into_words=True,
    )
    xyz_positions = torch.Tensor(xyz_positions)
    if USE_CUDA:
        inputs = {
            "input_ids": input_ls.input_ids.cuda(),
            "xyz_position": xyz_positions.cuda(),
            "attention_mask": input_ls.attention_mask.cuda(),
            "labels": target_ls.input_ids.cuda(),
        }
    else:
        inputs = {
            "input_ids": input_ls.input_ids,
            "xyz_position": xyz_positions,
            "attention_mask": input_ls.attention_mask,
            "labels": target_ls.input_ids,
        }

    return inputs


'''
    Functions for preparing dataset
'''
# 制作数据（仅提取变构位点json中的第一条链，原始pdb中的对应链）
def pre_single_a(target_dir: str, pdb_dir: str, output_json: str):
    target_jsons = os.listdir(target_dir)
    data = []
    for target in tqdm(target_jsons):
        if "_1.json" in target:
            with open(target_dir + target, "r") as f:
                ls = json.load(f)
            if len(ls) > 0:
                data.append(
                    {
                        "pdbid": (ls[0])["pdbid"][-6:-2],
                        "chain": (ls[0])["chain"],
                        "target_orders": (ls[0])["orders"],
                    }
                )
                with open(pdb_dir + target[-11:-7].upper() + ".json", "r") as f:
                    origin = json.load(f)
                for single in origin:
                    if single["chain"] == (data[-1])["chain"]:
                        data[-1]["origin_orders"] = single["orders"]
                        data[-1]["residues"] = single["residues"]
                        data[-1]["positions"] = single["positions"]
                        break
    with open(output_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 处理所有的数据
def pre_data(target_dir: str, pdb_dir: str, output_json: str):
    target_jsons = os.listdir(target_dir)
    data = []
    pdbid_chain = []
    for target in tqdm(target_jsons):
        with open(target_dir + target, "r") as f:
            ls = json.load(f)
        with open(pdb_dir + target[-11:-7].upper() + ".json", "r") as f:
            origin = json.load(f)
        for item in ls:
            now_pdbid_chain = item["pdbid"][-6:-2] + item["chain"]
            if now_pdbid_chain not in pdbid_chain:
                temp = {
                    "pdbid": item["pdbid"][-6:-2],
                    "chain": item["chain"],
                    "target_orders": item["orders"],
                }
                data.append(temp)
                for single in origin:
                    if single["chain"] == (data[-1])["chain"]:
                        data[-1]["origin_orders"] = single["orders"]
                        data[-1]["residues"] = single["residues"]
                        data[-1]["positions"] = single["positions"]
                        break
                pdbid_chain.append(now_pdbid_chain)
            else:
                pos = pdbid_chain.index(now_pdbid_chain)
                temp_orders = item["orders"].strip().split()
                previous_orders = (data[pos])["target_orders"].strip().split()
                temp_orders.extend(previous_orders)
                temp_orders = list(set(temp_orders))
                temp_orders = list(map(int, temp_orders))
                temp_orders = sorted(temp_orders)
                temp_orders = list(map(str, temp_orders))
                (data[pos])["target_orders"] = " ".join(temp_orders)

    with open(output_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def pre_data_uni(target_dir: str, pdb_dir: str, output_json: str):
    target_jsons = os.listdir(target_dir)
    data = []
    pdbid_chain = []
    for target in tqdm(target_jsons):
        with open(target_dir + target, "r") as f:
            ls = json.load(f)
        if len(ls) > 1:
            continue
        with open(pdb_dir + target[-11:-7].upper() + ".json", "r") as f:
            origin = json.load(f)
        for item in ls:
            now_pdbid_chain = item["pdbid"][-6:-2] + item["chain"]
            if now_pdbid_chain not in pdbid_chain:
                temp = {
                    "pdbid": item["pdbid"][-6:-2],
                    "chain": item["chain"],
                    "target_orders": item["orders"],
                }
                data.append(temp)
                for single in origin:
                    if single["chain"] == (data[-1])["chain"]:
                        data[-1]["origin_orders"] = single["orders"]
                        data[-1]["residues"] = single["residues"]
                        data[-1]["positions"] = single["positions"]
                        break
                pdbid_chain.append(now_pdbid_chain)
            else:
                pos = pdbid_chain.index(now_pdbid_chain)
                temp_orders = item["orders"].strip().split()
                previous_orders = (data[pos])["target_orders"].strip().split()
                temp_orders.extend(previous_orders)
                temp_orders = list(set(temp_orders))
                temp_orders = list(map(int, temp_orders))
                temp_orders = sorted(temp_orders)
                temp_orders = list(map(str, temp_orders))
                (data[pos])["target_orders"] = " ".join(temp_orders)

    with open(output_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def pre_data_rcsb(rcsb_dir: str, output_json: str, split: int = None):
    rcsb_jsons = os.listdir(rcsb_dir)
    data = []
    order = 0
    if split != None:
        for rcsb in tqdm(rcsb_jsons):
            with open(rcsb_dir + rcsb, "r") as f:
                ls = json.load(f)
            for item in ls:
                data.append(item)
            if len(data) > split:
                print("Writing to " + output_json[:-5] + "_" + str(order) + ".json")

                with open(output_json[:-5] + "_" + str(order) + ".json", "w") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    del data
                    data = []
                    order += 1

        if len(data) > 0:
            with open(output_json[:-5] + "_" + str(order) + ".json", "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        for rcsb in tqdm(rcsb_jsons):
            with open(rcsb_dir + rcsb, "r") as f:
                ls = json.load(f)
            for item in ls:
                data.append(item)

        with open(output_json, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def transform_data(data_path: str):
    with open(data_path, "r") as load_f:
        data = json.load(load_f)
    inputs = []
    targets = []
    for item in tqdm(data):
        residues = []
        positions = []
        orders = []
        temp_resi = item["residues"].strip().split()
        temp_target = item["target_orders"].strip().split()
        temp_orders = item["origin_orders"].strip().split()
        temp_positions = item["positions"]
        current_order = int(temp_orders[0])
        orders.append(str(current_order))
        positions.append(temp_positions[0])
        residues.append(temp_resi[0])

        for i in range(1, len(temp_positions)):
            now_order = int(temp_orders[i])
            if current_order + 1 != now_order:
                for j in range(now_order - current_order - 1):
                    residues.append("[UNK]")
                    positions.append({"x": "0.0", "y": "0.0", "z": "0.0"})
                    orders.append(str(current_order + j + 1))
            residues.append(temp_resi[i])
            positions.append(temp_positions[i])
            orders.append(temp_orders[i])
            current_order = now_order
        x_s = []
        y_s = []
        z_s = []
        max_x = 0.0
        max_y = 0.0
        max_z = 0.0
        for item in positions:
            x_s.append(float(item["x"]))
            y_s.append(float(item["y"]))
            z_s.append(float(item["z"]))
            if abs(x_s[-1] - 0) > max_x:
                max_x = abs(x_s[-1] - 0)
            if abs(y_s[-1] - 0) > max_y:
                max_y = abs(y_s[-1] - 0)
            if abs(z_s[-1] - 0) > max_z:
                max_z = abs(z_s[-1] - 0)
        max_ls = [max_x, max_y, max_z]
        max_delta = max(max_ls)
        for pos in range(len(x_s)):
            x_s[pos] = x_s[pos] / max_delta
            y_s[pos] = y_s[pos] / max_delta
            z_s[pos] = z_s[pos] / max_delta
        input = {"sequence": residues, "x_s": x_s, "y_s": y_s, "z_s": z_s}
        inputs.append(input)
        # tpis: Generate the target for EncoderDecoderModel to predicted
        # target = ['0' for _ in range(len(orders))]
        # for t_order in temp_target:
        # target[orders.index(t_order)] = '1'

        # tpis: Generate the target for TokenClassificationModel to category
        target = [1 for _ in range(len(orders))]
        for t_order in temp_target:
            target[orders.index(t_order)] = 2

        targets.append(target)

    return inputs, targets

def transform_pretrain_data(data_dir: str, out_dir: str):
    rcsb_jsons = os.listdir(data_dir)
    for rcsb_json in rcsb_jsons:
        with open(data_dir + rcsb_json, "r") as load_f:
            data = json.load(load_f)
        inputs = []
        for item in tqdm(data):
            residues = []
            positions = []
            orders = []
            temp_resi = item["residues"].strip().split()
            temp_orders = item["orders"].strip().split()
            temp_positions = item["positions"]
            current_order = int(temp_orders[0])
            orders.append(str(current_order))
            positions.append(temp_positions[0])
            residues.append(temp_resi[0])

            for i in range(1, len(temp_positions)):
                now_order = int(temp_orders[i])
                if current_order + 1 != now_order:
                    for j in range(now_order - current_order - 1):
                        residues.append("[UNK]")
                        positions.append({"x": "0.0", "y": "0.0", "z": "0.0"})
                        orders.append(str(current_order + j + 1))
                residues.append(temp_resi[i])
                positions.append(temp_positions[i])
                orders.append(temp_orders[i])
                current_order = now_order
            if len(residues) > 1024:
                continue
            x_s = []
            y_s = []
            z_s = []
            max_x = 0.0
            max_y = 0.0
            max_z = 0.0
            for item in positions:
                x_s.append(float(item["x"]))
                y_s.append(float(item["y"]))
                z_s.append(float(item["z"]))
                if abs(x_s[-1] - 0) > max_x:
                    max_x = abs(x_s[-1] - 0)
                if abs(y_s[-1] - 0) > max_y:
                    max_y = abs(y_s[-1] - 0)
                if abs(z_s[-1] - 0) > max_z:
                    max_z = abs(z_s[-1] - 0)
            max_ls = [max_x, max_y, max_z]
            max_delta = max(max_ls)
            for pos in range(len(x_s)):
                x_s[pos] = x_s[pos] / max_delta
                y_s[pos] = y_s[pos] / max_delta
                z_s[pos] = z_s[pos] / max_delta
            input = {"sequence": residues, "x_s": x_s, "y_s": y_s, "z_s": z_s}
            inputs.append(input)
        with open(out_dir + rcsb_json, "w") as f:
            json.dump(inputs, f, ensure_ascii=False)

def split_train_test(inputs: list, targets: list, train_file: str, test_file: str):
    data = []
    for i in range(len(inputs)):
        data.append({"input": inputs[i], "target": targets[i]})
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    with open(train_file, "w") as f:
        json.dump(train_set, f, ensure_ascii=False)
    with open(test_file, "w") as f:
        json.dump(test_set, f, ensure_ascii=False)