import json
import os
import operator
import pandas as pd
from tqdm import tqdm
import numpy as np


residue_dict = {
    "GLY": "G",
    "ALA": "A",
    "VAL": "V",
    "LEU": "L",
    "ILE": "I",
    "PHE": "F",
    "TRP": "W",
    "TYR": "Y",
    "ASP": "D",
    "HIS": "H",
    "ASN": "N",
    "GLU": "E",
    "LYS": "K",
    "GLN": "Q",
    "MET": "M",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "CYS": "C",
    "PRO": "P",
    "SEC": "U",
    "PYL": "O",
}

"""
    Tokenizer building part
"""


# extracting residue sequence form pdb
def extract_residue_sequence(pdb_path: str):
    result_lists = []
    item_list = []
    current_chain = ""
    resi_order = ""
    with open(pdb_path, "r") as f:
        for line in f.readlines():
            single = line
            if single[0:4] == "ATOM":
                if len(item_list) == 0:
                    resi = single[17:20].strip()
                    resi_order = single[22:26].strip()
                    if len(resi) == 3:
                        item_list.append((pdb_path.strip().split("/")[-1])[:-4])
                        item_list.append(single[21])
                        try:
                            item_list.append(residue_dict[resi])
                        except Exception as e:
                            print(
                                "pdbid: {0}\tchain: {1}\terror: {2}".format(
                                    item_list[0], item_list[1], e
                                )
                            )

                        current_chain = single[21]
                elif (
                    single[21] == current_chain and resi_order != single[22:26].strip()
                ):
                    resi = single[17:20].strip()
                    resi_order = single[22:26].strip()
                    if len(resi) == 3:
                        try:
                            item_list.append(residue_dict[resi])
                        except Exception as e:
                            print(
                                "pdbid: {0}\tchain: {1}\terror: {2}".format(
                                    item_list[0], item_list[1], e
                                )
                            )
                elif single[21] != current_chain:
                    for pre in result_lists:
                        if operator.eq(pre[2:], item_list[2:]):
                            item_list = []
                    if len(item_list) > 2:
                        result_lists.append(item_list)
                        item_list = []
                    else:
                        item_list = []
                    resi_order = ""

                    resi = single[17:20].strip()
                    if len(resi) == 3:
                        item_list.append((pdb_path.strip().split("/")[-1])[:-4])
                        item_list.append(single[21])
                        try:
                            item_list.append(residue_dict[resi])
                        except Exception as e:
                            print(
                                "pdbid: {0}\tchain: {1}\terror: {2}".format(
                                    item_list[0], item_list[1], e
                                )
                            )
                        current_chain = single[21]
                        resi_order = single[22:26].strip()
        for pre in result_lists:
            if operator.eq(pre[2:], item_list[2:]):
                item_list = []
        if len(item_list) > 2:
            result_lists.append(item_list)
            item_list = []

    return result_lists


def build_tokenizer_dataset(path: str, outpath: str):
    filenames_total = os.listdir(path)
    file_count = len(filenames_total)
    part_index = 0
    part_idx = 0
    while (part_index + 10000) < file_count:
        data = []
        filenames = filenames_total[part_index : part_index + 10000]
        for file in tqdm(filenames):
            if ".pdb" in file or ".ent" in file:
                sequence_list = extract_residue_sequence(pdb_path=path + file)
                for item in sequence_list:
                    data.append(item)

            idx = 0
            file_idx = 0
            max_length = len(data)
            while (idx + 10000) < max_length:
                with open(
                    outpath + str(part_idx) + "_" + str(file_idx) + ".json", "w"
                ) as f:
                    for line in data[idx : idx + 10000]:
                        json.dump(
                            {
                                "pdbid": line[0],
                                "chain": line[1],
                                "residue sequence": " ".join(line[2:]),
                            },
                            f,
                            ensure_ascii=False,
                            indent=4,
                        )
                        f.write("\n")
                    idx += 10000
                    file_idx = int(file_idx)
                    file_idx += 1
            with open(
                outpath + str(part_idx) + "_" + str(file_idx) + ".json", "w"
            ) as f:
                for line in data[idx:max_length]:
                    json.dump(
                        {
                            "pdbid": line[0],
                            "chain": line[1],
                            "residue sequence": " ".join(line[2:]),
                        },
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )
                    f.write("\n")
        part_index += 10000
        part_idx = int(part_idx)
        part_idx += 1
    data = []
    filenames = filenames_total[part_index:file_count]
    for file in tqdm(filenames):
        if ".pdb" in file or ".ent" in file:
            sequence_list = extract_residue_sequence(pdb_path=path + file)
            for item in sequence_list:
                data.append(item)

        idx = 0
        file_idx = 0
        max_length = len(data)
        while (idx + 10000) < max_length:
            with open(
                outpath + str(part_idx) + "_" + str(file_idx) + ".json", "w"
            ) as f:
                for line in data[idx : idx + 10000]:
                    json.dump(
                        {
                            "pdbid": line[0],
                            "chain": line[1],
                            "residue sequence": " ".join(line[2:]),
                        },
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )
                    f.write("\n")
                idx += 10000
                file_idx = int(file_idx)
                file_idx += 1
        with open(outpath + str(part_idx) + "_" + str(file_idx) + ".json", "w") as f:
            for line in data[idx:max_length]:
                json.dump(
                    {
                        "pdbid": line[0],
                        "chain": line[1],
                        "residue sequence": " ".join(line[2:]),
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
                f.write("\n")


def tokenizer_json_to_txt(path: str):
    filnames = os.listdir(path=path)
    for file in filnames:
        if ".json" in file:
            data = []
            with open(path + file, "r") as f:
                js = ""
                for i, s in enumerate(f):
                    js += s
                    i += 1
                    if i % 5 == 0:
                        item = json.loads(js)
                        data.append(item["residue sequence"])
                        js = ""
        with open(path + file[:-5] + ".txt", "w") as f:
            for line in data:
                f.writelines(line)
                f.write("\n")


# This file contains allosteric site description of all available entries contained in ASD
def transform_txt_to_csv(path: str, outpath: str):
    with open(path, "r") as f:
        list = []
        for line in f.readlines():
            list.append(line.strip().split("\t"))

    df = pd.DataFrame(list[1:], columns=list[0])
    df.to_csv(outpath, index=False, header=True)


def load_shsmu_site(path: str):
    with open(path, "r") as f:
        js = ""
        data = []
        for i, s in enumerate(f):
            js += s
            i += 1
            if i % 16 == 0:
                item = json.loads(js)
                data.append(item)
                js = ""
    return data


"""
    Extract residue and compute average pisition of atom (x,y,z) from pdb
"""


def extract_residue_avg(pdb_path: str):
    result_lists = []
    position_lists = []
    order_lists = []
    item_list = []
    x_list = []
    y_list = []
    z_list = []
    avg_list = []
    o_list = ["start"]
    current_chain = ""
    resi_order = ""
    with open(pdb_path, "r") as f:
        for line in f.readlines():
            single = line
            if single[0:4] == "ATOM":
                if len(item_list) == 0:
                    resi = single[17:20].strip()
                    resi_order = single[22:26].strip()
                    if len(resi) == 3:
                        item_list.append((pdb_path.strip().split("/")[-1])[:-4])
                        item_list.append(single[21])

                        try:
                            item_list.append(residue_dict[resi])
                            x_list.append(float(single[30:38].strip()))
                            y_list.append(float(single[38:46].strip()))
                            z_list.append(float(single[46:54].strip()))
                            o_list.append(resi_order)
                        except Exception as e:
                            print(
                                "pdbid: {0}\tchain: {1}\terror: {2}".format(
                                    item_list[0], item_list[1], e
                                )
                            )
                            item_list.append('[UNK]')
                            x_list.append(float(single[30:38].strip()))
                            y_list.append(float(single[38:46].strip()))
                            z_list.append(float(single[46:54].strip()))
                            o_list.append(resi_order)

                        current_chain = single[21]
                elif (
                    single[21] == current_chain and o_list[-1] == single[22:26].strip()
                ):
                    x_list.append(float(single[30:38].strip()))
                    y_list.append(float(single[38:46].strip()))
                    z_list.append(float(single[46:54].strip()))
                elif (
                    single[21] == current_chain and resi_order != single[22:26].strip()
                ):
                    if len(x_list) > 0:
                        x_total = 0.0
                        y_total = 0.0
                        z_total = 0.0
                        length = len(x_list)

                        for _ in x_list:
                            x_total += _
                        for _ in y_list:
                            y_total += _
                        for _ in z_list:
                            z_total += _
                        avg_list.append(
                            {
                                "x": "{:.3f}".format(x_total / length),
                                "y": "{:.3f}".format(y_total / length),
                                "z": "{:.3f}".format(z_total / length),
                            }
                        )
                        x_list = []
                        y_list = []
                        z_list = []
                    resi = single[17:20].strip()
                    resi_order = single[22:26].strip()
                    if len(resi) == 3:
                        try:
                            item_list.append(residue_dict[resi])
                            x_list.append(float(single[30:38].strip()))
                            y_list.append(float(single[38:46].strip()))
                            z_list.append(float(single[46:54].strip()))
                            o_list.append(resi_order)
                        except Exception as e:
                            print(
                                "pdbid: {0}\tchain: {1}\terror: {2}".format(
                                    item_list[0], item_list[1], e
                                )
                            )
                            item_list.append('[UNK]')
                            x_list.append(float(single[30:38].strip()))
                            y_list.append(float(single[38:46].strip()))
                            z_list.append(float(single[46:54].strip()))
                            o_list.append(resi_order)
                elif single[21] != current_chain:
                    # although exist same chain, save all chain of allosteric site (Unkown which chain is experment tested)
                    # for pre in result_lists:
                    #     if operator.eq(pre[2:], item_list[2:]):
                    #         item_list = []
                    if len(x_list) > 0:
                        x_total = 0.0
                        y_total = 0.0
                        z_total = 0.0
                        length = len(x_list)

                        for _ in x_list:
                            x_total += _
                        for _ in y_list:
                            y_total += _
                        for _ in z_list:
                            z_total += _
                        avg_list.append(
                            {
                                "x": "{:.3f}".format(x_total / length),
                                "y": "{:.3f}".format(y_total / length),
                                "z": "{:.3f}".format(z_total / length),
                            }
                        )
                        x_list = []
                        y_list = []
                        z_list = []
                    if len(item_list) > 2:
                        result_lists.append(item_list)
                        item_list = []
                        order_lists.append(o_list[1:])
                        o_list = ["start"]
                        position_lists.append(avg_list)
                        avg_list = []
                    else:
                        item_list = []
                        o_list = ["start"]
                        avg_list = []
                    # resi_order = ""

                    resi = single[17:20].strip()
                    if len(resi) == 3:
                        item_list.append((pdb_path.strip().split("/")[-1])[:-4])
                        item_list.append(single[21])
                        try:
                            item_list.append(residue_dict[resi])
                            x_list.append(float(single[30:38].strip()))
                            y_list.append(float(single[38:46].strip()))
                            z_list.append(float(single[46:54].strip()))
                            o_list.append(single[22:26].strip())
                        except Exception as e:
                            print(
                                "pdbid: {0}\tchain: {1}\terror: {2}".format(
                                    item_list[0], item_list[1], e
                                )
                            )
                            item_list.append('[UNK]')
                            x_list.append(float(single[30:38].strip()))
                            y_list.append(float(single[38:46].strip()))
                            z_list.append(float(single[46:54].strip()))
                            o_list.append(single[22:26].strip())
                        current_chain = single[21]
                        resi_order = single[22:26].strip()
                    else:
                        resi_order = ""
        # although exist same chain, save all chain of allosteric site (Unkown which chain is experment tested)
        # for pre in result_lists:
        #     if operator.eq(pre[2:], item_list[2:]):
        #         item_list = []
        if len(x_list) > 0:
            x_total = 0.0
            y_total = 0.0
            z_total = 0.0
            length = len(x_list)

            for _ in x_list:
                x_total += _
            for _ in y_list:
                y_total += _
            for _ in z_list:
                z_total += _
            avg_list.append(
                {
                    "x": "{:.3f}".format(x_total / length),
                    "y": "{:.3f}".format(y_total / length),
                    "z": "{:.3f}".format(z_total / length),
                }
            )
            x_list = []
            y_list = []
            z_list = []
        if len(item_list) > 2:
            result_lists.append(item_list)
            item_list = []
            order_lists.append(o_list[1:])
            o_list = ["start"]
            position_lists.append(avg_list)
            avg_list = []

    return result_lists, position_lists, order_lists

def build_allosteric_dataset(dir_path: str, outdir: str):
    filenames = os.listdir(dir_path)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for file in tqdm(filenames):
        if '.pdb' or '.ent' in file:
            results, positions, orders = extract_residue_avg(dir_path+file)
            dicts = []
            with open(outdir+file[:-4]+'.json', 'w') as f:
                for i in range(len(results)):
                    # json.dump({
                    #     'pdbid': (results[i])[0],
                    #     'chain': (results[i])[1],
                    #     'residues': ' '.join((results[i])[2:]),
                    #     'orders': ' '.join(orders[i]),
                    #     'positions': positions[i]
                    # }, f, ensure_ascii=False, indent=4)
                    # f.write('\n')
                    dicts.append({
                        'pdbid': (results[i])[0],
                        'chain': (results[i])[1],
                        'residues': ' '.join((results[i])[2:]),
                        'orders': ' '.join(orders[i]),
                        'positions': positions[i]
                    })
                json.dump(dicts, f, ensure_ascii=False, indent=4)
                

def extra_by_ncac(path: str, outpath: str):
    result_list = []
    item_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            temp_list = line.strip().split()
            if len(temp_list) == 12:
                if temp_list[4] == "A" and temp_list[2] == "N":
                    item_list.append(
                        [
                            temp_list[5],
                            temp_list[2],
                            temp_list[3],
                            temp_list[6],
                            temp_list[7],
                            temp_list[8],
                        ]
                    )
                elif temp_list[4] == "A" and temp_list[2] == "CA":
                    item_list.append(
                        [
                            temp_list[5],
                            temp_list[2],
                            temp_list[3],
                            temp_list[6],
                            temp_list[7],
                            temp_list[8],
                        ]
                    )
                elif temp_list[4] == "A" and temp_list[2] == "C":
                    item_list.append(
                        [
                            temp_list[5],
                            temp_list[2],
                            temp_list[3],
                            temp_list[6],
                            temp_list[7],
                            temp_list[8],
                        ]
                    )

            if len(item_list) == 3:
                postion_1 = 0.0
                postion_2 = 0.0
                postion_3 = 0.0
                for e in item_list:
                    postion_1 += float(e[3])
                    postion_2 += float(e[4])
                    postion_3 += float(e[5])

                result_list.append(
                    [
                        item_list[0][0],
                        item_list[0][1],
                        "{:.3f}".format(postion_1 / 3),
                        "{:.3f}".format(postion_2 / 3),
                        "{:.3f}".format(postion_3 / 3),
                    ]
                )
                item_list = []

    with open(outpath, "w") as f:
        for item in result_list:
            f.write(" ".join(item))
            f.write("\n")


# extra_by_ncac(path='../data/test/2xcg.pdb', outpath='../data/test/2xcg')
# transform_txt_to_json('../data/ASD_Release_201909_AS.txt', '../data/ASD_Release_201909_AS.csv')
# build_tokenizer_dataset(path='../data/shsmu_allosteric_site/rscb_pdb/', outpath='../data/tokenizer/')
# tokenizer_json_to_txt('../data/tokenizer/')

"""
    Extract residue sequence from pocket
"""


def extract_from_fpocketPDB(pdb_path: str):
    pdb_dic = {}
    pdb_dic["id"] = (pdb_path.strip().split("/")[-1])[:-4]

    with open(pdb_path, "r") as f:
        for line in f.readlines():
            single = line
            if single[0:4] == "ATOM":
                chainID = single[21]
                resnanme = single[17:20].strip()
                resi = single[22:26].strip()
                if not chainID in pdb_dic:
                    pdb_dic[chainID] = [resnanme + resi]
                else:
                    pdb_dic[chainID].append(resnanme + resi)
                    pdb_dic[chainID] = list(set(pdb_dic[chainID]))

    return pdb_dic


def extract_from_fpocket(fpocket_path: str, save_path: str):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dir_list = os.listdir(fpocket_path)
    pockets_all = []
    for dir in dir_list:
        pockets_dic = {}
        pockets_dic["ID"] = dir
        pockets = []
        dir_path = os.path.join(fpocket_path, dir)
        file_list = os.listdir(dir_path)
        for file in file_list:
            pockets.append(extract_from_fpocketPDB(os.path.join(dir_path, file)))
        pockets_dic["Pockets"] = pockets
        np.save(os.path.join(save_path, dir + ".npy"), pockets_dic)

        pockets_all.append(pockets_dic)

    return pockets_all
