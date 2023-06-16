import json
import os
import operator
import pandas as pd
from tqdm import tqdm


'''
    Tokenizer building part
'''

# extracting residue sequence form pdb
def extract_residue_sequence(pdb_path:str):
    result_lists = []
    item_list = []
    current_chain = ''
    with open(pdb_path, 'r') as f:
        for line in f.readlines():
            single = line.strip().split()
            if single[0] == 'SEQRES':
                if len(item_list) == 0:
                    item_list.append((pdb_path.strip().split('/')[-1])[:-4])
                    item_list.append(single[2])
                    item_list.extend([_ for _ in single[4:]])
                    current_chain = single[2]
                elif single[2] == current_chain:
                    item_list.extend([_ for _ in single[4:]])
                elif single[2] != current_chain:
                    if len(item_list[2]) != 3:
                        item_list = []
                    else:
                        for pre in result_lists:
                            if operator.eq(pre[2:], item_list[2:]):
                                item_list = []
                    if len(item_list) > 0:
                        result_lists.append(item_list)
                    item_list = []
                    item_list.append((pdb_path.strip().split('/')[-1])[:-4])
                    item_list.append(single[2])
                    current_chain = single[2]
                    item_list.extend([_ for _ in single[4:]])
        if len(item_list[2]) != 3:
            item_list = []
        else:
            for pre in result_lists:
                if operator.eq(pre[2:], item_list[2:]):
                    item_list = []
        if len(item_list) > 0:
            result_lists.append(item_list)

    return result_lists

def build_tokenizer_dataset(path:str, outpath:str):
    filenames = os.listdir(path)
    data = []

    for file in tqdm(filenames):
        if '.pdb' in file or '.ent' in file:
            sequence_list = extract_residue_sequence(pdb_path=path+file)
            for item in sequence_list:
                data.append(item)
    
        idx = 0
        file_idx = 0
        max_length = len(data)
        while idx+10000 < max_length:
            with open(outpath+str(file_idx)+'.json', 'w') as f:
                for line in data[idx:idx+10000]:
                    json.dump({'pdbid': line[0], 'chain':line[1], 'residue sequence': ' '.join(line[2:])}, f, ensure_ascii=False, indent=4)
                    f.write('\n')
                idx += 10000
                file_idx = int(file_idx) 
                file_idx += 1
        with open(outpath+str(file_idx)+'.json', 'w') as f:
            for line in data[idx:max_length]:
                json.dump({'pdbid': line[0], 'chain':line[1], 'residue sequence': ' '.join(line[2:])}, f, ensure_ascii=False, indent=4)
                f.write('\n')

def tokenizer_json_to_txt(path:str):
    filnames = os.listdir(path=path)
    for file in filnames:
        if '.json' in file:
            data = []
            with open(path+file, 'r') as f:
                js = ""
                for i, s in enumerate(f):
                    js += s
                    i += 1
                    if i % 5 == 0:
                        item = json.loads(js)
                        data.append(item['residue sequence'])
                        js = ""
        with open(path+file[:-5]+'txt', 'w') as f:
            for line in data:
                f.writelines(line)
                f.write('\n')
                


# This file contains allosteric site description of all available entries contained in ASD
def transform_txt_to_json(path:str, outpath:str):
    with open(path, 'r') as f:
        list = []
        for line in f.readlines():
            list.append(line.strip().split('\t'))

    df = pd.DataFrame(list[1:], columns=list[0])
    df.to_csv(outpath, index=False, header=True)

def load_shsmu_site(path:str):
    with open(path, 'r') as f:
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

def extra_by_ncac(path:str, outpath:str):
    result_list = []
    item_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            temp_list = line.strip().split()
            if len(temp_list) == 12:
                if temp_list[4] == 'A' and temp_list[2] == 'N':
                    item_list.append([temp_list[5], temp_list[2], temp_list[3], temp_list[6], temp_list[7], temp_list[8]])
                elif temp_list[4] == 'A' and temp_list[2] == 'CA':
                    item_list.append([temp_list[5], temp_list[2], temp_list[3], temp_list[6], temp_list[7], temp_list[8]])
                elif temp_list[4] == 'A' and temp_list[2] == 'C':
                    item_list.append([temp_list[5], temp_list[2], temp_list[3], temp_list[6], temp_list[7], temp_list[8]])
                
            if len(item_list) == 3:
                postion_1 = 0.0
                postion_2 = 0.0
                postion_3 = 0.0
                for e in item_list:
                    postion_1 += float(e[3])
                    postion_2 += float(e[4])
                    postion_3 += float(e[5])
                    
                result_list.append([item_list[0][0], item_list[0][1], '{:.3f}'.format(postion_1/3), '{:.3f}'.format(postion_2/3), '{:.3f}'.format(postion_3/3)])
                item_list = []
    
    with open(outpath, 'w') as f:
        for item in result_list:
            f.write(' '.join(item))
            f.write('\n')

# extra_by_ncac(path='../data/test/2xcg.pdb', outpath='../data/test/2xcg')
# transform_txt_to_json('../data/ASD_Release_201909_AS.txt', '../data/ASD_Release_201909_AS.csv')
# build_tokenizer_dataset(path='../data/shsmu_allosteric_site/rscb_pdb/', outpath='../data/tokenizer/')
tokenizer_json_to_txt('../data/tokenizer/')