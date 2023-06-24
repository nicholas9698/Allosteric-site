import os
import json
from tqdm import tqdm


def pre_single_a(target_dir:str, pdb_dir:str, output_json:str):
    target_jsons = os.listdir(target_dir)
    data = []
    for target in tqdm(target_jsons):
        if '_1.json' in target:
            
            with open(target_dir+target, 'r') as f:
                ls = json.load(f)
            if len(ls) > 0:
                data.append({'pdbid':(ls[0])['pdbid'][-6:-2], 'chain':(ls[0])['chain'], 'target_orders':(ls[0])['orders']})
                with open(pdb_dir+target[-11:-7].upper()+'.json', 'r') as f:
                    origin = json.load(f)
                for single in origin:
                    if single['chain'] == (data[-1])['chain']:
                        data[-1]['origin_orders'] = single['orders']
                        data[-1]['residues'] = single['residues']
                        data[-1]['positions'] = single['positions']
                        break
    with open(output_json, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_data(data_path:str):
    with open(data_path, 'r') as load_f:
        data = json.load(load_f)
    inputs = []
    targets = []
    for item in data:
        residues = []
        positions = []
        orders = []
        temp_resi = item['residues'].strip().split()
        temp_target = item['target_orders'].strip().split()
        temp_orders = item['origin_orders'].strip().split()
        temp_positions = item['positions']
        current_order = int(temp_orders[0])
        orders.append(str(current_order))
        positions.append(temp_positions[0])
        residues.append(temp_resi[0])

        for i in range(1,len(temp_positions)):
            now_order = int(temp_orders[i])
            if current_order+1 != now_order:
                for j in range(now_order-current_order-1):
                    residues.append('[UNK]')
                    positions.append({'x': '0.0', 'y': '0.0', 'z': '0.0'})
                    orders.append(str(current_order+j+1))
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
            x_s.append(float(item['x']))
            y_s.append(float(item['y']))
            z_s.append(float(item['z']))
            if abs(x_s[-1]-0) > max_x:
                max_x = abs(x_s[-1]-0)
            if abs(y_s[-1]-0) > max_y:
                max_y = abs(y_s[-1]-0)
            if abs(z_s[-1]-0) > max_z:
                max_z = abs(z_s[-1]-0)
        max_ls = [max_x, max_y, max_z]
        max_delta = max(max_ls)
        for pos in range(len(x_s)):
            x_s[pos] = x_s[pos] / max_delta
            y_s[pos] = y_s[pos] / max_delta
            z_s[pos] = z_s[pos] / max_delta
        input = {'sequence': residues, 'x_s': x_s, 'y_s': y_s, 'z_s': z_s}
        inputs.append(input)
        target = [1 for _ in range(len(orders))]
        for t_order in temp_target:
            target[orders.index(t_order)] = 2
        
        targets.append(target)

    return inputs, targets
