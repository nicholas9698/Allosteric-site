'''
Author: nicho-UJN nicholas9698@outlook.com
Date: 2023-07-20 10:13:29
LastEditors: nicho-UJN nicholas9698@outlook.com
LastEditTime: 2023-09-24 11:48:15
FilePath: /Allosteric-site/utils/tools.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import math
import torch
import numpy as np
from transformers import BertTokenizer
from Bio import Align

# compute time
def time_since(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

# transform prob to label
def get_labels(probs: torch.Tensor, pocket_classification: bool = False):
    if pocket_classification:
        results = torch.max(probs, dim=1)
        return results[1].tolist()
    else:
        results = torch.max(probs, dim=2) 
        return results[1].tolist()

# logits adjustment
def compute_adjustment(targets_loader: dict, tro: float, USE_CUDA: bool):
    """compute the base probabilities"""

    label_freq = {}

    targets = targets_loader['labels']
    for target in targets:
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)

    if USE_CUDA:
        adjustments = adjustments.cuda()
    return adjustments

# Dynamic mask of ResidueRobertaMLM
def mask_tokens(inputs:torch.LongTensor=None, tokenizer:BertTokenizer=None, mlm_probability:float=0.15, min_residue_index:int=5, special_tokens_mask: torch.LongTensor=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(min_residue_index, len(tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
        
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

def compute_seq_identity(seq1: str, seq2: str):
    aligner = Align.PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    seq1_aligned = alignments[0][0]
    align_score = alignments.score
    seq_identity = align_score / len(seq1_aligned)

    return seq_identity