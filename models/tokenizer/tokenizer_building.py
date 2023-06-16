import os
from tokenizers import ByteLevelBPETokenizer


data_path = '../../data/tokenizer/'
paths = []
filenames = os.listdir(data_path)
for file in filenames:
    if '.txt' in file:
        paths.append(data_path+file)

# initialize
tokenizer = ByteLevelBPETokenizer()

# training
tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>']) 

if not os.path.exists('residue'):
    os.mkdir('residue')
tokenizer.save_model('residue')