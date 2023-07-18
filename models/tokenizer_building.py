import os
import json
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

data_path = "/mnt/g/Little-LL/pretrain_tokenizer/"
paths = []
filenames = os.listdir(data_path)
for file in filenames:
    if ".txt" in file:
        paths.append(data_path + file)

# initialize
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# normalize
tokenizer.normalizer = BertNormalizer(
    clean_text=True, handle_chinese_chars=True, lowercase=True
)

# split by whitespace
pre_tokenizer = Whitespace()
tokenizer.pre_tokenizer = pre_tokenizer

# processing way to single or pair
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

# Define WordPieceTrainer
trainer = WordPieceTrainer(
    vocab_size=100, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
)

# training
tokenizer.train(files=paths, trainer=trainer)

if not os.path.exists("residue"):
    os.mkdir("residue")
tokenizer.save("residue/tokenizer.json")

with open("residue/tokenizer.json", "r") as f:
    tokenizer = json.load(f)

with open("residue/vocab.txt", "w") as f:
    vocab = tokenizer["model"]["vocab"].keys()
    for word in vocab:
        f.writelines(word)
        f.write("\n")
