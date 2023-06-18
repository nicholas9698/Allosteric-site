import os
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

data_path = '../../data/tokenizer/'
paths = []
filenames = os.listdir(data_path)
for file in filenames:
    if '.txt' in file:
        paths.append(data_path+file)

# initialize
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# normalize
tokenizer.normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=True, lowercase=True)

# split by whitespace
pre_tokenizer = Whitespace()
tokenizer.pre_tokenizer = pre_tokenizer

# processing way to single or pair
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2)
    ]
)

# Define WordPieceTrainer
trainer = WordPieceTrainer(
    vocab_size=100,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# training
tokenizer.train(files=paths, trainer=trainer) 

if not os.path.exists('residue'):
    os.mkdir('residue')
tokenizer.save_model('residue')