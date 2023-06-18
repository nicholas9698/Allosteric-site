from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="./residue/")

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

lorem_ipsum = (
    "MET ALA GLU PRO GLY ILE ASP LYS LEU PHE GLY MET VAL "
    "ASP SER LYS TYR ARG LEU THR VAL VAL VAL ALA LYS ARG "
    "ALA GLN GLN LEU LEU ARG HIS GLY PHE LYS ASN THR VAL "
    "LEU GLU PRO GLU GLU ARG PRO LYS MET GLN THR LEU GLU "
    "GLY LEU PHE ASP ASP PRO ASN ALA VAL THR TRP ALA MET "
    "LYS GLU LEU LEU THR GLY ARG LEU VAL PHE GLY GLU ASN "
    "LEU VAL PRO GLU ASP ARG LEU GLN LYS GLU MET GLU ARG "
    "LEU TYR PRO VAL GLU ARG GLU GLU"
).split()
lorem_ipsum = " ".join(residue_dict[_] for _ in lorem_ipsum)

input = tokenizer(
    lorem_ipsum, return_tensors="pt", add_special_tokens=False, is_split_into_words=True
)
print(input["input_ids"][0], len(input["input_ids"][0]), len(lorem_ipsum))

list = []
for id in input["input_ids"][0]:
    list.append(tokenizer._convert_id_to_token(int(id)))

print(list, len(list))
print(tokenizer._convert_id_to_token(int((input["input_ids"][0])[0])))
