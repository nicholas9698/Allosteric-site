# Allosteric-site
This is the offcial repo for the xxx paper "[xxx](https://google.com)".

## Pre-training

1. Follow the instruction `Data downloading` in `data_processing.ipynb` to download the pre-training pdb corpus from [rcsb.org](https://www.rcsb.org/).
   
2. Execute the `Build the dataset for pretrain ResidueRobertaMLM` in `data_processing.ipynb` to process the pdb file and build the pre-training data.

3.  Run the following to pre-train the Residue-RoBERTa:
```
python -u pretrain_ResidueRoberta.py
```

When an error occurs, the `resume_pretraining.py` can be executed to continue the pre-training.
```
python -u pretrain_ResidueRoberta.py
```

> Our pre-trained checkpoints can be obtained from https://drive.google.com/drive/folders/1Q6cd4mTw7Imd9fdiz8qttbF_27_oGMPI?usp=drive_link

## Train and Test

### Allosteric site prediction

Run the following to train the model to directly predict allosteric sites in 3D protein sequences:
```
python -u train_with_TokenClassification.py
```

Run the following to train the model with *Logit Adjustment*:
> *Menon, Aditya Krishna, et al.*, 
> *[Long-tail learning via logit adjustment](https://openreview.net/forum?id=37nvvqkCo5)*,
> *ICLR 2021*
```
python -u train_with_TokenClassification_LA.py
```

### Allosteric pocke prediction

Run the following to train the model to predict allosteric pockets with 3D protein sequences:
```
python -u train_with_SequenceClassification.py
```

### Test Results

We reproduce the main results of **Allosteric site classification** and **Allosteric pocket classification** in the following tables:

| Metric | Site classification | Pocket classification |
| :--- | :---: | :---: |
Residue
| residue acc | 97.6 | - |
| residue precision | 80.1 | - |
| residue recall | 86.1 | - |
| residue f1 | 80.0 | - |
| sequence acc | 20.1 | - |
Pocket
| pocket acc | 80.0 | 97.0 |
| pocket precision | 80.1 | 90.0 |
| pocket recall | 86.1 | 96.0 |
| pocket f1 | 80.0 | 84.0 |

## Dataset

The processed data of allosteric sites we use is uploaded to GitHub (`data/allosteric_site/`).

And the origin pdb data is from [Allosteric Database](https://mdl.shsmu.edu.cn/ASD/module/mainpage/mainpage.jsp) as shown in `data/ASD_Release_201909_AS.txt`
> *Liu, Xinyi, et al.*,
> *[ASD: a comprehensive database of allosteric proteins and modulators](https://doi.org/10.1093/nar/gkq1022)*,
> *Nucleic Acids Research*

## Citation

If you find this work useful, please cite our paper:
```
@inproceedings{}
```