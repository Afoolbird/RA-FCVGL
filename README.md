# DA-FCVGL
This repository provides the code for "**Digging Into Informative Regions with Adaptive Gradient Modulation for Fine-grained Cross View Geo-localization**

<img src="figs/Overview.png"/>

## Dataset
Please prepare [VIGOR](https://github.com/Jeff-Zilence/VIGOR) and [CVACT](https://github.com/Liumouliu/OriCNN). You may need to modify specific path in dataloader.

## Requirement
	- Python >= 3.6, numpy, matplotlib, pillow, ptflops, timm
	- PyTorch >= 1.8.1, torchvision >= 0.11.1

## Training and Evaluation
Simply run the scripts like:

    sh run_VIGOR_uncertainty_mso.sh

You may need to specify the GPUs for training in "train.py". Change the "--dataset" to train on other datasets. The code follows the multiprocessing distributed training style from [PyTorch](https://github.com/pytorch/examples/tree/main/imagenet) and [Moco](https://github.com/facebookresearch/moco), but it only uses one GPU by default for training. You may need to tune the learning rate for multi-GPU training, e.g. [linear scaling rule](https://arxiv.org/pdf/1706.02677.pdf).     

## Reference
    - http://mvrl.cs.uky.edu/datasets/cvusa/
    - https://github.com/Jeff-Zilence/VIGOR
    - https://github.com/Liumouliu/OriCNN
    - https://github.com/facebookresearch/deit
    - https://github.com/facebookresearch/moco
    - https://github.com/davda54/sam
    - https://github.com/david-husx/crossview_localisation.git

