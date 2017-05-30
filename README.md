# HardNet model implementation

HardNet model implementation in PyTorch for paper "Working hard to know your neighbor's margins: Local descriptor learning loss"

## Requirements

Please use Python 2.7, install OpenCV and additional libraries from requirements.txt

## Datasets and Training

To download datasets and start learning descriptor:

```bash
git clone https://github.com/DagnyT/hardnet
./run_me.sh
```

Logs are stored in tensorboard format in directory logs/

## Pre-trained models

Pre-trained models can be found in folder pretrained:  train_liberty and train_liberty_with_aug


## Citation

Please cite us if you use this code:

```
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
```
