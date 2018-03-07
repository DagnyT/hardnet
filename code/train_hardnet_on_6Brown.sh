#!/bin/bash
RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/logs"
DATAMODELS="$RUNPATH/data/models"

mkdir -p "$DATASETS"
mkdir -p "$DATALOGS"
mkdir -p "$DATAMODELS"

cd "$RUNPATH"
python -utt ./code/HardNetMultipleDatasets.py --training-set=all  --gpu-id=3 --fliprot=True --model-dir="$DATAMODELS/model_6Brown_30M_ortho06_lr10" --dataroot "$DATASETS/PhotoTourism/" --lr=10.0 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=30000000 --imageSize 32 --log-dir "$DATALOGS" --experiment-name=brown6_aug_lr10/ | tee -a "$DATALOGS/log_HardNet_orthlr10_aug_6Brown.log"
