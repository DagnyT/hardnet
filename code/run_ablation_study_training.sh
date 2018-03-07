#!/bin/bash

RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/logs"

mkdir -p "$DATASETS"
mkdir -p "$DATALOGS"

cd "$RUNPATH"
python ./code/HardNet.py --dataroot "$DATASETS/PhotoTourism" --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/"  --fliprot=False --experiment-name=liberty_train/ --loss=triplet_margin --batch-reduce=min | tee -a "$DATALOGS/log_HardNet_Lib_as_is.log"
python ./code/HardNet.py --dataroot "$DATASETS/PhotoTourism" --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/"  --fliprot=False --experiment-name=liberty_train/ --loss=triplet_margin --batch-reduce=average | tee -a "$DATALOGS/log_HardNet_Lib_average_negative.log"
python ./code/HardNet.py --dataroot "$DATASETS/PhotoTourism" --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/"  --fliprot=False --experiment-name=liberty_train/ --loss=triplet_margin --batch-reduce=random | tee -a l"$DATALOGS/og_HardNet_Lib_random_negative.log"
python ./code/HardNet.py --dataroot "$DATASETS/PhotoTourism" --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/"  --fliprot=False --experiment-name=liberty_train/ --loss=softmax --batch-reduce=min | tee -a "$DATALOGS/log_HardNet_Lib_softmax.log"
