#!/bin/bash
git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
mv W1BS wxbs-descriptors-benchmark/data/
python HardNet.py --fliprot=False --experiment-name=/liberty_train/ | tee -a log_HardNet_Lib.log
python HardNet.py --fliprot=True --experiment-name=/liberty_train_with_aug/  | tee -a log_HardNetPlus_Lib.log