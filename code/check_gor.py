# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

import numpy as np
import scipy.io as sio
import time
import os
import sys
import pandas as pd
import subprocess
import shlex
import argparse
####################################################################
# Parse command line
####################################################################
def usage():
    print >> sys.stderr 
    sys.exit(1)

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", nargs='?', type=float, default = 0.1,
                    help="learning rate")

parser.add_argument("--training", nargs='?', type=str, default = 'notredame',
                    help="Training dataset name")

parser.add_argument("--test", nargs='?', type=str, default = 'yosemite',
                    help="Training dataset name")
args = parser.parse_args()
gpu_set = ['0']
parameter_set = ['0.0','1.0']
#gpu_set = ['2']
#parameter_set = ['1.0']
#parameter_set = ['0.0','1.0','5.0','10.0','20.0','50.0','100.0','200']
number_gpu = len(gpu_set)

#datasets = ['notredame', 'yosemite', 'liberty']
datasets = ['liberty']
process_set = []


for dataset in datasets:
    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))
    
        #command = 'python HardNet_provenance.py --fliprot=False --provenance --n-triplets=1000000 --epochs 10 --alpha=0.0 --beta={} --loss_type=1 --gpu-id {}'\
        #        .format(parameter,gpu_set[idx%number_gpu])
        
        command = 'python HardNet.py --training-set {} --fliprot=False --batch-reduce=random_global --n-triplets=1000000 --batch-size=128 --epochs 10 --alpha={} --w1bsroot=None --gpu-id {} --log-dir ../ubc_log/  --enable-logging=True --model-dir ../ubc_model/ --hard_mining=True'\
                .format(dataset, parameter,gpu_set[idx%number_gpu])
    
        print(command)
        p = subprocess.Popen(shlex.split(command))#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        #(stdoutput,erroutput) = p.communicate()
        process_set.append(p)
        
        if (idx+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
    
        time.sleep(60)
    
    for sub_process in process_set:
        sub_process.wait()

