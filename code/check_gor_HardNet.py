"""
Check the correctness of gor on HardNet loss using multiple GPUs
Usage: check_gor_HardNet.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

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

gpu_set = ['0','1']
parameter_set = ['False','True']
number_gpu = len(gpu_set)

#datasets = ['notredame', 'yosemite', 'liberty']
datasets = ['notredame']
process_set = []


for dataset in datasets:
    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))
        command = 'python HardNet.py --training-set {} --fliprot=False --n-triplets=1000000 --batch-size=128 --epochs 10 --gor={} --w1bsroot=None --gpu-id {} --log-dir ../ubc_log/ --enable-logging=True --batch-reduce=min --model-dir ../ubc_model/ '\
                .format(dataset, parameter, gpu_set[idx%number_gpu])
    
        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)
        
        if (idx+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
    
        time.sleep(60)
    
    for sub_process in process_set:
        sub_process.wait()

