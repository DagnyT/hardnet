import sys
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import sys
sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
import cv2
import math
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random
import time
import numpy as np
import glob
import os
descr_name = 'HardNet'
USE_CUDA = True
  
# all types of patches 
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self,base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/65
            setattr(self, t, np.split(im, self.N))

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),

        )
        #self.features.apply(weights_init)

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

ww = ["../pretrained/train_yosemite/checkpoint_yosemite_no_aug.pth",
"../pretrained/train_liberty/checkpoint_liberty_no_aug.pth",
"../pretrained/train_notredame/checkpoint_notredame_no_aug.pth",
"../pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth",
"../pretrained/train_notredame_with_aug/checkpoint_notredame_with_aug.pth",
"../pretrained/train_yosemite_with_aug/checkpoint_yosemite_with_aug.pth",
"../pretrained/pretrained_all_datasets/HardNet++.pth"
]
try:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    seqs = glob.glob(sys.argv[1]+'/*')
    seqs = [os.path.abspath(p) for p in seqs]   
except:
    print('Wrong input format. Try python hpatches_extract_HardNet.py /home/ubuntu/dev/hpatches/hpatches-benchmark/data/hpatches-release /home/old-ufo/dev/hpatches/hpatches-benchmark/data/descriptors')
    sys.exit(1)
    
w = 65

model = HardNet()
if USE_CUDA:
    model.cuda()

for model_weights in ww:
    desc_suffix = model_weights.split('/')[-1].replace('.pth', '').replace('checkpoint_', '')
    curr_desc_name = descr_name + '_' + desc_suffix
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    for seq_path in seqs:
        seq = hpatches_sequence(seq_path)
        path = os.path.join(output_dir, os.path.join(curr_desc_name,seq.name))
        if not os.path.exists(path):
            os.makedirs(path)
        descr = np.zeros((seq.N,128)) # trivial (mi,sigma) descriptor
        for tp in tps:
            print(seq.name+'/'+tp)
            if os.path.isfile(os.path.join(path,tp+'.csv')):
                continue
            n_patches = 0
            for i,patch in enumerate(getattr(seq, tp)):
                n_patches+=1
            t = time.time()
            patches_for_net = np.zeros((n_patches, 1, 32, 32))
            uuu = 0
            for i,patch in enumerate(getattr(seq, tp)):
                patches_for_net[i,0,:,:] = cv2.resize(patch[0:w,0:w],(32,32))
            ###
            model.eval()
            outs = []
            bs = 128;
            n_batches = n_patches / bs + 1
            for batch_idx in range(n_batches):
                st = batch_idx * bs
                if batch_idx == n_batches - 1:
                    if (batch_idx + 1) * bs > n_patches:
                        end = n_patches
                    else:
                        end = (batch_idx + 1) * bs
                else:
                    end = (batch_idx + 1) * bs
                if st >= end:
                    continue
                data_a = patches_for_net[st: end, :, :, :].astype(np.float32)
                data_a = torch.from_numpy(data_a)
                if USE_CUDA:
                    data_a = data_a.cuda()
                data_a = Variable(data_a, volatile=True)
                # compute output
                out_a = model(data_a)
                outs.append(out_a.data.cpu().numpy().reshape(-1, 128))
            res_desc = np.concatenate(outs)
            print(res_desc.shape, n_patches)
            res_desc = np.reshape(res_desc, (n_patches, -1))
            out = np.reshape(res_desc, (n_patches,-1))
            np.savetxt(os.path.join(path,tp+'.csv'), out, delimiter=',', fmt='%10.5f')   # X is an array