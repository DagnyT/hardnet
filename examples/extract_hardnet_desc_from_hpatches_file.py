#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
import cv2
import math
import numpy as np


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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8),
            nn.BatchNorm2d(128, affine=False),

        )
        #self.features.apply(weights_init)

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.sum(flat, dim=1) / (32. * 32.)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)
    

model_weights = '../pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth'

model = HardNet()
model.cuda()

checkpoint = torch.load(model_weights)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
except:
    print "Wrong input format. Try ./extract_hardnet_desc_from_hpatches_file.py imgs/ref.png out.txt"
    sys.exit(1)

image = cv2.imread(input_img_fname,0)
h,w = image.shape
print(h,w)

n_patches =  h/w


descriptors_for_net = np.zeros((n_patches, 128))
t = time.time()
patches = np.ndarray((n_patches, 1, 32, 32), dtype=np.float32)
for i in range(n_patches):
    patch =  image[i*(w): (i+1)*(w), 0:w]
    patches[i,0,:,:] = cv2.resize(patch,(32,32)) / 255.
patches -= 0.443728476019
patches /= 0.20197947209
bs = 128;
outs = []
n_batches = n_patches / bs + 1
t = time.time()

for batch_idx in range(n_batches):
    if batch_idx == n_batches - 1:
        if (batch_idx + 1) * bs > n_patches:
            end = n_patches
        else:
            end = (batch_idx + 1) * bs
    else:
        end = (batch_idx + 1) * bs
    data_a = patches[batch_idx * bs: end, :, :, :].astype(np.float32)
    data_a = torch.from_numpy(data_a)
    data_a = data_a.cuda()
    data_a = Variable(data_a, volatile=True)
    # compute output
    out_a = model(data_a)
    descriptors_for_net[batch_idx * bs: end,:] = out_a.data.cpu().numpy().reshape(-1, 128)
print descriptors_for_net.shape
et  = time.time() - t
print 'processing', et, et/float(n_patches), ' per patch'
np.savetxt(output_fname, descriptors_for_net, delimiter=' ', fmt='%10.5f')    
