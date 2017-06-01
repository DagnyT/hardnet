import numpy as np
# Make sure that caffe is on the python path:
caffe_root = '/home/ubuntu/dev/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


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
from copy import deepcopy
import random
import time
import numpy as np
import glob
import os

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.expand_as(x)
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

    def forward(self, input):
        flat = input.view(input.size(0), -1)
        mp = torch.sum(flat, dim=1) / (32. * 32.)
        sp = torch.std(flat, dim=1) + 1e-7
        x_features = self.features(
            (input - mp.unsqueeze(-1).unsqueeze(-1).expand_as(input)) / sp.unsqueeze(-1).unsqueeze(1).expand_as(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)
    

model_weights = '../../pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth'

model = HardNet()
model.cuda()

checkpoint = torch.load(model_weights)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

caffe.set_mode_cpu()
net = None
net = caffe.Net('HardNet.prototxt', caffe.TEST)

gg = {}
gg['counter'] = 1
def copy_weights(m,):
    if isinstance(m, nn.Conv2d):
        counter = gg['counter']
        l_name = 'conv' + str(counter)
        print l_name,  m.weight.data.cpu().numpy().shape;
        net.params[l_name][0].data[:] = np.rollaxis( m.weight.data.cpu().numpy(),3,2);
        net.params[l_name][1].data[:] = m.bias.data.cpu().numpy();
    if isinstance(m, nn.BatchNorm2d):
        counter = gg['counter']
        l_name = 'conv' + str(counter) + '_BN'
        print l_name
        net.params[l_name][0].data[:] = m.running_mean.cpu().numpy();
        net.params[l_name][1].data[:] = m.running_var.cpu().numpy();
        net.params[l_name][2].data[:] = 1.
        gg['counter'] += 1


model.features.apply(copy_weights)
net.save('HardNet+.caffemodel')

    
