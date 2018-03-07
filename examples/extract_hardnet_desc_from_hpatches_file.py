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
    

if __name__ == '__main__':
    DO_CUDA = True
    try:
          input_img_fname = sys.argv[1]
          output_fname = sys.argv[2]
          if len(sys.argv) > 3:
              DO_CUDA = sys.argv[3] != 'cpu'
    except:
          print("Wrong input format. Try ./extract_hardnet_desc_from_hpatches_file.py imgs/ref.png out.txt gpu")
          sys.exit(1)
    model_weights = '../pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth'
    model = HardNet()
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if DO_CUDA:
        model.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()
    image = cv2.imread(input_img_fname,0)
    h,w = image.shape
    print(h,w)

    n_patches =  int(h/w)

    print('Amount of patches: {}'.format(n_patches))

    t = time.time()
    patches = np.ndarray((n_patches, 1, 32, 32), dtype=np.float32)
    for i in range(n_patches):
        patch =  image[i*(w): (i+1)*(w), 0:w]
        patches[i,0,:,:] = cv2.resize(patch,(32,32)) / 255.
    patches -= 0.443728476019
    patches /= 0.20197947209
    bs = 128
    outs = []
    n_batches = int(n_patches / bs) + 1
    t = time.time()
    descriptors_for_net = np.zeros((len(patches), 128))
    for i in range(0, len(patches), bs):
        data_a = patches[i: i + bs, :, :, :].astype(np.float32)
        data_a = torch.from_numpy(data_a)
        if DO_CUDA:
            data_a = data_a.cuda()
        data_a = Variable(data_a)
        # compute output
        with torch.no_grad():
            out_a = model(data_a)
        descriptors_for_net[i: i + bs,:] = out_a.data.cpu().numpy().reshape(-1, 128)
    print(descriptors_for_net.shape)
    assert n_patches == descriptors_for_net.shape[0]
    et  = time.time() - t
    print('processing', et, et/float(n_patches), ' per patch')
    np.savetxt(output_fname, descriptors_for_net, delimiter=' ', fmt='%10.5f')
