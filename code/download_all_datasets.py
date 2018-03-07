
import os
import errno
import numpy as np
from PIL import Image
import torchvision.datasets as dset

import sys
from copy import deepcopy
import argparse
import math
import torch.utils.data as data
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import cv2
import copy
from Utils import str2bool

from dataset import TripletPhotoTour
root='../data/sets'
train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         batch_size=128,
                         root=root,
                         name='notredame',
                         download=True,
                         transform=None),
                         batch_size=128,
                         shuffle=False)
train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         batch_size=128,
                         root=root,
                         name='yosemite',
                         download=True,
                         transform=None),
                         batch_size=128,
                         shuffle=False)
train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         batch_size=128,
                         root=root,
                         name='liberty',
                         download=True,
                         transform=None),
                         batch_size=128,
                         shuffle=False)
train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         batch_size=128,
                         root=root,
                         name='notredame_harris',
                         download=True,
                         transform=None),
                         batch_size=128,
                         shuffle=False)
train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         batch_size=128,
                         root=root,
                         name='yosemite_harris',
                         download=True,
                         transform=None),
                         batch_size=128,
                         shuffle=False)
train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         batch_size=128,
                         root=root,
                         name='liberty_harris',
                         download=True,
                         transform=None),
                         batch_size=128,
                         shuffle=False)
