#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-

"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this code, please cite
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin
"""

from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
from EvalMetrics import ErrorRateAt95Recall
from Loggers import Logger, FileLogger
from W1BS import w1bs_extract_descs_and_save
from Utils import L2Norm, cv2_scale, np_reshape
from Utils import str2bool
import torch.utils.data as data
import torch.utils.data as data_utils
import torch.nn.functional as F
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net

import faiss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--w1bsroot', type=str,
                    default='data/sets/wxbs-descriptors-benchmark/code/',
                    help='path to dataset')
parser.add_argument('--dataroot', type=str,
                    default='data/sets/',
                    help='path to dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--enable-logging',type=bool, default=True,
                    help='folder to output model checkpoints')
parser.add_argument('--log-dir', default='data/logs/',
                    help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default= 'liberty_train_hard_mining/',
                    help='experiment path')
parser.add_argument('--decor',type=str2bool, default = False,
                    help='L2Net decorrelation penalty')
parser.add_argument('--training-set', default= 'liberty',
                    help='Other options: notredame, yosemite')
parser.add_argument('--num-workers', default= 8,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--anchorave', type=bool, default=False,
                    help='anchorave')
parser.add_argument('--hardnegatives', type=int, default=7,
                    help='the height / width of the input image to network')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=bool, default=True,
                    help='turns on anchor swap')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--act-decay', type=float, default=0,
                    help='activity L2 decay, default 0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--fliprot', type=str2bool, default=False,
                    help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

dataset_names = ['liberty', 'notredame', 'yosemite']

# check if path to w1bs dataset testing module exists
if os.path.isdir(args.w1bsroot):
    sys.path.insert(0, args.w1bsroot)
    import utils.w1bs as w1bs
    TEST_ON_W1BS = True

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

LOG_DIR = args.log_dir + args.experiment_name
# create loggin directory
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)


class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum()) / input.size(0)


from matplotlib.pyplot import figure, imshow, axis

def showImagesHorizontally(images):
    fig = figure()
    for i in range(len(images)):
        a=fig.add_subplot(1,len(images),i+1)
        image = images[i]
        imshow(image,cmap='Greys_r')
        axis('off')

class TripletPhotoTour(dset.PhotoTour):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform

        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes - 1)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes - 1)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes - 1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes - 1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            n3 = np.random.randint(0, len(indices[c2]) - 1)
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = transform_img(n)

        # transform images if required
        # if args.fliprot:
        #     do_flip = random.random() > 0.5
        #     do_rot = random.random() > 0.5
        #
        #     if do_rot:
        #         img_a = img_a.permute(0,2,1)
        #         img_p = img_p.permute(0,2,1)
        #
        #     if do_flip:
        #         img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
        #         img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

class TripletPhotoTourHardNegatives(dset.PhotoTour):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, negative_indices, train=True, transform=None, batch_size = None, *arg, **kw):
        super(TripletPhotoTourHardNegatives, self).__init__(*arg, **kw)
        self.transform = transform

        self.train = train
        self.n_triplets = args.n_triplets
        self.negative_indices = negative_indices
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets, self.negative_indices)


    @staticmethod
    def generate_triplets(labels, num_triplets, negative_indices):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]

        # add only unique indices in batch
        already_idxs = set()
        count  = 0
        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes - 1)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes - 1)
            already_idxs.add(c1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            indx = indices[c1][n1]
            if(len(negative_indices[indx])>0):

                negative_indx = random.choice(negative_indices[indx])
                negative_indices[indx].remove(negative_indx)

                if(indx in negative_indices[negative_indx]):
                    negative_indices[negative_indx].remove(indx)

            else:
                count+=1
                c2 = np.random.randint(0, n_classes - 1)
                while c1 == c2:
                    c2 = np.random.randint(0, n_classes - 1)
                n3 = np.random.randint(0, len(indices[c2]) - 1)
                negative_indx = indices[c2][n3]

            triplets.append([indices[c1][n1], indices[c1][n2], negative_indx])

        print(count)
        print('triplets are generated. amount of triplets: {}'.format(len(triplets)))
        return torch.LongTensor(np.array(triplets))


    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = transform_img(n)

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()

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
        self.features.apply(weights_init)

    def forward(self, input):
        flat = input.view(input.size(0), -1)
        mp = torch.sum(flat, dim=1) / (32. * 32.)
        sp = torch.std(flat, dim=1) + 1e-7
        x_features = self.features(
            (input - mp.unsqueeze(-1).unsqueeze(-1).expand_as(input)) / sp.unsqueeze(-1).unsqueeze(1).expand_as(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.7)
        nn.init.constant(m.bias.data, 0.01)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal(m.weight.data, gain=0.01)
        nn.init.constant(m.bias.data, 0.)

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.training_set)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])

    trainPhotoTourDataset =  TripletPhotoTour(train=True,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             download=True,
                             transform=transform)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             TripletPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     transform=transform),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return trainPhotoTourDataset, test_loaders

def train(train_loader, model, optimizer, epoch, logger):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, data_p, data_n) in pbar:

        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()

        data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)

        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)

        #hardnet loss
        loss = loss_random_sampling(out_a, out_p, out_n, margin=args.margin)

        if args.decor:
            loss += CorrelationPenaltyLoss()(out_a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if(logger!=None):
         logger.log_value('loss', loss.data[0]).step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data[0]))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        out_a, out_p = model(data_a), model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

        if batch_idx % args.log_interval == 0:
            pbar.set_description(logger_test_name+' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    if (args.enable_logging):
        if(logger!=None):
            logger.log_value(logger_test_name+' fpr95', fpr95)
    return

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
        1.0 - float(group['step']) * float(args.batch_size) / (args.n_triplets * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(trainPhotoTourDataset, test_loaders, model, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    if (args.enable_logging):
        file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()

    optimizer1 = create_optimizer(model.features, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])

    first_init = False

    for epoch in range(start, end):

        model.eval()

        if(not first_init):

            descriptors = pre_init_with_sift(trainPhotoTourDataset)
            np.save('descriptors_sift.npy', descriptors)
            descriptors = np.load('descriptors_sift.npy')

            hard_negatives = get_hard_negatives(trainPhotoTourDataset, descriptors)
            np.save('descriptors_min_dist_sift.npy', hard_negatives)
            hard_negatives = np.load('descriptors_min_dist_sift.npy')

            first_init = True

        else:
            # #
            descriptors = get_descriptors_for_dataset(model, trainPhotoTourDataset)
                # #
            np.save('descriptors.npy', descriptors)
            descriptors = np.load('descriptors.npy')
                #
            hard_negatives = get_hard_negatives(trainPhotoTourDataset, descriptors)
            np.save('descriptors_min_dist.npy', hard_negatives)
            hard_negatives = np.load('descriptors_min_dist.npy')

        trainPhotoTourDatasetWithHardNegatives = TripletPhotoTourHardNegatives(train=True,
                                                              negative_indices=hard_negatives,
                                                              batch_size=args.batch_size,
                                                              root=args.dataroot,
                                                              name=args.training_set,
                                                              download=True,
                                                              transform=transform)

        train_loader = torch.utils.data.DataLoader(trainPhotoTourDatasetWithHardNegatives,
                                                   batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)

        train(train_loader, model, optimizer1, epoch, logger)

        # iterate over test loaders and test results
        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])

        if TEST_ON_W1BS :
            # print(weights_path)
            patch_images = w1bs.get_list_of_patch_images(
                DATASET_DIR=args.w1bsroot.replace('/code', '/data/W1BS'))
            desc_name = 'curr_desc'

            for img_fname in patch_images:
                w1bs_extract_descs_and_save(img_fname, model, desc_name, cuda = args.cuda,
                                            mean_img=args.mean_image,
                                            std_img=args.std_image)

            DESCS_DIR = args.w1bsroot.replace('/code', "/data/out_descriptors")
            OUT_DIR = args.w1bsroot.replace('/code', "/data/out_graphs")

            force_rewrite_list = [desc_name]
            w1bs.match_descriptors_and_save_results(DESC_DIR=DESCS_DIR, do_rewrite=True,
                                                    dist_dict={},
                                                    force_rewrite_list=force_rewrite_list)
            if(args.enable_logging):
                w1bs.draw_and_save_plots_with_loggers(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name],
                                         logger=file_logger,
                                         tensor_logger = None)
            else:
                w1bs.draw_and_save_plots(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name],
                                         really_draw = False)


class PhototourTrainingData(data.Dataset):

    def __init__(self, data):
        self.data_files = data

    def __getitem__(self, item):
        res = self.data_files[item]
        return res

    def __len__(self):
        return len(self.data_files)

def BuildKNNGraphByFAISS_GPU(db,k):
    dbsize, dim = db.shape
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    res = faiss.StandardGpuResources()
    nn = faiss.GpuIndexFlatL2(res, dim, flat_config)
    nn.add(db)
    dists,idx = nn.search(db, k+1)
    return idx[:,1:],dists[:,1:]

from pytorch_sift import SIFTNet

def pre_init_with_sift(trainPhotoTourDataset):

    patch_size = 65
    ON_GPU = True
    SIFT = SIFTNet(patch_size=patch_size, do_cuda=ON_GPU)
    SIFT.eval()

    if ON_GPU:
        SIFT.cuda()

    transformed = []
    for img in trainPhotoTourDataset.data:
        transformed.append(np.expand_dims(cv2.resize(img.cpu().numpy(), (65,65)), axis=0))

    phototour_loader = data_utils.DataLoader(PhototourTrainingData(transformed), batch_size=256, shuffle=False)
    descriptors = []

    pbar = tqdm(enumerate(phototour_loader))
    for batch_idx, data_a in pbar:

        if ON_GPU:
            torch_patches = Variable(data_a.type(torch.FloatTensor).cuda(), volatile=True)
        else:
            torch_patches = Variable(data_a.type(torch.FloatTensor), volatile=True)

        res = SIFT(torch_patches)
        sift = np.round(512. * res.data.cpu().numpy()).astype(np.float32)
        descriptors.extend(sift)

    return np.array(descriptors)


def get_descriptors_for_dataset(model, trainPhotoTourDataset):
    model.eval()
    transformed = []

    for img in trainPhotoTourDataset.data:
        transformed.append(trainPhotoTourDataset.transform(img.numpy()))
    print(len(transformed))
    phototour_loader = data_utils.DataLoader(PhototourTrainingData(transformed), batch_size=128, shuffle=False)
    descriptors = []
    pbar = tqdm(enumerate(phototour_loader))
    for batch_idx, data_a in pbar:

        if args.cuda:
            model.cuda()
            data_a = data_a.cuda()

        data_a = Variable(data_a, volatile=True),
        out_a = model(data_a[0])
        descriptors.extend(out_a.data.cpu().numpy())

    return descriptors


def remove_descriptors_with_same_index(min_dist_indices, indices, labels, descriptors):

    res_min_dist_indices = []

    for current_index in range(0, len(min_dist_indices)):
        # get indices of the same 3d points
        point3d_indices = labels[indices[current_index]]
        indices_to_remove = []
        for indx in min_dist_indices[current_index]:
            # add to removal list indices of the same 3d point and same images in other 3d point
            if(indx in point3d_indices or (descriptors[indx] == descriptors[current_index]).all()):
                indices_to_remove.append(indx)

            # check if 3dpoint indices equals to any of min dist indices
            for point3d_indx in point3d_indices:
                if ((descriptors[point3d_indx]==descriptors[indx]).all()):
                    indices_to_remove.append(indx)

        curr_desc = [x for x in min_dist_indices[current_index] if x not in indices_to_remove]
        res_min_dist_indices.append(curr_desc)


    return res_min_dist_indices


def get_hard_negatives(trainPhotoTourDataset, descriptors):

    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds

    labels = create_indices(trainPhotoTourDataset.labels)
    indices = {}
    for key, value in labels.iteritems():
        for ind in value:
            indices[ind] = key

    print('getting closest indices .... ')
    descriptors_min_dist, inidices = BuildKNNGraphByFAISS_GPU(descriptors, args.hardnegatives)
    print(descriptors_min_dist[0])

    print('removing descriptors with same indices .... ')
    descriptors_min_dist = remove_descriptors_with_same_index(descriptors_min_dist, indices, labels, descriptors)
    print(descriptors_min_dist[0])

    return descriptors_min_dist

if __name__ == '__main__':

            LOG_DIR = args.log_dir + args.experiment_name
            logger, file_logger = None, None
            model = TNet()

            if(args.enable_logging):
                #logger = Logger(LOG_DIR)
                file_logger = FileLogger(LOG_DIR)

            test_dataset_names = copy.copy(dataset_names)
            test_dataset_names.remove(args.training_set)

            trainPhotoTourDataset, test_loaders = create_loaders()
            main(trainPhotoTourDataset, test_loaders, model, logger, file_logger)
