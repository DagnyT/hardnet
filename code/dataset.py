# Training settings
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

def find_files(_data_dir, _image_ext):
    """Return a list with the file names of the images containing the patches
    """
    files = []
    # find those files with the specified extension
    for file_dir in os.listdir(_data_dir):
        if file_dir.endswith(_image_ext):
            files.append(os.path.join(_data_dir, file_dir))
    return sorted(files)  # sort files in ascend order to keep relations
def np2torch(npr):
    if len(npr.shape) == 4:
        return torch.from_numpy(np.rollaxis(npr, 3, 1))
    elif len(npr.shape) == 3:
        torch.from_numpy(np.rollaxis(npr, 2, 0))
    else:
        return torch.from_numpy(npr)
def read_patch_file(fname, patch_w = 65, patch_h = 65, start_patch_idx = 0):
    img = Image.open(fname).convert('RGB')
    width, height = img.size
    #print (img.size, patch_w, patch_h)
    assert ((height % patch_h == 0) and (width % patch_w == 0))
    patch_idxs = []
    patches = []
    current_patch_idx = start_patch_idx
    for y in range(0, height, patch_h):
        patch_idxs.append([])
        curr_patches = []
        for x in range(0, width, patch_w):
            patch = np.array(img.crop((x, y, x + patch_w, y + patch_h))).mean(axis = 2, keepdims = True)
            #print(patch.astype(np.float32).std(), patch.mean())
            if (patch.mean() != 0) and (patch.astype(np.float32).std() > 1e-2):
                curr_patches.append(patch.astype(np.uint8))
                patch_idxs[-1].append(current_patch_idx)
                current_patch_idx+=1
        if len(curr_patches) > 1:
            patches = patches + curr_patches
        else:
            for i in range(len(curr_patches)):
                current_patch_idx -=1
            patch_idxs = patch_idxs[:-1] 
    return np2torch(np.array(patches)), patch_idxs, patch_idxs[-1][-1]

def read_image_dir(dir_name, ext, patch_w, patch_h, good_fnames):
    fnames = find_files(dir_name, ext)
    patches = []
    idxs = []
    current_max_idx = 0
    for f in fnames:
        if f.split('/')[-1].replace('.png', '') not in good_fnames:
            continue
        try:
            torch_patches, p_idxs_list, max_idx = read_patch_file(f, patch_w, patch_h, current_max_idx)
        except:
            continue
        current_max_idx = max_idx + 1
        #if patches is None:
        #    patches = torch_patches
        #    idxs = p_idxs_list
        #else:
        patches.append(torch_patches)
        idxs = idxs + p_idxs_list
        print (f, len(idxs))
    print( 'torch.cat')
    patches = torch.cat(patches, dim = 0)
    print ('done')
    return patches, idxs


class HPatchesDM(data.Dataset):
    image_ext = 'png'
    def __init__(self, root, name, train=True, transform=None,
                 download=True, pw = 65, ph = 65,
                 n_pairs = 1000, batch_size = 128, split_name = 'b'):
        self.root = os.path.expanduser(root)
        self.name = name
        self.n_pairs = n_pairs
        self.split_name = split_name
        self.batch_size = batch_size
        self.train = train
        self.data_dir = os.path.join(self.root, name)
        if self.train:
            self.data_file = os.path.join(self.root, '{}.pt'.format(self.name  + '_train' ))
        else:
            self.data_file = os.path.join(self.root, '{}.pt'.format(self.name  + '_test' ))            
        self.transform = transform
        self.patch_h = ph
        self.patch_w = pw
        self.batch_size = batch_size
        if download:
            self.download()

        if not self._check_datafile_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # load the serialized data
        self.patches, self.idxs = torch.load(self.data_file)
        print('Generating {} triplets'.format(self.n_pairs))
        self.pairs = self.generate_pairs(self.idxs, self.n_pairs)
        return
    def generate_pairs(self, labels, n_pairs):
        pairs = []
        n_classes = len(labels)
        # add only unique indices in batch
        already_idxs = set()
        for x in tqdm(range(n_pairs)):
            if len(already_idxs) >= self.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            while len(labels[c1]) < 3:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            if len(labels[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(labels[c1]))
                while (self.patches[labels[c1][n1],:,:,:].float().std() < 1e-2):
                    n1 = np.random.randint(0, len(labels[c1]))
                n2 = np.random.randint(0, len(labels[c1]))
                while (self.patches[labels[c1][n2],:,:,:].float().std() < 1e-2):
                    n2 = np.random.randint(0, len(labels[c1]))
            pairs.append([labels[c1][n1], labels[c1][n2]])
        return torch.LongTensor(np.array(pairs))
    def __getitem__(self, index):
        def transform_pair(i1,i2):
            if self.transform is not None:
                return self.transform(i1.cpu().numpy()), self.transform(i2.cpu().numpy())
            else:
                return i1,i2
        t = self.pairs[index]
        a, p = self.patches[t[0],:,:,:], self.patches[t[1],:,:,:]
        a1,p1 = transform_pair(a,p)
        return (a1,p1)

    def __len__(self):
        return len(self.pairs)

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)

    def _check_downloaded(self):
        return os.path.exists(self.data_dir)

    def download(self):
        if self._check_datafile_exists():
            print('# Found cached data {}'.format(self.data_file))
            return
        # process and save as torch files
        print('# Caching data {}'.format(self.data_file))
        import json
        from pprint import pprint
        #print self.urls['splits']
        with open(os.path.join(self.root, 'splits.json')) as splits_file:    
            data = json.load(splits_file)
        if self.train:
            self.img_fnames = data[self.split_name]['train']
        else:
            self.img_fnames = data[self.split_name]['test']
        dataset = read_image_dir(self.data_dir, self.image_ext, self.patch_w, self.patch_h, self.img_fnames)
        print('saving...')
        with open(self.data_file, 'wb') as f:
            torch.save(dataset, f)
        return
class TotalDatasetsLoader(data.Dataset):

    def __init__(self, datasets_path, train = True, transform = None, batch_size = None, n_triplets = 5000000, fliprot = False, *arg, **kw):
        super(TotalDatasetsLoader, self).__init__()

        datasets_path = [os.path.join(datasets_path, dataset) for dataset in os.listdir(datasets_path)]

        datasets = [torch.load(dataset) for dataset in datasets_path]

        data, labels = datasets[0][0], datasets[0][1]

        for i in range(1,len(datasets)):
            data = torch.cat([data,datasets[i][0]])
            labels = torch.cat([labels, datasets[i][1]+torch.max(labels)+1])

        del datasets

        self.data, self.labels = data, labels
        self.transform = transform
        self.train = train
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        self.fliprot = fliprot
        if self.train:
                print('Generating {} triplets'.format(self.n_triplets))
                self.triplets = self.generate_triplets(self.labels, self.n_triplets, self.batch_size)


    def generate_triplets(self, labels, num_triplets, batch_size):
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
                if len(already_idxs) >= batch_size:
                    already_idxs = set()
                c1 = np.random.randint(0, n_classes)
                while c1 in already_idxs:
                    c1 = np.random.randint(0, n_classes)
                already_idxs.add(c1)
                c2 = np.random.randint(0, n_classes)
                while c1 == c2:
                    c2 = np.random.randint(0, n_classes)
                if len(indices[c1]) == 2:  # hack to speed up process
                    n1, n2 = 0, 1
                else:
                    n1 = np.random.randint(0, len(indices[c1]))
                    n2 = np.random.randint(0, len(indices[c1]))
                    while n1 == n2:
                        n2 = np.random.randint(0, len(indices[c1]))
                n3 = np.random.randint(0, len(indices[c2]))
                triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
            return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
            def transform_img(img):
                if self.transform is not None:
                    img = (img.numpy())/255.0
                    img = self.transform(img)
                return img

            t = self.triplets[index]
            a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

            img_a = transform_img(a)
            img_p = transform_img(p)

            # transform images if required
            if self.fliprot:
                do_flip = random.random() > 0.5
                do_rot = random.random() > 0.5

                if do_rot:
                    img_a = img_a.permute(0,2,1)
                    img_p = img_p.permute(0,2,1)

                if do_flip:
                    img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                    img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
            return img_a, img_p

    def __len__(self):
            if self.train:
                return self.triplets.size(0)

class TripletPhotoTour(dset.PhotoTour):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    urls = {
        'notredame_harris': [
            'http://matthewalunbrown.com/patchdata/notredame_harris.zip',
            'notredame_harris.zip',
            '69f8c90f78e171349abdf0307afefe4d'
        ],
        'yosemite_harris': [
            'http://matthewalunbrown.com/patchdata/yosemite_harris.zip',
            'yosemite_harris.zip',
            'a73253d1c6fbd3ba2613c45065c00d46'
        ],
        'liberty_harris': [
            'http://matthewalunbrown.com/patchdata/liberty_harris.zip',
            'liberty_harris.zip',
            'c731fcfb3abb4091110d0ae8c7ba182c'
        ],
        'notredame': [
            'http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip',
            'notredame.zip',
            '509eda8535847b8c0a90bbb210c83484'
        ],
        'yosemite': [
            'http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip',
            'yosemite.zip',
            '533b2e8eb7ede31be40abc317b2fd4f0'
        ],
        'liberty': [
            'http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip',
            'liberty.zip',
            'fdd9152f138ea5ef2091746689176414'
        ],
    }
    mean = {'notredame': 0.4854, 'yosemite': 0.4844, 'liberty': 0.4437, 'notredame_harris': 0.4854, 'yosemite_harris': 0.4844, 'liberty_harris': 0.4437}
    std = {'notredame': 0.1864, 'yosemite': 0.1818, 'liberty': 0.2019, 'notredame_harris': 0.1864, 'yosemite_harris': 0.1818, 'liberty_harris': 0.2019}
    lens = {'notredame': 468159, 'yosemite': 633587, 'liberty': 450092, 'liberty_harris': 379587, 'yosemite_harris': 450912 , 'notredame_harris': 325295}
    def __init__(self, train=True, transform=None, batch_size = None, n_triplets = 5000, load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = 1000
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)
    def generate_triplets(self,labels, num_triplets):
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
            if len(already_idxs) >= self.batch_size:
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
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
        if True:#args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
                if self.out_triplets:
                    img_n = img_n.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,:,::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

