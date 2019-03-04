import os
import numpy as np
import cv2
import time
from tqdm import tqdm
import torch
from torch.autograd import Variable
from skimage.io import imread

def w1bs_extract_descs_and_save(input_img_fname, model, desc_name, mean_img=0.443728476019, std_img=0.20197947209, cuda = False, out_dir = None):
    if out_dir is None:
        out_fname = input_img_fname.replace("data/W1BS", "data/out_descriptors").replace(".bmp", "." + desc_name)
        out_dir = os.path.dirname(out_fname)
    else:
        out_fname = out_dir + input_img_fname[input_img_fname.find('data/W1BS'):].replace("data/W1BS", "").replace(".bmp", "." + desc_name)
        out_fname = out_fname.replace('//', '/')
        out_dir = os.path.dirname(out_fname)
    if len(out_dir) > 0:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    image = imread(input_img_fname, 0)
    h, w = image.shape
    # print(h,w)
    n_patches = int(h / w)
    patches_for_net = np.zeros((n_patches, 1, 32, 32))
    for i in range(n_patches):
        patch = cv2.resize(image[i * (w): (i + 1) * (w), 0:w], (32, 32))
        patches_for_net[i, 0, :, :] = patch[0:w, 0:w]
    patches_for_net = patches_for_net/255
    patches_for_net -= mean_img  # np.mean(patches_for_net)
    patches_for_net /= std_img  # np.std(patches_for_net)
    t = time.time()
    ###
    model.eval()
    outs = []
    labels, distances = [], []
    pbar = tqdm(enumerate(patches_for_net))
    bs = 128
    n_batches = int(n_patches / bs) + 1
    for batch_idx in range(n_batches):
        if batch_idx == n_batches - 1:
            if (batch_idx + 1) * bs > n_patches:
                end = n_patches
            else:
                end = (batch_idx + 1) * bs
        else:
            end = (batch_idx + 1) * bs
        data_a = patches_for_net[batch_idx * bs: end, :, :, :].astype(np.float32)
        data_a = torch.from_numpy(data_a)
        if cuda:
            data_a = data_a.cuda()
        data_a = Variable(data_a, volatile=True)
        out_a = model(data_a)
        outs.append(out_a.data.cpu().numpy().reshape(-1, 128))
    ###
    res_desc = np.concatenate(outs)
    print(res_desc.shape, n_patches)
    res_desc = np.reshape(res_desc, (n_patches, -1))
    np.savetxt(out_fname, res_desc, delimiter=' ', fmt='%10.7f')
    return
