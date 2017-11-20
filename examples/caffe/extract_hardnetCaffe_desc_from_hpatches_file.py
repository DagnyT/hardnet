#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import sys
import numpy as np
try:
    caffe_root = '/home/ubuntu/dev/caffe/' 
    sys.path.insert(0, caffe_root + 'python')
    sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
    import caffe
    caffe_imported = True
except ImportError:
    caffe_imported = None
if caffe_imported is None:
    raise ImportError("Please, install caffe.")
import os
import sys
import time
import cv2
PATCH_SIZE = 32
DESC_SIZE = 128
BATCH_SIZE = 256
PROTOTXT_FNAME = 'HardNet.prototxt'
WEIGHT_FNAME = 'HardNet++.caffemodel'


def preprocess_patch(patch):
    out = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE)).astype(np.float32) 
    return out.reshape(1, PATCH_SIZE, PATCH_SIZE)

def extract_tfeats(net,patches):
    n_desc = len(patches)
    n_batches = int(np.ceil(float(n_desc) / BATCH_SIZE))
    descriptors = np.zeros((n_desc,DESC_SIZE))
    for i in range(n_batches):
        start = i * BATCH_SIZE
        if i < n_batches - 1:
            end = (i + 1) * BATCH_SIZE
        else:
            end = n_desc
        current_batch_size = end - start;
        currect_patches = patches[start:end, :, :, :]
        net.blobs['data'].reshape(current_batch_size, 1, PATCH_SIZE, PATCH_SIZE)
        net.blobs['data'].data[...] = currect_patches
        descriptors[start:end,:] = net.forward()['final'].reshape(current_batch_size, DESC_SIZE)
    return descriptors

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Wrong input format. Try python extract_hardnetCaffe_desc_from_hpatches_file.py ../../imgs/ref.png ref_caffe.TFEAT')
        sys.exit(1)

    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]

    t = time.time()
    image = cv2.imread(input_img_fname,0) #hpatch image is patch column 65*n x 65
    h,w = image.shape
    n_patches = h/w
    print('{0} patches to describe in {1}'.format(n_patches, input_img_fname))
    patches = np.zeros((n_patches,1,PATCH_SIZE, PATCH_SIZE))
    for i in range(n_patches):
        patches[i,:,:,:] = preprocess_patch(image[i*(w): (i+1)*(w), 0:w]) 
    caffe.set_mode_gpu()
    net = None
    net = caffe.Net(PROTOTXT_FNAME, caffe.TEST, weights = WEIGHT_FNAME)
    print 'Initialization and preprocessing time', time.time() - t
    t = time.time()
    out_descs = extract_tfeats(net,patches)
    print 'extraction time', time.time() - t 
    np.savetxt(output_fname, out_descs, delimiter=' ', fmt='%10.5f')    
