import os
import numpy as np
import torch
import torch.utils.data as data
import cv2

types = ['e1','e2','e3','e4','e5','ref','h1']
images_to_exclude = ['v_adam', 'v_boat', 'v_graffiti', 'v_there','i_dome']

def mean_image(patches):
    mean = np.mean(patches)
    return mean

def std_image(patches):
    std = np.std(patches)
    return std

class HPatches(data.Dataset):

    def __init__(self, train=True, transform=None, download=False):
        self.train = train
        self.transform = transform

    def read_image_file(self, data_dir):
        """Return a Tensor containing the patches
        """
        patches = []
        labels = []
        counter = 0
        hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        for directory in hpatches_sequences:
           if (directory not in images_to_exclude):
            print(directory)
            for type in types:
                sequence_path = os.path.join(data_dir, directory,type)+'.png'
                image = cv2.imread(sequence_path, 0)
                h, w = image.shape
                n_patches = int(h / w)
                for i in range(n_patches):
                    patch = image[i * (w): (i + 1) * (w), 0:w]
                    patch = cv2.resize(patch, (64, 64))
                    patch = np.array(patch, dtype=np.uint8)
                    patches.append(patch)
                    labels.append(i+counter)
            counter += n_patches

        return torch.ByteTensor(np.array(patches, dtype=np.uint8)), torch.LongTensor(labels)

if __name__ == '__main__':

    # need to be specified
    path_to_hpatches_directory = '/home/dagnyt/hpatches-release/'

    hPatches = HPatches()
    images, labels = hPatches.read_image_file(path_to_hpatches_directory)
    datasets_path = os.path.abspath(__file__ + "/../../../")
    with open(os.path.join(datasets_path,('datasets/hpatches_dataset.pt')), 'wb') as f:
        torch.save((images, labels), f)
    print('Saved')