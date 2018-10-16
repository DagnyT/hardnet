import os
import numpy as np
import torch
import torch.utils.data as data
import tqdm
import random
from copy import deepcopy

class TotalDatasetsLoader(data.Dataset):

    def __init__(self, datasets_path, train = True, transform = None, batch_size = None, n_triplets = 100000, fliprot = False, *arg, **kw):
        super(TotalDatasetsLoader, self).__init__(*arg, **kw)

        datasets = [torch.load(dataset) for dataset in datasets_path]
        data, labels = datasets[0][0], datasets[0][1]

        for i in range(1,len(datasets)):
            data = torch.cat([data,datasets[i][0]])
            labels = torch.cat([labels, datasets[i][1]+torch.max(labels)+1])

        del datasets
        self.transform = transform
        self.train = train
        self.n_triplets = n_triplets
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
            if fliprot:
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


if __name__ == '__main__':
    datasets_path = os.path.join(os.path.abspath(__file__ + "/../../../"),'datasets')
    datasets = [os.path.join(datasets_path, dataset) for dataset in os.listdir(datasets_path)]
    TotalDatasetsLoader(datasets)