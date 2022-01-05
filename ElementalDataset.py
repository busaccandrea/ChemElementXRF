import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, Subset
from time import time
import numpy as np
from torch.utils.data.sampler import Sampler
from glob import glob
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         row, class_ = sample['row'], sample['class']

#         return {'row': torch.from_numpy(row),
#                 'class': torch.from_numpy(class_)}

def join_ds(data_files, label_files):
    is_first = True
    pointers = [0]
    for i, ds in enumerate(data_files):
        tmp = np.load(ds)
        tmpl = np.load(label_files[i])
        if is_first:
            dataset = tmp
            labels = tmpl
            is_first = False
        else:
            dataset = np.concatenate((dataset, tmp), axis=0)
            labels = np.concatenate((labels, tmpl), axis=0)
        pointers += [tmp.shape[0] + pointers[-1]]
    return dataset, labels, pointers


class ElementalDataset(Dataset):
    """ Build a custom dataset for ChemElementRegressor model """
    def __init__(self, preprocessing, normalize_data, normalize_labels):
        self.data_files = glob('./data/*/calibrated/data_1024.npy')# + glob('./data/synthetic/*/calibrated/data_1024.npy')
        self.label_files = glob('./data/*/labels/labels.npy')# + glob('./data/synthetic/*/labels.npy')
        self.data_set, self.labels, self.pointers = join_ds(self.data_files, self.label_files)
        self.labels[self.labels<0] = 0
        self.max_labels = np.max(self.labels, axis=0)

        # preprocessing
        self.preprocessing(preprocessing)

        # normalization
        self.normalization(normalize_data, normalize_labels)
        

    def __getitem__(self, index):
        ds = np.random.randint(0,len(self.data_files))        
        start = self.pointers[ds]
        stop = self.pointers[ds + 1]
        index = np.random.randint(start, stop)
        val = self.labels[index]
        row = self.data_set[index]
        val = torch.tensor([val])

        return {'row': row, 'counts': val, 'index': ds}
    
    def __len__(self):
        return self.data_set.shape[0]

    def preprocessing(self, p):
        if p=='log':
            self.data_set[self.data_set<1] = 1
            self.data_set = np.log(self.data_set)
        elif p=='sqrt':
            self.data_set[self.data_set<0] = 0
            self.data_set = np.sqrt(self.data_set)
        elif not p=='none':
            print('Invalid preprocessing option')

    def normalization(self, norm_data, norm_labels):
        if norm_data == 1:
            self.data_set = minmax_scale(self.data_set, axis=1)
        if norm_labels == 1:
            self.labels = minmax_scale(self.labels, axis=0)


class BalancedSampler(Sampler):
    def __init__(self, data_set:ElementalDataset, batch_size):
        self.data_set = data_set
        self.dataset_len = data_set.__len__()
        self.batch_size = batch_size
        self.n_samples_per_class = int(batch_size/len(self.data_set.data_files))

    def __iter__(self):
        random_indices = []
        time_start = time()
        while len(random_indices) < self.dataset_len:
            for i in range(0, len(self.data_set.pointers)-1):
                number_of_rows = self.data_set.pointers[i+1] - self.data_set.pointers[i]
                random_indices += list(self.data_set.pointers[i] + np.random.choice(number_of_rows, size=self.n_samples_per_class, replace=False))

            if len(random_indices) < self.batch_size:
                n = self.batch_size - len(random_indices)
                number_of_rows = self.data_set.pointers[-1] - self.data_set.pointers[-2]
                random_indices += list(self.data_set.pointers[-2] + np.random.choice(number_of_rows, size=n, replace=False))
        # np.random.shuffle(random_indices)
        # random_indices = random_indices[:self.dataset_len]
        # random_indices = iter(random_indices[:self.dataset_len])
        print('batchtime', time()-time_start)
        return iter(random_indices)

    def __len__(self):
        return self.dataset_len

# ds = ElementalDataset()
# BalancedSampler(ds, batch_size=256).__iter__()