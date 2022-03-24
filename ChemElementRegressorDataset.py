import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, Subset
from time import time
import numpy as np
from torch.utils.data.sampler import Sampler
import torchtest
from sklearn.preprocessing import minmax_scale, PowerTransformer
from matplotlib import pyplot as plt


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        row, class_ = sample['row'], sample['class']

        return {'row': torch.from_numpy(row),
                'class': torch.from_numpy(class_)}


class ChemElementRegressorDataset(Dataset):
    """ Build a custom dataset for ChemElementRegressor model """
    def __init__(self, data, labels, preprocessing, normalize_data, normalize_labels):
        self.data_set = data
        self.labels = np.array(labels, dtype=float)
        self.max_labels = np.max(self.labels, axis=0)
        
        self.data_set[self.data_set<0] = 0
        self.labels[self.labels<0] = 0

        self.normalization(normalize_data, normalize_labels)
        self.preprocessing(preprocessing)

 
    def __len__(self):
        return self.data_set.shape[0]


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        else:
            index = [index]

        counts = self.labels[index]
        
        row = self.data_set[index]
        counts = torch.from_numpy(counts)

        return {'row': row, 'counts': counts}

    def preprocessing(self, p):
        if p=='log':
            self.data_set[self.data_set<1] = 1
            self.data_set = np.log(self.data_set)
        elif p=='sqrt':
            self.data_set[self.data_set<0] = 0
            self.data_set = np.sqrt(self.data_set)
        elif p=='exp':
            self.data_set = np.exp(self.data_set)
        elif not p=='none':
            print('Invalid preprocessing option')

    def normalization(self, norm_data, norm_labels):
        if norm_data == 1:
            self.data_set = minmax_scale(self.data_set, axis=1)
        if norm_labels == 1:
            self.labels = minmax_scale(self.labels, axis=0)

                
class BalancedRandomSampler(Sampler):
    def __init__(self, data_set:ChemElementRegressorDataset, high_low_ratio=0.5):
        self.data_set = data_set
        self.dataset_len = len(self.data_set)

        self.ratio = high_low_ratio
        
        self.high_count_idxs = self.data_set.high_count_idxs # where label is > threshold
        self.low_count_idxs = self.data_set.low_count_idxs # where label is <= threshold

    def __iter__(self):
        highspectra_choice = np.random.choice(self.high_count_idxs, int(self.dataset_len * self.ratio)) # pick from high_counts_idxs dataset_len/2 elements
        lowspectra_choice = np.random.choice(self.low_count_idxs, int(self.dataset_len * (1 - self.ratio)))

        highspectra = highspectra_choice
        lowspectra = lowspectra_choice

        idxs = np.hstack([highspectra, lowspectra])
        np.random.shuffle(idxs)
        idxs = idxs.astype(int)
        idxs = iter(idxs[:self.dataset_len])

        return iter(idxs)

    def __len__(self):
        return self.dataset_len