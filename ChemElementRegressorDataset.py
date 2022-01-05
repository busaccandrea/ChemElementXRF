import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, Subset
from time import time
import numpy as np
from torch.utils.data.sampler import Sampler
import torchtest
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        row, class_ = sample['row'], sample['class']

        return {'row': torch.from_numpy(row),
                'class': torch.from_numpy(class_)}


class ChemElementRegressorDataset(Dataset):
    """ Build a custom dataset for ChemElementRegressor model """
    def __init__(self, data, labels, transform=ToTensor()):
        self.data = data
        self.labels = np.array(labels, dtype=float)
        self.max_labels = np.max(self.labels, axis=0)
        
        # self.data = minmax_scale(self.data, axis=1)*2
        # self.labels = minmax_scale(self.labels, axis=0)*2

        # print('\n\n\n', self.labels.shape,'\n\n')

        # x, _, _ = plt.hist(self.labels.reshape(
        #     (self.labels.shape[0]*self.labels.shape[1], self.labels.shape[2])), bins=2000)
        # # show histogram of labels
        # # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
        # # plt.show()

        # self.threshold = np.argmax(x)
        # print('Threshold high/low counts:', np.max(x), self.threshold, self.threshold * 0.6)

        # self.high_count_idxs = np.where(self.labels>self.threshold)[0] # where label is > threshold
        # self.low_count_idxs = np.where(self.labels<=self.threshold)[0] # where label is <= threshold
        # print('ratio beetween high and low counts:', len(self.high_count_idxs),'/', len(self.low_count_idxs))

        # self.trasform = transform
        # self.pick_high = True

 
    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        else:
            index = [index]
        
        # val = self.labels[index]
        # value = int(val)

        # if self.pick_high:
        #     if value >= self.threshold:
        #         self.pick_high = False
        #     else:
        #         # choose randomly between self.high count indexes
        #         index = np.random.choice(self.high_count_idxs, 1)
        # else:
        #     if value < self.threshold:
        #         self.pick_high = True
        #     else:
        #         # choose randomly between self.low count indexes
        #         index = np.random.choice(self.low_count_idxs, 1)

        counts = self.labels[index]
        
        row = self.data[index]
        counts = torch.from_numpy(counts)

        return {'row': row, 'counts': counts}

                
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