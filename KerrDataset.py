import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import minmax_scale


class KerrDataset(Dataset):
    def __init__(self, x, y, preprocessing='none', normalize_data=0, normalize_labels=0):
        self.data_set, self.labels = x, y
        self.labels[self.labels<0] = 0
        self.max_labels = np.max(self.labels, axis=0)

        # preprocessing
        self.preprocessing(preprocessing)

        # normalization
        self.normalization(normalize_data, normalize_labels)
        
    def __getitem__(self, index):
        val = self.labels[index]
        row = self.data_set[index]

        return {'row': row, 'counts': val}
    
    def __len__(self):
        return self.data_set.shape[0]

    def preprocessing(self, p):
        if p=='log':
            self.data_set[self.data_set<1] = 1
            self.data_set = np.log(self.data_set)
        elif p=='sqrt':
            self.data_set[self.data_set<0] = 0
            self.data_set = np.sqrt(self.data_set)
        elif p=='exp':
            self.data_set[self.data_set<0] = 0
            self.data_set = np.exp(self.data_set)    
        elif not p=='none':
            print('Invalid preprocessing option')
        self.labels = torch.from_numpy(self.labels.astype(float))

    def normalization(self, norm_data, norm_labels):
        if norm_data == 1:
            self.data_set = minmax_scale(self.data_set, axis=1)
        if norm_labels == 1:
            self.labels = minmax_scale(self.labels, axis=0)
