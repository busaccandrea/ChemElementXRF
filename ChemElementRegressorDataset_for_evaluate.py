import torch
from torch.utils.data import Dataset


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        row, class_ = sample['row'], sample['class']

        return {'row': torch.from_numpy(row),
                'class': torch.from_numpy(class_)}


class ChemElementRegressorDataset_for_evaluate(Dataset):
    """ Build a custom dataset for ChemElementRegressor model """
    def __init__(self, data, transform=ToTensor()):
        self.data = data
        self.trasform = transform
        
    # mandatory override of methods __len__ and __getitem__
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        else:
            index = [index]
        return self.data[index]