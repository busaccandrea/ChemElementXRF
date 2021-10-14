import scipy.io as scio
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from scipy.sparse import data
from sklearn.model_selection import train_test_split
import h5py
import edf_read
from glob import glob
from time import time

def check_existing_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Folder not found, created.')

def __pack_labels(path_to_labels):
    """ pack all .tif files in a single numpy.array """
    labels_files = glob(path_to_labels + '*.tif')
    
    labels = []
    for label_file in labels_files:
        labels.append(np.array(Image.open(label_file)))
    
    labels = np.array(labels)
    labels = np.swapaxes(labels, axis1=0, axis2=1)
    labels = np.swapaxes(labels, axis1=1, axis2=2)

    return labels
        


def create_h5_dataset(data_folder, train_test_ratio=0.8, seed=2233):
    """ given the data and label folder, saves the dataset as .h5 file. """
    
    edf_folder = data_folder + 'Edf/'
    labels_folder = data_folder + 'labels/'
    
    # load data from EDF.
    data = edf_read.read_edf(edf_folder) # shape (n,m,2048)
    # get labels
    labels = __pack_labels(labels_folder)

    # the shape after the split is still a cube.
    input_train, input_test, target_train, target_test = train_test_split(data, labels, test_size=1-train_test_ratio, random_state=seed)
    
    # save h5 for train set
    print('Saving train.h5')
    with h5py.File(data_folder + 'train.h5','w') as f:
        f.create_dataset('inputs', data = input_train)
        f.create_dataset('targets', data = target_train)
    
    # save h5 for test set
    print('Saving test.h5')
    with h5py.File(data_folder + 'test.h5','w') as f:
        f.create_dataset('inputs', data = input_test)
        f.create_dataset('targets', data = target_test)