import os
import torch
import numpy as np
from CNN import N_ElementCNN, OneElementCNN
from sklearn.preprocessing import normalize
from torch.utils.data import random_split
from ChemElementRegressorDataset import ChemElementRegressorDataset, BalancedRandomSampler
import csv
from sklearn.model_selection import train_test_split
from PIL import Image

    # Braque analisys: E:\MondrianVenezia2021\BraqueAnalisiBis\Edf1
if __name__ == '__main__':

    # data file name to use for training
    path_to_data = 'data/Braque/'
    data = np.load(path_to_data + 'data_sampled.npy')

    # cutting the last row of the data to make it the same of label's.
    data = data[:-1]
    print('Data loaded. Shape:', data.shape)

    if data.ndim > 2:
        [rows, cols, spectrum_length] = data.shape
        data = data.reshape((rows*cols, spectrum_length))
    
    print('Data loaded. Shape:', data.shape)
    # pre-processing of inputs
    data = normalize(data, axis=1, norm='max')
    
    # elements = [
    #     1,
    #     2,
    #     3,
    #     4, # index out of bound
    #     5, # index out of bound
    #     6,
    #     7,
    #     8,
    #     9
    # ]

    # elements = ['Ca','Cu','Fe','Hg','K','Mn','Pb','Sn','Sr','Ti']
    elements = ['Cu','Fe','Hg','K','Mn','Pb','Sn','Sr','Ti']
    for i, element in enumerate(elements):
        print("Training of element:", element)
        # define which device is available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        labels = np.array(Image.open(path_to_data + 'labels/'+ element + '.tiff'))

        [l_rows, l_cols] = labels.shape
        print('label\'s shape:', labels.shape)
        labels = labels.reshape((l_rows*l_cols,1))
        
        # csvfile = path_to_data + 'labels/' + str(element) + '.csv'
        
        # training parameters
        epochs = 10
        batch_size = 1024
        learning_rates = [0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006 ] # using a list to give different learning rates for each element
        
        # define dataset object
        n_high_counts = 0
        # labels = []
        row_numbers = []
        """ with open(csvfile, newline='') as infile: # this 'with' operation takes 0.01 sec
            csvreader = csv.reader(infile, delimiter=' ')
            for row in csvreader:
                # print('DEBUG: ROW', row)
                row_numbers.append(float(row[0]))
                labels.append(float(row[-1]))
        n_low_counts = data.shape[0] - n_high_counts """
        # print('DEBUG: n_low_counts:', n_low_counts, '\nDEBUG: n_high_counts:', n_high_counts)
        
        # define ratio between train set and test set
        trainset_ratio = 0.8
        trainsize = int(data.shape[0] * trainset_ratio)
        testsize = data.shape[0] - trainsize
        
        # split dataset into training set and test set.
        xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=1-trainset_ratio, random_state=2233)
        print('test set length:', testsize, '\ntraining set length:', trainsize, '\ndatashape', data.shape)
        xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=1-trainset_ratio, random_state=2233)
        training_set = ChemElementRegressorDataset(data=xtrain, labels=ytrain) #, row_numbers=row_numbers) #row numbers
        test_set = ChemElementRegressorDataset(data=xtest, labels=ytest) #, row_numbers=row_numbers) #row numbers
        
        print('creating train dataset...')
        training_set = ChemElementRegressorDataset(data=xtrain, labels=ytrain)
        print('creating test dataset...')
        test_set = ChemElementRegressorDataset(data=xtest, labels=ytest)
        
        training_sampler = BalancedRandomSampler(training_set, high_low_ratio=0.5) # pass the sampler to train method
        test_sampler = BalancedRandomSampler(test_set, high_low_ratio=0.5) # pass the sampler to train method

        # define model
        model_filename = path_to_data + 'models' + str(element) + '.p'
        if os.path.isfile(model_filename):
            print('Model and checkpoint files found. Loading.')
            model = torch.load(model_filename)
        else:
            print('Model not found. Creating...')
            input_size = data.shape[1] # size of each sample
            # model = ChemElementRegressor(input_size=input_size)
            model = ChemElementRegressor_Convbased(input_size=input_size)
            print('Done.')

        print('Moving model to', device)
        model.to(device)
        model.double()

        # train
        print('Training model for', epochs, 'epochs...')
        train_regressor(model, training_set, test_set, epochs=epochs, batch_size=batch_size, lr=learning_rates[i], sampler={'training':training_sampler,'test':test_sampler}, model_filename=str(element), output_path=path_to_data+'models/')
        print('\nDone.')