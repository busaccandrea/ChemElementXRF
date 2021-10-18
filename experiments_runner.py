import yaml
from torch import optim
from torch.nn import modules
import yaml
import torch
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from torchnet.meter import AverageValueMeter
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import h5py
from ChemElementRegressorDataset import ChemElementRegressorDataset
from utilities import check_existing_folder
import train_cnn
from glob import glob


# def train(model, train_set, test_set, batch_size, sampler, lr=0.003, epochs=5, momentum=0.99, loss, model_filename='regressor.p', output_path='data/'):
def main(experiments):
    """ experiments: path to experiment files. """
    with open ('./cnn_general_config.yaml') as general_yaml:   
        general_config_file = yaml.load(general_yaml, Loader=yaml.FullLoader)

    # load dataset files (h5)
    with h5py.File(general_config_file['test_set'], "r") as f:
        (x,y,z) = f['data'][:].shape
        test_data = f['data'][:].reshape((x*y, z))
        print(test_data.shape)
        (x,y,z) = f['labels'][:].shape
        test_labels = f['labels'][:].reshape((x*y, z))
        test_set = ChemElementRegressorDataset(data=test_data, labels=test_labels)
    
    with h5py.File(general_config_file['train_set'], 'r') as g:
        (x,y,z) = g['data'][:].shape
        training_data = g['data'][:].reshape((x*y, z))
        (x,y,z) = g['labels'][:].shape
        training_labels = g['labels'][:].reshape((x*y, z))

        training_set = ChemElementRegressorDataset(data=training_data, labels=training_labels)
    
    # loop over experiments (.yaml files) 
    # in this case the path is './run/nelements_mse/*/config.yaml'
    for experiment in experiments:
        # train the model
        train_cnn.main(yaml_file=experiment, train_set=training_set, test_set=test_set)


if __name__=='__main__':
    experiments = glob('./run/nelements_mse/*/config.yaml')
    print('config files:\n', experiments)
    main(experiments=experiments)