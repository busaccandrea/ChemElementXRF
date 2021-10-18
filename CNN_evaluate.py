import torch
import numpy as np
from PIL import Image
from time import time
from sklearn.preprocessing import normalize
from torch.functional import norm
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from ChemElementRegressorDataset import ChemElementRegressorDataset
from glob import glob
from os import path
from ChemElementRegressorDataset_for_evaluate import ChemElementRegressorDataset_for_evaluate
from utilities import check_existing_folder


if __name__=='__main__':
    

    # define device to use
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device available:', device)
    running_times = []

    data_loading_time = time()
    print('loading data...')
    data_np = np.load('./data/DOggionoGiulia/Edf/data.npy').reshape((-1, 2048))
    print('loaded in', time() - data_loading_time)
    
    ground_truth = np.load('./data/DOggionoGiulia/labels/labels.npy')

    print('creating dataset and loader...')
    dataset_time = time()
    data_set = ChemElementRegressorDataset_for_evaluate(data=data_np)
    eval_loader = DataLoader(dataset=data_set, batch_size=2048, shuffle=False)
    print('created in', time() - dataset_time)

    experiments = glob('./run/nelements_mse/*/')
    for experiment in experiments:
        nn_list = glob(experiment + './*.pth')
        for nn in nn_list:
            epoch = path.basename(nn).split('.')[0].split('_')[-1]

            if path.isfile(experiment+'/eval/' + 'eval_ep'+str(epoch)+'.npy'):
                continue
            print('evaluating', path.basename(nn))
            
            experiment_timestart = time()
            model = torch.load(nn)
            model.to(device)
            model.eval()
            print('Model loaded and moved to :', device)
            
            first_time = True
            for i, batch_ in enumerate(eval_loader):
                batch_ = batch_.to(device)
                start = time()
                tmp_output = model(batch_)
                print('batch time', time()-start)
                tmp_output = torch.squeeze(tmp_output).to('cpu').detach()
                tmp_output[tmp_output<0] = 0
                concat_time = time()
                if first_time:
                    outputs = tmp_output
                    first_time = False
                else:
                    outputs = torch.cat((outputs, tmp_output), dim=0)
                print('concatenated in ', time()-concat_time)

            outputs = outputs.numpy()
            outputs = outputs.reshape((418, 418, 4))

            check_existing_folder(experiment+'/eval/')
            np.save(experiment+'/eval/' + 'eval_ep'+str(epoch)+'.npy', outputs)