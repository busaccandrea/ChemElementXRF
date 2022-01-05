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

def evaluate():
    # define device to use
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device available:', device)
    running_times = []

    data_loading_time = time()
    print('loading data...')
    datapath = './data/mockup/calibrated/'
    # data_np = np.load('./data/DOggionoGiulia/Edf/data.npy').reshape((-1, 2048))
    data_np = np.load(datapath+'data_1024.npy')
    ground_truth = np.load('./data/mockup/labels/labels.npy').reshape((130, 185, 10))
    # ground_truth = ground_truth.reshape((-1, ground_truth.shape[2]))
    print('loaded in', time() - data_loading_time)

    print('creating dataset and loader...')
    dataset_time = time()
    data_set = ChemElementRegressorDataset_for_evaluate(data=data_np)
    eval_loader = DataLoader(dataset=data_set, batch_size=4096, shuffle=False)
    print('created in', time() - dataset_time)
    classes = {
        '0':'Ca',
        '1':'Co',
        '2':'Cu',
        '3':'Fe',
        '4':'Hg',
        '5':'K',
        '6':'Mn',
        '7':'Pb',
        '8':'S',
        '9':'Sn'
        }
    experiments = glob('./run/22-12-21/prepr_none*/')
    for experiment in experiments:
        nn_list = glob(experiment + '*.pth')
        print('starting evaluation of experiment:', experiment)
        for nn in nn_list:
            epoch = path.basename(nn).split('.')[0].split('_')[-1]

            if 'model_099.pth' in nn:
                print('\nEvaluating', nn + '...')
                # if path.isfile(experiment+'/eval/' + 'eval_ep'+str(epoch)+'.npy'):
                #     print('already evaluated. Next one!\n')
                #     continue
                if int(epoch) == 99:
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
                        tmp_output = torch.squeeze(tmp_output).to('cpu').detach()
                        tmp_output[tmp_output<0] = 0
                        concat_time = time()
                        if first_time:
                            outputs = tmp_output
                            first_time = False
                        else:
                            outputs = torch.cat((outputs, tmp_output), dim=0)

                    outputs = outputs.numpy()
                    outputs = outputs.reshape((130, 185, 10))
                    print(outputs.shape)

                    check_existing_folder(experiment+'/eval/')
                    np.save(experiment+'/eval/' + 'mockup_eval_ep' + str(epoch).zfill(3) + '.npy', outputs)
                    """ for im in range(outputs.shape[2]):    
                        plt.figure('Model '+classes[str(im)])
                        plt.imshow(outputs[:,:,im])
                        plt.figure('Ground truth '+classes[str(im)])
                        plt.imshow(ground_truth[:,:, im])
                    # plt.figure('Model '+classes[str(5)])
                    # plt.imshow(outputs[:,:,5])
                    # plt.figure('Ground truth '+classes[str(5)])
                    # plt.imshow(ground_truth[:,:, 5])
                    plt.show() """
    

if __name__=='__main__':
    evaluate()