import torch
import numpy as np
from PIL import Image
from time import time
from sklearn.preprocessing import normalize
from torch.functional import norm
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from glob import glob
from os import path
from ChemElementRegressorDataset_for_evaluate import ChemElementRegressorDataset_for_evaluate
from KerrDataset import KerrDataset
from utilities import check_existing_folder


def _loop_over_data(model, data_set, eval_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_time = True
    for i, batch_ in enumerate(eval_loader):
        x = batch_['row'].to(device)
        y = batch_['counts'].to(device)

        y = torch.squeeze(y)
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, 1)
            
        tmp_output = model(x)
        tmp_output = torch.squeeze(tmp_output).to('cpu').detach()
        tmp_output[tmp_output<0] = 0
                        
        if first_time:
            outputs = tmp_output
            first_time = False
        else:
            outputs = torch.cat((outputs, tmp_output), dim=0)

    outputs = outputs.numpy()
    outputs = outputs.reshape(data_set.labels.shape)
    print(outputs.shape)
    return outputs
    

def evaluate(datapath, results_path):
    # define device to use
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device available:', device)
    running_times = []

    data_loading_time = time()
    print('loading data...')
    # data_np = np.load('./data/DOggionoGiulia/Edf/data.npy').reshape((-1, 2048))

    data_np = np.load(datapath+'calibrated/data_1024.npy')

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
    
    [s0, s1] = np.array(Image.open(glob(datapath+'labels/*.tif*')[0])).shape
    shapes = (s0, s1, 10)
    ground_truth = np.load(datapath+'labels/labels.npy').reshape(shapes)
    # ground_truth = ground_truth.reshape((-1, ground_truth.shape[2]))
    print('loaded in', time() - data_loading_time)

    print('creating dataset and loader...')
    dataset_time = time()
    data_set = ChemElementRegressorDataset_for_evaluate(data=data_np)
    eval_loader = DataLoader(dataset=data_set, batch_size=4096, shuffle=False)
    print('created in', time() - dataset_time)
    experiments = glob(results_path+'prepr_exp-lr_0.0001-normdata_1-normlabels_0/')
    print (experiments)
    for experiment in experiments:
        nn_list = glob(experiment + '*.pth')
        print('starting evaluation of experiment:', experiment)
        for nn in nn_list:
            epoch = path.basename(nn).split('.')[0].split('_')[-1]

            if 'model_036.pth' in nn:
                print('\nEvaluating', nn + '...')
                # if path.isfile(experiment+'/eval/' + 'eval_ep'+str(epoch)+'.npy'):
                #     print('already evaluated. Next one!\n')
                #     continue
                if int(epoch) == 36:
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
                    outputs = outputs.reshape(shapes)
                    print(outputs.shape)

                    check_existing_folder(experiment+'/eval/')
                    np.save(experiment+'/eval/' + 'elgreco_eval_ep' + str(epoch).zfill(3) + '.npy', outputs)

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


def eval_(output_folder, data, labels, shapes, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    names = {
        '0':'elgreco',
        '1':'giulia',
        '2':'mockup'
        }
    
    model.to(device)
    model.eval()
    first_time = True
    for index, data in enumerate(data):
        data_set = ChemElementRegressorDataset_for_evaluate(data=data[index])
        eval_loader = DataLoader(dataset=data_set, batch_size=4096, shuffle=False)
    
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
        outputs = outputs.reshape(shapes[index])
        print(outputs.shape)

        check_existing_folder(output_folder+'/eval/')
        np.save(output_folder+'/eval/' + names[str(i)] +'_eval.npy', outputs)


def kerr_eval(output_folder, data_list, labels_list, model, epoch_list=[]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if type(model) == str: # if model is a string-type it means that this method is invoked by the eval_epoch_list method
        epoch = path.basename(model).split('.')[0].split('_')[-1]
        model = torch.load(model)
        model.to(device)
        model.eval()

        if int(epoch) not in epoch_list:
            return

    else: epoch = ''
    
    classes = {
    '0':'Au',
    '1':'Ca',
    '2':'Cu',
    '3':'Fe',
    '4':'Hg',
    '5':'Pb'
    }
    
    names = {
        '0':'3318HenryVIIIAnalisys'
        }
    
    for index, data in enumerate(data_list):

        check_existing_folder(output_folder+'/eval/')

        if path.isfile(output_folder+'/eval/' + names[str(index)] +'_eval'+epoch+'.npy'):
            print(output_folder+'/eval/' + names[str(index)]+'_eval'+epoch+'.npy','already evaluated. skipped.')

            continue
        
        print('evaluating:', names[str(index)]+epoch)
        
        data_set = KerrDataset(x=data, y=labels_list[index])
        eval_loader = DataLoader(dataset=data_set, batch_size=2048, shuffle=False)

        outputs = _loop_over_data(model, data_set, eval_loader)

        np.save(output_folder+'/eval/' + names[str(index)] +'_eval'+epoch+'.npy', outputs)


def eval_epoch_list(model_folder, data_list, labels_list, epoch_list):
    start_time = time()
    model_list = glob(model_folder+'*.pth')

    print('evaluating requested epochs...')
        
    for nn in model_list:
        # print(nn)
        kerr_eval(output_folder=model_folder, data_list=data_list, labels_list=labels_list, model=nn, epoch_list=epoch_list)
        
    print('done in', time()-start_time)

if __name__=='__main__':

    nns_folders = glob('./run/kerr_*_channels/*/')

    data2evaluate = glob('data/kerr_to_evaluate/3318HenryVIIIAnalisys/calibrated/data_1024.npy')

    for nn_folder in nns_folders:
        print(nn_folder)
        datalist = []
        label_list = []

        for f in data2evaluate:
            datalist += [np.load(f)]
            head, _ = path.split(f)
            label_eval = glob(head+'/../labels/*.tif*')[0]
            label_list += [np.load(head+'/../labels/labels.npy')]
        
        epoch_list = [0,1,2,3,4,5,6,7,8,9,10,14,24]
        eval_epoch_list(model_folder=nn_folder, data_list=datalist, labels_list=label_list, epoch_list=epoch_list)