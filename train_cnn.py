from glob import glob
import os
import yaml
import torch
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from torchnet.meter import AverageValueMeter
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import CNN
import h5py
from ElementalDataset import ElementalDataset
from utilities import check_existing_folder
from torch.nn.modules.loss import MSELoss
from time import time
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

# def train(model, train_set, test_set, batch_size, sampler, lr=0.003, epochs=5, momentum=0.99, loss, model_filename='regressor.p', output_path='data/'):
def main(lr, train_set, test_set, train_sampler, output_folder, giulia_set=None, mockup_set=None):
        """ model_filename: the name of the model must not include extension of model file. """
        # the model
        model = CNN.N_ElementCNN(input_size=1024)
        
        last_epoch = 0
        n_epochs = 100

        check_existing_folder(output_folder)
        # if the folder already exists, I need to continue the training.
        model_file = glob(output_folder+'model*')
        if len(model_file)>0:
            model_file=model_file[-1]
            last_epoch = int(os.path.basename(model_file).split('.')[0].split('_')[-1])
            n_epochs = n_epochs - last_epoch - 1
            if n_epochs <= 0:
                print('Number of epochs already reached.')
                return
            model = torch.load(model_file)

        model.double()

        # the available device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # # plotter
        # loss_logger = VisdomPlotLogger(plot_type='line', env='ELEMENTAL', opts={'title': 'dec 07 lr: '+str(lr), 'legend':['train','test']})

        # # meters
        # loss_meter = AverageValueMeter()
        
        # # saver
        # visdom_saver = VisdomSaver(envs=['ELEMENTAL'])

        # optimizer
        optim = Adam(model.parameters(), lr, weight_decay=0.2)

        # loss 
        criterion = MSELoss()
        criterion_elements = MSELoss(reduction='none')

        # batch size
        train_batch_size = 256
        test_batch_size = 512
        
        # obtain loaders
        train_dataloader = DataLoader(train_set, batch_size=train_batch_size, drop_last=True)
        test_dataloader = DataLoader(test_set, batch_size=test_batch_size, drop_last=True)
        giulia_dataloader = DataLoader(giulia_set, batch_size=test_batch_size, drop_last=True)
        mockup_dataloader = DataLoader(mockup_set, batch_size=test_batch_size, drop_last=True)

        loader = {
            'train' : train_dataloader,
            'test' : test_dataloader,
            'giulia': giulia_dataloader,
            'mockup': mockup_dataloader
            }

        train_loss_curve_y = []
        train_loss_curve_x = []

        test_loss_curve_y = []
        test_loss_curve_x = []

        giulia_loss_curve_y = []
        giulia_loss_curve_x = []
        
        mockup_loss_curve_y = []
        mockup_loss_curve_x = []

        l_all_loss_curve_x = []
        l_all_loss_curve_y = torch.tensor([]).to(device)
        loss_curve = {
            'train':{'x': train_loss_curve_x, 'y': train_loss_curve_y},
            'test':{'x': test_loss_curve_x, 'y': test_loss_curve_y},
            'giulia':{'x': giulia_loss_curve_x, 'y': giulia_loss_curve_y},
            'mockup':{'x': mockup_loss_curve_x, 'y': mockup_loss_curve_y}
        }
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
        with open(output_folder+'log.log', 'w') as logfile:
            for e in range(n_epochs):
                epoch_time_start = time()
                logfile.write(str(datetime.now()) + ' epoch: ' + str(e)+'\n')
                for mode in ['train', 'test', 'giulia', 'mockup']:
                    # loss_meter.reset()
                    # feed the model with train/test set to train properly.
                    model.train() if mode == 'train' else model.eval()
                    with torch.set_grad_enabled(mode=='train'):
                        for i, batch_ in enumerate(loader[mode]):
                            x = batch_['row'].to(device)
                            y = batch_['counts'].to(device)
                            y = torch.squeeze(y)
                            y = torch.unsqueeze(y,1)
                            x = torch.squeeze(x)
                            x = torch.unsqueeze(x, 1)
                        
                            output = model(x)

                            # add loss for every element
                            l_all = torch.sqrt(torch.mean(criterion_elements(output, y.type_as(output)), dim=0))

                            l = torch.sqrt(criterion(output, y.type_as(output)))
                            
                            if mode == 'train':
                                l.backward()
                                optim.step()
                                optim.zero_grad()
                                
                                l_all_loss_curve_x += [e+(i+1)/(len(loader[mode])-1)]
                                l_all_loss_curve_y = torch.cat((l_all_loss_curve_y, l_all), dim=0)

                            # if mode == 'train':
                                # loss_logger.log(e+(i+1)/len(train_dataloader), l.item(), name='train')
                    # loss_logger.log(e+1, loss_meter.value()[0], name=mode)
                            
                            print('mode:', mode,'epoch:' , e, format(((i+1)/len(loader[mode])*100), '.3f'),'%', ' loss value:', format(l.item(), '.3f'), end='\r')
                            logfile.write(str(datetime.now()) + ' ' + mode + ' batch: ' + str(i)+ '/'+ str(len(loader[mode]))+ ' epoch:'+ str(e)+ format(((i+1)/len(loader[mode])*100), '.3f')+'%'+ ' loss value: '+ format(l.item(), '.3f')+'\n')
                            loss_curve[mode]['x'] += [e+(i+1)/(len(loader[mode])-1)]
                            loss_curve[mode]['y'] += [l.item()]
                            
                np.save(output_folder+'loss_y_train.npy', np.array(loss_curve['train']['y']))
                np.save(output_folder+'loss_x_train.npy', np.array(loss_curve['train']['x']))

                np.save(output_folder+'loss_y_test.npy', np.array(loss_curve['test']['y']))
                np.save(output_folder+'loss_x_test.npy', np.array(loss_curve['test']['x']))

                np.save(output_folder+'loss_y_giulia.npy', np.array(loss_curve['giulia']['y']))
                np.save(output_folder+'loss_x_giulia.npy', np.array(loss_curve['giulia']['x']))

                np.save(output_folder+'loss_y_mockup.npy', np.array(loss_curve['mockup']['y']))
                np.save(output_folder+'loss_x_mockup.npy', np.array(loss_curve['mockup']['x']))

                logfile.write(str(datetime.now()) + ' epoch '+ str(e)+ ' time:'+ format(time() - epoch_time_start, '.3f')+'\n')
                # visdom_saver.save()
                # save the model
                np.save(output_folder + 'loss_elements_y.npy',l_all_loss_curve_y.detach().cpu().numpy())
                np.save(output_folder+'loss_elements_x.npy', np.array(l_all_loss_curve_x))

                torch.save(model, output_folder + 'model' + '_' + str(last_epoch + e).zfill(3) + '.pth')
        return model


if __name__=='__main__':
    print('esegui experiments runner!')