from glob import glob
import os
import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import CNN
from utilities import check_existing_folder
from torch.nn.modules.loss import MSELoss
from time import time
import numpy as np
from datetime import datetime
import gc
import nvidia_smi

# def train(model, train_set, test_set, batch_size, sampler, lr=0.003, epochs=5, momentum=0.99, loss, model_filename='regressor.p', output_path='data/'):
def main(model, lr, train_set, test_set, validation_set,\
     output_folder, n_channels_start=-1, n_outputs=10, n_epochs=25, last_epoch=0):
        """ model_filename: the name of the model must not include extension of model file. """
        # clear the gpu memory to avoid out-of-memory problems
        torch.cuda.empty_cache()
        
        # the available device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # the model
        if model is None:
            model = CNN.N_ElementCNN_1(n_channels_start=n_channels_start, n_outputs=n_outputs)
            train_loss_curve_y = []
            train_loss_curve_x = []

            test_loss_curve_y = []
            test_loss_curve_x = []
            
            validation_loss_curve_y = []
            validation_loss_curve_x = []

            element_loss_curve_x = []
            element_loss_curve_y = []
            is_first = True
        else:
            train_loss_curve_y = list(np.load(output_folder+'loss_y_train.npy'))
            train_loss_curve_x = list(np.load(output_folder+'loss_x_train.npy'))

            test_loss_curve_y = list(np.load(output_folder+'loss_y_test.npy'))
            test_loss_curve_x = list(np.load(output_folder+'loss_x_test.npy'))
                
            validation_loss_curve_y = list(np.load(output_folder+'loss_y_validation.npy'))
            validation_loss_curve_x = list(np.load(output_folder+'loss_x_validation.npy'))

            element_loss_curve_x = list(np.load(output_folder+'loss_elements_x.npy'))
            element_loss_curve_y = torch.tensor(np.load(output_folder+'loss_elements_y.npy')).to(device)
            is_first = False


        model.to(device)
        model.double()

        # optimizer
        optim = Adam(model.parameters(), lr, weight_decay=0.2)

        # loss 
        criterion = MSELoss()
        criterion_elements = MSELoss(reduction='none')

        # batch size
        train_batch_size = 512
        test_batch_size = 64
        validation_batch_size = 64
        
        # obtain loaders
        train_dataloader = DataLoader(train_set, batch_size=train_batch_size, drop_last=True, pin_memory=True)
        test_dataloader = DataLoader(test_set, batch_size=test_batch_size, drop_last=True, pin_memory=True)
        validation_dataloader = DataLoader(validation_set, batch_size=validation_batch_size, drop_last=True, pin_memory=True)

        loader = {
            'train' : train_dataloader,
            'test' : test_dataloader,
            'validation' : validation_dataloader
            }
        
        loss_curve = {
            'train':{'x': train_loss_curve_x, 'y': train_loss_curve_y},
            'test':{'x': test_loss_curve_x, 'y': test_loss_curve_y},
            'validation':{'x': validation_loss_curve_x, 'y': validation_loss_curve_y}
        }

        debug_mode = False

        start_time = time()
        with open(output_folder+'log.log', 'w') as logfile:
            epoch_times = []
            logfile.write('n_epochs: ' +str(n_epochs))

            for e in range(last_epoch,n_epochs):
                epoch_time_start = time()
                logfile.write(str(datetime.now()) + ' epoch: ' + str(e)+'\n')
                
                for mode in ['train', 'test', 'validation']:
                    
                    model.train() if mode == 'train' else model.eval()
                    with torch.set_grad_enabled(mode=='train'):
                        for i, batch_ in enumerate(loader[mode]):
                            if debug_mode: print('batch estratto.')
                            x = batch_['row'].to(device)
                            if debug_mode: print('x to device.')

                            y = batch_['counts'].to(device)
                            if debug_mode: print('y to device.')

                            y = torch.squeeze(y)
                            x = torch.squeeze(x)
                            x = torch.unsqueeze(x, 1)
                            if debug_mode: print('x and y squeezed.')

                            output = model(x)
                            if debug_mode: print('output computed')

                            l = torch.sqrt(criterion(output, y.type_as(output)))
                            if debug_mode: print('loss computed')


                            if mode == 'train':
                                l.backward()
                                if debug_mode: print('l.backward')

                                optim.step()
                                if debug_mode: print('optim.step()')
                                
                                optim.zero_grad()
                                if debug_mode: print('optim.zero_grad()')
                                
                                l_all = torch.sqrt(torch.mean(criterion_elements(output, y.type_as(output)), dim=0)).unsqueeze(0)
                                if debug_mode: print('l_all computed')

                                element_loss_curve_x += [e+(i+1)/(len(loader[mode])-1)]
                                if debug_mode: print('element_loss computed')

                                if is_first: 
                                    element_loss_curve_y = l_all
                                    is_first=False
                                else: 
                                    element_loss_curve_y = torch.cat((element_loss_curve_y, l_all), dim=0)
                                    if debug_mode: print('torch cat elem loss curve y')
                            
                            print('mode:', mode,'epoch:' , e, format(((i+1)/len(loader[mode])*100), '.3f'),'%', ' loss value:', format(l.item(), '.3f'), end='\r')
                            logfile.write(str(datetime.now()) + ' ' + mode + ' batch: ' + str(i)+ '/'+ str(len(loader[mode]))+ ' epoch:'+ str(e) + ' ' + format(((i+1)/len(loader[mode])*100), '.3f')+'%'+ ' loss value: '+ format(l.item(), '.3f')+'\n')
                            if debug_mode: print('log file writed.')
                            
                            loss_curve[mode]['x'] += [e+(i+1)/(len(loader[mode])-1)]
                            if debug_mode: print('append to loss x', mode)
                            
                            loss_curve[mode]['y'] += [l.item()]
                            if debug_mode: print('append to loss y', mode)

                            l.detach()
                            del l, output
                            

                epoch_times.append(format(time() - epoch_time_start, '.3f'))
                if debug_mode: print('append to epoch times')
                np.save(output_folder+'epoch_times.npy', np.array(epoch_times))
                if debug_mode: print('saved epoch times')
                      
                np.save(output_folder+'loss_y_train.npy', np.array(loss_curve['train']['y']))
                if debug_mode: print('saved epoch loss_y_train')
                np.save(output_folder+'loss_x_train.npy', np.array(loss_curve['train']['x']))
                if debug_mode: print('saved epoch loss_x_train')

                np.save(output_folder+'loss_y_test.npy', np.array(loss_curve['test']['y']))
                if debug_mode: print('saved epoch loss_y_test')
                np.save(output_folder+'loss_x_test.npy', np.array(loss_curve['test']['x']))
                if debug_mode: print('saved epoch loss_x_test')

                np.save(output_folder+'loss_y_validation.npy', np.array(loss_curve['validation']['y']))
                if debug_mode: print('saved epoch loss_y_test')
                np.save(output_folder+'loss_x_validation.npy', np.array(loss_curve['validation']['x']))
                if debug_mode: print('saved epoch loss_x_test')

                logfile.write(str(datetime.now()) + ' epoch '+ str(e)+ ' time:'+ format(time() - epoch_time_start, '.3f')+'\n')
                if debug_mode: print('log file writed.')

                # save the model
                np.save(output_folder + 'loss_elements_y.npy',element_loss_curve_y.detach().cpu().numpy())
                np.save(output_folder+'loss_elements_x.npy', np.array(element_loss_curve_x))

                torch.save(model, output_folder + 'model' + '_' + str(e).zfill(3) + '.pth')
            
            logfile.write(str(datetime.now()) + 'executed in: ' + str(e) + ' time:' + format(time() - start_time, '.3f')+'\n')

        # return model
        del model
        del train_dataloader
        del validation_dataloader
        del test_dataloader
        gc.collect()


if __name__=='__main__':
    print('esegui experiments runner!')
    