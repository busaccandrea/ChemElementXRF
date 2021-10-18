import yaml
import torch
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from torchnet.meter import AverageValueMeter
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import CNN
import h5py
from utilities import check_existing_folder


# def train(model, train_set, test_set, batch_size, sampler, lr=0.003, epochs=5, momentum=0.99, loss, model_filename='regressor.p', output_path='data/'):
def main(yaml_file, train_set, test_set):
        """ model_filename: the name of the model must not include extension of model file. """

        # load yaml
        with open(yaml_file,'r') as file:
            print('opening', yaml_file)
            parameters = yaml.load(file,Loader=yaml.FullLoader)

        # the model
        model = getattr(CNN, parameters['model_name'])()
        model.double()

        # the available device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # plotter
        loss_logger = VisdomPlotLogger(plot_type='line', env='element_regressor', opts={'title': 'lr: '+str(parameters['learning_rate']), 'legend':['train','test']})

        # meters
        loss_meter = AverageValueMeter()
        
        # saver
        visdom_saver = VisdomSaver(envs=['element_regressor'])

        # optimizer
        optim = Adam(model.parameters(), float(parameters['learning_rate']), weight_decay=0.2)

        # loss 
        criterion = getattr(nn.modules.loss, parameters['loss'])()

        # batch size
        batch_size = int(parameters['batch_size'])
        
        # obtain loaders
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        output_path = parameters['output_path']

        epochs = int(parameters['epochs'])
        model.to(device)

        for e in range(epochs):
            for mode in ['train', 'test']:
                loss_meter.reset()
                print('epoch:', e, end='\r')

                loader = {
                    'train' : train_dataloader,
                    'test' : test_dataloader
                    }

                # feed the model with train/test set to train properly.
                model.train() if mode == 'train' else model.eval()
                with torch.set_grad_enabled(mode=='train'):
                    for i, batch_ in enumerate(loader[mode]):
                        # index of the row
                        x = batch_['row'].to(device)

                        # the value to predict
                        y = batch_['counts'].to(device)
                        y = torch.squeeze(y,1)
                    
                        output = model(x)
                        
                        l = torch.sqrt(criterion(output, y.type_as(output)))
                        if mode == 'train':
                            l.backward()
                            optim.step()
                            optim.zero_grad()
                        n = batch_['row'].shape[0]
                        loss_meter.add(l.item() * n, n)

                        if mode == 'train':
                            loss_logger.log(e+(i+1)/len(train_dataloader), l.item(), name='train')
                
                loss_logger.log(e+1, loss_meter.value()[0], name=mode)

            # save the model
            check_existing_folder(output_path+ str(parameters['learning_rate']) + '/')
            torch.save(model, output_path + str(parameters['learning_rate']) + '/' + parameters['model_name'] + '_' + str(e) + '.pth')
            visdom_saver.save()
        return model


if __name__=='__main__':
    main()