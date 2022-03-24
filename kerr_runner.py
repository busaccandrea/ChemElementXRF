import yaml
from time import time
from KerrDataset import KerrDataset
import train_cnn_kerr
import numpy as np
from glob import glob
from CNN_evaluate import kerr_eval
from utilities import check_existing_folder
from torch import load

from sklearn.model_selection import train_test_split
import os

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

def join_ds(data_files, label_files):
    is_first = True

    for i, ds in enumerate(data_files):
        tmp = np.load(ds)
        tmpl = np.load(label_files[i])
        if is_first:
            dataset = tmp
            labels = tmpl
            is_first = False
        else:
            dataset = np.concatenate((dataset, tmp), axis=0)
            labels = np.concatenate((labels, tmpl), axis=0)
        print(100*i/len(data_files), '%', end='\r')
    return dataset, labels

def main(experiments, data, labels, run_number):  
    """ experiments: path to experiment files. """
    options = {}
    with open(experiments+'/cnn_general_config.yaml') as configfile:
        documents = yaml.full_load(configfile)
        for item, doc in documents.items():
            options[item] = doc

    for p in options['preprocessing']:
        for nd in options['norm_data']:
            for nl in options['norm_labels']:
                for lr in options['learning_rates']:
                    output_folder = './run/'+exp_name+str(run)+'/lr_' + str(lr) + '/'
                    
                    n_channels_start = options['n_channels_start']
                    n_outputs = options['n_outputs']

                    n_epochs = int(384/n_channels_start)

                    check_existing_folder(output_folder)

                    # if there is a model file already, I have to continue the training.
                    model_file = glob(output_folder+'model*')
                    if len(model_file)>0:
                        model_file=model_file[-1]

                        last_epoch = int(os.path.basename(model_file).split('.')[0].split('_')[-1])
                        if last_epoch == n_epochs-1:
                            print('Number of epochs for',n_channels_start, run, 'already reached.')
                            continue
                        
                        model = load(model_file)

                    else: 
                        model=None
                        last_epoch=-1

                    t = time()

                    x_train, x_rem, y_train, y_rem = train_test_split(data, labels, train_size=0.7)
                    x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.66)
                    split_time= time()
                    print('split time', split_time - t)
                    training_set = KerrDataset(x=x_train, y=y_train, preprocessing=p, normalize_data=nd, normalize_labels=nl)
                    test_set = KerrDataset(x=x_test, y=y_test, preprocessing=p, normalize_data=nd, normalize_labels=nl)
                    validation_set = KerrDataset(x=x_valid, y=y_valid, preprocessing=p, normalize_data=nd, normalize_labels=nl)
                    print('dataset_time', time()- split_time)

                    print('EXPERIMENT SETTINGS:', exp_name+str(run), 'n_channels:', n_channels_start, 'n_elements:', n_outputs, 'lr', lr, 'n_epochs', n_epochs)
                    last_epoch +=1
                    print('epoch starts from', last_epoch,'to', n_epochs, 'excluded')
                    # train the model
                    train_cnn_kerr.main(lr=lr, train_set=training_set, test_set=test_set,\
                        validation_set=validation_set, output_folder=output_folder, n_channels_start=n_channels_start, n_outputs=n_outputs, n_epochs=n_epochs, last_epoch=last_epoch, model=model)

if __name__=='__main__':
    exp_names = []
    
    nchan = [6]
    # nchan = [6,8,16,24,32,48]

    for n in nchan:
        exp_names.append('kerr_'+str(n).zfill(3)+'_channels_run_')

    data_files = glob('./data/*/calibrated/data_1024.npy')
    label_files = glob('./data/*/labels/labels.npy')

    print('getting dataset...')
    join_time = time()
    data, labels= join_ds(data_files=data_files, label_files=label_files)
    print('jointime:', time()-join_time)

    for run in range(6,7):
        for exp_name in exp_names:
            start_time = time()
            experiments = './run/'+exp_name+str(run)

            main(experiments=experiments, run_number=run, data=data, labels=labels)
            print('\n\nExecuted in', time()-start_time)