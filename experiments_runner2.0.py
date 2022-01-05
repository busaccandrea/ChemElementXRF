import yaml
from time import time
from ElementalDataset import ElementalDataset, BalancedSampler
from ChemElementRegressorDataset import ChemElementRegressorDataset
import train_cnn
import numpy as np
from CNN_evaluate import evaluate


def main(experiments):
    """ experiments: path to experiment files. """
    test_data = np.load('./data/testset/ElGreco/calibrated/data_1024.npy')
    test_labels = np.load('./data/testset/ElGreco/labels/labels.npy')
    number_of_rows = test_data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=int(number_of_rows*0.30), replace=False)
    test_data = test_data[random_indices, :]
    test_labels = test_labels[random_indices, :]
    
    random_indices = []
    number_of_rows = test_data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=24000, replace=False)
    giulia_data = np.load('./data/DOggionoGiulia/calibrated/data_1024.npy')
    giulia_labels = np.load('./data/DOggionoGiulia/labels/labels.npy')
    giulia_data = giulia_data[random_indices, :]
    giulia_labels = giulia_labels[random_indices, :]
    
    mock_data = np.load('./data/mockup/calibrated/data_1024.npy')
    mock_labels = np.load('./data/mockup/labels/labels.npy')

    giulia_set = ChemElementRegressorDataset(data=giulia_data, labels=giulia_labels)
    mockup_set = ChemElementRegressorDataset(data=mock_data, labels=mock_labels)
    options = {}
    with open(experiments) as configfile:
        documents = yaml.full_load(configfile)
        for item, doc in documents.items():
            options[item] = doc
    
    for p in options['preprocessing']:
        for nd in options['norm_data']:
            for nl in options['norm_labels']:
                training_set = ElementalDataset(p, nd, nl)
                train_sampler = BalancedSampler(training_set, batch_size=512)
                test_set = ChemElementRegressorDataset(data=test_data, labels=test_labels)

                for lr in options['learning_rates']:
                    output_folder = './run/prepr_' + p + '-lr_' + str(lr) + '-normdata_' + str(nd) + '-normlabels_' + str(nl) + '/'
                    print('EXPERIMENT SETTINGS:', '\npreprocessing:', p, 'normdata:', nd, 'normlabels', nl, 'lr', lr)
                    
                    # train the model
                    train_cnn.main(lr=lr, train_set=training_set, train_sampler=train_sampler, test_set=test_set, output_folder=output_folder, giulia_set=giulia_set, mockup_set=mockup_set)
                    evaluate()
                    

if __name__=='__main__':
    start_time = time()
    experiments = './run/cnn_general_config.yaml'
    main(experiments=experiments)
    print('\n\nExecuted in', time()-start_time)