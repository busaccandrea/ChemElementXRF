from os import path
from PIL import Image
import numpy as np
import os.path as pth
from glob import glob



if __name__=='__main__':
    experiment_folders = glob('./run/nelements_mse/*/')
    for exp_f in experiment_folders:
        evaluated_list = glob(exp_f+'eval/*.npy')
        for evaluated in evaluated_list:
            outputs = np.load(evaluated)
            for i in range(0, outputs.shape[2]):
                img = Image.fromarray(outputs[:,:,i])
                filename = pth.basename(evaluated).split('.')[0].split('_')[1] + '.tiff'
                img.save(exp_f+'eval/png/'+filename, format='tiff')