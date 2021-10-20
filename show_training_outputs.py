from os import path
from PIL import Image
from matplotlib import image
import numpy as np
from os import path
from glob import glob
from matplotlib import pyplot as plt, image
import os


if __name__=='__main__':
    experiment_folders = glob('./run/nelements_mse/*/')

    # 0 =
    elements_to_show = [0,1,2]
    saveimg = False
    showimg = False

    for exp_f in experiment_folders:
        evaluated_list = glob(exp_f+'eval/*.npy')

        if not elements_to_show: elements_to_show = list(range(6))

        for element in elements_to_show:
            for evaluated in evaluated_list:
                outputs = np.load(evaluated)

                folder, tail = path.split(evaluated)
                
                filename = folder + tail.split('.')[0].split('_')[1] + '_el_'+str(element)
                print(filename)

                img = outputs[:,:,element]
                
                plt.figure()
                plt.title(filename)

                if showimg: plt.imshow(img)
                if saveimg: plt.imsave(evaluated + filename+'.png', img)
            if showimg: plt.show()