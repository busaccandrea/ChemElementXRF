from os import path
from PIL import Image
from matplotlib import image
import numpy as np
import os.path as pth
from glob import glob
from matplotlib import pyplot as plt, image
import os


if __name__=='__main__':
    path_to_labels = './data/DOggionoGiulia/labels/all/'



    images = glob(path_to_labels + '*.tif*')
    for i, img in enumerate(images):
        plt.figure(i)
        a = np.array(Image.open(img))
        plt.imshow(a)
        plt.title(path.basename(img))
    plt.show()