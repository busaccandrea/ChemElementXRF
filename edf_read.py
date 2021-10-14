import numpy as np
from glob import glob
import h5py

def read_edf_line(name,n,shape):
    """ Reads an .edf file.
    params:
        name: the .edf filename. 
    n: number of lines composing the header which needs to be skipped.
        default value: 14
        
    shape: the shape of all spectra
        default value: =(-1,2048)
    
    returns: numpy.array """
    with open(name,'rb') as f:
        for _ in range(n):
            f.readline()

        x = np.frombuffer(f.read(),'d')
        x = x.reshape(*shape)

    return x

def read_edf(path_to_edf_files,n=14,shape=(-1,2048)):
    """ This method reads a list of edf files.
    returns: numpy.array
    params:
        files: a list containing .edf file names.

        n: number of lines composing the header which needs to be skipped.
            default value: 14
        
        shape: the shape of all spectra
            default value: =(-1,2048)
            
        usage example:
            from glob import glob
            filelist = glob('./data/mockup/Edf/*edf')
            data = read_edf(files=filelist)

    """
    edf_files = glob(path_to_edf_files + '*.edf')
    x = []
    for i, file in enumerate(edf_files):
        print('reading', i+1, 'out of', len(edf_files), end='\r')
        x += [read_edf_line(file,n,shape)]

    return np.asarray(x)
