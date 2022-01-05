from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt 
from scipy.optimize import curve_fit
from time import time
import os


def fce_linear(x,a,b):
    return a * x + b


def fce_quad(x,a,b,c):
    return a * x + b * x**2 + c


def fce_trip(x,a,b,c,d):
    return a * x + b * x**2 + c * x**3 +d


# calibration
def calibrate(n_bins, calibration_filepath):
    calibration_points= np.loadtxt(calibration_filepath, unpack=True) # points manually selected from pymca
    channels, energies = calibration_points
    
    g_x_axis_ = np.arange(start=0, stop=2048, dtype=float)

    opt_quad_g, _= curve_fit(fce_quad, channels, energies)
    g_x_axis = fce_quad(g_x_axis_, opt_quad_g[0], opt_quad_g[1], opt_quad_g[2])

    energy = np.linspace(start=0.0, stop=30.0, num=n_bins)

    return energy, g_x_axis


def _integrate(energy, interpolation_f):
    bins = []
    integ = []

    for index in range(1, len(energy)):
    # take points between two values (in a bin)
        a = energy[index-1]
        b = energy[index]
        bins += [(b+a)/2]

        y_a = interpolation_f(a)
        y_b = interpolation_f(b)

        integ += [(y_a + y_b)/2]
        if index==len(energy)-1:
            integ += [y_a]
    integ = np.array(integ).T
    bins += [bins[-1]+bins[0]]
    return integ, bins

def calibrate_data(data_path, plot_results=False):    
    # load data
    if data_path[-1] !='/':
        print('data_path must end with a \'/\'')
        return

    data = np.load(data_path + 'data.npy').reshape((-1, 2048))
    print('data loaded shape:', data.shape)
    calibration_file = data_path + 'calibration.ini'
    
    if not os.path.exists(data_path + 'calibrated/'):
        os.makedirs(data_path + 'calibrated/')
        print('Folder not found, created.')
    
    start = time()
    n_bins = 256
    energy256, g_x_axis256 = calibrate(n_bins=n_bins, calibration_filepath=calibration_file)
    print('Done.\nIntegrating with', n_bins, 'bins...')
    interpolation_f = interp1d(g_x_axis256, data, fill_value='extrapolate')
    integ256, bins256 = _integrate(energy256, interpolation_f)
    print('done in:',time()-start)
    np.save(data_path + 'calibrated/data_'+str(n_bins)+'.npy', integ256)
    np.save(data_path + 'calibrated/x_axis'+str(n_bins)+'_bins.npy', bins256)

    start = time()
    n_bins = 512
    energy512, g_x_axis512 = calibrate(n_bins=n_bins, calibration_filepath=calibration_file)
    print('Done.\nIntegrating with', n_bins, 'bins...')
    interpolation_f = interp1d(g_x_axis512, data, fill_value='extrapolate')
    integ512, bins512 = _integrate(energy512, interpolation_f)
    print('done in:',time()-start)
    np.save(data_path + 'calibrated/data_'+str(n_bins)+'.npy', integ512)
    np.save(data_path + 'calibrated/x_axis'+str(n_bins)+'_bins.npy', bins512)

    start = time() 
    n_bins = 1024
    energy1024, g_x_axis1024 = calibrate(n_bins=n_bins, calibration_filepath=calibration_file)
    print('Done.\nIntegrating with', n_bins, 'bins...')
    interpolation_f = interp1d(g_x_axis1024, data, fill_value='extrapolate')
    integ1024, bins1024 = _integrate(energy1024, interpolation_f)
    print('done in:',time()-start)
    np.save(data_path + 'calibrated/data_'+str(n_bins)+'.npy', integ1024)
    np.save(data_path + 'calibrated/x_axis'+str(n_bins)+'_bins.npy', bins1024)
    
    # n_bins = 2048
    # energy2048, g_x_axis2048 = calibrate(n_bins=n_bins, calibration_filepath=calibration_file)
    # print('Done.\nIntegrating with', n_bins, 'bins...')
    # interpolation_f = interp1d(g_x_axis2048, data, fill_value='extrapolate')
    # integ2048, bins2048 = _integrate(energy2048, interpolation_f)
    # print('done in:',time()-start)
    # np.save(data_path + 'calibrated/data_'+str(n_bins)+'.npy', integ2048)
    # np.save(data_path + 'calibrated/x_axis'+str(n_bins)+'_bins.npy', bins2048)

    if plot_results:
        g_x_axis = np.linspace(0.0, 30.0, 2048)
        for i,_ in enumerate(integ1024):
            plt.title('row:'+str(i))
            plt.plot(g_x_axis, data[i], '-o', ms=2, label='real')
            
            plt.plot(bins1024, integ1024[i], '-o', ms=2, label='1024')
            
            plt.plot(bins512, integ512[i], '-o', ms=2, label='512')
            
            plt.plot(bins256, integ256[i], '-o', ms=2, label='256')

            plt.legend()
            plt.show()

    return integ1024, bins1024, integ512, bins512, integ256, bins256

if __name__=='__main__':
    path_to_data = './data/ElGreco/'
    
    calibrate_data(path_to_data, plot_results=True)