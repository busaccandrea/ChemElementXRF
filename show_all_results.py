from os import path
from PIL import Image
from matplotlib import image
import matplotlib.gridspec as gridspec
import numpy as np
import os.path as pth
from glob import glob
from matplotlib import pyplot as plt
import os



def plot_loss(path_to_losses):
    x_train = np.load(path_to_losses+'loss_x_train.npy')
    y_train = np.load(path_to_losses+'loss_y_train.npy')
    x_test = np.load(path_to_losses+'loss_x_test.npy')
    y_test = np.load(path_to_losses+'loss_y_test.npy')
    plt.plot(x_train, y_train, label='train')
    plt.plot(x_test, y_test, label='test')
    plt.legend()
    # plt.show()

if __name__=='__main__':
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
        
    # for f in glob('./run/01-12-2021_giulia_mockup_synth/0.001//eval/eval_ep99.npy'):
    labels = np.load('./data/testset/ElGreco/labels/labels.npy')
    giulia_labels = np.load('./data/DOggionoGiulia/labels/labels.npy')
    mockup_labels = np.load('./data/mockup/labels/labels.npy')

    exp_folders = glob('./run/22-12-21/prepr_none*/')
    for exp_folder in exp_folders:
        print('\n',exp_folder)
        # plot_loss(exp_folder)
        
        f = np.load(exp_folder + '/eval/eval_ep099.npy')
        giulia_f = np.load(exp_folder + '/eval/giulia_eval_ep099.npy')
        mockup_f = np.load(exp_folder + '/eval/mockup_eval_ep099.npy')
        
        """ # # elgreco images
        fig,ax = plt.subplots(f.shape[2]+1, 6, figsize=(200,200))
        plt.subplots_adjust(wspace=0.005, hspace=0.001, left=0.0001, right=1)
        ax[0,0].set_axis_off()
        ax[0,1].set_axis_off()
        ax[0,0].text(x=0.5 , y=0.5, s='model', fontsize=100)
        ax[0,1].text(x=0.5, y=0.5, s='GT', fontsize=100)
        for i in range(0, f.shape[2]):
            a = f[:,:,i]

            ax[i+1,0].set_axis_off()
            ax[i+1,1].set_axis_off()
            # ax[i,2].set_axis_off()
            # ax[i,0].text(x=0.9 , y=0.5, s=classes[str(i)], horizontalalignment='right')
            ax[i+1,0].imshow(a)
            ax[i+1,1].imshow(labels[:,i].reshape((407,724)))
        plt.savefig(exp_folder+'/results.pdf', format='pdf')

        # # elgreco images
        fig,ax = plt.subplots(f.shape[2]+1, 2, figsize=(200,200))
        plt.subplots_adjust(wspace=0.005, hspace=0.001, left=0.0001, right=1)
        ax[0,0].set_axis_off()
        ax[0,1].set_axis_off()
        ax[0,0].text(x=0.5 , y=0.5, s='model', fontsize=100)
        ax[0,1].text(x=0.5, y=0.5, s='GT', fontsize=100)
        for i in range(0, f.shape[2]):
            a = f[:,:,i]

            ax[i+1,0].set_axis_off()
            ax[i+1,1].set_axis_off()
            # ax[i,2].set_axis_off()
            # ax[i,0].text(x=0.9 , y=0.5, s=classes[str(i)], horizontalalignment='right')
            ax[i+1,0].imshow(a)
            ax[i+1,1].imshow(labels[:,i].reshape((407,724)))
        plt.savefig(exp_folder+'/results.pdf', format='pdf')
        
        # # giulia images
        fig,ax = plt.subplots(f.shape[2]+1, 2, figsize=(200,200))
        plt.subplots_adjust(wspace=0.005, hspace=0.001, left=0.0001, right=1)
        ax[0,0].set_axis_off()
        ax[0,1].set_axis_off()
        ax[0,0].text(x=0.5 , y=0.5, s='model', fontsize=100)
        ax[0,1].text(x=0.5, y=0.5, s='GT', fontsize=100)
        for i in range(0, giulia_f.shape[2]):
            a = giulia_f[:,:,i]
            ax[i+1,0].set_axis_off()
            ax[i+1,1].set_axis_off()
            # ax[i,2].set_axis_off()
            # ax[i,0].text(x=0.9 , y=0.5, s=classes[str(i)], horizontalalignment='right')
            ax[i+1,0].imshow(a)
            ax[i+1,1].imshow(giulia_labels[:,i].reshape((418,418)))
        plt.savefig(exp_folder+'/giulia_results.pdf', format='pdf')

        # # mockup images
        fig,ax = plt.subplots(f.shape[2]+1, 2, figsize=(200,200))
        plt.subplots_adjust(wspace=0.005, hspace=0.001, left=0.0001, right=1)
        ax[0,0].set_axis_off()
        ax[0,1].set_axis_off()
        ax[0,0].text(x=0.5 , y=0.5, s='model', fontsize=100)
        ax[0,1].text(x=0.5, y=0.5, s='GT', fontsize=100)
        for i in range(0, mockup_f.shape[2]):
            a = mockup_f[:,:,i]
            ax[i+1,0].set_axis_off()
            ax[i+1,1].set_axis_off()
            # ax[i,2].set_axis_off()
            # ax[i,0].text(x=0.9 , y=0.5, s=classes[str(i)], horizontalalignment='right')
            ax[i+1,0].imshow(a)
            ax[i+1,1].imshow(mockup_labels[:,i].reshape((130,185)))
        plt.savefig(exp_folder+'/mockup_results.pdf', format='pdf')
 """

        # # histograms
        for i in range(0, f.shape[2]):
            # print(f[:,:,i].mean(), f[:,:,i].std(), f[:,:,i].min(), f[:,:,i].max())
            fig,ax = plt.subplots(3,2)

            a = f[:,:,i]
            plt.title('element:'+classes[str(i)])
            ax[0,0].imshow(a)
            ax[0,1].imshow(labels[:,i].reshape((407,724)))

            giulia_a = giulia_f[:,:,i]
            ax[1,0].imshow(giulia_a)
            ax[1,1].imshow(giulia_labels[:,i].reshape((418,418)))

            mockup_a = mockup_f[:,:,i]
            ax[2,0].imshow(mockup_a)
            ax[2,1].imshow(mockup_labels[:,i].reshape((130,185)))


            fig,ax = plt.subplots(2,3)
            fig.suptitle(classes[str(i)])
            max_range = 200
            ran = (0, max_range)
            n_bins = max_range
            hist_gt, bin_gt = np.histogram(labels[:,i], range=ran, bins=n_bins)
            hist_model, bin_model = np.histogram(a, range=ran, bins=n_bins)
            ax[0,0].plot(bin_gt[:-1], hist_gt, label='experiment')
            ax[0,0].plot(bin_model[:-1], hist_model, label='model')
            ax[0,0].legend(frameon=False)
            ax[0,0].set_title('elgreco')
            ax[0,0].set_ylim(0,np.max(hist_model))
            ax[0,0].set_xlabel('counts')
            ax[0,0].set_ylabel('H(counts)')
            

            hist_gt, bin_gt = np.histogram(giulia_labels[:,i], range=ran, bins=n_bins)
            hist_model, bin_model = np.histogram(giulia_a, range=ran, bins=n_bins)
            ax[0,1].plot(bin_gt[:-1], hist_gt, label='experiment')
            ax[0,1].plot(bin_model[:-1], hist_model, label='model')
            ax[0,1].legend(frameon=False)
            ax[0,1].set_ylim(0,np.max(hist_model))
            ax[0,1].set_title('giulia')
            ax[0,1].set_xlabel('counts')
            ax[0,1].set_ylabel('H(counts)')

            
            hist_gt, bin_gt = np.histogram(mockup_labels[:,i], range=ran, bins=n_bins)
            hist_model, bin_model = np.histogram(mockup_a, range=ran, bins=n_bins)
            ax[0,2].plot(bin_gt[:-1], hist_gt, label='experiment')
            ax[0,2].plot(bin_model[:-1], hist_model, label='model')
            ax[0,2].legend(frameon=False)
            ax[0,2].set_title('mockup')
            ax[0,2].set_ylim(0,np.max(hist_model))
            ax[0,2].set_xlabel('counts')
            ax[0,2].set_ylabel('H(counts)')

            hist_gt, bin_gt = np.histogram(labels[:,i], range=ran, bins=n_bins)
            hist_model, bin_model = np.histogram(a, range=ran, bins=n_bins)
            ax[1,0].semilogy(bin_gt[:-1], hist_gt, label='experiment')
            ax[1,0].semilogy(bin_model[:-1], hist_model, label='model')
            ax[1,0].legend(frameon=False)
            ax[1,0].set_title('elgreco')
            ax[1,0].set_ylim(0,np.max(hist_model))
            ax[1,0].set_xlabel('counts')
            ax[1,0].set_ylabel('H(counts)')

            hist_gt, bin_gt = np.histogram(giulia_labels[:,i], range=ran, bins=n_bins)
            hist_model, bin_model = np.histogram(giulia_a, range=ran, bins=n_bins)
            ax[1,1].semilogy(bin_gt[:-1], hist_gt, label='experiment')
            ax[1,1].semilogy(bin_model[:-1], hist_model, label='model')
            ax[1,1].legend(frameon=False)
            ax[1,1].set_ylim(0,np.max(hist_model))
            ax[1,1].set_title('giulia')
            ax[1,1].set_xlabel('counts')
            ax[1,1].set_ylabel('H(counts)')
            
            hist_gt, bin_gt = np.histogram(mockup_labels[:,i], range=ran, bins=n_bins)
            hist_model, bin_model = np.histogram(mockup_a, range=ran, bins=n_bins)
            ax[1,2].semilogy(bin_gt[:-1], hist_gt, label='experiment')
            ax[1,2].semilogy(bin_model[:-1], hist_model, label='model')
            ax[1,2].legend(frameon=False)
            ax[1,2].set_title('mockup')
            ax[1,2].set_ylim(0,np.max(hist_model))
            ax[1,2].set_xlabel('counts')
            ax[1,2].set_ylabel('H(counts)')
       
        # # # giulia
        # for i in range(0, giulia_f.shape[2]):
        #     print(giulia_f[:,:,i].mean(), giulia_f[:,:,i].std(), giulia_f[:,:,i].min(), giulia_f[:,:,i].max())
        #     fig,ax = plt.subplots(2,2)

        #     plt.title('element:'+classes[str(i)])
        #     a = giulia_f[:,:,i]

        #     ax[0,0].imshow(a)
        #     ax[0,1].imshow(giulia_labels[:,i].reshape((407,724)))

        #     hist_gt, bin_gt = np.histogram(giulia_labels[:,i], range=(0,10000), bins=5000)
        #     hist_model, bin_model = np.histogram(a, range=(0,10000), bins=5000)
        #     ax[1,0].plot(bin_gt[:-1], hist_gt, label='experiment')
        #     ax[1,0].plot(bin_model[:-1], hist_model, label='model')
        #     ax[1,1].plot(bin_gt[:-1], hist_gt, label='experiment')
        #     ax[1,1].plot(bin_model[:-1], hist_model, label='model')
        #     ax[1,1].set_xlim(0,300)

        #     plt.legend(frameon=False)


        """ plt.figure()
        plt.title('train loss')
        plt.plot(np.load(exp_folder + '/loss_x_train.npy'), np.load(exp_folder + '/loss_y_train.npy'))
        plt.xlabel('epochs')
        plt.ylabel('loss value')
        # plt.figure()
        # plt.title('giulia loss')
        # plt.plot(np.load(exp_folder + '/loss_x_giulia.npy'), np.load(exp_folder + '/loss_y_giulia.npy'))
        # plt.figure()
        # plt.title('mockup loss')
        # plt.plot(np.load(exp_folder + '/loss_x_mockup.npy'), np.load(exp_folder + '/loss_y_mockup.npy'))
        plt.figure()
        plt.title('test loss')
        plt.plot(np.load(exp_folder + '/loss_x_test.npy'), np.load(exp_folder + '/loss_y_test.npy'))
        plt.xlabel('epochs')
        plt.ylabel('loss value')
        # plt.savefig(exp_folder+'/loss.pdf', format='pdf')
        plt.close() """

        plt.show ()