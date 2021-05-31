#!/usr/bin/env python
# coding: utf-8

# In[1]:



import tensorflow as tf
import os

# data_folder = '/home/dados229/luciana/BayesianProjects/BayesianChallenge2/deepbayesianstronglensing-master/data/DataChallenge2'
os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

import numpy as np
from time import time
import matplotlib
# from utils._time import ElapsedTime
import keras.backend as K
import warnings
warnings.filterwarnings("ignore")
from astropy.io import fits
from keras.utils import Progbar
import matplotlib.pyplot as plt
from keras.utils import to_categorical
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#data_dir = '/home/dados229/cenpes/DataChallenge2' #### diretorio original do Patrick
data_dir = '/home/dados2T/DataChallenge2'
#data_dirALT = '/home/dados229/cenpes/DataChallenge2ALTERADO' #### diretório para teste do pre processamento do Patrick
data_dirALT = '/home/dados2T/DataChallenge2'
catalog_name = 'image_catalog2.0train_corrigido.csv'

""" Load catalog before images """
import pandas as pd 
catalog = pd.read_csv(os.path.join(data_dir, catalog_name), header = 0) # 28 for old catalog

""" Now load images using catalog's IDs """
from skimage.transform import resize
# channels = ['H','J','Y']
channels = ['VIS']
nsamples = len(catalog['ID'])
idxs2keep = []

""" Criterions  """
is_lens = (catalog['n_source_im'] > 0) & (catalog['mag_eff'] > 1.6) & (catalog['n_pix_source'] > 20)
print(is_lens[:3])# 700
is_lens = 1.0*is_lens


reload = False #### se quiser criar novamento os images_hjy.npy e/ou images_vis.npy
PP_VIS = True
PP_HJY = True
Plot_Stamps_HJY = False
Plot_Stamps_VIS = False


if len(channels) > 1:
    CL = ''.join(channels)
    CL = CL.lower()
else:
    CL = channels[0].lower()

################################################################################
#########  LOAD STAMP
if reload:
    """ Try to load numpy file with images """
    if os.path.isfile(os.path.join(data_dirALT,'images_' + CL +'.npy')) and not reload:
        #images = np.load(os.path.join(data_dirALT,'images_' + CL +'.npy'))
        # print(images.shape)
        #idxs2keep = list(np.load(os.path.join(data_dirALT,'idxs2keep.npy')))
        print("Memory low")
    else:
        images = None
        """ Loop thru indexes """
        pbar = Progbar(nsamples-1)
        for iid,cid in enumerate(catalog['ID']): #enumerate(catalog['ID']):

            """ Loop thru channels"""
            for ich,ch in enumerate(channels):

                """ Init image dir and name """
                image_file = os.path.join(data_dir,
                                          'Train',
                                          'Public',
                                          'EUC_' + ch,
                                          'imageEUC_{}-{}.fits'.format(ch,cid))

                if os.path.isfile(image_file):

                    """ Import data with astropy """
                    image_data = fits.getdata(image_file, ext=0)
                    #image_data = resize(image_data, (100,100))

                    """ Initialize images array in case we haven't done it yet """
                    if images is None:
                        images = np.zeros((nsamples,*image_data.shape,len(channels)))

                    """ Set data in array """
                    images[iid,:,:,ich] = image_data
                    if iid not in idxs2keep:
                        idxs2keep.append(iid)
                else:
                    print('\tSkipping index: {} (ID: {})'.format(iid,cid))
                    break


            if iid%100 == 0 and iid != 0:
                pbar.update(iid)

        """ Now save to numpy file """
        np.save(os.path.join(data_dirALT,'images_' + CL +'.npy'), images)
        np.save(os.path.join(data_dirALT,'idxs2keep.npy'), np.array(idxs2keep))
        print("saved: " + os.path.join(data_dirALT,'images_' + CL +'.npy'))
        del images
        del idxs2keep




################################################################################
#########  PRE PROCESS AND NORMALIZE ------ VIS (1 channel)
if PP_VIS:
    #data_dir = '/home/dados229/cenpes/DataChallenge2'
    catalog_name = 'image_catalog2.0train_corrigido.csv'
    # print("Load images VIS and HJY")
    images_vis = np.load(os.path.join(data_dirALT,'images_vis.npy'))
    # images_hjy = np.load(os.path.join(data_dirALT,'images_hjy.npy'))
    idxs2keep = list(np.load(os.path.join(data_dirALT,'idxs2keep.npy')))

    import pandas as pd
    catalog = pd.read_csv(os.path.join(data_dir, catalog_name), header = 0) # 28 for old catalog

    catalog = catalog.loc[idxs2keep]
    images_vis = images_vis[idxs2keep]
    # images_hjy = images_hjy[idxs2keep]

    # print(images_vis.shape, images_hjy.shape)

    vis_p_max = np.percentile(images_vis, 98)
    # vis_p_max

    vis_p_min = -(np.percentile(-images_vis, 99.9))
    # vis_p_min

    #vis_min = images_vis.min()
    #vis_max = images_vis.max()

    #print(f"Vis_Min: {vis_min} | Vis_Max: {vis_max}")

    plt.hist(images_vis.ravel(), bins=np.linspace(vis_p_min, vis_p_max, 100))
    plt.xlim(vis_p_min, vis_p_max)

    # plt.show()

    vis_min = images_vis.min()
    vis_max = images_vis.max()

    print(f"Vis_Min: {vis_min} | Vis_Max: {vis_max}")

    plt.hist(images_vis.ravel(), bins=np.linspace(vis_min, vis_max + 0.25, 100))
    # plt.show()

    #VIS
    #
    images_vis = np.clip(images_vis, vis_p_min, vis_p_max)
    images_vis = (images_vis - vis_p_min)/(vis_p_max - vis_p_min)
    print(images_vis.min(), images_vis.max())

    np.save(os.path.join(data_dirALT, 'images_vis_new.npy'), images_vis)
    print(' saved.')

    del images_vis


################################################################################
#########  PRE PROCESS AND NORMALIZE ------ HJY - (3 channels)
if PP_HJY:
    #data_dir = '/home/dados229/cenpes/DataChallenge2'
    catalog_name = 'image_catalog2.0train_corrigido.csv'
    # print("Load images VIS and HJY")
    # images_vis = np.load(os.path.join(data_dirALT, 'images_vis.npy'))
    images_hjy = np.load(os.path.join(data_dirALT, 'images_hjy.npy'))
    idxs2keep = list(np.load(os.path.join(data_dirALT, 'idxs2keep.npy')))

    import pandas as pd

    catalog = pd.read_csv(os.path.join(data_dir, catalog_name), header=0)  # 28 for old catalog

    # ### Delete invalid indexes

    catalog = catalog.loc[idxs2keep]
    images_hjy = images_hjy[idxs2keep]

    ########## max and min channel h
    h_max = images_hjy[:, :, :, 0].max()
    print(h_max)
    h_p_max = np.percentile(images_hjy[:, :, :, 0], 98)
    # h_p_max

    h_min = images_hjy[:, :, :, 0].min()
    print(h_min)
    h_p_min = -(np.percentile(-images_hjy[:, :, :, 0], 99.9))
    # h_p_min


    plt.hist(images_hjy[:, :, :, 0].ravel(), bins=np.linspace(h_p_min, h_p_max, 100))
    plt.xlim(h_p_min, h_p_max)
    # plt.show()

    ########## max and min channel j
    j_max = images_hjy[:, :, :, 1].max()
    print(j_max)
    j_p_max = np.percentile(images_hjy[:, :, :, 1], 98)
    # j_p_max

    j_min = images_hjy[:, :, :, 1].min()
    print(j_min)
    j_p_min = -(np.percentile(-images_hjy[:, :, :, 1], 99.9))
    # j_p_min


    plt.hist(images_hjy[:, :, :, 1].ravel(), bins=np.linspace(j_p_min, j_p_max, 100))
    plt.xlim(j_p_min, j_p_max)
    # plt.show()

    ########## max and min channel y
    y_max = images_hjy[:, :, :, 2].max()
    print(y_max)
    y_p_max = np.percentile(images_hjy[:, :, :, 2], 98)
    # y_p_max

    y_min = images_hjy[:, :, :, 2].min()
    print(y_min)
    y_p_min = -(np.percentile(-images_hjy[:, :, :, 2], 99.9))
    # y_p_min

    plt.hist(images_hjy[:, :, :, 2].ravel(), bins=np.linspace(y_p_min, y_p_max, 100))
    plt.xlim(y_p_min, y_p_max)
    # plt.show()

    ####### clip and normalize channel h == channel 0
    images_hjy[:, :, :, 0] = np.clip(images_hjy[:, :, :, 0], h_p_min, h_p_max)
    images_hjy[:, :, :, 0] = (images_hjy[:, :, :, 0] - h_p_min) / (h_p_max - h_p_min)
    print(images_hjy[:, :, :, 0].min(), images_hjy[:, :, :, 0].max())

    ####### clip and normalize channel j == channel 1
    images_hjy[:, :, :, 1] = np.clip(images_hjy[:, :, :, 1], j_p_min, j_p_max)
    images_hjy[:, :, :, 1] = (images_hjy[:, :, :, 1] - j_p_min) / (j_p_max - j_p_min)
    print(images_hjy[:, :, :, 1].min(), images_hjy[:, :, :, 1].max())

    ####### clip and normalize channel y == channel 2
    images_hjy[:, :, :, 2] = np.clip(images_hjy[:, :, :, 2], y_p_min, y_p_max)
    images_hjy[:, :, :, 2] = (images_hjy[:, :, :, 2] - y_p_min) / (y_p_max - y_p_min)
    print(images_hjy[:, :, :, 2].min(), images_hjy[:, :, :, 2].max())



    np.save(os.path.join(data_dirALT, 'images_hjy_new.npy'), images_hjy)
    print(' saved.')

if Plot_Stamps_HJY:
    #data_dir = '/home/dados229/cenpes/DataChallenge2'

    images_hjy_normalized = np.load(os.path.join(data_dirALT, 'images_hjy_normalized.npy'))
    print(images_hjy_normalized.shape)




    print("Plot Mosaic stamps")
    plt.figure(figsize=(32, 32))
    RD = np.random.randint(images_hjy_normalized.shape[0], size=9)
    for j, i in enumerate(RD):
        img = images_hjy_normalized[i, :, :, :]
        plt.subplot(3, 3, j + 1)
        plt.imshow(img, cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('9 components extracted', fontsize=16)
    plt.tight_layout()
    # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.savefig(os.path.join(data_dirALT,"images_hjy_normalized.png"))

    del images_hjy_normalized

if Plot_Stamps_VIS:
    #data_dir = '/home/dados229/cenpes/DataChallenge2'

    images_vis_normalized = np.load(os.path.join(data_dirALT, 'images_vis_normalized.npy'))
    print(images_vis_normalized.shape)
    print("Plot Mosaic stamps")
    plt.figure(figsize=(32, 32))
    RD = np.random.randint(images_vis_normalized.shape[0], size=9)
    for j, i in enumerate(RD):

        plt.subplot(3, 3, j + 1)
        plt.imshow(images_vis_normalized[i, :, :, 0], cmap="gray")
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('9 components extracted', fontsize=16)
    # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dirALT, "images_vis_normalized.png"))

    del images_vis_normalized


###### make Y (is_lens categorical)
idxs2keep = list(np.load(os.path.join(data_dirALT, 'idxs2keep.npy')))
print(len(idxs2keep))
is_lens = is_lens[idxs2keep]
is_lens = to_categorical(is_lens, 2)
np.save(os.path.join(data_dirALT, 'Y.npy'), is_lens)
print(is_lens.shape)




