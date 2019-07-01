# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:41:10 2018

@author: ASUS
"""

import numpy as np
import cv2
import glob

#%%

def load_train_raw_data(dirImage):
    #dirImage = r"f:/4-1/4-1/Thesis/OK/train/raw all/"
    #dirImage = r"../train/raw all/"
    filenames= []
    filenames += glob.glob(dirImage + "/*" + ".jpg")
    images = []
    for file in filenames:
        img = np.asarray(cv2.imread(file,0))
        img= cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
        images.append(img)
    images = np.asarray(images)
    print(np.shape(images))
    images = images.reshape(-1, 128, 128, 1)    
    print(np.shape(images))  
    return images

#%%
    
def load_train_mask_data(dirImage):
    #dirImage = r"f:/4-1/4-1/Thesis/OK/train/mask all/"
    #dirImage = r"../train/mask all/"
    filenames= []
    filenames += glob.glob(dirImage + "/*" + ".tif")
    images = []
    for file in filenames:
        img = np.asarray(cv2.imread(file,0))
        img= cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
        images.append(img)
    images = np.asarray(images)
    print(np.shape(images))
    images = images.reshape(-1, 128, 128, 1)    
    print(np.shape(images))  
    return images

#%%
    
