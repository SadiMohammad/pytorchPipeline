# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:39:55 2018

@author: ASUS
"""

#%%

# import libraries

import time

#import os,glob
import numpy as np
#import cv2
#from numpy import zeros, newaxis
#from PIL import Image

#from keras import backend as K

from helpers import load_train_raw_data,load_train_mask_data
from losses import dice_coef, dice_coef_loss, semi_dice_coef, semi_dice_coef_loss
#from models_1 import FCN_Vgg16_32s, dummy_model, segnet, ultra_unet, get_unet, next_unet,seg_unet, lung_net, feel_net
from models_1 import lung_net
# from models_1 import anything_net
#from models_1 import feel_net
from preprocess import preprocess_load_train_raw_data

#from __future__ import print_function

#import os
#import cv2
#import numpy as np

#from keras.models import Model
#from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Add, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
#from keras import regularizers
#from keras import backend as K

#import matplotlib.pyplot as plt


#%%

# parameter setup

img_rows = 128
img_cols = 128

#%%
# loading data

#time
print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

dirImage_raw = r"../train/raw all/"
dirImage_mask = r"../train/mask all/"

imgs_train_raw = load_train_raw_data(dirImage_raw, img_rows, img_cols)
imgs_train_mask = load_train_mask_data(dirImage_mask, img_rows, img_cols)

#    imgs_train = preprocess(imgs_train)
#    imgs_train = np.reshape(imgs_train,(100,64,64,1))
#    imgs_mask_train = preprocess(imgs_mask_train)
#    imgs_mask_train = np.reshape(imgs_mask_train,(100,64,64,1))

imgs_train_raw = imgs_train_raw.astype('float32')
mean = np.mean(imgs_train_raw)  # mean for data centering
std = np.std(imgs_train_raw)  # std for data normalization

imgs_train_raw -= mean
imgs_train_raw /= std

imgs_train_mask = imgs_train_mask.astype('float32')
imgs_train_mask /= 255.  # scale masks to [0, 1]

print('-'*30)
print('Creating and compiling model...')
print('-'*30)

#%%

# defining model

model_name = lung_net.__name__
model = lung_net(img_rows, img_cols)

#model = FCN_Vgg16_32s(weight_decay=0.01, batch_momentum=0.9, classes=21)
model.summary()
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

moment=time.strftime("%Y-%b-%d__%H_%M_%S",time.localtime())
model_checkpoint = ModelCheckpoint(r'../code/anything_weight/weights_'+ model_name+ moment+'--{epoch:02d}-{val_loss:.4f}'+'.h5', monitor='val_loss', save_best_only=True)

#%%

# training model

model.load_weights(r'../code/lung_weight/weights_lung_net2019-Feb-12__00_26_20--50--0.7973.h5')

print('-'*30)
print('Fitting model...')
print('-'*30)
history= model.fit(imgs_train_raw, imgs_train_mask, batch_size=16, epochs=300, verbose=1, callbacks=[model_checkpoint],validation_split=0.2, shuffle=True)





