# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:43:10 2018

@author: ASUS
"""


import keras.backend as K


#%%

def iou_coef(y_true, y_pred,smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = (K.sum(y_true_f) + K.sum(y_pred)) - intersection
    return (intersection + smooth) / ( union + smooth)

def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)

#%%
    
def dice_coef(y_true, y_pred,smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
