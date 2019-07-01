# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:46:19 2018

@author: ASUS
"""

#%% importing

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Add, Activation, BatchNormalization, Reshape, UpSampling2D, Concatenate
from keras import regularizers
from keras import backend as K
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D, Softmax4D


#%%

'''
def FCN_Vgg16_32s(weight_decay=0.01, batch_momentum=0.9, classes=21, img_rows = 128, img_cols = 128):
    
    inputs = Input((img_rows, img_cols, 1))
    # Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=regularizers.l2(weight_decay))(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=regularizers.l2(weight_decay))(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=regularizers.l2(weight_decay))(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=regularizers.l2(weight_decay))(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=regularizers.l2(weight_decay))(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    # Block 5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=regularizers.l2(weight_decay))(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=regularizers.l2(weight_decay))(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=regularizers.l2(weight_decay))(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

    # Convolutional layers transfered from fully-connected layers
    fc1 = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=regularizers.l2(weight_decay))(pool5)
    drp1 = Dropout(0.5)(fc1)
    fc2 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=regularizers.l2(weight_decay))(drp1)
    drp2 = Dropout(0.5)(fc2)
    
    #classifying layer
    classifier_layer_32s = Conv2D(classes, kernel_size=(1,1) ,  strides=(1,1), name='Classifier_layer_32s', kernel_initializer='he_normal', activation='sigmoid', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))(drp2)
    
    #out = BilinearUpSampling2D(size=(32, 32))(class_layer)
    #############################
    up_16s_21 = Conv2DTranspose(21, (2, 2), strides=(2, 2), name='up_16s_21',padding='same')(classifier_layer_32s)
    pool4_21 = Conv2D(classes, kernel_size=(1,1) ,  strides=(1,1), name='Classifier_layer_Pool4_21', kernel_initializer='he_normal', activation='sigmoid', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))(pool4)
    out_16s = Add(name="Add_16s")([up_16s_21, pool4_21])
    #############################
    up_8s_21 = Conv2DTranspose(21, (2, 2), strides=(2, 2), name='up_8s_21',padding='same')(out_16s)
    pool3_21 = Conv2D(classes, kernel_size=(1,1) ,  strides=(1,1), name='Classifier_layer_Pool3_21', kernel_initializer='he_normal', activation='sigmoid', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))(pool3)
    out_8s = Add(name="Add_8s")([up_8s_21, pool3_21])
    ############################
    
    out_128 = Conv2DTranspose(classes, kernel_size=(8,8), strides=(8,8), name='out_128', kernel_initializer='he_normal', activation='sigmoid', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))(out_8s)
    out_act = Activation('softmax')(out_128)
    out = Conv2D(1, (1, 1), activation='sigmoid', name ='Out_layer')(out_act)
    
    model = Model(inputs=[inputs], outputs=[out])
    return model

#%%


def dummy_model(weight_decay=0.01, batch_momentum=0.9, classes=21, img_rows = 128, img_cols = 128):

    inputs = Input((img_rows, img_cols, 1))
    
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=regularizers.l2(weight_decay))(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)
    
    out_128 = Conv2DTranspose(classes, kernel_size=(8,8), strides=(8,8), name='out_128', kernel_initializer='he_normal', activation='sigmoid', padding='valid', kernel_regularizer=regularizers.l2(weight_decay))(pool1)
    out_act = Activation('softmax')(out_128)
    out = Conv2D(1, (1, 1), activation='sigmoid', name ='Out_layer')(out_act)
    
    model = Model(inputs=[inputs], outputs=[out])
    
    return model


#%%
    
def segnet(img_rows, img_cols, n_labels= 20, kernel=3, pool_size=(2, 2)):
    # encoder
    inputs = Input((img_rows, img_cols, 1))

    conv_1 = Conv2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Conv2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Conv2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Conv2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Conv2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Conv2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Conv2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Conv2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Conv2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build encoder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Conv2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Conv2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Conv2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Conv2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Conv2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Conv2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Conv2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Conv2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Conv2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Conv2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Conv2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Conv2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

#    conv_26 = Conv2D(n_labels, (1, 1), padding="valid")(conv_25)
#    conv_26 = BatchNormalization()(conv_26)
#    
    conv_27 = Conv2D(1, (1, 1))(conv_25)
    
    outputs = Activation("sigmoid")(conv_27)
    
    print("Build decoder done..")
    
    model = Model(inputs=[inputs], outputs=[outputs], name="SegNet")
    
    return model


#%%

def ultra_unet(img_rows, img_cols):
    
    inputs = Input((img_rows, img_cols, 1))
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = Conv2D(32, (3, 3), strides=(2, 2), activation ='relu', padding="same")(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = Conv2D(64, (3, 3), strides=(2, 2), activation ='relu', padding="same")(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = Conv2D(128, (3, 3), strides=(2, 2), activation ='relu', padding="same")(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = Conv2D(256, (3, 3), strides=(2, 2), activation ='relu', padding="same")(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = concatenate([Reshape((16, 16, 256))(conv5), conv4], axis = 3)
    
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = concatenate([Reshape((32, 32, 128))(conv6), conv3], axis = 3)
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    
    up8 = concatenate([Reshape((64, 64, 64))(conv7), conv2], axis = 3)
    
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    
    up9 = concatenate([Reshape((128, 128, 32))(conv8), conv1], axis = 3)
    
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)   

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

#%%
'''
    
#def get_unet(img_rows, img_cols):
#    inputs = Input((img_rows, img_cols, 1))
#    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#
#    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
#
#    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
#
#    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
#
#    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
#    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
#    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
#
#    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
#
#    model = Model(inputs=[inputs], outputs=[conv10])
#
#    return model

#%%

'''    
def next_unet(img_rows, img_cols):
    
    inputs = Input((img_rows, img_cols, 1))
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    
    pool1 = Conv2D(32, (3, 3), strides=(2, 2), activation ='relu', padding="same")(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    
    pool2 = Conv2D(64, (3, 3), strides=(2, 2), activation ='relu', padding="same")(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    
    pool3 = Conv2D(128, (3, 3), strides=(2, 2), activation ='relu', padding="same")(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    
    pool4 = Conv2D(256, (3, 3), strides=(2, 2), activation ='relu', padding="same")(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = concatenate([Conv2D(256, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation = 'bilinear')(conv5)), conv4], axis = 3)
    
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = concatenate([Conv2D(128, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation = 'bilinear')(conv6)), conv3], axis = 3)
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2D(64, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation = 'bilinear')(conv7)), conv2], axis = 3)
    
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    
    up9 = concatenate([Conv2D(32, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation = 'bilinear')(conv8)), conv1], axis = 3)
    
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)   

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


#%%
    
def seg_unet(img_rows, img_cols):
    
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    
    pool1, mask1 = MaxPoolingWithArgmax2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    
    pool2, mask2 = MaxPoolingWithArgmax2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    
    pool3, mask3 = MaxPoolingWithArgmax2D((2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    
    pool4, mask4 = MaxPoolingWithArgmax2D((2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    
    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    
    pool5, mask5 = MaxPoolingWithArgmax2D((2, 2))(conv5)

    unpool1 = MaxUnpooling2D((2,2))([pool5, mask5])
    
    conv6 = Conv2D(256, (3, 3), padding='same')(unpool1)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    
    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    unpool2 = MaxUnpooling2D((2,2))([conv6, mask4])
    
    conv7 = Conv2D(128, (3, 3), padding='same')(unpool2)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    
    conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    unpool3 = MaxUnpooling2D((2,2))([conv7, mask3])
    
    conv8 = Conv2D(64, (3, 3), padding='same')(unpool3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    
    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)

    unpool4 = MaxUnpooling2D((2,2))([conv8, mask2])
    
    conv9 = Conv2D(32, (3, 3), padding='same')(unpool4)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    
    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    
    unpool5 = MaxUnpooling2D((2,2))([conv9, mask1])
    
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same')(unpool5)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv10)
    
    conv11 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=[inputs], outputs=[conv11])

    return model

'''
#%%

def lung_net(img_rows, img_cols):
  inputs = Input((img_rows, img_cols, 1))
  
  block_1 = BatchNormalization()(inputs)
  block_1 = Add()([inputs, block_1])
  
  conv_1 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_1)
  block_2 = BatchNormalization()(conv_1)
  block_2 =  Add()([conv_1, block_2])
  block_2 = Activation('relu')(block_2)
  
  conv_2 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_2)
  block_3 = BatchNormalization()(conv_2)
  block_3 =  Add()([conv_2, block_3])
  block_3 = Activation('relu')(block_3)
  
  conv_3 = Conv2D(32, (3, 3), dilation_rate=(2, 2), kernel_initializer='he_normal', padding='same')(block_3)
  block_4 = BatchNormalization()(conv_3)
  block_4 =  Add()([conv_3, block_4])
  block_4 = Activation('relu')(block_4)
  
  conv_4 = Conv2D(32, (3, 3), dilation_rate=(3, 3), kernel_initializer='he_normal', padding='same')(block_4)
  block_5 = BatchNormalization()(conv_4)
  block_5 =  Add()([conv_4, block_5])
  block_5 = Activation('relu')(block_5)
  
  conv_5 = Conv2D(32, (3, 3), dilation_rate=(5, 5), kernel_initializer='he_normal', padding='same')(block_5)
  block_6 = BatchNormalization()(conv_5)
  block_6 =  Add()([conv_5, block_6])
  block_6 = Activation('relu')(block_6)
  
  conv_6 = Conv2D(32, (3, 3), dilation_rate=(8, 8), kernel_initializer='he_normal', padding='same')(block_6)
  block_7 = BatchNormalization()(conv_6)
  block_7 =  Add()([conv_6, block_7])
  block_7 = Activation('relu')(block_7)
  
  conv_7 = Conv2D(32, (3, 3), dilation_rate=(13, 13), kernel_initializer='he_normal', padding='same')(block_7)
  block_8 = BatchNormalization()(conv_7)
  block_8 =  Add()([conv_7, block_8])
  block_8 = Activation('relu')(block_8)
  
  conv_8 = Conv2D(32, (3, 3), dilation_rate=(21, 21), kernel_initializer='he_normal', padding='same')(block_8)
  block_9 = BatchNormalization()(conv_8)
  block_9 =  Add()([conv_8, block_9])
  block_9 = Activation('relu')(block_9)
  
  conv_9 = Conv2D(32, (3, 3), dilation_rate=(34, 34), kernel_initializer='he_normal', padding='same')(block_9)
  block_10 = BatchNormalization()(conv_9)
  block_10 =  Add()([conv_9, block_10])
  block_10 = Activation('relu')(block_10)
  
  conv_10 = Conv2D(32, (3, 3), dilation_rate=(55, 55), kernel_initializer='he_normal', padding='same')(block_10)
  block_11 = BatchNormalization()(conv_10)
  block_11 =  Add()([conv_10, block_11])
  block_11 = Activation('relu')(block_11)
  
  concat = concatenate([inputs, block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9, block_10, block_11], axis=-1)
  dropout = Dropout(0.5)(concat)
  
  conv_11 = Conv2D(128, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(dropout)
  block_12 = BatchNormalization()(conv_11)
  block_12 =  Add()([conv_11, block_12])
  block_12 = Activation('relu')(block_12)
  
  conv_12 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_12)
  block_13 = BatchNormalization()(conv_12)
  block_13 =  Add()([conv_12, block_13])
  block_13 = Activation('relu')(block_13)
  
#   conv_13 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_13)
#   block_14 = BatchNormalization()(conv_13)
#   block_14 =  Add()([conv_13, block_14])
#   block_14 = Activation('relu')(block_14)
  
  conv_13 = Conv2D(1, (1, 1), kernel_initializer='he_normal')(block_13)
  
  output = Activation('sigmoid')(conv_13)
  
  model = Model(inputs=[inputs], outputs=[output])

  return model

#    
    
    
 #%%   
    
# def feel_net(img_rows, img_cols):
    # inputs = Input((img_rows, img_cols, 1))
    
    # block_1 = BatchNormalization()(inputs)
    # block_1 = Add()([inputs, block_1])
    
    # conv_1 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_1)
    # block_2 = BatchNormalization()(conv_1)
    # block_2 =  Add()([conv_1, block_2])
    # block_2 = Activation('relu')(block_2)
    
    # conv_2 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_2)
    # block_3 = BatchNormalization()(conv_2)
    # block_3 =  Add()([conv_2, block_3])
    # block_3 = Activation('relu')(block_3)
    
    # conv_3 = Conv2D(32, (3, 3), dilation_rate=(2, 2), kernel_initializer='he_normal', padding='same')(block_3)
    # block_4 = BatchNormalization()(conv_3)
    # block_4 =  Add()([conv_3, block_4])
    # block_4 = Activation('relu')(block_4)
    
    # conv_4 = Conv2D(32, (3, 3), dilation_rate=(3, 3), kernel_initializer='he_normal', padding='same')(block_4)
    # block_5 = BatchNormalization()(conv_4)
    # block_5 =  Add()([conv_4, block_5])
    # block_5 = Activation('relu')(block_5)
    
    # conv_5 = Conv2D(32, (3, 3), dilation_rate=(5, 5), kernel_initializer='he_normal', padding='same')(block_5)
    # block_6 = BatchNormalization()(conv_5)
    # block_6 =  Add()([conv_5, block_6])
    # block_6 = Activation('relu')(block_6)
    
    # conv_6 = Conv2D(32, (3, 3), dilation_rate=(7, 7), kernel_initializer='he_normal', padding='same')(block_6)
    # block_7 = BatchNormalization()(conv_6)
    # block_7 =  Add()([conv_6, block_7])
    # block_7 = Activation('relu')(block_7)
    
    # conv_7 = Conv2D(32, (3, 3), dilation_rate=(11, 11), kernel_initializer='he_normal', padding='same')(block_7)
    # block_8 = BatchNormalization()(conv_7)
    # block_8 =  Add()([conv_7, block_8])
    # block_8 = Activation('relu')(block_8)
    
    # conv_8 = Conv2D(32, (3, 3), dilation_rate=(13, 13), kernel_initializer='he_normal', padding='same')(block_8)
    # block_9 = BatchNormalization()(conv_8)
    # block_9 =  Add()([conv_8, block_9])
    # block_9 = Activation('relu')(block_9)
    
    # conv_9 = Conv2D(32, (3, 3), dilation_rate=(17, 17), kernel_initializer='he_normal', padding='same')(block_9)
    # block_10 = BatchNormalization()(conv_9)
    # block_10 =  Add()([conv_9, block_10])
    # block_10 = Activation('relu')(block_10)
    
    # conv_10 = Conv2D(32, (3, 3), dilation_rate=(19, 19), kernel_initializer='he_normal', padding='same')(block_10)
    # block_11 = BatchNormalization()(conv_10)
    # block_11 =  Add()([conv_10, block_11])
    # block_11 = Activation('relu')(block_11)
    
    # conv_11 = Conv2D(32, (3, 3), dilation_rate=(23, 23), kernel_initializer='he_normal', padding='same')(block_11)
    # block_12 = BatchNormalization()(conv_11)
    # block_12 =  Add()([conv_11, block_12])
    # block_12 = Activation('relu')(block_12)
    
    # conv_12 = Conv2D(32, (3, 3), dilation_rate=(29, 29), kernel_initializer='he_normal', padding='same')(block_12)
    # block_13 = BatchNormalization()(conv_12)
    # block_13 =  Add()([conv_12, block_13])
    # block_13 = Activation('relu')(block_13)
    
    # conv_13 = Conv2D(32, (3, 3), dilation_rate=(31, 31), kernel_initializer='he_normal', padding='same')(block_13)
    # block_14 = BatchNormalization()(conv_13)
    # block_14 =  Add()([conv_13, block_14])
    # block_14 = Activation('relu')(block_14)
    
    # conv_14 = Conv2D(32, (3, 3), dilation_rate=(37, 37), kernel_initializer='he_normal', padding='same')(block_14)
    # block_15 = BatchNormalization()(conv_14)
    # block_15 =  Add()([conv_14, block_15])
    # block_15 = Activation('relu')(block_15)
    
    # conv_15 = Conv2D(32, (3, 3), dilation_rate=(41, 41), kernel_initializer='he_normal', padding='same')(block_15)
    # block_16 = BatchNormalization()(conv_15)
    # block_16 =  Add()([conv_15, block_16])
    # block_16 = Activation('relu')(block_16)
    
    # conv_16 = Conv2D(32, (3, 3), dilation_rate=(43, 43), kernel_initializer='he_normal', padding='same')(block_16)
    # block_17 = BatchNormalization()(conv_16)
    # block_17 =  Add()([conv_16, block_17])
    # block_17 = Activation('relu')(block_17)
    
    # conv_17 = Conv2D(32, (3, 3), dilation_rate=(47, 47), kernel_initializer='he_normal', padding='same')(block_17)
    # block_18 = BatchNormalization()(conv_17)
    # block_18 =  Add()([conv_17, block_18])
    # block_18 = Activation('relu')(block_18)
    
    # conv_18 = Conv2D(32, (3, 3), dilation_rate=(53, 53), kernel_initializer='he_normal', padding='same')(block_18)
    # block_19 = BatchNormalization()(conv_18)
    # block_19 =  Add()([conv_18, block_19])
    # block_19 = Activation('relu')(block_19)
    
    # concat = concatenate([inputs, block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9, block_10, block_11, block_12, block_13, block_14, block_15, block_16, block_17, block_18, block_19], axis=-1)
    # dropout = Dropout(0.5)(concat)
    
    # conv_19 = Conv2D(256, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(dropout)
    # block_20 = BatchNormalization()(conv_19)
    # block_20 =  Add()([conv_19, block_20])
    # block_20 = Activation('relu')(block_20)
    
    # conv_20 = Conv2D(128, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_20)
    # block_21 = BatchNormalization()(conv_20)
    # block_21 =  Add()([conv_20, block_21])
    # block_21 = Activation('relu')(block_21)
    
    # conv_21 = Conv2D(64, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_21)
    # block_22 = BatchNormalization()(conv_21)
    # block_22 =  Add()([conv_21, block_22])
    # block_22 = Activation('relu')(block_22)
    
    # conv_22 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_22)
    # block_23 = BatchNormalization()(conv_22)
    # block_23 =  Add()([conv_22, block_23])
    # block_23 = Activation('relu')(block_23)
    
    # conv_23 = Conv2D(1, (1, 1), kernel_initializer='he_normal')(block_23)
    
    # output = Activation('sigmoid')(conv_23)
    
    # model = Model(inputs=[inputs], outputs=[output])

    # return model



#%%
# def anything_net(img_rows, img_cols):
   # inputs = Input((img_rows, img_cols, 1))
   
   # block_1 = BatchNormalization()(inputs)
   # block_1 = Add()([inputs, block_1])
   
   # conv_1 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_1)
   # block_2 = BatchNormalization()(conv_1)
   # block_2 =  Add()([conv_1, block_2])
   # block_2 = Activation('relu')(block_2)
   
   # concat_1 = Add()([block_1, block_2])
   
   # conv_2 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(concat_1)
   # block_3 = BatchNormalization()(conv_2)
   # block_3 =  Add()([conv_2, block_3])
   # block_3 = Activation('relu')(block_3)
   
   # concat_2 = Add()([block_1, block_3])
   
   # conv_3 = Conv2D(32, (3, 3), dilation_rate=(2, 2), kernel_initializer='he_normal', padding='same')(concat_2)
   # block_4 = BatchNormalization()(conv_3)
   # block_4 =  Add()([conv_3, block_4])
   # block_4 = Activation('relu')(block_4)
   
   # concat_3 = Add()([block_1, block_4])
   
   # conv_4 = Conv2D(32, (3, 3), dilation_rate=(3, 3), kernel_initializer='he_normal', padding='same')(concat_3)
   # block_5 = BatchNormalization()(conv_4)
   # block_5 =  Add()([conv_4, block_5])
   # block_5 = Activation('relu')(block_5)
   
   # concat_4 = Add()([block_1, block_5])
   
   # conv_5 = Conv2D(32, (3, 3), dilation_rate=(5, 5), kernel_initializer='he_normal', padding='same')(concat_4)
   # block_6 = BatchNormalization()(conv_5)
   # block_6 =  Add()([conv_5, block_6])
   # block_6 = Activation('relu')(block_6)
   
   # concat_5 = Add()([block_1, block_6])
   
   # conv_6 = Conv2D(32, (3, 3), dilation_rate=(8, 8), kernel_initializer='he_normal', padding='same')(concat_5)
   # block_7 = BatchNormalization()(conv_6)
   # block_7 =  Add()([conv_6, block_7])
   # block_7 = Activation('relu')(block_7)
   
   # concat_6 = Add()([block_1, block_7])
   
   # conv_7 = Conv2D(32, (3, 3), dilation_rate=(13, 13), kernel_initializer='he_normal', padding='same')(concat_6)
   # block_8 = BatchNormalization()(conv_7)
   # block_8 =  Add()([conv_7, block_8])
   # block_8 = Activation('relu')(block_8)
   
   # concat_7 = Add()([block_1, block_8])
   
   # conv_8 = Conv2D(32, (3, 3), dilation_rate=(21, 21), kernel_initializer='he_normal', padding='same')(concat_7)
   # block_9 = BatchNormalization()(conv_8)
   # block_9 =  Add()([conv_8, block_9])
   # block_9 = Activation('relu')(block_9)
   
   # concat_8 = Add()([block_1, block_9])
   
   # conv_9 = Conv2D(32, (3, 3), dilation_rate=(34, 34), kernel_initializer='he_normal', padding='same')(concat_8)
   # block_10 = BatchNormalization()(conv_9)
   # block_10 =  Add()([conv_9, block_10])
   # block_10 = Activation('relu')(block_10)
   
   # concat_9 = Add()([block_1, block_10])
   
   # conv_10 = Conv2D(32, (3, 3), dilation_rate=(55, 55), kernel_initializer='he_normal', padding='same')(concat_9)
   # block_11 = BatchNormalization()(conv_10)
   # block_11 =  Add()([conv_10, block_11])
   # block_11 = Activation('relu')(block_11)
   
   # concat_10 = Add()([block_1, block_11])
   
   # concat = concatenate([inputs, block_1, concat_1, concat_2, concat_3, concat_4, concat_5, concat_6, concat_7, concat_8, concat_9, concat_10], axis=-1)
   # dropout = Dropout(0.5)(concat)
   
   # conv_11 = Conv2D(128, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(dropout)
   # block_12 = BatchNormalization()(conv_11)
   # block_12 =  Add()([conv_11, block_12])
   # block_12 = Activation('relu')(block_12)
   
   # conv_12 = Conv2D(64, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_12)
   # block_13 = BatchNormalization()(conv_12)
   # block_13 =  Add()([conv_12, block_13])
   # block_13 = Activation('relu')(block_13)
   
   # conv_13 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(block_13)
   # block_14 = BatchNormalization()(conv_13)
   # block_14 =  Add()([conv_13, block_14])
   # block_14 = Activation('relu')(block_14)
   
   # conv_14 = Conv2D(1, (1, 1), kernel_initializer='he_normal')(block_14)
   
   # output = Activation('sigmoid')(conv_14)
   
   # model = Model(inputs=[inputs], outputs=[output])

   # return model

    
    