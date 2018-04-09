from __future__ import print_function


from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, Permute, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, UpSampling2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


from keras.optimizers import SGD


from keras.utils import plot_model

import cv2, numpy as np
import sys


nb_classes = 17
nb_epoch = 5
batch_size = 32
img_rows = 572
img_cols = 572
samples_per_epoch = 1190
nb_val_samples = 170

def create_model():

    #(samples, channels, rows, cols)
    input_img = Input(shape=(3, img_rows, img_cols))
    #3*572*572
    x = Conv2D(64, 3, strides=(1, 1), activation='relu', padding='valid')(input_img)
    x = Conv2D(64, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #64*568*568
    p1 = Cropping2D(cropping=((88, 88), (88, 88)))(x)
    #64*392*392


    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #64*284*284
    x = Conv2D(128, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(128, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #128*280*280
    p2 = Cropping2D(cropping=((40, 40), (40, 40)))(x)
    #128*200*200

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #128*140*140
    x = Conv2D(256, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(256, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #256*136*136
    p3 = Cropping2D(cropping=((16, 16), (16, 16)))(x)
    #256*104*104

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #256*68*68
    x = Conv2D(512, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(512, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #512*64*64
    p4 = Cropping2D(cropping=((4, 4), (4, 4)))(x)
    #512*56*56

    x = MaxPooling2D((2,2), strides=(2,2))(x)
    #512*32*32
    x = Conv2D(1024, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(1024, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #1024*28*28
    p5 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(x)
    #512*56*56 

    x = add([p5, p4])
    #512*56*56
    x = Conv2D(512, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(512, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #512*52*52
    p6 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(x)
    #256*104*104

    x = add([p6, p3])
    #256*104*104
    x = Conv2D(256, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(256, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #256*100*100
    p7 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(x)
    #128*200*200

    x = add([p7, p2])
    #128*200*200
    x = Conv2D(128, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(128, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #128*196*196
    p8 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(x)
    #64*392*392

    x = add([p8, p1])
    #64*392*392
    x = Conv2D(64, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(64, 3, strides=(1, 1), activation='relu', padding='valid')(x)
    #64*388*388

    x = Conv2D(2, 1, strides=(1, 1), activation='relu', padding='valid')(x)
    #2*388*388

    out = x
    
    model = Model(input_img, out)
    return model

model = create_model()

model.summary()


plot_model(model, to_file='u-net_model.png', show_shapes=True)
