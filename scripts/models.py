from tensorflow.keras import layers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
#from tensorflow.keras.layers.merge import Concatenate, Average
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental import RandomFourierFeatures
import numpy as np
import augmentations
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
import tensorflow as tf


def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
# added MH    
    conv0 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv0 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)    
    
#    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool0)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    
    # added MH
    up9a = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv9), conv0], axis=3)
    conv9a = Conv2D(16, (3, 3), activation='relu', padding='same')(up9a)
    conv9a = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9a)

#    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9a)

    return Model(inputs=[inputs], outputs=[conv10])

def unet_L(input_size=(256,256,1)):
    inputs = Input(input_size)
    
# added MH    
    conv0 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv0 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)    
    pool0 = bn_act(pool0)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool0)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = bn_act(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = bn_act(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = bn_act(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = bn_act(pool4)
    
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = bn_act(pool5)

    convB = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
    convB = Conv2D(1024, (3, 3), activation='relu', padding='same')(convB)

    upB = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(convB), conv5], axis=3)
    convB2 = Conv2D(512, (3, 3), activation='relu', padding='same')(upB)
    convB2 = Conv2D(512, (3, 3), activation='relu', padding='same')(convB2)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(convB2), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    
    # added MH
    up9a = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv9), conv0], axis=3)
    conv9a = Conv2D(16, (3, 3), activation='relu', padding='same')(up9a)
    conv9a = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9a)

#    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9a)

    return Model(inputs=[inputs], outputs=[conv10])

def bn_act(x, act=True):
    # temporary reinstate batchnorm
    #x = layers.BatchNormalization()(x)
    x = tfa.layers.InstanceNormalization()(x)
    
    if act == True:
        x = layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = layers.UpSampling2D((2, 2))(x)
    c = layers.Concatenate()([u, xskip])
    return c


def ResUNet_M(input_size=(256,256,1)):
    f = [16, 32, 64, 128, 256, 512]
    
    inputs = Input(input_size)
    
    #inputs = keras.layers.Input((image_size, image_size, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    
    
    ## Bridge
    b0 = conv_block(e6, f[5], strides=1)
    b1 = conv_block(b0, f[5], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e5)
    d1 = residual_block(u1, f[5])
    
    u2 = upsample_concat_block(d1, e4)
    d2 = residual_block(u2, f[4])
    
    u3 = upsample_concat_block(d2, e3)
    d3 = residual_block(u3, f[3])
    
    u4 = upsample_concat_block(d3, e2)
    d4 = residual_block(u4, f[2])
    
    u5 = upsample_concat_block(d4, e1)
    d5 = residual_block(u5, f[1])
    
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d5)
    model = Model(inputs, outputs)
    return model

def ResUNet_S(input_size=(256,256,1)):
#    f = [32, 64, 128, 256, 512]
    f = [16, 32, 64, 128, 256] #, 512, 1024]#512]

    
    inputs = Input(input_size)
    
    #inputs = keras.layers.Input((image_size, image_size, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)    
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs)
    return model

def ResUNet_L(input_size=(256,256,1)):
    # f = [16, 32, 64, 128, 256, 512] normal res-unet
 
    f = [16, 32, 64, 128, 256, 512, 1024]#512]
    
    inputs = Input(input_size)
    
    #inputs = keras.layers.Input((image_size, image_size, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    e7 = residual_block(e6, f[6], strides=2)
    
    ## Bridge
    b0 = conv_block(e7, f[6], strides=1)
    b1 = conv_block(b0, f[6], strides=1)
    
    ## Decoder
    u0 = upsample_concat_block(b1, e6)
    d0 = residual_block(u0, f[6])

    u1 = upsample_concat_block(d0, e5)
    d1 = residual_block(u1, f[5])
    
    u2 = upsample_concat_block(d1, e4)
    d2 = residual_block(u2, f[4])
    
    u3 = upsample_concat_block(d2, e3)
    d3 = residual_block(u3, f[3])
    
    u4 = upsample_concat_block(d3, e2)
    d4 = residual_block(u4, f[2])
    
    u5 = upsample_concat_block(d4, e1)
    d5 = residual_block(u5, f[1])

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d5)
    model = Model(inputs, outputs)
    return model
    
def DenseUNet_L(input_size=(256,256,1)):
    
    inputs = Input(input_size)
    
    conv0_1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #BatchNorm0_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv0_1)
    BatchNorm0_1 = tfa.layers.InstanceNormalization()(conv0_1)
      
    ReLU0_1 = Activation('relu')(BatchNorm0_1)
    conv0_2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU0_1)
    drop0_2 = Dropout(0)(conv0_2)
    Merge0 = Concatenate()([conv0_1,drop0_2])
    pool0 = MaxPooling2D(pool_size=(2, 2))(Merge0)
    
    conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
    #BatchNorm1_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv1_1)
    BatchNorm1_1 = tfa.layers.InstanceNormalization()(conv1_1)
    
    ReLU1_1 = Activation('relu')(BatchNorm1_1)
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU1_1)
    drop1_2 = Dropout(0)(conv1_2)
    Merge1 = Concatenate()([conv1_1,drop1_2])
    pool1 = MaxPooling2D(pool_size=(2, 2))(Merge1)

    conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #BatchNorm2_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv2_1)
    BatchNorm2_1 = tfa.layers.InstanceNormalization()(conv2_1)
      
    ReLU2_1 = Activation('relu')(BatchNorm2_1)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU2_1)
    drop2_2 = Dropout(0)(conv2_2)
    Merge2 = Concatenate()([conv2_1,drop2_2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(Merge2)

    conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #BatchNorm3_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv3_1)
    BatchNorm3_1 = tfa.layers.InstanceNormalization()(conv3_1)

    ReLU3_1 = Activation('relu')(BatchNorm3_1)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU3_1)
    drop3_2 = Dropout(0)(conv3_2)
    Merge3 = Concatenate()([conv3_1,drop3_2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(Merge3)

    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #BatchNorm4_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv4_1)
    BatchNorm4_1 = tfa.layers.InstanceNormalization()(conv4_1)

    ReLU4_1 = Activation('relu')(BatchNorm4_1)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU4_1)
    drop4_2 = Dropout(0)(conv4_2)
    Merge4 = Concatenate()([conv4_1,drop4_2])
    drop4 = Dropout(0.5)(Merge4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    BatchNorm5_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv5_1)
    BatchNorm5_1 = tfa.layers.InstanceNormalization()(conv5_1)

    ReLU5_1 = Activation('relu')(BatchNorm5_1)
    conv5_2 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU5_1)
    drop5_2 = Dropout(0)(conv5_2)
    Merge5 = Concatenate()([conv5_1,drop5_2])
    drop5 = Dropout(0.5)(Merge5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate()([drop4,up6])
    conv6_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #BatchNorm6_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv6_1)
    BatchNorm6_1 = tfa.layers.InstanceNormalization()(conv6_1)
    ReLU6_1 = Activation('relu')(BatchNorm6_1)
    conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU6_1)
    drop6_2 = Dropout(0)(conv6_2)
    Merge6 = Concatenate()([conv6_1,drop6_2])

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge6))
    merge7 = Concatenate()([Merge3,up7])
    conv7_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #BatchNorm7_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv7_1)
    BatchNorm7_1 = tfa.layers.InstanceNormalization()(conv7_1)
    ReLU7_1 = Activation('relu')(BatchNorm7_1)
    conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU7_1)
    drop7_2 = Dropout(0)(conv7_2)
    Merge7 = Concatenate()([conv7_1,drop7_2])

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge7))
    merge8 = Concatenate()([Merge2,up8])
    conv8_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #BatchNorm8_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv8_1)
    BatchNorm8_1 = tfa.layers.InstanceNormalization()(conv8_1)
    ReLU8_1 = Activation('relu')(BatchNorm8_1)
    conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU8_1)
    drop8_2 = Dropout(0)(conv8_2)
    Merge8 = Concatenate()([conv8_1,drop8_2])

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge8))
    merge9 = Concatenate()([Merge1,up9])
    conv9_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #BatchNorm9_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv9_1)
    BatchNorm9_1 = tfa.layers.InstanceNormalization()(conv9_1)    
    ReLU9_1 = Activation('relu')(BatchNorm9_1)
    conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU9_1)
    drop9_2 = Dropout(0)(conv9_2)
    Merge9 = Concatenate()([conv9_1,drop9_2])
    
    up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge9))
    merge10 = Concatenate()([Merge0,up10])
    conv10_1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    #BatchNorm10_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv10_1)
    BatchNorm10_1 = tfa.layers.InstanceNormalization()(conv10_1)
    ReLU10_1 = Activation('relu')(BatchNorm10_1)
    conv10_2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU10_1)
    drop10_2 = Dropout(0)(conv10_2)
    Merge10 = Concatenate()([conv10_1,drop10_2])
    
    

    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge10)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs, conv11)
    return model

def DenseUNet_M(input_size=(256,256,1)):
    
    inputs = Input(input_size)
       
    conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    BatchNorm1_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv1_1)
    ReLU1_1 = Activation('relu')(BatchNorm1_1)
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU1_1)
    drop1_2 = Dropout(0)(conv1_2)
    Merge1 = Concatenate()([conv1_1,drop1_2])
    pool1 = MaxPooling2D(pool_size=(2, 2))(Merge1)

    conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    BatchNorm2_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv2_1)
    ReLU2_1 = Activation('relu')(BatchNorm2_1)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU2_1)
    drop2_2 = Dropout(0)(conv2_2)
    Merge2 = Concatenate()([conv2_1,drop2_2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(Merge2)

    conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    BatchNorm3_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv3_1)
    ReLU3_1 = Activation('relu')(BatchNorm3_1)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU3_1)
    drop3_2 = Dropout(0)(conv3_2)
    Merge3 = Concatenate()([conv3_1,drop3_2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(Merge3)

    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    BatchNorm4_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv4_1)
    ReLU4_1 = Activation('relu')(BatchNorm4_1)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU4_1)
    drop4_2 = Dropout(0)(conv4_2)
    Merge4 = Concatenate()([conv4_1,drop4_2])
    drop4 = Dropout(0.5)(Merge4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    BatchNorm5_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv5_1)
    ReLU5_1 = Activation('relu')(BatchNorm5_1)
    conv5_2 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU5_1)
    drop5_2 = Dropout(0)(conv5_2)
    Merge5 = Concatenate()([conv5_1,drop5_2])
    drop5 = Dropout(0.5)(Merge5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate()([drop4,up6])
    conv6_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    BatchNorm6_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv6_1)
    ReLU6_1 = Activation('relu')(BatchNorm6_1)
    conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU6_1)
    drop6_2 = Dropout(0)(conv6_2)
    Merge6 = Concatenate()([conv6_1,drop6_2])

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge6))
    merge7 = Concatenate()([Merge3,up7])
    conv7_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    BatchNorm7_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv7_1)
    ReLU7_1 = Activation('relu')(BatchNorm7_1)
    conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU7_1)
    drop7_2 = Dropout(0)(conv7_2)
    Merge7 = Concatenate()([conv7_1,drop7_2])

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge7))
    merge8 = Concatenate()([Merge2,up8])
    conv8_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    BatchNorm8_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv8_1)
    ReLU8_1 = Activation('relu')(BatchNorm8_1)
    conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU8_1)
    drop8_2 = Dropout(0)(conv8_2)
    Merge8 = Concatenate()([conv8_1,drop8_2])

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge8))
    merge9 = Concatenate()([Merge1,up9])
    conv9_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    BatchNorm9_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv9_1)
    ReLU9_1 = Activation('relu')(BatchNorm9_1)
    conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU9_1)
    drop9_2 = Dropout(0)(conv9_2)
    Merge9 = Concatenate()([conv9_1,drop9_2])
    
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)# was sigmoid

    model = Model(inputs, conv11)
    return model


def DenseUNet_S(input_size=(256,256,1)):
    
    inputs = Input(input_size)
    
    conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    BatchNorm2_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv2_1)
    ReLU2_1 = Activation('relu')(BatchNorm2_1)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU2_1)
    drop2_2 = Dropout(0)(conv2_2)
    Merge2 = Concatenate()([conv2_1,drop2_2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(Merge2)
       
    conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    BatchNorm3_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv3_1)
    ReLU3_1 = Activation('relu')(BatchNorm3_1)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU3_1)
    drop3_2 = Dropout(0)(conv3_2)
    Merge3 = Concatenate()([conv3_1,drop3_2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(Merge3)

    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    BatchNorm4_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv4_1)
    ReLU4_1 = Activation('relu')(BatchNorm4_1)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU4_1)
    drop4_2 = Dropout(0)(conv4_2)
    Merge4 = Concatenate()([conv4_1,drop4_2])
    drop4 = Dropout(0.5)(Merge4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    BatchNorm5_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv5_1)
    ReLU5_1 = Activation('relu')(BatchNorm5_1)
    conv5_2 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU5_1)
    drop5_2 = Dropout(0)(conv5_2)
    Merge5 = Concatenate()([conv5_1,drop5_2])
    drop5 = Dropout(0.5)(Merge5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate()([drop4,up6])
    conv6_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    BatchNorm6_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv6_1)
    ReLU6_1 = Activation('relu')(BatchNorm6_1)
    conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU6_1)
    drop6_2 = Dropout(0)(conv6_2)
    Merge6 = Concatenate()([conv6_1,drop6_2])

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge6))
    merge7 = Concatenate()([Merge3,up7])
    conv7_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    BatchNorm7_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv7_1)
    ReLU7_1 = Activation('relu')(BatchNorm7_1)
    conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU7_1)
    drop7_2 = Dropout(0)(conv7_2)
    Merge7 = Concatenate()([conv7_1,drop7_2])
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge7))
    merge8 = Concatenate()([Merge2,up8])
    conv8_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    BatchNorm8_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv8_1)
    ReLU8_1 = Activation('relu')(BatchNorm8_1)
    conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU8_1)
    drop8_2 = Dropout(0)(conv8_2)
    Merge8 = Concatenate()([conv8_1,drop8_2])
   
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge8)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)# was sigmoid

    model = Model(inputs, conv11)
    return model
    
    
# resunet++
def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="same")(y)
    return y

def attetion_block(g, x):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    filters = x.shape[-1]

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul

#class ResUnetPlusPlus:
def ResUNetPlusPlus(input_size=(256,256,1)):

        n_filters = [16, 32, 64, 128, 256]
        inputs = Input(input_size)

        c0 = inputs
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model
    
def ResWNet_L(input_size=(256,256,1)):
    # f = [16, 32, 64, 128, 256, 512] normal res-unet
 
    f = [16, 32, 64, 128, 256, 512, 1024]#512]
    
    inputs = Input(input_size)
    
    #inputs = keras.layers.Input((image_size, image_size, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    e7 = residual_block(e6, f[6], strides=2)
    
    
    ## Bridge
    b0 = conv_block(e7, f[6], strides=1)
    
    #interim decode
    int_u = upsample_concat_block(b0, e6)
    int_d = residual_block(int_u, f[6])
    
    #interim bridge
    int_b = conv_block(int_d, f[6], strides=1)
    
    #interim endcode
    int_e = residual_block(int_b, f[6], strides=2)
    
    # final bridge
    b1 = conv_block(int_e, f[6], strides=1)
    
    ## Decoder
    u0 = upsample_concat_block(b1, e6)
    d0 = residual_block(u0, f[6])

    u1 = upsample_concat_block(d0, e5)
    d1 = residual_block(u1, f[5])
    
    u2 = upsample_concat_block(d1, e4)
    d2 = residual_block(u2, f[4])
    
    u3 = upsample_concat_block(d2, e3)
    d3 = residual_block(u3, f[3])
    
    u4 = upsample_concat_block(d3, e2)
    d4 = residual_block(u4, f[2])
    
    u5 = upsample_concat_block(d4, e1)
    d5 = residual_block(u5, f[1])
    
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d5)
    model = Model(inputs, outputs)
    return model

def RippleNet_L(input_size=(256,256,1)):
    # f = [16, 32, 64, 128, 256, 512] normal res-unet
 
    f = [16, 32, 64, 128, 256, 512, 1024]#512]
    
    inputs = Input(input_size)
    
    #inputs = keras.layers.Input((image_size, image_size, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    e7 = residual_block(e6, f[6], strides=2)
    
    
    ## Bridge
    b0 = conv_block(e7, f[6], strides=1)
    
    
    #interim decode/encode - A
    int_uA = upsample_concat_block(b0, e6)
    int_dA = residual_block(int_uA, f[6])
    int_bA = conv_block(int_dA, f[6], strides=1)
    int_eA = residual_block(int_bA, f[6], strides=2)
    
    b1_A = conv_block(int_eA, f[6], strides=1)

    #interim decode/encode - B
    int_uB = upsample_concat_block(b1_A, e6)
    int_dB = residual_block(int_uB, f[6])
    int_bB = conv_block(int_dB, f[6], strides=1)
    int_eB = residual_block(int_bB, f[6], strides=2)
    
    b1_B = conv_block(int_eB, f[6], strides=1)
    
    
    
    #interim bridge
    #int_b = conv_block(int_d, f[6], strides=1)
    
    #interim decode/endcode - B
    #int_e = residual_block(int_b, f[6], strides=2)
    
    # final bridge
    
    b1 = conv_block(b1_B, f[6], strides=1)
    
    
    ## Decoder
    u0 = upsample_concat_block(b1, e6)
    d0 = residual_block(u0, f[6])

    u1 = upsample_concat_block(d0, e5)
    d1 = residual_block(u1, f[5])
    
    u2 = upsample_concat_block(d1, e4)
    d2 = residual_block(u2, f[4])
    
    u3 = upsample_concat_block(d2, e3)
    d3 = residual_block(u3, f[3])
    
    u4 = upsample_concat_block(d3, e2)
    d4 = residual_block(u4, f[2])
    
    u5 = upsample_concat_block(d4, e1)
    d5 = residual_block(u5, f[1])
    
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d5)
    model = Model(inputs, outputs)
    return model    
    
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X
def convolutional_block(X, f, filters, stage, block, s = 1):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X
    
def CANet_L(input_size=(256,256,1)):
    X_input = Input(input_size)
    
    # Zero-Padding
    #X = ZeroPadding2D((3, 3))(X_input)

    # MHStage 0
    X = Conv2D(8, (3, 3), name='convinit', kernel_initializer=glorot_uniform(seed=0))(X_input)
    
    X = BatchNormalization(axis=3, name='bn_convinit')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    X = Conv2D(16, (3, 3), name='conv0', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis=3, name='bn_conv0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)


    # Stage 1
    X = Conv2D(64, (3, 3), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X) #(X_input)
    
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
    X_ag_cat = convolutional_block(X, f=3, filters=[4, 4, 8], stage=2, block='a', s=4)
    X = identity_block(X_ag_cat, 3, [4, 4, 8], stage=2, block='b')
    X = identity_block(X, 3, [4, 4, 8], stage=2, block='c')

        # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [8, 8, 16], stage = 3, block='a', s = 1)
    X = identity_block(X, 3, [8, 8, 16], stage=3, block='b')
    X = identity_block(X, 3, [8, 8, 16], stage=3, block='c')
    X = identity_block(X, 3, [8, 8, 16], stage=3, block='d')

        # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [16, 16, 32], stage = 4, block='a', s = 1)
    X = identity_block(X, 3, [16, 16, 32], stage=4, block='b')
    X = identity_block(X, 3, [16, 16, 32], stage=4, block='c')
    X = identity_block(X, 3, [16, 16, 32], stage=4, block='d')
    X = identity_block(X, 3, [16, 16, 32], stage=4, block='e')
    X = identity_block(X, 3, [16, 16, 32], stage=4, block='f')

        # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [32, 32, 64], stage = 5, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 64], stage=5, block='b')
    X = identity_block(X, 3, [32, 32, 64], stage=5, block='c')

    x_GF = AveragePooling2D(1,(16,16))(X)                       #Global Average pooling changes the dimensions of the output. Instead use average   
    
    x_GF2 = BatchNormalization()(x_GF)                               # pooling with the kernal dimensions equal to that of the entire layer
    x_GF3 = Activation('relu')(x_GF2)    
    
    GF_Conv_1 = Conv2D(32, (1, 1), activation = 'relu', padding = 'same', name = 'GF_conv1' )(x_GF3)
    GF_Conv_T = (Conv2DTranspose(32 ,(16,16), use_bias = False))(GF_Conv_1)

    print(X.shape)
    print(GF_Conv_T.shape)

    c1 = Concatenate()([X,GF_Conv_T])
    CF_x1 = AveragePooling2D((2, 2),  (2, 2), name = 'CF1_pool' )(c1)
    CF_Conv_11 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'CF1_conv1' )(CF_x1)
    CF_Conv_12 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'CF1_conv2')(CF_Conv_11)
    CF_Conv_13 = Conv2D(32, (1, 1), activation = 'relu', padding = 'same', name = 'CF1_conv3' )(CF_Conv_12)
    CF_x2 = Activation('relu')(CF_Conv_13)
    CF_Conv_14 = Conv2D(32, (1, 1), activation = 'relu', padding = 'same', name = 'CF1-conv4' )(CF_x2)
    CF_x3 = Activation('sigmoid')(CF_Conv_14)
    CF_x4 = Multiply()([CF_Conv_12 , CF_Conv_14])
    CF_x5 = Add()([CF_Conv_12,CF_x4])
    CF_Conv_T1 = (Conv2DTranspose(32 ,  (9, 9), use_bias = False ))(CF_x5) 


    c12= concatenate([X,CF_Conv_T1])
    x = AveragePooling2D((2, 2), (2, 2), name = 'CF2_pool' )(c12)
    CF_Conv_21 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'CF2_conv1' )(x)
    CF_Conv_22 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'CF2_conv2' )(CF_Conv_11)
    CF_Conv_23 = Conv2D(32, (1, 1), activation = 'relu', padding = 'same', name = 'CF2_conv3')(CF_Conv_12)
    x = Activation('relu')(CF_Conv_13)
    CF_Conv_24 = Conv2D(32, (1, 1), activation = 'relu', padding = 'same', name = 'CF2_conv4' )(x)
    x =  Activation('sigmoid')(CF_Conv_24)
    x = Multiply()([CF_Conv_22 , CF_Conv_24])
    x = Add()([CF_Conv_22,x])
    CF_Conv_T2 = (Conv2DTranspose(32 , (9, 9), use_bias = False ))(x)


    c3 = concatenate([X,CF_Conv_T2])
    x = AveragePooling2D((2, 2),  (2, 2), name = 'CF3_pool')(c1)
    CF_Conv_31 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'CF3_conv1')(x)
    CF_Conv_32 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'CF3_conv2')(CF_Conv_11)
    CF_Conv_33 = Conv2D(32, (1, 1), activation = 'relu', padding = 'same', name = 'CF3_conv3' )(CF_Conv_12)
    x = Activation('relu')(CF_Conv_13)
    CF_Conv_34 = Conv2D(32, (1, 1), activation = 'relu', padding = 'same', name = 'CF3_conv4' )(x)
    x = Activation('sigmoid')(CF_Conv_34)
    x =  Multiply()([CF_Conv_32 , CF_Conv_34])
    x = Add()([CF_Conv_32,x])
    CF_Conv_T3 = (Conv2DTranspose(32 ,  (9, 9), use_bias = False ))(x)


    o = Add(name = 'add1')([GF_Conv_T, CF_Conv_T1, CF_Conv_T2, CF_Conv_T3])

    FS_Conv_1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'FS_conv1')(o)
    x = AveragePooling2D(1,(2, 2), name = 'FS_pool')(FS_Conv_1)
    FS_Conv_2 = Conv2D(32, (1, 1), activation = 'relu', padding = 'same', name = 'FS_conv2')(x)
    bn = BatchNormalization()(FS_Conv_2)
    s = Activation('sigmoid')(bn)
    FS_Conv_T = Conv2DTranspose(32 ,  (9, 9), use_bias = False )(s)
    x = Multiply()([FS_Conv_1,FS_Conv_T])


    # 32 --> 16  --> 8
    mid_Conv = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'FS_Conv3')(x)


    AG_Conv_L1 = Conv2D(32, (7, 1), activation = 'relu', padding = 'same', name = 'AG_conv1' )(X_ag_cat)
    AG_Conv_L2 = Conv2D(32, (1, 7), activation = 'relu', padding = 'same', name = 'AG_conv2')(AG_Conv_L1)

    AG_Conv_R1 = Conv2D(32, (1, 7), activation = 'relu', padding = 'same', name = 'AG_conv3' )(X_ag_cat)
    AG_Conv_R2 = Conv2D(32, (7, 1), activation = 'relu', padding = 'same', name = 'AG_conv4')(AG_Conv_R1)

    o2 = Add()([AG_Conv_L2,AG_Conv_R2])

    #AG_Conv_F1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'AG_conv5' )(o2)
    AG_Conv_F1 = Conv2D(32, (3, 3), activation = 'sigmoid', padding = 'same', name = 'AG_conv5' )(o2)


    o3 = Add()([o2,AG_Conv_F1])


    Conc_A =  (UpSampling2D((2,2), interpolation='bilinear'))(mid_Conv) #(Conv2DTranspose(nClasses , (113,113), use_bias = False ))(mid_Conv)
    #Conc_A = Conv2D(16, (3, 3), activation = 'relu', padding = 'same', name = 'CF_conv0A' )(Conc_A)
    
    Conc_B = (UpSampling2D((2,2), interpolation='bilinear'))(o3)
    #Conc_B = Conv2D(16, (3, 3), activation = 'relu', padding = 'same', name = 'CF_conv0B' )(Conc_B)
    
    CF = concatenate([Conc_A,Conc_B])
    
    #CF = Conv2D(16, (3, 3), activation = 'relu', padding = 'same', name = 'CF_conv0' )(CF)
    #CF = Conv2D(8, (3, 3), activation = 'relu', padding = 'same', name = 'CF_conv1' )(CF)
    
    
    # MH
    CF = (UpSampling2D((2,2), interpolation='bilinear'))(CF)
    CF = (UpSampling2D((2,2), interpolation='bilinear'))(CF)
    
    # one more upsample layer
    # added MH
    #CF = Conv2D(16, (3, 3), activation='relu', padding='same')(CF)
    #CF = Conv2D(16, (3, 3), activation='relu', padding='same')(CF)
    
    
    #Final = Conv2D(39, (3, 3), activation = 'relu', padding = 'same', name = 'FinalBlock1')(CF)
    # only one classs
    Final = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same', name = 'FinalBlock1')(CF)
         
    o_Final  = UpSampling2D((4,4), interpolation = 'bilinear')(Final)
    Final = (Activation('softmax'))(o_Final)
        
    model = Model(inputs = X_input, outputs = Final, name='CANet')
    
    return model
    
def gauss2D(shape=(3,3),sigma=0.5):

    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h    
           
def LOGResUNet_M(input_size=(256,256,1)):
    f = [16, 32, 64, 128, 256, 512] # best
    
    #f = [8, 16, 32, 64, 128, 256] # zero accuracy/high fps
    
    #f = [16, 32, 64, 128, 256, 512] 
    
       
    inputs = Input(input_size)  
    
    # let's try some pre-processing
    #crop = augmentations.RandomResizedCrop()
    #inputs = crop(inputs)  
    
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    e6 = residual_block(e5, f[5], strides=2)
    
    ## Bridge
    b0 = conv_block(e6, f[5], strides=1)
    b1 = conv_block(b0, f[5], strides=1)
       
    ## Decoder
    u1 = upsample_concat_block(b1, e5)
    d1 = residual_block(u1, f[5])
    
    u2 = upsample_concat_block(d1, e4)
    d2 = residual_block(u2, f[4])
    
    u3 = upsample_concat_block(d2, e3)
    d3 = residual_block(u3, f[3])
    
    u4 = upsample_concat_block(d3, e2)
    d4 = residual_block(u4, f[2])
    
    u5 = upsample_concat_block(d4, e1)
    d5 = residual_block(u5, f[1])
    
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d5)
    model = Model(inputs, outputs)
    
    return model
    
# IMPLEMENT A DEEPLAB MODEL
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output
    
def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 1))
#    resnet50 = keras.applications.ResNet50(
#        weights="imagenet", include_top=False, input_tensor=model_input
#    )
    resnet50 = keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=model_input
    )


    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

# IMPLEMENT A PRETRAINED INCEPTION MODEL
