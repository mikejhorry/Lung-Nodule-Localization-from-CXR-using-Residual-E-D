#!/usr/bin/env python
# coding: utf-8
# Mike Horry

import os
import argparse

#import models

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from glob import glob
import random
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import seaborn as sns
#import imutils
from datetime import datetime
from skimage import data, img_as_float, exposure, io
from skimage.metrics import structural_similarity
import mlflow
import mlflow.tensorflow
import json

modelroot = './models/'

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", required=True, help="Test Data Set")
ap.add_argument("-d", "--dataset", required=True, help="Source Data Set")
ap.add_argument("-a", "--augmentflag", default=False, help="Augment Data")
ap.add_argument("-k", "--kern", default=10, required=True, help="Morph Kernal size")
ap.add_argument("-ext", "--external", default=False, help="External Mode Training/Testing")
ap.add_argument("-m", "--model", required=True, help="RESNET-S RESNET-M RESNET-L unet")
ap.add_argument("-e", "--epochs", required=True, help="Number of Epochs")
ap.add_argument("-he", "--histogram", default=False, help="Equalize Histogram")
ap.add_argument("-in", "--instance", default=False, help="Use Instance Normalization")
ap.add_argument("-lr", "--learningrate", required=True, help="Learning Rate")
ap.add_argument("-bs", "--batchsize", required=True, help="Batch Size")
ap.add_argument("-x", "--dim", default=1024, help="Dimension")
ap.add_argument("-inf", "--inference", default=False, help="Inferencing Mode")
ap.add_argument("-best", "--best", default=False, help="Load Best Models from JSON")
ap.add_argument("-od", "--odataset", required=False, help="Override Dataset")


args = vars(ap.parse_args())

# globals
#classifier = args["classifier"]
test = args["test"]
EPOCHS = int(args["epochs"])
augmentflag = args["augmentflag"]
#resumeflag = args["resumeflag"]
dataset = args["dataset"] #+ "/" + test #+ "/" #+ test
override_dataset = args["odataset"]
#datasetVal = dataset + "_VALIDATION"
externalMode = args["external"]
BS=int(args["batchsize"])
LR=float(args["learningrate"])
DIM = int(args["dim"])
KERN = int(args["kern"])
MODEL = args['model']
#INSTANCENORM = args["instance"]
#INSTANCENORM = True
#HE = args['histogram']
INSTANCENORM = args["instance"]
HE = args['histogram']
INFERENCE = args["inference"]
BEST = args["best"]

multiplier = 1

tolerance = 128

areaconstraint = 256

if DIM == 1024:
    multiplier = 0.5
    tolerance = 64
    areaconstraint = 64

if DIM == 512:
    multiplier = 0.25
    tolerance = 32
    areaconstraint = 16

# hack
dim=DIM

# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d%m%Y-%H:%M:%S")
print("Date and Time: ", dt_string)

print('LR=' + str(LR))

EXPERIMENT = test + "-Image Size=" + str(DIM) + "-Model=" + MODEL + "-LR="+str(LR) + "-" + "BS="+str(BS)
print("Experiment: " + EXPERIMENT)

# k-fold setup
NUM_FOLDS=10 #(0,1,2,3,4,5,6,7,8,9)
#NUM_FOLDS=1 #tuning only

# init seeds so that results are repeatable
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# set folds to one for externalMode
if externalMode:
    print("Running External Mode Training")
    NUM_FOLDS=1
    if override_dataset:
        dataset = override_dataset

#Models under test
def bn_act(x, act=True):   
    if INSTANCENORM:
        x = tfa.layers.InstanceNormalization()(x)
    else:
        #print("Using BN")
        x = tensorflow.keras.layers.BatchNormalization()(x)    
    if act == True:
        x = tensorflow.keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tensorflow.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = tensorflow.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tensorflow.keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tensorflow.keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    c = tensorflow.keras.layers.Concatenate()([u, xskip])
    return c

def ResUNet_S():
    f = [16, 32, 64, 128, 256]
    inputs = tensorflow.keras.layers.Input((dim, dim, 1))
    
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
    
    outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = tensorflow.keras.models.Model(inputs, outputs)
    return model

def ResUNet_M():
    f = [16, 32, 64, 128, 256, 512]
    inputs = tensorflow.keras.layers.Input((dim, dim, 1))
    
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
    
    
    outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d5)
    model = tensorflow.keras.models.Model(inputs, outputs)
    return model

def ResUNet_L(input_size=(256,256,1)):
 
    f = [16, 32, 64, 128, 256, 512, 1024]
    
    #inputs = Input((dim, dim, 1))
    
    inputs = tensorflow.keras.layers.Input((dim, dim, 1))
    
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

    outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d5)
    model = tensorflow.keras.models.Model(inputs, outputs)
    return model


def unet():

    inputs = tensorflow.keras.layers.Input((dim, dim, 1))

    #inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
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

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

def sliding_window(elements, window_size):
    
    if len(elements) <= window_size:
       return elements
    for i in range(len(elements)- window_size + 1):
        combos.append(elements[i:i+window_size])
       
def getROI(frame, x, y, r):
    return frame[int(y-r/2):int(y+r/2), int(x-r/2):int(x+r/2)]        

def checkJSRTMetaDataForTP(filename, foundX, foundY, tolerance, multiplier):
    df = pd.read_csv('/data/mjhorry/Experiments/JSRT Nodule Segmentation/JSRT+NIH_Metadata.csv')
    df.set_index('study_id')
    #df = df.dropna()
    df_rec = df.loc[df['study_id']==filename]
    
    if filename.startswith('JPCLN'):
        x = multiplier * df_rec['x'].iloc[0]
        y = multiplier * df_rec['y'].iloc[0]
    else:
        #NIH is only 1024, so co-ords x 2
        x = 2 * multiplier * df_rec['x'].iloc[0]
        y = 2 * multiplier * df_rec['y'].iloc[0]
    
    trueX = abs(foundX - x) < tolerance
    
    trueY = abs(foundY - y) < tolerance
    
    if trueX and trueY:
        return True
    else:
        return False    

def train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict_images, aug_dict_masks,
        image_color_mode="grayscale",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(DIM,DIM),
        seed=1):
                        
    image_datagen = ImageDataGenerator(**aug_dict_images)
    mask_datagen = ImageDataGenerator(**aug_dict_masks)
            
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def val_generator(batch_size, train_path, image_folder, mask_folder,
        image_color_mode="grayscale",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(DIM,DIM),
        seed=1):
        
    aug_dict = dict()
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    test_gen = zip(image_generator, mask_generator)
    
    counter=0

    for (img, mask) in test_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img, mask): 
    '''
    img = img / 255    
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    '''
    img = (img - 127.0) / 127.0
    mask = (mask > 127).astype(np.float32)
    
    
    
    return (img, mask)

def dice_coef(y_true, y_pred, smooth=1): #1e-6
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)    
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
          
# SSIM Loss function
def ssim_loss(y_true, y_pred):
  return -tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 2.0)

# Jaccard loss function
def jaccard_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (keras.sum(y_true_f) + keras.sum(y_pred_f) - intersection + 1.0)

def jaccard_coef_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred) 

# gaussian noise function to reduce overfitting and train to ignore film grain
def add_noise(img):
    epsilon = 0.1
    noise = np.random.randint(0, 255, size=(DIM, DIM, 1))
    img += (epsilon * noise)
    np.clip(img, 0., 255.)
    return img
    
# function to load test images    
def test_load_image(test_file, target_size=(DIM,DIM)):
    img = cv2.imread(test_file, cv2.IMREAD_UNCHANGED)
      
    if len(img.shape) > 2:
        img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
               
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

def test_generator(test_files, target_size=(DIM,DIM)):
    for test_file in test_files:
        yield test_load_image(test_file, target_size)
        
def save_result(save_path, npyfile, test_files, fold, scoreonly=False):
    
    totaltruepositives = 0
    totalfalsepositives = 0
    
    custom_kern = np.array(
                  [[1,0],
                   [0,1]],np.uint8)
    
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)

        # morphologially close to remove noise
        # this reduced sensitivity and FPs
        if (KERN > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERN, KERN))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            
        filename, fileext = os.path.splitext(os.path.basename(result_file))
        
        # calculate the dice score for this prediction    
        savefile = filename + "_predict-" + EXPERIMENT + fileext
        
        # replace masks with predict masks -> predicted
        result_filename = os.path.join(save_path, savefile)
               
        pred_arr = img
        pred_arr = cv2.resize(pred_arr, (DIM,DIM))
        
        true_arr =  cv2.imread(test_files[i])   
        true_arr = cv2.resize(true_arr, (DIM,DIM))
        true_arr = cv2.cvtColor(true_arr, cv2.COLOR_BGR2GRAY)
                
        (score, diff) = structural_similarity(pred_arr, true_arr, full=True)
                
        # count the TPs using contour detection
        items = cv2.findContours(pred_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)  # 81 - 12.5        
        
        tp_cnts = items[0] if len(items) == 2 else items[1]
        
        # number of contours
        fp_detections = 0 
         
        # loop over the contours
        tp=False
        tp_found = False
               
        # convert img to color and apply a map       
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
               
        for c in tp_cnts:
        # compute the center of the contour     
            # fit elipses to increment fps
            if(len(c)>4):
                                
                # fp reduction scheme by shape analysis
                ellipse = cv2.fitEllipse(c)
                (center,axes,orientation) = ellipse
                majoraxis_length = max(axes)
                minoraxis_length = min(axes)
                if majoraxis_length > 0:
                    eccentricity=(np.sqrt(1-(minoraxis_length/majoraxis_length)**2))
                else:
                    eccentricity=1
                  
                area = np.pi * axes[0] * axes[1] / 4
                if(eccentricity <= 0.95):# 0.95 for JSRT):
                    #if(area > 256):# 256 for JSRT):
                    if(area > areaconstraint):# 256 for JSRT):
                     
                       # is it a tp
                       M = cv2.moments(c)
                
                       if M["m00"]:
                           cX = int(M["m10"] / M["m00"])
                           cY = int(M["m01"] / M["m00"])
                           tp = checkJSRTMetaDataForTP(os.path.basename(result_file), cX, cY, tolerance, multiplier)
                           roi = getROI(img, center[0], center[1], minoraxis_length)
                           if roi.any():
                               intensity = np.average(roi)
                           else:
                               intensity = 0    
                           #print(intensity)
                           if(intensity >= 0.5): #0.5 for JSRT 
                            if tp:
                               if not tp_found:
                                   print(os.path.basename(result_file) + " is a TP")
                               tp_found = True
                               # draw the circle
                               # print("tp median tp intensity is : " + str(intensity) + "-drawing ellipse")
                               # FOR PAPER - DON'T DRAW THE ELLIPSE
                               #cv2.ellipse(img,ellipse,(0,0,255),5)
                            else:
                               #cv2.ellipse(img,ellipse,(0,255,0),5)
                               fp_detections = fp_detections + 1
                       
        poscount = '0'        
        if(tp_found):
            poscount='1'

        cv2.imwrite(result_filename, img)
        
        # add a line to the results file
        # resultsfile = "./results/all-results-BN.csv"
        #resultsfile = "./results/all-results_10FPLIMIT.csv"
        resultsfile = "./results/all-results_FP_Reduce.csv"
        
        #print(result_file)
        
        # always use the last part of the dest path for the test
        ultimatetest = result_file.split(os.path.sep)[-5]
        
        # append test in external mode so that it's possible to distinguish results
        if externalMode:
            ultimatetest = ultimatetest + '_' + test
        
        if not scoreonly:
            f = open(resultsfile,'a')
            f.write(ultimatetest + "," + MODEL + "," + str(DIM) + "," + str(fold) + "," + str(LR) + "," + os.path.basename(result_file).split('.')[0] + "," + str(KERN) + "," + str(score) + "," + poscount + "," + str(fp_detections))
            f.write("\n")
            f.close()
        
        totaltruepositives += int(poscount)
        totalfalsepositives += int(fp_detections)
            
    return (int(totaltruepositives), int(totalfalsepositives))      
        
# Select test and validation files
def add_suffix(base_file, suffix):
    filename, fileext = os.path.splitext(base_file)
    return "%s_%s%s" % (filename, suffix, fileext)

acc_per_fold, loss_per_fold = [], []    
   
# K-fold Cross Validation model evaluation
history_all = []

# dataset is the crossval directory
for fold in range(NUM_FOLDS):
    SEGMENTATION_DIR = dataset
    # test is the split, train is the fold
    SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "split" + str(fold))
    SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "fold" + str(fold))
    SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "images")
    SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "masks")
    SEGMENTATION_TEST_MASK_DIR = os.path.join(SEGMENTATION_DIR, "split" + str(fold), 'masks')
    SEGMENTATION_TEST_PREDICT_DIR = os.path.join(SEGMENTATION_DIR, "split" + str(fold), 'predicted')
    
    #BATCH_SIZE=bs

    train_files = glob(os.path.join(SEGMENTATION_IMAGE_DIR, "*.*"))
    test_files = glob(os.path.join(SEGMENTATION_TEST_DIR, "*.*"))
    mask_files = glob(os.path.join(SEGMENTATION_MASK_DIR, "*.*"))
    test_mask_files = glob(os.path.join(SEGMENTATION_TEST_MASK_DIR, "*.*"))

    print(SEGMENTATION_TEST_DIR)

    test_files = [test_file for test_file in glob(os.path.join(SEGMENTATION_TEST_DIR, 'images', "*.*"))  if ("_mask" not in test_file  and "_dilate" not in test_file and "_predict" not in test_file)]
       
    validation_gen = val_generator(BS,
                            SEGMENTATION_TEST_DIR,
                            'images',
                            'masks', 
                            target_size=(DIM,DIM))                 

    #augrange = 0.05 #0.1

    # Prepare the U-Net model and train the model. It will take a while...
    if augmentflag:
        images_train_generator_args = dict(
                            #rotation_range=0.1,
                        #    width_shift_range=augrange,#0.05,  was commented
                        #    height_shift_range=augrange,#0.05, was commented
                        #    shear_range=augrange,#0.05, was commented
                        #    zoom_range=augrange,#0.1#0.05, was 0.15
                            #brightness_range=(0.4, 0.6),
                            horizontal_flip=True, 
                        #    vertical_flip=True, #added as test
                            fill_mode='nearest',
                            preprocessing_function = add_noise
        )
        
    else:
        print('******** augmentations off ************')
        train_generator_args = dict()
        
    train_gen = train_generator(BS,
                            SEGMENTATION_TRAIN_DIR,
                            'images',
                            'masks', 
                            images_train_generator_args,
                            images_train_generator_args,
                            target_size=(DIM,DIM))#, save_to_dir=os.path.abspath(SEGMENTATION_AUG_DIR))
    
    # automatically log the test runs
    # mlflow.autolog()
    # mlflow.set_experiment(EXPERIMENT)

    # define the base models
    if MODEL == 'RESUNET-M':
        print("USING RESUNET-M")
        model = ResUNet_M()
        modelA = ResUNet_M()
        modelB = ResUNet_M()
        modelC = ResUNet_M()
    elif MODEL == 'RESUNET-S':
        print("USING RESUNET-S")
        model = ResUNet_S()
        modelA = ResUNet_S()
        modelB = ResUNet_S()
        modelC = ResUNet_S()
    elif MODEL == 'RESUNET-L':
        print("USING RESUNET-L")
        model = ResUNet_L()
        modelA = ResUNet_L()
        modelB = ResUNet_L()
        modelC = ResUNet_L()
    else:
        # for regression testing
        print("USING UNET")
        model = unet()
        modelA = unet()
        modelB = unet()
        modelC = unet()
        
    # setup metrics and loss functions
    metrics = [dice_coef, jaccard_coef,
               'binary_accuracy']#, 
               #tf.keras.metrics.Precision(), 
               #tf.keras.metrics.Recall()]

    loss = [dice_coef_loss, 
            jaccard_coef_loss,
            'binary_crossentropy']
        
    adam = tensorflow.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=adam, loss=loss, metrics=metrics)

    # setup the callbacks
    weight_path=modelroot + "{}-{}.h5".format(EXPERIMENT + "-FOLD" + str(fold), 'epoch-{epoch:04d}')
    #weight_path_best=modelroot + "{}.h5".format('BEST-NODULESEG-' + EXPERIMENT)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                       patience=4, 
                                       verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)

    checkpointALL = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                                 save_best_only=False, mode='auto', save_freq='epoch')

    callbacks_list = [checkpointALL, reduceLROnPlat]

    #initialize random seeds so results are repeatable
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    
    if not INFERENCE:
        
        print("IN TRAINING MODE")
                                  
        res = model.fit(train_gen,
                              steps_per_epoch=len(train_files) / BS, 
                              epochs=EPOCHS, 
                              callbacks=callbacks_list,
                              validation_data = validation_gen,
                              validation_steps=len(test_files) / BS)
                    
        # plot the training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
        ax1.grid(True)
        ax2.grid(True)

        ax1.set_xlabel('Training Epochs')
        ax1.set_ylabel('Loss')

        ax2.set_xlabel('Training Epochs')
        ax2.set_ylabel('Accuracy (%)')

        ax1.plot(res.history['loss'], '-', label = 'Loss')
        ax1.plot(res.history['val_loss'], '-', label = 'Validation Loss')
        ax1.legend()
        ax2.plot(100 * np.array(res.history['binary_accuracy']), '-', 
                 label = 'Accuracy')
        ax2.plot(100 * np.array(res.history['val_binary_accuracy']), '-',
                 label = 'Validation Accuracy')
        ax2.legend();

        fig.savefig('./diagrams/' + EXPERIMENT + "-FOLD" + str(fold) +  '_ACCURACY_LOSS.png')

        # dice and jaccard
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize = (10, 5))

        ax3.grid(True)
        ax4.grid(True)

        ax3.set_xlabel('Training Epochs')
        ax3.set_ylabel('Dice Score (%)')

        ax4.set_xlabel('Training Epochs')
        ax4.set_ylabel('Jaccard Score (%)')

        ax3.plot(100 * np.array(res.history['dice_coef']), '-', 
                 label = 'Dice')
        ax3.plot(100 * np.array(res.history['val_dice_coef']), '-',
                 label = 'Validation Dice')
        ax3.legend();
        ax4.plot(100 * np.array(res.history['jaccard_coef']), '-', 
                 label = 'Jaccard')
        ax4.plot(100 * np.array(res.history['val_jaccard_coef']), '-',
                 label = 'Validation Jaccard')
        ax4.legend();

        fig.savefig('./diagrams/' + EXPERIMENT + "-FOLD" + str(fold) + '_DICE_JACCARD.png')        
        
        print("Maximum Training Accuracy = " + str(max(res.history['binary_accuracy'])))
        print("Maximum Training Dice = " + str(max(res.history['dice_coef'])))
        print("Maximum Validation Accuracy = " + str(max(res.history['val_binary_accuracy'])))
        print("Maximum Validation Dice = " + str(max(res.history['val_dice_coef'])))

        # create a report file
        reportfile = "./logs/training_report-" + EXPERIMENT + "-FOLD" + str(fold) + ".txt"
        f = open(reportfile,'a')
        f.write("\n")
        f.write("Date: " + dt_string + "\n")
        f.write("Test: " + EXPERIMENT + "\n")
        f.write("Epochs: " + str(EPOCHS) + "\n")
        f.write("Maximum Training Accuracy = " + str(max(res.history['binary_accuracy'])))
        f.write("\n")
        f.write("Maximum Training Dice = " + str(max(res.history['dice_coef'])))
        f.write("\n")
        f.write("Maximum Validation Accuracy = " + str(max(res.history['val_binary_accuracy'])))
        f.write("\n")
        f.write("Maximum Validation Dice = " + str(max(res.history['val_dice_coef'])))
        f.write("\n")

        f.close()

        # dump the training history to file in case we need it later
        historyfile = "./logs/training_history-" + EXPERIMENT + "-FOLD" + str(fold) + ".txt"
        f = open(historyfile, 'a')
    
        f.write("\n")
        f.write("Date: " + dt_string + "\n")
        f.write("Experiment: " + EXPERIMENT + "\n")
        f.write("Epochs: " + str(EPOCHS) + "\n")
        f.write(str(res.history))
        f.write("\n")
        f.close()
        
        # generate the predicted nodule masks if in test mode
    if INFERENCE:
                
        print("IN INFERENCE MODE")
        # this is the validation calculation - # search the saved models for highest Dice score
        # optimisation
        WINDOW = 3
        
        FPLIMIT = 168 #10 PER IMAGE IN FOLD for JSRT

        #FPLIMIT = 1000 #NIH

        # can experiment with differnt step lengths
        lst = [*range(1, EPOCHS,1)]

        combos = []

        alldice = [0]
        alltpos = [0]
        #allfpos = [500]
                
        bestcombo = ''
        modelfileA_best = ''
        modelfileB_best = ''
        modelfileC_best = ''
        
        resultsDict = {}

        sliding_window(lst, WINDOW)
        
        # this should be the number of windows
        
        #for i in range(15):
        # print(len(combos))
        # this should be length of files
        #for i in range(len(combos)):
        for i in range(len(test_files) + 1):  # added 1 for all TP	
            resultsDict[i] = resultsDict.setdefault(i, 1000)  # was 1000
                
        #print(resultsDict) 
                

        #DEBUG = False
        
        #BEST = False
       
        if not BEST:

            for i in combos:    
                iterdicelist = []

                modelfileA = modelroot + EXPERIMENT + "-FOLD" + str(fold) + '-epoch-{:04d}.h5'.format(i[0])
                modelfileB = modelroot + EXPERIMENT + "-FOLD" + str(fold) + '-epoch-{:04d}.h5'.format(i[1])
                modelfileC = modelroot + EXPERIMENT + "-FOLD" + str(fold) + '-epoch-{:04d}.h5'.format(i[2])
                
                model_list = []
                modelA.load_weights(modelfileA)
                model_list.append(modelA)
                
                modelB.load_weights(modelfileB)                
                model_list.append(modelB)
                
                modelC.load_weights(modelfileC)            
                model_list.append(modelC)

                model_input = tf.keras.Input(shape=(DIM, DIM, 1))
                model_outputs = [model(model_input) for model in model_list]

                ensemble_output = tf.keras.layers.Maximum()(model_outputs)

                model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

                test_gen = test_generator(test_files, target_size=(DIM,DIM))
                results = model.predict(test_gen, len(test_files), verbose=1)
                
                tpcount, fpcount = save_result(SEGMENTATION_TEST_PREDICT_DIR, results, test_mask_files, fold, True)
                
#                if resultsDict[tpcount] >= fpcount:
#                    resultsDict[tpcount] = fpcount
                
                #print(resultsDict)
                
                # commented 26/7 - causing a bug ... tp11 fp 110 should have been accepted - BUGGGG
                if(fpcount < FPLIMIT):                
                    alltpos.append(tpcount)
                    # added to avoid bug where this is first.

                print('checked ' + str(i[0]) + ':' + str(i[1]) + ':' + str(i[2]) + ' TP=' + str(tpcount) + ' FP=' + str(fpcount))          
                 
                # set an fp limit of 10 (140) and see what we get 
                if tpcount >= max(alltpos):
                    print("resultsdict for tps is " + str(resultsDict[tpcount]))
                    if resultsDict[tpcount] >= fpcount and fpcount < FPLIMIT:   #add fp limit
                        # update the dict
                        resultsDict[tpcount] = fpcount
                                           
                        print('bestfp ' + str(fpcount))  
                        print('adding new best combo')
                    
                        bestcombo = "{}:{}:{}".format(i[0],i[1],i[2])
                        print('best combo so far: ' + bestcombo)
                
                        modelfileA_best = modelfileA
                        modelfileB_best = modelfileB
                        modelfileC_best = modelfileC
                            
                for j in range(len(results)):
                    score = dice_coef(results[j], results[j]).numpy() * 100
                    iterdicelist.append(score)
                    
                        
                # calculate the average DICE
                averagedice = sum(iterdicelist) / len(iterdicelist)
                #print('checked ' + str(i[0]) + ':' + str(i[1]) + ':' + str(i[2]) + ' Average Dice is ' + str(averagedice))          
                #if averagedice < 80.0:
                #    print("adding candidate combo")
                #    alldice.append(averagedice)
 
                # ignore the leading full lung lit up predictions
                #if averagedice >= max(alldice):
                #    bestcombo = "{}:{}:{}".format(i[0],i[1],i[2])
                #    modelfileA_best = modelfileA
                #    modelfileB_best = modelfileB
                #    modelfileC_best = modelfileC
                    
            #print("Combo {}:{}:{} - Average dice is: {:.2f}".format(i[0],i[1],i[2],averagedice))  
            #print(bestcombo)
            
        else:
            # debug
            # load the dict for this fold
            loadfile = open('./optimization/' + EXPERIMENT + '.json')
            optdata = json.load(loadfile)
            
            epochA = optdata[str(fold)]["a"]
            epochB = optdata[str(fold)]["b"]
            epochC = optdata[str(fold)]["c"]

            modelfileA_best = modelroot + EXPERIMENT + "-FOLD" + str(fold) + '-epoch-' + epochA + '.h5'
            modelfileB_best = modelroot + EXPERIMENT + "-FOLD" + str(fold) + '-epoch-' + epochB + '.h5'
            modelfileC_best = modelroot + EXPERIMENT + "-FOLD" + str(fold) + '-epoch-' + epochC + '.h5'

        model_list = []

        #modelA = ResUNet_M()
        modelA.load_weights(modelfileA_best)
        #print("loading" + modelfileA_best)
        model_list.append(modelA)

        #modelB = ResUNet_M()
        modelB.load_weights(modelfileB_best)
        #print("loading" + modelfileB_best)
        model_list.append(modelB)

        #modelC = ResUNet_M()
        modelC.load_weights(modelfileC_best)
        #print("loading" + modelfileC_best)
        model_list.append(modelC)

        model_input = tf.keras.Input(shape=(DIM, DIM, 1))
        model_outputs = [model(model_input) for model in model_list]
                
        ensemble_output = tf.keras.layers.Maximum()(model_outputs)

        model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)     
    
        test_gen = test_generator(test_files, target_size=(DIM,DIM))
        results = model.predict(test_gen, len(test_files), verbose=1)
        tpcount, fpcount = save_result(SEGMENTATION_TEST_PREDICT_DIR, results, test_mask_files, fold, False)
        
        #print(str(tpcount) + '-' + str(fpcount))
        print("Best Combo {} - TP is: {:d} - FP is: {:d}".format(bestcombo, tpcount, fpcount))  
        #print(bestcombo)

