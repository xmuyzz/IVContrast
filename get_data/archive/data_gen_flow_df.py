#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image
import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator


# ----------------------------------------------------------------------------------
# train and val generator
# ----------------------------------------------------------------------------------
def train_val_generator(train_img_dir, val_img_dir, batch_size, target_size):
  
    train_df = pd.read_pickle(os.path.join(train_img_dir, 'train_df.p'))
    val_df   = pd.read_pickle(os.path.join(val_img_dir, 'val_df.p'))
    pd.options.display.max_colwidth = 100

    datagen = ImageDataGenerator(
        rescale=1./1024.,                #1./255.,
        validation_split=None,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
        )

##    print(train_df['image'][0])
##    image = Image.open(train_df['image'][0])
##    image.show()

   ### Train generator
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='D:/Contrast_CT_Project/train_img',     #"../input/train/",
        x_col='image',
        y_col='label',
        #subset='training',
        subset=None,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode='categorical',
        target_size=target_size,
        interpolation='nearest',
        color_mode='grayscale'
        )
    print('Train generator created')

    ### Val generator
    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=None,      #"../input/train/",
        x_col='image',
        y_col='label',
        #subset='validation',
        subset=None,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode='categorical',
        target_size=target_size,
        interpolation='nearest',
        color_mode='grayscale'
        )    
    print('Validation generator created')

    return train_generator, val_generator

# ----------------------------------------------------------------------------------
# train and val generator
# ----------------------------------------------------------------------------------
def test_gen(test_img_dir, batch_size, target_size):

    test_df = pd.read_pickle(os.path.join(test_img_dir, 'test_df.p'))
    test_datagen = ImageDataGenerator(rescale=1./1024.)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=None,     #"../input/train/",
        x_col='image',
        y_col='label',
        #has_ext=False,
        class_mode='categorical',
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        target_size=target_size,
        interpolation='nearest',
        color_mode='grayscale'
        )     
    print('Test generator created')

    return test_generator

          





    

    
