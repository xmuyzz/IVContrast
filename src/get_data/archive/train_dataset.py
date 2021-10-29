import glob
import shutil
import os
import pandas as pd
import nrrd
import re
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image

    
#----------------------------------------------------------------------------------------
# training dataset
#----------------------------------------------------------------------------------------
def train_dataset(train_img_dir, crop, crop_shape, return_type1, return_type2, 
                  interp_type, input_channel, output_size, output_dir, norm_type):

    ### load data
    x_train = pd.read_pickle(os.path.join(train_img_dir, 'x_train.p'))
    y_train = pd.read_pickle(os.path.join(train_img_dir, 'y_train.p'))
	
    ### get image slice and save them as numpy array
    count = 0
    list_slice_number = []
    list_fn = []
    arr = np.empty([0, 96, 96])
    
    for train_file in x_train:
        count += 1
        print(count)

        ### create consistent patient ID
        if train_file.split('/')[-1].split('_')[0] == 'PMH':
            patient_id = 'PMH' + train_file.split('/')[-1].split('-')[1][2:5].strip()
        elif train_file.split('/')[-1].split('-')[1] == 'CHUM':
            patient_id = 'CHUM' + train_file.split('/')[-1].split('_')[1].split('-')[2].strip()
        elif train_file.split('/')[-1].split('-')[1] == 'CHUS':
            patient_id = 'CHUS' + train_file.split('/')[-1].split('_')[1].split('-')[2].strip()
        
        if crop == True:
            ## crop image from (512, 512, ~160) to (192, 192, 110)
            img_crop = crop_image(
                nrrd_file=train_file,
                crop_shape=crop_shape,
                return_type=return_type1,
                output_dir=None
                )
            img_nrrd = img_crop
        elif crop == False:
            img_nrrd = sitk.ReadImage(train_file)

        ### resize image to (36, 96, 96)
        ### sitk axis order (x, y, z), np axis order (z, y, x)
        resized_arr = resize_3d(
                img_nrrd=img_nrrd,
                interp_type=interp_type,
                output_size=output_size,
                patient_id=patient_id,
                return_type=return_type2,
                save_dir=None
                ) 
        #print(data.shape)
        #data = resized_arr[0:32, :, :]
        data = resized_arr[0:32, :, :]
        ### clear signals lower than -1024
        data[data <= -1024] = -1024
        ### strip skull, skull UHI = ~700
        data[data > 700] = 0
        ### normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
        if norm_type == 'np_interp':
            arr_img = np.interp(data, [-200, 200], [0, 1])
        elif norm_type == 'np_clip':
            arr_img = np.clip(data, a_min=-200, a_max=200)
            MAX, MIN = arr_img.max(), arr_img.min()
            arr_img = (arr_img - MIN) / (MAX - MIN)
        ## stack all image arrays to one array for CNN input
        arr = np.concatenate([arr, arr_img], 0)

        ### create patient ID and slice index for img
        list_slice_number.append(data.shape[0])
        for i in range(data.shape[0]):
            img = data[i, :, :]
            fn = patient_id + '_' + 'slice%s'%(f'{i:03d}')
            list_fn.append(fn)
    
    ### covert 1 channel input to 3 channel inputs for CNN
    if input_channel == 1:
        train_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        print('train_arr shape:', train_arr.shape)
        if crop == True:
            fn = 'train_arr_crop.npy'
        elif crop == False:
            fn = 'train_arr.npy'
        np.save(os.path.join(train_img_dir, fn), train_arr)
    elif input_channel == 3:
        train_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        train_arr = np.transpose(train_arr, (1, 2, 3, 0))
        print('train_arr shape:', train_arr.shape)
        if crop == True:
            fn = 'train_arr_new.npy'
        elif crop == False:
            fn = 'train_arr_3ch.npy'
        np.save(os.path.join(train_img_dir, fn), train_arr)
    
    ### generate labels for CT slices
    list_label = []
    list_img = []
    for label, slice_number in zip(y_train, list_slice_number):
        list_1 = [label] * slice_number
        list_label.extend(list_1)
    ### makeing dataframe containing img dir and labels
    train_df = pd.DataFrame({'fn': list_fn, 'label': list_label})
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    print(train_df[0:100])
    train_df.to_pickle(os.path.join(train_img_dir, 'train_df.p'))
    print('train data size:', train_df.shape[0])









