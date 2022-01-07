import glob
import shutil
import os
import pandas as pd
import nrrd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import pickle
#import skimage.transform as st
from tensorflow.keras.utils import to_categorical
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image

#----------------------------------------------------------------------------------------
# val dataset
#----------------------------------------------------------------------------------------
def val_dataset(val_img_dir, crop, crop_shape, return_type1, return_type2, interp_type, 
                input_channel, output_size, output_dir, norm_type):

    x_val = pd.read_pickle(os.path.join(val_img_dir, 'x_val.p'))
    y_val = pd.read_pickle(os.path.join(val_img_dir, 'y_val.p'))
    
   ### generate CT slices and save them as numpy array
    count = 0
    list_fn = []
    list_slice_number = []
    arr = np.empty([0, 96, 96])
    
    for val_file in x_val:
        count += 1
        print(count)
        ### create consistent patient ID format
        if val_file.split('/')[-1].split('_')[0] == 'PMH':
            patient_id = 'PMH' + val_file.split('/')[-1].split('-')[1][2:5].strip()
        elif val_file.split('/')[-1].split('-')[1] == 'CHUM':
            patient_id = 'CHUM' + val_file.split('/')[-1].split('_')[1].split('-')[2].strip()
        elif val_file.split('/')[-1].split('-')[1] == 'CHUS':
            patient_id = 'CHUS' + val_file.split('/')[-1].split('_')[1].split('-')[2].strip()
        
        ### crop image or not
        if crop == True:
            ## crop image from (512, 512, ~160) to (192, 192, 110)
            img_crop = crop_image(
                nrrd_file=val_file,
                crop_shape=crop_shape,
                return_type=return_type1,
                output_dir=None
                )
            img_nrrd = img_crop
        elif crop == False:
            img_nrrd = sitk.ReadImage(val_file)

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
        ### extract img from nrrd files
        #print(data.shape)
        data = resized_arr[0:32, :, :]
        ### normalize CT values to be 0 - 1
        data[data <= -1024] = -1024
        data[data > 700] = 0
        if norm_type == 'np_interp':
            arr_img = np.interp(data, [-200, 200], [0, 1])
        elif norm_type == 'np_clip':
            arr_img = np.clip(data, a_min=-200, a_max=200)
            MAX, MIN = arr_img.max(), arr_img.min()
            arr_img = (arr_img - MIN) / (MAX - MIN)
        ### stack all the array to a 4D array
        arr = np.concatenate([arr, arr_img], 0)
        
        ### find img slices for all scans
        list_slice_number.append(data.shape[0])
        ### create patient ID and slice index for img
        for i in range(data.shape[0]):
            img = data[:, :, i]
            fn = patient_id + '_' + 'slice%s'%(f'{i:03d}')
            list_fn.append(fn)
    print('successfully create val data array!')

    ### choose the input channels
    if input_channel == 1:
        val_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        print('val arr shape:', val_arr.shape)
        if crop == True:
            fn = 'val_arr_crop.npy'
        elif crop == False:
            fn = 'val_arr.npy'
        np.save(os.path.join(val_img_dir, fn), val_arr)

    elif input_channel == 3:
        val_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        val_arr = np.transpose(val_arr, (1, 2, 3, 0))
        print('val arr shape:', val_arr.shape)
        if crop == True:
            fn = 'val_arr_new.npy'
        elif crop == False:
            fn = 'val_arr_3ch.npy'
        np.save(os.path.join(val_img_dir, fn), val_arr)
    
   ### generate labels for CT slices
    list_label = []
    list_img = []
    for label, slice_number in zip(y_val, list_slice_number):
        list_1 = [label] * slice_number
        list_label.extend(list_1)
    val_df = pd.DataFrame({'fn': list_fn, 'label': list_label})
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    print(val_df[0:100])
    ### save dataframe to pickle
    val_df.to_pickle(os.path.join(val_img_dir, 'val_df.p'))
    print('val data size:', val_df.shape[0])

    #return val_df






