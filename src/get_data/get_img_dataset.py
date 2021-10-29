
"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step1
  ----------------------------------------------
  ----------------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.8.5
  ----------------------------------------------
  
"""


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
import SimpleITK as sitk
import h5py

#----------------------------------------------------------------------------------------
# training dataset
#----------------------------------------------------------------------------------------
def img_dataset(nrrds, IDs, labels, fn_arr_1ch, fn_arr_3ch, fn_df, slice_range, return_type1, 
                return_type2, interp_type, input_channel, output_size, norm_type,
                pro_data_dir, run_type):

    """
    Generate np array as CNN inputs and associated labels

    @params:
      nrrds        - required : raw CT scan file in nrrd format
      IDs          - required : patient ID
      slice_range  - required : image slice range in z direction for cropping
      output_size  - required : whether a manual segmentation for the volume is available or not
      run_type     - required : train, val, test, or external val     
    
    """	

    ### get image slice and save them as numpy array
    count = 0
    slice_numbers = []
    list_fn = []
    arr = np.empty([0, 192, 192])
    
    for nrrd, patient_id in zip(nrrds, IDs):
        count += 1
        print(count)
 
#        ### resize image to (36, 96, 96)
#        resized_arr = resize_3d(
#                img_nrrd=img_crop,
#                interp_type=interp_type,
#                output_size=output_size,
#                patient_id=patient_id,
#                return_type=return_type2,
#                save_dir=None
#                ) 
        #print(data.shape)
        nrrd = sitk.ReadImage(nrrd, sitk.sitkFloat32)
        img_arr = sitk.GetArrayFromImage(nrrd)
        #data = img_arr[30:78, :, :]
        #data = img_arr[17:83, :, :]
        data = img_arr[slice_range, :, :]
        ### clear signals lower than -1024
        data[data <= -1024] = -1024
        ### strip skull, skull UHI = ~700
        data[data > 700] = 0
        ### normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
        if norm_type == 'np_interp':
            data = np.interp(data, [-200, 200], [0, 1])
        elif norm_type == 'np_clip':
            data = np.clip(data, a_min=-200, a_max=200)
            MAX, MIN = data.max(), data.min()
            data = (data - MIN) / (MAX - MIN)
        ## stack all image arrays to one array for CNN input
        arr = np.concatenate([arr, data], 0)

        ### create patient ID and slice index for img
        slice_numbers.append(data.shape[0])
        for i in range(data.shape[0]):
            img = data[i, :, :]
            fn = patient_id + '_' + 'slice%s'%(f'{i:03d}')
            list_fn.append(fn)
    
    ### covert 1 channel input to 3 channel inputs for CNN
    if input_channel == 1:
        img_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        print('img_arr shape:', img_arr.shape)
        np.save(os.path.join(pro_data_dir, fn_arr_1ch), img_arr)
    elif input_channel == 3:
        img_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img_arr = np.transpose(img_arr, (1, 2, 3, 0))
        print('img_arr shape:', img_arr.shape)
        np.save(os.path.join(pro_data_dir, fn_arr_3ch), img_arr)
        #fn = os.path.join(pro_data_dir, 'exval_arr_3ch.h5')
        #h5f = h5py.File(fn, 'w')
        #h5f.create_dataset('dataset_exval_arr_3ch', data=img_arr)

    ### generate labels for CT slices
    if run_type == 'pred':
        ### makeing dataframe containing img dir and labels
        img_df = pd.DataFrame({'fn': list_fn})
        img_df.to_csv(os.path.join(pro_data_dir, fn_df))
        print('data size:', img_df.shape[0])
    else:
        list_label = []
        list_img = []
        for label, slice_number in zip(labels, slice_numbers):
            list_1 = [label] * slice_number
            list_label.extend(list_1)
        ### makeing dataframe containing img dir and labels
        img_df = pd.DataFrame({'fn': list_fn, 'label': list_label})
        pd.options.display.max_columns = 100
        pd.set_option('display.max_rows', 500)
        print(img_df[0:100])
        img_df.to_csv(os.path.join(pro_data_dir, fn_df))
        print('data size:', img_df.shape[0])

#----------------------------------------------------------------------------------------
# training dataset
#----------------------------------------------------------------------------------------
def get_img_dataset(data_tot, ID_tot, label_tot, slice_range, return_type1, return_type2, 
                    interp_type, input_channel, output_size, norm_type, pro_data_dir,
                    run_type, fns_arr_3ch, fns_df):
    
    """
    data_tot = ['data_train', 'data_val', 'data_test']
    ID_tot = ['ID_train', 'ID_val', 'ID_test']
    label_tot = ['label_train', 'label_val', 'label_test']
    
    fns_arr_1ch = ['train_arr_1ch.npy', 'val_arr_1ch.npy', 'test_arr_1ch.npy']
    fns_arr_3ch = ['train_arr_3ch.npy', 'val_arr_3ch.npy', 'test_arr_3ch.npy']
    fns_df = ['train_img_df.csv', 'val_img_df.csv', 'test_img_df.csv']
    """

    for nrrds, IDs, labels, fn_arr_1ch, fn_arr_3ch, fn_df in zip(
        data_tot, ID_tot, label_tot, fns_arr_1ch, fns_arr_3ch, fns_df):
        
        img_dataset(
            nrrds=nrrds, 
            IDs=IDs,
            labels=labels,
            fn_arr_1ch=fn_arr_1ch,
            fn_arr_3ch=fn_arr_3ch,
            fn_df=fn_df,
            slice_range=slice_range, 
            return_type1=return_type1,
            return_type2=return_type2,
            interp_type=interp_type, 
            input_channel=input_channel,
            output_size=output_size,
            norm_type=norm_type,
            pro_data_dir=pro_data_dir,
            run_type=run_type
            )





