import glob
import shutil
import os
import pandas as pd
import numpy as np
import nrrd
import re
import matplotlib
import matplotlib.pyplot as plt
import pickle
from time import gmtime, strftime
from datetime import datetime
import timeit
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image
from utils.respacing import respacing
from utils.nrrd_reg import nrrd_reg_rigid_ref
from get_data.get_img_dataset import img_dataset

#----------------------------------------------------------------------------------------
# external val dataset using lung CT
#----------------------------------------------------------------------------------------
def pred_pat_dataset(crop_shape, new_spacing, fixed_img_dir, slice_range, ahmed_data_dir,
                     fns_pat_df, data_dirs, reg_dirs):
    
    ## create raw data dirs and IDs
    dfs = []
    for data_dir in data_dirs:
        IDs = []
        fns = [fn for fn in sorted(glob.glob(data_dir + '/*nrrd'))]
        for fn in fns:
            ID = fn.split('/')[-1].split('_image')[0].strip()
            IDs.append(ID)
        df = pd.DataFrame({'ID': IDs, 'file': fns})
        print('scan number:', df.shape[0])
        dfs.append(df)
    print('df number:', len(dfs))

#    ### registration, respacing, cropping   
#    for df, reg_dir in zip(dfs, reg_dirs): 
#        for fn, ID in zip(df['file'], df['ID']):
#            print(ID)
#            ## respacing      
#            img_nrrd = respacing(
#                nrrd_dir=fn,
#                interp_type='linear',
#                new_spacing=new_spacing,
#                patient_id=ID,
#                return_type='nrrd',
#                save_dir=None
#                )
#            ## registration
#            img_reg = nrrd_reg_rigid_ref(
#                img_nrrd=img_nrrd,
#                fixed_img_dir=fixed_img_dir,
#                patient_id=ID,
#                save_dir=None
#                )
#            ## crop image from (500, 500, 116) to (180, 180, 60)
#            img_crop = crop_image(
#                nrrd_file=img_reg,
#                patient_id=ID,
#                crop_shape=crop_shape,
#                return_type='nrrd',
#                save_dir=reg_dir
#                )
    
    ## create registration data dirs and IDs  
    for reg_dir, fn_pat_df, df in zip(reg_dirs, fns_pat_df, dfs): 
        fns = [fn for fn in sorted(glob.glob(reg_dir + '/*nrrd'))]
        IDs = df['ID'].to_list()
        #print('ID:', len(IDs))
        #print('file:', len(fns))
        df = pd.DataFrame({'ID': IDs, 'file': fns})
        df.to_csv(os.path.join(ahmed_data_dir, fn_pat_df))

#----------------------------------------------------------------------------------------
# image level dataset
#----------------------------------------------------------------------------------------
def pred_img_dataset(ahmed_data_dir, slice_range, fns_pat_df, fns_img_df, fns_arr):

    for fn_pat_df, fn_arr, fn_img_df in zip(fns_pat_df, fns_arr, fns_img_df):
    
        df = pd.read_csv(os.path.join(ahmed_data_dir, fn_pat_df))
        fns = df['file']
        IDs = df['ID']
        print('total patient:', len(IDs))         
        
        img_dataset(
            nrrds=fns,
            IDs=IDs,
            labels=None,
            fn_arr_1ch=None,
            fn_arr_3ch=fn_arr,
            fn_df=fn_img_df,
            slice_range=slice_range,
            return_type1='nrrd',
            return_type2='nrrd',
            interp_type='linear',
            input_channel=3,
            output_size=None,
            norm_type='np_clip',
            pro_data_dir=ahmed_data_dir,
            run_type='pred'
            )    
    
#---------------------------------------------------------------------
# generate CT slices and save them as numpy array
#---------------------------------------------------------------------
if __name__ == '__main__':

    ahmed_data_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data_pro'
    fixed_img_dir = os.path.join(ahmed_dir, 'NSCLC001.nrrd')
    downsample_size = (96, 96, 36)
    crop_shape = [192, 192, 140]
    new_spacing = [1, 1, 3]
    slice_range = range(50, 120)

    os.mkdir(ahmed_data_dir) if not os.path.isdir(ahmed_data_dir) else None

    start = timeit.default_timer()

    pred_pat_dataset(
        crop_shape=crop_shape, 
        new_spacing=new_spacing, 
        fixed_img_dir=fixed_img_dir, 
        slice_range=slice_range, 
        ahmed_data_dir=ahmed_data_dir
        )

    pred_img_dataset(
        ahmed_data_dir=ahmed_data_dir,
        slice_range=slice_range
        )

    stop = timeit.default_timer()
    print('Run Time:', np.around((stop - start)/60, 0), 'mins')
