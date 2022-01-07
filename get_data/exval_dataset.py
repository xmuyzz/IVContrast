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
def exval_pat_dataset(NSCLC_pin_file, NSCLC_label_file, NSCLC_data_dir, crop_shape, 
                      return_type1, return_type2, interp_type, input_channel, output_size, 
                      NSCLC_reg_dir, norm_type, data_exclude, new_spacing, fixed_img_dir,
                      pro_data_dir):
    
    df_label = pd.read_csv(os.path.join(pro_data_dir, NSCLC_label_file))
    df_label.dropna(subset=['ctdose_contrast', 'top_coder_id'], how='any', inplace=True)
    df_id = pd.read_csv(os.path.join(pro_data_dir, NSCLC_pin_file))
    
    ## create df for dir, ID and labels on patient level
    fns = []
    IDs = []
    labels = []
    list_fn = [fn for fn in sorted(glob.glob(NSCLC_data_dir + '/*nrrd'))]
    for fn in list_fn:
        ID = fn.split('/')[-1].split('_')[2][0:5].strip()
        for label, top_coder_id in zip(df_label['ctdose_contrast'], df_label['top_coder_id']):
            tc_id = top_coder_id.split('_')[2].strip()
            if tc_id == ID:
                IDs.append(ID)
                labels.append(label)
                fns.append(fn)
    ## exclude scans with certain conditions
    print("ID:", len(IDs))
    print("file:", len(fns))
    print("label:", len(labels))
    print('contrast scan in ex val:', labels.count(1))
    print('non-contrast scan in ex val:', labels.count(0))
    df = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    df.to_csv(os.path.join(pro_data_dir, 'exval_pat_df.csv'))
    print('total test scan:', df.shape[0])
    
    ## delete excluded scans and repeated scans
    if data_exclude != None:
        df_exclude = df[df['ID'].isin(data_exclude)]
        print('exclude scans:', df_exclude)
        df.drop(df[df['ID'].isin(test_exclude)].index, inplace=True)
        print('total test scans:', df.shape[0])
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    #print(df[0:50])
   
    ### registration, respacing, cropping 
    for fn, ID in zip(df['file'], df['ID']):
        print(ID)
        ## respacing      
        img_nrrd = respacing(
            nrrd_dir=fn,
            interp_type=interp_type,
            new_spacing=new_spacing,
            patient_id=ID,
            return_type=return_type1,
            save_dir=None
            )
        ## registration
        img_reg = nrrd_reg_rigid_ref(
            img_nrrd=img_nrrd,
            fixed_img_dir=fixed_img_dir,
            patient_id=ID,
            save_dir=None
            )
        ## crop image from (500, 500, 116) to (180, 180, 60)
        img_crop = crop_image(
            nrrd_file=img_reg,
            patient_id=ID,
            crop_shape=crop_shape,
            return_type='nrrd',
            save_dir=NSCLC_reg_dir
            )

#----------------------------------------------------------------------------------------
# external val dataset using lung CT
#----------------------------------------------------------------------------------------
def exval_img_dataset(slice_range, interp_type, input_channel, norm_type, pro_data_dir, split,
                      return_type1, return_type2, fn_arr_1ch, fn_arr_3ch, fn_df):
    
    df = pd.read_csv(os.path.join(pro_data_dir, 'exval_pat_df.csv'))
    fns = df['file']
    labels = df['label']
    IDs = df['ID']
    
    ## split dataset for fine-tuning model and test model
    if split == True:
        data_exval1, data_exval2, label_exval1, label_exval2, ID_exval1, ID_exval2 = train_test_split(
            fns,
            labels,
            IDs,
            stratify=labels,
            shuffle=True,
            test_size=0.2,
            random_state=42
            )
        nrrds = [data_exval1, data_exval2]
        labels = [label_exval1, label_exval2]
        IDs = [ID_exval1, ID_exval2]
        fn_arrs = ['exval_arr_3ch1.npy', 'exval_arr_3ch2.npy']
        fn_dfs = ['exval_img_df1.csv', 'exval_img_df2.csv'] 
        ## creat numpy array for image slices
        for nrrd, label, ID, fn_arr, fn_df in zip(nrrds, labels, IDs, fn_arrs, fn_dfs):
            img_dataset(
                nrrds=nrrd,
                IDs=ID,
                labels=label,
                fn_arr_1ch=None,
                fn_arr_3ch=fn_arr,
                fn_df=fn_df,
                slice_range=slice_range,
                return_type1=return_type1,
                return_type2=return_type2,
                interp_type=interp_type,
                input_channel=input_channel,
                output_size=None,
                norm_type=norm_type,
                pro_data_dir=pro_data_dir,
                run_type='exval'
                )
        print("train and test datasets created!")   
    
    ## use entire exval data to test model
    elif split == False:
        nrrds = fns
        labels = labels
        IDs = IDs    
        img_dataset(
            nrrds=nrrds,
            IDs=IDs,
            labels=labels,
            fn_arr_1ch=None,
            fn_arr_3ch='exval_arr_3ch.npy',
            fn_df='exval_img_df.csv',
            slice_range=slice_range,
            return_type1=return_type1,
            return_type2=return_type2,
            interp_type=interp_type,
            input_channel=input_channel,
            output_size=None,
            norm_type=norm_type,
            pro_data_dir=pro_data_dir,
            run_type='exval'
            )    
        print('total patient:', len(IDs))
        print("exval datasets created!")

#---------------------------------------------------------------------
# generate CT slices and save them as numpy array
#---------------------------------------------------------------------
if __name__ == '__main__':

    exval_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval'
    data_pro_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data_pro'
    NSCLC_data_dir = '/mnt/aertslab/DATA/Lung/TOPCODER/nrrd_data'
    NSCLC_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/NSCLC_data_reg'
    fixed_img_dir = os.path.join(exval_dir, 'NSCLC001.nrrd')
    NSCLC_label_file = 'NSCLC_label.csv'
    NSCLC_pin_file = 'harvard_rt.csv'
    interp_type = 'linear'
    norm_type = 'np_clip'
    downsample_size = (96, 96, 36)
    input_channel = 3
    return_type1 = 'nrrd'
    return_type2 = 'nrrd'
    crop_shape = [192, 192, 140]
    new_spacing = [1, 1, 3]
    data_exclude = None
    fn_arr_1ch='exval_arr_1ch1.npy'
    fn_arr_3ch='exval_arr_3ch1.npy'
    fn_df='exval_img_df1.csv'
    slice_range = range(50, 100)
    split = False


    os.mkdir(NSCLC_reg_dir) if not os.path.isdir(NSCLC_reg_dir) else None

    start = timeit.default_timer()

#    exval_dataset(
#        exval_dir=exval_dir, 
#        NSCLC_pin_file=NSCLC_pin_file, 
#        NSCLC_label_file=NSCLC_label_file, 
#        NSCLC_data_dir=NSCLC_data_dir, 
#        crop_shape=crop_shape,
#        return_type1=return_type1, 
#        return_type2=return_type2, 
#        interp_type=interp_type, 
#        input_channel=input_channel, 
#        output_size=downsample_size,
#        NSCLS_reg_dir=NSCLC_reg_dir, 
#        norm_type=norm_type, 
#        data_exclude=data_exclude
#        )

    exval_img_dataset(
        exval_dir=exval_dir, 
        fn_arr_1ch=fn_arr_1ch, 
        fn_arr_3ch=fn_arr_3ch, 
        fn_df=fn_df, 
        slice_range=slice_range, 
        interp_type=interp_type,
        input_channel=input_channel, 
        norm_type=norm_type, 
        data_pro_dir=data_pro_dir
        )


    stop = timeit.default_timer()
    print('Run Time:', np.around((stop - start)/60, 0), 'mins')
