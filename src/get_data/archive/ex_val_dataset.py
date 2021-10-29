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
from tensorflow.keras.utils import to_categorical
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image


#----------------------------------------------------------------------------------------
# external val dataset using lung CT
#----------------------------------------------------------------------------------------
def exval_dataset(exval_dir, NSCLC_pin_file, NSCLC_label_file, NSCLC_data_dir, crop_shape, 
                  return_type1, return_type2, interp_type, input_channel, output_size, 
                  NSCLS_reg_dir, norm_type, data_exclude):
    
    df_label = pd.read_csv(os.path.join(exval_dir, nsclc_label_file))
    df_label.dropna(subset=['ctdose_contrast'], inplace=True)
    df_id = pd.read_csv(os.path.join(exval_dir, nsclc_pin_file))
    
    ## save only IDs with labels
    IDs = []
    labels = []
    for label, pin in zip(df_label['ctdose_contrast'], df_label['pin']):
        for patient, topcoder_id in zip(df_id['patient'], df_id['topcoder_od']):
            if pin == patient:
                IDs.append(topcoder_id)
                labels.append(label)
    print('contrast scan in ex val:', labels.count(1))
    print('non-contrast scan in ex val:', labels.count(0))
    
    ## append all the data dirs
    fns = []
    list_fn = [fn for fn in sorted(glob.glob(NSCLC_data_dir + '/*nrrd'))]
    for fn in list_fn:
        ID = 'NSCLC' + fn.split('/')[-1].split('_')[2][2:5].strip()
        if ID in IDs:
            fns.append(fn)

    ## exclude scans with certain conditions
    df = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    print('total test scan:', df.shape[0])
    df_exclude = df[df['ID'].isin(data_exclude)]
    print('exclude scans:', df_exclude)
    df.drop(df[df['ID'].isin(test_exclude)].index, inplace=True)
    print('total test scans:', df.shape[0])
    ## delete repeated scans from same patients
    df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    #print(df[0:50])
   
    ### registration, respacing, cropping 
    for fn, ID in zip(df['file'], df['ID']):
        print(ID)
        ## set up save dir
        if ID[:-3] == 'NSCLC':
            save_dir = NSCLC_reg_dir
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

    ## train test split        
    fns = [fn for fn in sorted(glob.glob(NSCLC_reg_dir + '/*nrrd'))]
    label = df['label']
    ID = df['ID']
    data_exval1, data_exval2, label_xval1, label_exval2, ID_exval1, ID_exval2 = train_test_split(
        fns,
        label,
        stratify=label,
        test_size=0.5,
        random_state=42
        )
    print("train and test datasets created!")
   
    
    

    #---------------------------------------------------------------------
    # generate CT slices and save them as numpy array
    #---------------------------------------------------------------------


if __name__ == '__main__':

    exval_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval'
    NSCLC_data_dir = '/mnt/aertslab/DATA/Lung/TOPCODER/nrrd_data'
    NSCLC_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/NSCLC_data_reg'
    fixed_img_dir = os.path.join(PMH_reg_dir, 'PMH050.nrrd')
    NSCLC_label_file = 'nsclc_label.csv'
    NSCLC_pin_file = 'harvard_rt'
    interp_type = 'linear'
    norm_type = 'np_clip'
    downsample_size = (96, 96, 36)
    input_channel = 3
    return_type = 'sitk'
    crop_shape = [192, 192, 100]
    data_exclude = None

    os.mkdir(NSCLC_reg_dir) if not os.path.isdir(test_img_dir) else print("folder exists!")

    start = timeit.default_timer()

    exval_dataset(
        exval_dir=exval_dir, 
        NSCLC_pin_file=NSCLC_pin_file, 
        NSCLC_label_file=NSCLC_label_file, 
        NSCLC_data_dir=NSCLC_data_dir, 
        crop_shape=crop_shape,
        return_type1=return_type1, 
        return_type2=return_type2, 
        interp_type=interp_type, 
        input_channel=input_channel, 
        output_size=downsample_size,
        NSCLS_reg_dir=NSCLC_reg_dir, 
        norm_type=norm_type, 
        data_exclude=data_exclude
        )


    stop = timeit.default_timer()
    print('Run Time:', np.around((stop - start)/60, 0), 'mins')
