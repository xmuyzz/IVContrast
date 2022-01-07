
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
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from time import gmtime, strftime
from datetime import datetime
import timeit


#-------------------------------------------------------------------------
# process
#-------------------------------------------------------------------------
def pat_df(label_dir, label_file, cohort, data_reg_dir, MDACC_data_dir):

    ## labels
    labels = []
    df_label = pd.read_csv(os.path.join(label_dir, label_file))
    df_label['Contrast'] = df_label['Contrast'].map({'Yes': 1, 'No': 0})
    if cohort == 'CHUM':
        for file_ID, label in zip(df_label['File ID'], df_label['Contrast']):
            scan = file_ID.split('_')[2].strip()
            if scan == 'CT-SIM':
                labels.append(label)
            elif scan == 'CT-PET':
                continue
    elif cohort == 'CHUS':
        labels = df_label['Contrast'].to_list()
    elif cohort == 'PMH':
        labels = df_label['Contrast'].to_list()
    elif cohort == 'MDACC':
        fns = [fn for fn in sorted(glob.glob(MDACC_data_dir + '/*nrrd'))]
        IDs = []
        for fn in fns:
            ID = 'MDACC' + fn.split('/')[-1].split('-')[2][1:4].strip()
            IDs.append(ID)
        labels = df_label['Contrast'].to_list()
        print('MDACC label:', len(labels))
        print('MDACC ID:', len(IDs))
        ## check if labels and data are matched
        for fn in fns:
            fn = fn.split('/')[-1]
            if fn not in df_label['File ID'].to_list():
                print(fn)
        ## make df and delete duplicate patient scans
        df = pd.DataFrame({'ID': IDs, 'labels': labels})
        df.drop_duplicates(subset=['ID'], keep='last', inplace=True)
        labels = df['labels'].to_list()
        #print('MDACC label:', len(labels))
    
    ## data
    fns = [fn for fn in sorted(glob.glob(data_reg_dir + '/*nrrd'))]
    
    ## patient ID
    IDs = []
    for fn in fns:
        ID = fn.split('/')[-1].split('.')[0].strip()
        IDs.append(ID)
    ## check id and labels
    if cohort == 'MDACC':
        list1 = list(set(IDs) - set(df['ID'].to_list()))
        print(list1)
    ## create dataframe
    print('cohort:', cohort)
    print('ID:', len(IDs))
    print('file:', len(fns))
    print('label:', len(labels))
    df = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    
    return df
#----------------------------------------------------------------------------
# run loop to get train, val, test dataframe
#----------------------------------------------------------------------------
def get_pat_dataset(label_dir, CHUM_label_csv, CHUS_label_csv, PMH_label_csv, 
                    MDACC_label_csv, CHUM_reg_dir, CHUS_reg_dir, PMH_reg_dir, 
                    MDACC_reg_dir, MDACC_data_dir, pro_data_dir):

    cohorts = ['CHUM', 'CHUS', 'PMH', 'MDACC']
    label_files = [CHUM_label_csv, CHUS_label_csv, PMH_label_csv, MDACC_label_csv]
    data_reg_dirs = [CHUM_reg_dir, CHUS_reg_dir, PMH_reg_dir, MDACC_reg_dir]
    df_tot = []
    for cohort, label_file, data_reg_dir in zip(cohorts, label_files, data_reg_dirs):
        df = pat_df(
            label_dir=label_dir,
            label_file=label_file,
            cohort=cohort,
            data_reg_dir=data_reg_dir,
            MDACC_data_dir=MDACC_data_dir
            )
        df_tot.append(df) 

    ## get df for different cohorts
    df_CHUM  = df_tot[0]
    df_CHUS  = df_tot[1]
    df_PMH   = df_tot[2]
    df_MDACC = df_tot[3]

    ## train-val split
    df = pd.concat([df_PMH, df_CHUM, df_CHUS], ignore_index=True)
    data = df['file']
    label = df['label']
    ID = df['ID']
    data_train, data_val, label_train, label_val, ID_train, ID_val = train_test_split(
        data,
        label,
        ID,
        stratify=label,
        test_size=0.3,
        random_state=42
        )

    ## test patient data
    data_test = df_MDACC['file']
    label_test = df_MDACC['label']
    ID_test = df_MDACC['ID']
  
    ## save train, val, test df on patient level
    train_pat_df = pd.DataFrame({'ID': ID_train, 'file': data_train, 'label': label_train})
    val_pat_df = pd.DataFrame({'ID': ID_val, 'file': data_val, 'label': label_val})
    test_pat_df = pd.DataFrame({'ID': ID_test, 'file': data_test, 'label': label_test})
    train_pat_df.to_csv(os.path.join(pro_data_dir, 'train_pat_df.csv'))
    val_pat_df.to_csv(os.path.join(pro_data_dir, 'val_pat_df.csv'))
    test_pat_df.to_csv(os.path.join(pro_data_dir, 'test_pat_df.csv'))

    ## save data, label and ID as list
    data_tot = [data_train, data_val, data_test]
    label_tot = [label_train, label_val, label_test]
    ID_tot = [ID_train, ID_val, ID_test]

    return data_tot, label_tot, ID_tot

#--------------------------------------------------------------------------
# run main function
#--------------------------------------------------------------------------
if __name__ == '__main__':

    proj_dir        = '/media/bhkann/HN_RES1/HN_CONTRAST/label_files'
    output_dir      = '/media/bhkann/HN_RES1/HN_CONTRAST/output'
    log_dir         = '/media/bhkann/HN_RES1/HN_CONTRAST/log'
    checkpoint_dir  = '/media/bhkann/HN_RES1/HN_CONTRAST/log'
    chum_data_dir   = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_CHUM'
    chus_data_dir   = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_CHUS'
    pmh_data_dir    = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_PMH'
    mdacc_data_dir  = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_mdacc'
    train_img_dir   = '/media/bhkann/HN_RES1/HN_CONTRAST/train_img_dir'
    val_img_dir     = '/media/bhkann/HN_RES1/HN_CONTRAST/val_img_dir'
    test_img_dir    = '/media/bhkann/HN_RES1/HN_CONTRAST/test_img_dir'
    chum_label_csv  = 'label_CHUM.csv'
    chus_label_csv  = 'label_CHUS.csv'
    pmh_label_csv   = 'label_PMH.csv'
    mdacc_label_csv = 'label_MDACC.csv'

    interp_type = 'linear'
    norm_type = 'np_clip'
    downsample_size = (96, 96, 36)
    input_channel = 3
    return_type = 'sitk'
    crop_shape = [192, 192, 110]
    crop = True
    train_exclude = ['CHUS101', 'CHUS102']

    os.mkdir(proj_dir)      if not os.path.isdir(proj_dir)      else None
    os.mkdir(output_dir)    if not os.path.isdir(output_dir)    else None
    os.mkdir(log_dir)       if not os.path.isdir(log_dir)       else None
    os.mkdir(train_img_dir) if not os.path.isdir(train_img_dir) else None
    os.mkdir(val_img_dir)   if not os.path.isdir(val_img_dir)   else None
    os.mkdir(test_img_dir)  if not os.path.isdir(test_img_dir)  else None

    start = timeit.default_timer()

    train_val_split(
        proj_dir=proj_dir,
        pmh_data_dir=pmh_data_dir,
        chum_data_dir=chum_data_dir,
        chus_data_dir=chus_data_dir,
        pmh_label_csv=pmh_label_csv,
        chum_label_csv=chum_label_csv,
        chus_label_csv=chus_label_csv,
        train_img_dir=train_img_dir,
        val_img_dir=val_img_dir,
        train_exclude=train_exclude
        )

    stop = timeit.default_timer()
    print('Run Time:', np.around((stop - start), 2), 'seconds')





