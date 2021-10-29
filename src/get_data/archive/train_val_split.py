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

#----------------------------------------------------------------------------------------
# training dataset
#----------------------------------------------------------------------------------------         
def train_val_split(proj_dir, pmh_data_dir, chum_data_dir, chus_data_dir, pmh_label_csv,
                    chum_label_csv, chus_label_csv, train_img_dir, val_img_dir, 
                    train_exclude):

    ## PMH labels
    pmh_label = pd.read_csv(os.path.join(proj_dir, pmh_label_csv))
    pmh_label['Contrast'] = pmh_label['Contrast'].map({'Yes': 1, 'No': 0})
    labels = pmh_label['Contrast'].to_list()
    #labels = to_categorical(labels)
    #print(labels)
    ## PMH data
    fns = [fn for fn in sorted(glob.glob(pmh_data_dir + '/*nrrd'))]
    #print(fns)
    IDs = []
    for fn in fns:
        ID = 'PMH' + fn.split('/')[-1].split('-')[1][2:5].strip()
        IDs.append(ID)
    df_pmh = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    pd.options.display.max_colwidth = 100
    #print(df_pmh)
    file = df_pmh['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('PMH data:', len(IDs))
    print('PMH datset created!')
    
    ## CHUM labels 
    labels = []
    chum_label = pd.read_csv(os.path.join(proj_dir, chum_label_csv))
    chum_label['Contrast'] = chum_label['Contrast'].map({'Yes': 1, 'No': 0})
    for i in range(chum_label.shape[0]):
        file = chum_label['File ID'].iloc[i]
        scan = file.split('_')[2].strip()
        if scan == 'CT-SIM':
            labels.append(chum_label['Contrast'].iloc[i])
        elif scan == 'CT-PET':
            continue
    #print('labels:', len(labels))
    fns = []
    for fn in sorted(glob.glob(chum_data_dir + '/*nrrd')):
        scan_ = fn.split('/')[-1].split('_')[2].strip()
        if scan_ == 'CT-SIM':
            fns.append(fn)
        else:
            continue
    #print('file:', len(fns))
    IDs = []
    for fn in fns:
        ID = 'CHUM' + fn.split('/')[-1].split('_')[1].split('-')[2].strip()
        IDs.append(ID)
    #print('ID:', len(IDs))
    df_chum = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    #print(df_chum)
    pd.options.display.max_colwidth = 100
    file = df_chum['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('CHUM data:', len(IDs))
    print('CHUM datset created!')

    ## CHUS labels
    labels = []
    chus_label = pd.read_csv(os.path.join(proj_dir, chus_label_csv))
    chus_label['Contrast'] = chus_label['Contrast'].map({'Yes': 1, 'No': 0})
    labels = chus_label['Contrast'].to_list()
    #print(labels)
    fns = []
    for fn in sorted(glob.glob(chus_data_dir + '/*nrrd')):
        scan = fn.split('/')[-1].split('_')[2].strip()
        if scan == 'CT-SIMPET':
            fns.append(fn)
        else:
            continue
    #print(fns)
    IDs = []
    for fn in fns:
        ID = 'CHUS' + fn.split('/')[-1].split('_')[1].split('-')[2].strip()
        IDs.append(ID)
    df_chus = pd.DataFrame({'ID': IDs, 'file': fns, 'label': labels})
    pd.options.display.max_colwidth = 100
    #print(df_chus)
    file = df_chus['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('CHUS data:', len(IDs))
    print('CHUS dataset created.')

    ## combine dataset for train-val split
    df = pd.concat([df_pmh, df_chum, df_chus], ignore_index=True)
    print(df.shape[0])
    #print(df[700:])
    df_exclude = df[df['ID'].isin(train_exclude)]
    print(df_exclude)
    df.drop(df[df['ID'].isin(train_exclude)].index, inplace=True)
    print(df.shape[0])
    #print(df[700:])

    x = df['file']
    y = df['label']
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        stratify=y,
        test_size=0.3,
        random_state=42
        )           
    #print(y_train)
    #print(x_val)
    x_train.to_pickle(os.path.join(train_img_dir, 'x_train.p'))
    x_val.to_pickle(os.path.join(val_img_dir, 'x_val.p'))

    #return x_train, x_val, y_train, y_val
    
    
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





