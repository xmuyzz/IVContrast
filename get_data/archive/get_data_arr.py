import glob
import shutil
import os
import pandas as pd
import nrrd
import re
import cv2
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import pickle
#import skimage.transform as st
from tensorflow.keras.utils import to_categorical

#----------------------------------------------------------------------------------------
# training dataset
#----------------------------------------------------------------------------------------         
def train_val_split(proj_dir, PMH_data_dir, CHUM_data_dir, CHUS_data_dir, PMH_label_csv,
                      CHUM_label_csv, CHUS_label_csv, train_img_dir, val_img_dir):

    ### PMH dataset
    PMH_label = pd.read_csv(os.path.join(proj_dir, PMH_label_csv))
    PMH_label['Contrast'] = PMH_label['Contrast'].map({'Yes': 'C', 'No': 'N'})
    labels = PMH_label['Contrast'].to_list()
    #labels = to_categorical(labels)
    #print(labels)
    fns = [fn for fn in sorted(glob.glob(PMH_data_dir + '/*nrrd'))]
    #print(fns)
    df_PMH = pd.DataFrame({'file': fns, 'label': labels})
    pd.options.display.max_colwidth = 100
    print(df_PMH)
    file = df_PMH['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    
    ### CHUM dataset
    try:
        labels = []
        CHUM_label = pd.read_csv(os.path.join(proj_dir, CHUM_label_csv))
        CHUM_label['Contrast'] = CHUM_label['Contrast'].map({'Yes': 'C', 'No': 'N'})
        for i in range(CHUM_label.shape[0]):
            file = CHUM_label['Patient ID'].iloc[i]
            scan = file.split('/')[-1].split('_')[2].strip()
            if scan == 'CT-SIM':
                labels.append(CHUM_label['Contrast'].iloc[i])
            elif scan == 'CT-PET':
                continue
        #print(labels)
        fns = []
        for fn in sorted(glob.glob(train_data_dir + '/*nrrd')):
            scan = fn.split('/')[-1].split('_')[2].strip()
            if scan == 'CT-SIM':
                fns.append(fn)
            else:
                continue
        #print(fns)
        df_CHUM = pd.DataFrame({'file': fns, 'label': labels})
        pd.options.display.max_colwidth = 100
        print(df)
        file = df['file'][0]
        data, header = nrrd.read(file)
        print(data.shape)
    except:
        print('CHUM dataset failed.')

    ### CHUS dataset
    try:
        labels = []
        CHUS_label = pd.read_csv(os.path.join(proj_dir, CHUS_label_csv))
        CHUS_label['Contrast'] = CHUS_label['Contrast'].map({'Yes': 'C', 'No': 'N'})
        labels = CHUS_label['Contrast'].to_list()
        #print(labels)
        fns = []
        for fn in sorted(glob.glob(CHUS_data_dir + '/*nrrd')):
            scan = fn.split('/')[-1].split('_')[2].strip()
            if scan == 'CT-SIMPET':
                fns.append(fn)
            else:
                continue
        #print(fns)
        df_CHUS = pd.DataFrame({'file': fns, 'label': labels})
        pd.options.display.max_colwidth = 100
        print(df_CHUS)
        file = df_CHUS['file'][0]
        data, header = nrrd.read(file)
        print(data.shape)
    except:
        print('CHUS dataset failed.')

    ### combine dataset for train-val split
    df_tot = pd.concat([df_PMH, df_CHUM, df_CHUS])
    x = df_tot['file']
    y = df_tot['label']
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
        )           
    print(x_train)
    print(y_train)
    print(x_val)
    print(y_val)
    x_train.to_pickle(os.path.join(train_img_dir, 'x_train.p'))
    y_train.to_pickle(os.path.join(train_img_dir, 'y_train.p'))
    x_val.to_pickle(os.path.join(val_img_dir, 'x_val.p'))
    y_val.to_pickle(os.path.join(val_img_dir, 'y_val.p'))

    #return x_train, x_val, y_train, y_val
    
#----------------------------------------------------------------------------------------
# training dataset
#----------------------------------------------------------------------------------------
def train_dataset(train_img_dir):

    list_slice_number = []
    list_label = []
    list_img = []
    list_fn = []
    count = 0
    x_train = pd.read_pickle(os.path.join(train_img_dir, x_train))
    y_train = pd.read_pickle(os.path.join(val_img_dir, y_train))
    try:
        for train_file in x_train:
            count += 1
            print(count)
            ### create consistent patient ID format
            if train_file.split('/')[-1].split('_')[0] == 'PMH':
                patient_id = 'PMH' + train_file.split('/')[-1].split('-')[1][2:4].strip()
            elif train_file.split('/')[-1].split('-')[1] == 'CHUM':
                patient_id = 'CHUM' + train_file.split('/')[-1].split('-')[2].strip()
            elif train_file.split('/')[-1].split('-')[1] == 'CHUS':
                patient_id = 'CHUS' + train_file.split('/')[-1].split('-')[2].strip()
            ### extract img from nrrd files 
            data, header = nrrd.read(train_file)
            data = data.transpose(2, 0, 1)
            data[data <= -1024] = -1024
            data = cv2.resize(data, dsize=(36, 64, 64), interpolation=cv2.INTER_CUBIC)
            arr = np.empty([0, 64, 64])
            arr = np.concatenate([arr, data], 0)
            ### find img slices for all scans
            list_slice_number.append(data.shape[0])
            ### create patient ID and slice index for img
            for i in range(data.shape[0]):
                img = data[:, :, i]
                fn = patient_id + '_' + 'slice%s'%(f'{i:03d}')
                #fn = patient_id + '_' + 'slice%s'%(f'{i:03d}') + '.npy'
                #img_dir = os.path.join(train_img_dir, fn)
                #matplotlib.image.imsave(img_dir, img, cmap='gray')
                #np.save(img_dir, img)
                #list_img.append(img_dir)
                list_fn.append(fn)
    except:
        print('train dataset failed.')
    else:
        print('train dataset created.')
    print(arr.shape)
    numpy.save(os.path.join(train_img_dir, 'train_arr.npy'), train_arr)
    #arr = np.repeat(arr[..., np.newaxis], 3, axis=1)
    #arr = st.resize(arr, (arr.shape[0], 224, 224, 3))
    #print(arr.shape)
    ### generate labels for CT slices
    for label, slice_number in zip(y_train, list_slice_number):
        list_1 = [label] * slice_number
        list_label.extend(list_1)
    #print(len(list_img))
    #print(len(list_label))
    ### makeing dataframe containing img directories and labels
    #train_df = pd.DataFrame({'image': list_img, 'label': list_label})
    train_df = pd.DataFrame({'fn': list_fn, 'label': list_label})
    print(train_df[0:10])
    ### save dataframe to pickle
    train_df.to_pickle(os.path.join(train_img_dir, 'train_df.p'))
    print('train data size:', train_df.shape[0])

    #return train_df

#----------------------------------------------------------------------------------------
# val dataset
#----------------------------------------------------------------------------------------
def val_dataset(val_img_dir):

    list_slice_number = []
    list_label = []
    list_img = []
    list_fn = []
    count = 0
    x_val = pd.read_pickle(os.path.join(val_img_dir, x_val))
    y_val = pd.read_pickle(os.path.join(val_img_dir, y_val))
    ### generate CT slices and save them as jpg
    try:
        for val_file in x_val:
            count += 1
            print(count)
            ### create consistent patient ID format
            if val_file.split('/')[-1].split('_')[0] == 'PMH':
                patient_id = 'PMH' + val_file.split('/')[-1].split('-')[1][2:4].strip()
            elif val_file.split('/')[-1].split('-')[1] == 'CHUM':
                val_id = 'CHUM' + val_file.split('/')[-1].split('-')[2].strip()
            elif val_file.split('/')[-1].split('-')[1] == 'CHUS':
                val_id = 'CHUS' + val_file.split('/')[-1].split('-')[2].strip()
            ### extract img from nrrd files 
            data, header = nrrd.read(train_file)
            data = data.transpose(2, 0, 1)
            data[data <= -1024] = -1024
            data = cv2.resize(data, dsize=(36, 64, 64), interpolation=cv2.INTER_CUBIC)
            arr = np.empty([0, 64, 64])
            arr = np.concatenate([arr, data], 0)
            ### find img slices for all scans
            list_slice_number.append(data.shape[0])
            ### create patient ID and slice index for img
            for i in range(data.shape[0]):
                img = data[:, :, i]
                fn = patient_id + '_' + 'slice%s'%(f'{i:03d}')
                #fn = patient_id + '_' + 'slice%s'%(f'{i:03d}') + '.npy'
                #img_dir = os.path.join(train_img_dir, fn)
                #matplotlib.image.imsave(img_dir, img, cmap='gray')
                #np.save(img_dir, img)
                #list_img.append(img_dir)
                list_fn.append(fn)
    except:
        print('val dataset failed.')
    else:
        print('val dataset created.')
    print(arr.shape)
    numpy.save(os.path.join(val_img_dir, 'val_arr.npy'), val_arr)
    #arr = np.repeat(arr[..., np.newaxis], 3, axis=1)
    #arr = st.resize(arr, (arr.shape[0], 224, 224, 3))
    #print(arr.shape)
    ### generate labels for CT slices
    for label, slice_number in zip(y_val, list_slice_number):
        list_1 = [label] * slice_number
        list_label.extend(list_1)
    #print(len(list_img))
    #print(len(list_label))
    ### makeing dataframe containing img directories and labels
    #train_df = pd.DataFrame({'image': list_img, 'label': list_label})
    val_df = pd.DataFrame({'fn': list_fn, 'label': list_label})
    print(val_df[0:10])
    ### save dataframe to pickle
    val_df.to_pickle(os.path.join(val_img_dir, 'val_df.p'))
    print('val data size:', val_df.shape[0])

    #return val_df

#----------------------------------------------------------------------------------------
# test dataset
#----------------------------------------------------------------------------------------
def test_dataset(test_label_dir, test_label_file, test_data_dir, test_img_dir):
    
    list_slice_number = []
    list_label = []
    list_img = []
    count = 0
    df_label = pd.read_csv(os.path.join(proj_dir, mdacc_label_file))
    df_label['Contrast'] = df_label['Contrast'].map({'Yes': 'C', 'No': 'N'})
    labels = df_label['Contrast'].to_list()
    #print(labels)
    fns = [fn for fn in sorted(glob.glob(test_data_dir + '/*nrrd'))]
    #print(fns)
    df = pd.DataFrame({'file': fns, 'label': labels})
    pd.options.display.max_colwidth = 100
    #print(df)
    file = df['file'][0]
    #data, header = nrrd.read(test_file)
    #print(data.shape)
    x_test = df['file']
    y_test = df['label']
    ### generate CT slices and save them as jpg
    try:
        for test_file in x_test:
            count += 1
            print(count)
            patient_id = 'MDACC' + train_file.split('/')[-1].split('-')[2][1:3].strip()
            data, header = nrrd.read(test_file)
            data = data.transpose(2, 0, 1)
            data[data <= -1024] = -1024
            data = cv2.resize(data, dsize=(36, 64, 64), interpolation=cv2.INTER_CUBIC)
            arr = np.empty([0, 64, 64])
            arr = np.concatenate([arr, data], 0)
            ### find img slices for all scans
            list_slice_number.append(data.shape[0])
            ### create patient ID and slice index for img
            for i in range(data.shape[0]):
                img = data[:, :, i]
                fn = patient_id + '_' + 'slice%s'%(f'{i:03d}')
                #fn = patient_id + '_' + 'slice%s'%(f'{i:03d}') + '.npy'
                #img_dir = os.path.join(train_img_dir, fn)
                #matplotlib.image.imsave(img_dir, img, cmap='gray')
                #np.save(img_dir, img)
                #list_img.append(img_dir)
                list_fn.append(fn)
    except:
        print('test dataset failed.')
    else:
        print('test dataset created.')
    print(arr.shape)
    numpy.save(os.path.join(test_img_dir, 'test_arr.npy'), val_arr)
    ### generate labels for CT slices
    for label, slice_number in zip(y_test, list_slice_number):
        list_1 = [label] * slice_number
        list_label.extend(list_1)
    print(len(list_img))
    test_df = pd.DataFrame({'image': list_img, 'label': list_label})
    print(test_df[0:10])
    test_df.to_pickle(os.path.join(test_img_dir, 'test_df.p'))
    print('test data size:', test_df.shape[0])

    #return test_df








