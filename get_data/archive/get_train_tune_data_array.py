#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import nrrd
import SimpleITK as stik
import glob
from PIL import Image
from time import gmtime, strftime
from collections import Counter
import skimage.transform as st
from datetime import datetime
from time import gmtime, strftime

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
def data_preprocess(train_data_dir, train_label_dir, train_label_file):

    list_slice_number = []
    list_index_per_scan = []
    arr = np.empty([0, 512, 512])
    
    for file in sorted(glob.glob(train_data_dir + '/*.nrrd')):
        data, header = nrrd.read(file)
        data = data.transpose(2, 0 ,1)
        arr = np.concatenate([arr, data], 0)
        
        ### calculate slice numbers for labels of each image slice
        list_slice_number.append(data.shape[0])

        ### list for slice # for each scan
        list_index_per_scan.extend(list(range(data.shape[0])))

    print(arr.shape)
    arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    arr = st.resize(arr, (arr.shape[0], 224, 224, 3))
    print(arr.shape)

    ### create dataframe for reference on patient ID and slice #
    #print(list_index_per_scan)

    list_ID = []
    list_id = os.listdir(train_data_dir)
    print('total patient:', len(list_id))
    
    for ID in list_id:
        ID = ID[:13]
        list_ID.append(ID)
        
    #print(list_ID)

    ### generate labels
    df_label = pd.read_csv(os.path.join(train_label_dir, train_label_file))
    df_label['Contrast'] = df_label['Contrast'].map({'Yes': 1, 'No': 0})
    list_label_patient = df_label['Contrast'].to_list()

    list_label_slice = []
    list_ID_slice = []
    #print(list_slice_number)

    ### create labels and index for image on the slice level
    for label, ID, slice_number in zip(list_label_patient, list_ID, list_slice_number):

        list_1 = [label] * slice_number
        #print(len(list_1))
        list_2 = [ID] * slice_number
        list_label_slice.extend(list_1)
        list_ID_slice.extend(list_2)
            
    #print('patient label list:', list_label_patient)
    #print('image label list:', list_label_slice)
    print('total label:', len(list_label_slice))
    #print(list_ID_slice)

    ### create dataframe for reference on patient ID and slice
    df = pd.DataFrame(
                      {'patient_ID': list_ID_slice,
                       'slice_number': list_index_per_scan,
                       'label': list_label_slice}
                      )

    #Image.fromarray(list_file[10][:, :], 'L').show()

    #print(df)                   
    #label = np.asarray(list_label_slice)
    label = df['label']
    index = np.asarray(range(arr.shape[0]))
    patient_ID = df['patient_ID']
    data = arr
    print(data.shape)
    #print(label.shape)
    #print(indices)

    return data, label, patient_ID, index, df

# ----------------------------------------------------------------------------------
# find patient ID and image slice
# ----------------------------------------------------------------------------------
def find_patient_slice(df, data):

    ### find specific patient ID and slice # based on data index
    n = 566
    Patient_ID_slice = df.iloc[[n]]
    print('find patient ID and slice #:\n', Patient_ID_slice)

    plt.imshow(datax_val[n, :, :, 0], interpolation='nearest', cmap='gray')
    plt.show()

# ----------------------------------------------------------------------------------
# training and tuning datasets
# ----------------------------------------------------------------------------------
def train_tune_dataset(train_size, data, label, patient_ID, batch_size):

    ### train and tune dataset split based on patient unique ID
    gss = GroupShuffleSplit(n_splits=2, train_size=train_size, random_state=42)
    train_idx, val_idx = next(gss.split(data, label, groups=patient_ID))

    x_train   = data[train_idx]
    x_val     = data[val_idx]
    y_train   = label.loc[train_idx]
    y_val     = label.loc[val_idx]
    ID_train  = patient_ID[train_idx]
    ID_val    = patient_ID[val_idx]

    y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
    y_val   = np.asarray(y_val).astype('float32').reshape((-1, 1))
    
##    x_train, x_val, y_train, y_val, ID_train, ID_val = train_test_split(
##                                                      data,
##                                                      label,
##                                                      patient_ID,
##                                                      test_size=0.3, 
##                                                      stratify=label, 
##                                                      random_state=42
##                                                      )

    ### Keras data generator for image rotate, aumentation, rescale, zoom to prevent overfit
##    train_datagen = ImageDataGenerator(
##                                       rescale=1./2048,
##                                       zoom_range=0.3,
##                                       rotation_range=50,
##                                       width_shift_range=0.2,
##                                       height_shift_range=0.2,
##                                       shear_range=0.2, 
##                                       horizontal_flip=True,
##                                       fill_mode='nearest'
##                                       )
##
##    val_datagen = ImageDataGenerator(rescale=1./2048)
##    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
##    val_generator  = val_datagen.flow(x_val, y_val, batch_size=20)

    pd.set_option('display.max_columns', 500)
    #print('val label:', y_val)
    #print('val ID:', ID_val)
    #print('val data:', x_val[1])
    #print('train ID:', ID_train)
##    plt.imshow(x_val[5, :, :, 0], interpolation='nearest', cmap='gray')
##    plt.show()
##    plt.imshow(x_val[50, :, :, 0], interpolation='nearest', cmap='gray')
##    plt.show()
##    plt.imshow(x_val[100, :, :, 0], interpolation='nearest', cmap='gray')
##    plt.show()
    
    #return x_train, y_train, x_val, y_val, idx_val, train_generator, val_generator
    return x_train, y_train, x_val, y_val



# ----------------------------------------------------------------------------------
# image quality control, still working on it
# ----------------------------------------------------------------------------------
def plot_sitk(patient_id, img_sitk, qc_curated_path):
  
    """
    Quality control - for the given ID, outputs a figure containing the three center slices
    (for the CT, one for each of the main view).
    @params:
    patient_id      - required : patient ID
    img_sitk        - required : SimpleITK image (CT), resulting from sitk.ImageFileReader().Execute()
    qc_curated_path - required : output directory for the png file

    """

    png_file = os.path.join(qc_curated_path, patient_id + 'img.png')
    img_cube = sitk.GetArrayFromImage(img_sitk)

    fig, ax = plt.subplots(1, 3, figsize = (32, 8))
    ax[0].imshow(img_cube[int(img_cube.shape[0]/2), :, :], cmap = 'gray')
    ax[1].imshow(img_cube[:, int(img_cube.shape[1]/2), :], cmap = 'gray')
    ax[2].imshow(img_cube[:, :, int(img_cube.shape[2]/2)], cmap = 'gray')

    plt.savefig(png_file, bbox_inches = 'tight')
    plt.close(fig)


    

    
