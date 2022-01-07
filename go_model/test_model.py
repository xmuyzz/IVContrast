
"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step4
  ----------------------------------------------
  ----------------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.8.5
  ----------------------------------------------
  
"""


import os
import timeit
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import nrrd
import scipy.stats as ss
import SimpleITK as stik
import glob
from PIL import Image
from collections import Counter
import skimage.transform as st
from datetime import datetime
from time import gmtime, strftime
import pickle
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




def test_model(model_dir, pro_data_dir, saved_model, threshold, input_channel, activation):    
    
    """
    Test CNN model

    @params:
      model_dir     - required : path to load CNN model
      pro_data_dir  - required : path to folder that saves all processed data
      saved_model   - required : CNN model name from training step
      input_channel - required : 1 or 3, usually 3
      threshold     - required : threshold to decide predicted label

    """
    ### load test data based on input channels 
    start = timeit.default_timer()
    if input_channel == 1:
       fn = 'test_arr_1ch.npy'
    elif input_channel == 3:
        fn = 'test_arr_3ch.npy'
    x_test = np.load(os.path.join(pro_data_dir, fn))

    ### load val labels
    df = pd.read_csv(os.path.join(pro_data_dir, 'test_img_df.csv'))
    print(df['label'][0:10])
    y_test  = np.asarray(df['label']).astype('int').reshape((-1, 1))

    ### load saved model and test data
    model = load_model(os.path.join(model_dir, saved_model))
    y_pred = model.predict(x_test)
    stop = timeit.default_timer()
    print('Run Time:', np.around((stop - start), 0), 'secs')
    score = model.evaluate(x_test, y_test)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('test loss:', loss)
    print('test acc:', acc)
    
    ## 'sigmoid' or 'softmax'
    if activation == 'sigmoid':
        y_pred = model.predict(x_test)
        y_pred_class = [1 * (x[0] >= threshold) for x in y_pred]
    elif activation == 'softmax':
        y_pred_prob = model.predict(x_test)
        y_pred = y_pred_prob[:, 1]
        y_pred_class = np.argmax(y_pred_prob, axis=1)

    ### save a dataframe for test and prediction
    ID = []
    for file in df['fn']:
        id = file.split('\\')[-1].split('_')[0].strip()
        ID.append(id)
    df['ID'] = ID
    df['y_test'] = y_test
    df['y_pred'] = y_pred
    df['y_pred_class'] = y_pred_class
    df_test_pred = df[['ID', 'fn', 'label', 'y_test', 'y_pred', 'y_pred_class']]
    df_test_pred.to_csv(os.path.join(pro_data_dir, 'test_img_pred.csv'))    
    
    return loss, acc

        



    

    
