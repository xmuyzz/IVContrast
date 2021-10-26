#---------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-------------------------------------------------------------------------------------------

import os
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


# ----------------------------------------------------------------------------------
# test model
# ----------------------------------------------------------------------------------
def exval_model(model_dir, pro_data_dir, threshold, run_type, activation, 
                saved_model, tuned_model):    

    ### load test data based on input channels 
    if run_type == 'exval':
        fn_arr = 'exval_arr_3ch2.npy'
        fn_df = 'exval_img_df2.csv'
        fn_pred = 'exval_img_pred.csv'
    elif run_type == 'exval2':
        fn_arr = 'rtog_0617_arr.npy'
        fn_df = 'rtog_img_df.csv'
        fn_pred = 'rtog_img_pred.csv'
    x_exval = np.load(os.path.join(pro_data_dir, fn_arr))

    ### load val labels
    df = pd.read_csv(os.path.join(pro_data_dir, fn_df))
    print(df['label'][0:10])
    print("eval size:", df.shape[0])
    y_exval = np.asarray(df['label']).astype('int').reshape((-1, 1))
    
    ### load saved model and test data
    if tuned_model == None:
        model = load_model(os.path.join(model_dir, saved_model))
    else:
        model = tuned_model
    score = model.evaluate(x_exval, y_exval)
    #score = model.evaluate(exval_gen)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('test loss:', loss)
    print('test acc:', acc)
    
    ## 'sigmoid' or 'softmax'
    if activation == 'sigmoid':
        y_pred = model.predict(x_exval)
        y_pred_class = [1 * (x[0] >= threshold) for x in y_pred]
    elif activation == 'softmax':
        y_pred_prob = model.predict(x_exval)
        y_pred = y_pred_prob[:, 1]
        y_pred_class = np.argmax(y_pred_prob, axis=1)

    ### save a dataframe for test and prediction
    ID = []
    for file in df['fn']:
        if run_type == 'exval':
            id = file.split('\\')[-1].split('_')[0].strip()
        elif run_type == 'exval2':
            id = file.split('\\')[-1].split('_s')[0].strip()
        ID.append(id)
    df['ID'] = ID
    df['y_exval'] = y_exval
    df['y_pred'] = y_pred
    df['y_pred_class'] = y_pred_class
    df_exval_pred = df[['ID', 'fn', 'label', 'y_exval', 'y_pred', 'y_pred_class']]
    df_exval_pred.to_csv(os.path.join(pro_data_dir, fn_pred))    
    print('ex val complete! Saved exval_img_pred file!!')

    return loss, acc

        



    

    
