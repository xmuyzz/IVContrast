import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
from collections import Counter
from datetime import datetime
from time import localtime, strftime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def val_model(saved_model, threshold, val_img_dir, output_dir, save_dir, input_channel, crop):

    ### load train data based on input channels
    if input_channel == 1 and crop == True:
        fn = 'val_arr_crop.npy'
    elif input_channel == 3 and crop == True:
        fn = 'val_arr_3ch_crop.npy'
    elif input_channel == 1 and crop == False:
        fn = 'val_arr.npy'
    elif input_channel == 3 and crop == False:
        fn = 'val_arr_3ch.npy'
    x_val = np.load(os.path.join(val_img_dir, fn))

    ### load val labels
    val_df = pd.read_pickle(os.path.join(val_img_dir, 'val_df.p'))
    y_val  = np.asarray(val_df['label']).astype('float32').reshape((-1, 1))

    ### load saved model and test data
    model = load_model(os.path.join(output_dir, saved_model))

    ### valudation acc and loss
    score = model.evaluate(x_val, y_val)
    loss = np.around(score[0], 3)
    acc  = np.around(score[1], 3)
    print('val loss:', loss)
    print('val acc:', acc)
    y_pred = model.predict(x_val)
    y_pred = np.around(y_pred, 3)
    y_pred_class = [1 * (x[0] >= threshold) for x in y_pred]

    ### confusion matrix
    cm = confusion_matrix(y_val, y_pred_class)
    print('confusion matrix:')
    print(cm)
    cm_norm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 2)
    print(cm_norm)

    ## sklearn classification report
    report = classification_report(y_val, y_pred_class, digits=3)
    print(report)

    ### save a dataframe for val and prediction
    ID = []
    df_val_pred = pd.read_pickle(os.path.join(val_img_dir, 'val_df.p'))
    #print(df_sum.columns)
    for file in df_val_pred['fn']:
        id = file.split('\\')[-1].split('_')[0].strip()
        ID.append(id)
    df_val_pred['ID'] = ID
    df_val_pred['y_pred'] = y_pred
    df_val_pred['y_pred_class'] = y_pred_class
    df_val_pred = df_val_pred[['ID', 'fn', 'label', 'y_pred', 'y_pred_class']] 
    df_val_pred.to_pickle(os.path.join(save_dir, 'df_val_pred.p'))
    df_val_pred.to_csv(os.path.join(save_dir, 'val_pred.csv'))

    return loss, acc, cm, cm_norm, report

if __name__ == '__main__':
    
    saved_model = 'SavedModel_2021_06_16_05_43_33'
    val_img_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/val_img_dir'
    output_dir =  '/media/bhkann/HN_RES1/HN_CONTRAST/output'
    val_save_dir =  '/mnt/aertslab/USERS/Zezhong/constrast_detection/val'
    input_channel = 3
    crop = True
    threshold = 0.707
    epoch = 500
    learning_rate = 1e-5
    batch_size = 32

    val_loss, val_acc, cm, cm_norm = val_model(
        saved_model=saved_model, 
        threshold=threshold, 
        val_img_dir=val_img_dir, 
        output_dir=output_dir, 
        save_dir=val_save_dir,
        input_channel=input_channel, 
        crop=crop
        )



