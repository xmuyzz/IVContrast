
"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step2
  ----------------------------------------------
  ----------------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.8.5
  ----------------------------------------------
  
"""


import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob
from collections import Counter
from datetime import datetime
from time import localtime, strftime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model



def finetune_model(input_channel, pro_data_dir, model_dir, saved_model,
                   batch_size, epoch, freeze_layer, run_model): 

    """
    finetune a CNN model

    @params:
      saved_model   - required : saved CNN model for finetuning
      run_model     - required : CNN model name to be saved
      model_dir     - required : folder path to save model
      input_channel - required : model input image channel, usually 3
      freeze_layer  - required : number of layers to freeze in finetuning 
    
    """

    ### load train data
    if input_channel == 1:
        fn = 'exval_arr_1ch.npy'
    elif input_channel == 3:
        fn = 'exval_arr_3ch1.npy'
    x_train = np.load(os.path.join(pro_data_dir, fn))
    ### load train labels
    train_df = pd.read_csv(os.path.join(pro_data_dir, 'exval_img_df1.csv'))
    y_train  = np.asarray(train_df['label']).astype('int').reshape((-1, 1))
    print("sucessfully load data!")

    ## load saved model
    model = load_model(os.path.join(model_dir, saved_model))
    model.summary()

    ### freeze specific number of layers
    if freeze_layer != None:
        for layer in model.layers[0:freeze_layer]:
            layer.trainable = False
        for layer in model.layers:
            print(layer, layer.trainable)
    else:
        for layer in model.layers:
            layer.trainable = True
    model.summary()

    ### fit data into dnn models
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=None,
        verbose=1,
        callbacks=None,
        validation_split=0.3,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0
        )
    
#    ### valudation acc and loss
#    score = model.evaluate(x_val, y_val)
#    loss = np.around(score[0], 3)
#    acc  = np.around(score[1], 3)
#    print('val loss:', loss)
#    print('val acc:', acc)

    #### save final model
    model_fn = 'Tuned' + '_' + str(run_model) + '_' + \
               str(strftime('%Y_%m_%d_%H_%M_%S', localtime()))
    model.save(os.path.join(model_dir, model_fn))
    tuned_model = model
    print('fine tuning model complete!!')
    print('saved fine-tuned model as:', model_fn)

    return tuned_model, model_fn

if __name__ == '__main__':
    
    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
    model_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/model'
    input_channel = 3
    batch_size = 32
    epoch = 10
    freeze_layer = -2
    saved_model = 'ResNet_2021_07_18_06_28_40'
    
    print("fine tune model")
    FineTune_model(
        input_channel=input_channel, 
        pro_data_dir=pro_data_dir, 
        model_dir=model_dir, 
        saved_model=saved_model,
        batch_size=batch_size, 
        epoch=epoch, 
        freeze_layer=freeze_layer
        )
    


    

    
