 
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
from go_model.callbacks import callbacks
from utils.plot_train_curve import plot_train_curve
from tensorflow.keras.optimizers import Adam

# ----------------------------------------------------------------------------------
# train model
# ----------------------------------------------------------------------------------
def train_model(model, run_model, train_gen, val_gen, x_val, y_val, batch_size, epoch, 
                model_dir, log_dir, opt, loss_func): 


    """
    train CNN model

    @params:
      model        - required : CNN model
      run_model    - required : CNN model name to be saved
      model_dir    - required : folder path to save model
      opt          - required : optimizer (adam)
      loss_funs    - required : loss function (binary_crossentropy) 
    
    """


    ## compile model
    print('complie model')
    model.compile(
                  optimizer=opt,
                  loss=loss_func,
                  metrics=['acc']
                  )
	
    ## call back functions
    my_callbacks = callbacks(log_dir)
    
    ## fit models
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n//batch_size,
        epochs=epoch,
        validation_data=val_gen,
        #validation_data=(x_val, y_val),
        validation_steps=val_gen.n//batch_size,
        #validation_steps=y_val.shape[0]//batch_size,
        verbose=2,
        callbacks=my_callbacks,
        validation_split=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0
        )
    
    ### valudation acc and loss
    score = model.evaluate(x_val, y_val)
    loss = np.around(score[0], 3)
    acc  = np.around(score[1], 3)
    print('val loss:', loss)
    print('val acc:', acc)

    #### save final model
    saved_model = str(run_model) + '_' + str(strftime('%Y_%m_%d_%H_%M_%S', localtime()))
    model.save(os.path.join(model_dir, saved_model))
    print(saved_model)

    return loss, acc, saved_model


    


    

    
