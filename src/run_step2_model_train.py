
"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step1
  ----------------------------------------------
  ----------------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.6.8
  ----------------------------------------------
  
  Deep-learning-based IV contrast detection
  in CT scans - all param.s are read
  from a config file stored under "/config"
  
"""

import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from models.cnn_model import cnn_model
from models.EffNet import EffNet
from models.ResNet import ResNet
from models.Inception import Inception
from models.VGGNet import VGGNet
from models.TLNet import TLNet
from get_data.data_gen_flow import train_generator
from get_data.data_gen_flow import val_generator
from go_model.train_model import train_model
from go_model.train_model import callbacks
from utils.write_txt import write_txt

#---------------------------------------------------------------------
# train model
#---------------------------------------------------------------------

def run_step2_train():
    
    # data generator for train and val data
    train_gen = train_generator(
        pro_data_dir=pro_data_dir,
        batch_size=batch_size,
        input_channel=input_channel
        )

    x_val, y_val, val_gen = val_generator(
        pro_data_dir=pro_data_dir,
        batch_size=batch_size,
        input_channel=input_channel
        )
    
    # choose models for training
    if run_model == 'cnn':
        my_model = cnn_model(
            input_shape=input_shape,
            lr=lr,
            activation=activation,
            loss_fn=loss_fn,
            opt=opt
            )
    elif run_model == 'ResNet101V2':
        my_model = ResNet(
            resnet='ResNet101V2',  #'ResNet50V2',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation,
            )
    elif run_model == 'EffNetB4':
        my_model = EffNet(
            effnet='EffNetB4',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation
            )
    elif run_model == 'TLNet':
        my_model = TLNet(
            resnet='ResNet101V2',
            input_shape=input_shape,
            activation=activation
            )    
    elif run_model == 'Xception':
        my_model = Inception(
            #inception='InceptionV3', 
            inception='Xception',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation
            )   
    elif run_model == 'VGG16':
        my_model = VGGNet(
            VGG='VGG16',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation
            )
    print(my_model) 
    ### train model
    loss, acc, saved_model = train_model(
        model=my_model,
        run_model=run_model,
        train_gen=train_gen,
        val_gen=val_gen,
        x_val=x_val,
        y_val=y_val,
        batch_size=batch_size,
        epoch=epoch,
        model_dir=model_dir,
        log_dir=log_dir,
        opt=opt,
        loss_func=loss_func
        )

    ### save validation results to txt
    write_txt(
        run_type=run_type,
        save_dir=train_dir,
        loss=loss,
        acc=acc,
        cm1=None,
        cm2=None,
        cm3=None,
        cm_norm1=None,
        cm_norm2=None,
        cm_norm3=None,
        report1=None,
        report2=None,
        report3=None,
        prc_auc1=None,
        prc_auc2=None,
        prc_auc3=None,
        stat1=None,
        stat2=None,
        stat3=None,
        run_model=run_model,
        saved_model=saved_model,
        epoch=epoch,
        batch_size=batch_size,
        lr=lr
        )


#---------------------------------------------------------------------
# run function
#---------------------------------------------------------------------
if __name__ == '__main__':

    base_conf_file_path = 'config/'
    conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']
    parser = argparse.ArgumentParser(description = 'Run pipeline step 2 - train model.')
    parser.add_argument(
        '--conf',
        required = False,
        help = 'Specify the YAML configuration file containing the run details.' \
                + 'Defaults to 'step2_model_train.yaml'',
        choices = conf_file_list,
        default = 'step2_model_train.yaml',
       )
    args = parser.parse_args()
    conf_file_path = os.path.join(base_conf_file_path, args.conf)

    with open(conf_file_path) as f:
      yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

    ## yaml config
    # input-output
    log_path = os.path.normpath(yaml_conf['io']['log_path'])
    checkpoint_path = os.path.normpath(yaml_conf['io']['checkpoint_path'])
    train_path = os.path.normpath(yaml_conf['io']['ahmed_data_path'])
    model_path = os.path.normpath(yaml_conf['io']['model_path'])
    pro_data_path = os.path.normpath(yaml_conf['io']['pro_data_path'])

    patient_df_name = yaml_conf['io']['patient_df_name']
    image_df_name = yaml_conf['io']['image_df_name']
    array_name = yaml_conf['io']['array_name']
    patient_predict_name = yaml_conf['io']['patient_predict_name']

    # preprocessing and inference parameters
    input_shape = yaml_conf['processing']['input_shape']
    input_channel = yaml_conf['processing']['input_channel']
    train_size = yaml_conf['processing']['train_size']
    random_state = yaml_conf['processing']['random_state']
    bootstrap = yaml_conf['processing']['bootstrap']
    
    # model config
    run_type = yaml_conf['processing']['run_type']
    transfer = yaml_conf['processing']['transfer']
    freeze_layer = yaml_conf['model']['freeze_layer']
    epoch = yaml_conf['model']['epoch']
    activation = yaml_conf['model']['activation']
    batch_size = yaml_conf['processing']['batch_size']
    lr = yaml_conf['processing']['lr']
    loss_func = yaml_conf['model']['loss_func']
    opt = yaml_conf['model']['opt']
    acc_metric = yaml_conf['model']['acc_metric']


    os.mkdir(log_path) if not os.path.isdir(log_path) else None
    os.mkdir(checkpoint_path) if not os.path.isdir(checkpoint_path) else None
    os.mkdir(model_path) if not os.path.isdir(model_path) else None    

    #------------------------------------------
    # run the localization pipeline
    #------------------------------------------
    print '\n--- STEP 2 - MODEL TRAIN ---\n'
    
    print('running models:', run_model)
    run_step2_train()

    
