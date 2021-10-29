
"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step3
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
from go_model.val_model import val_model
from utils.write_txt import write_txt
from utils.make_plots import make_plots



def run_step3_test():
    
    if run_type == 'val':
        save_dir = val_dir
    elif run_type == 'test':
        save_dir = test_dir

    loss, acc = val_model(
        model_dir=model_dir,
        pro_data_dir=pro_data_dir,
        saved_model=saved_model,
        threshold=thr_img,
        input_channel=input_channel,
        activation=activation
        ) 
    make_plots(
        run_type=run_type,
        thr_img=thr_img,
        thr_prob=thr_prob,
        thr_pos=thr_pos,
        bootstrap=bootstrap,
        pro_data_dir=pro_data_dir,
        save_dir=save_dir,
        loss=loss,
        acc=acc,
        run_model=run_model,
        saved_model=saved_model,
        epoch=epoch,
        batch_size=batch_size,
        lr=lr
        )

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
        default = 'step3_model_val.yaml',
       )
    args = parser.parse_args()
    conf_file_path = os.path.join(base_conf_file_path, args.conf)

    with open(conf_file_path) as f:
      yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

    ## yaml config
    # input-output
    log_path = os.path.normpath(yaml_conf['io']['log_path'])
    checkpoint_path = os.path.normpath(yaml_conf['io']['checkpoint_path'])
    val_path = os.path.normpath(yaml_conf['io']['val_path'])
    test_path = os.path.normpath(yaml_conf['io']['test_path'])
    model_path = os.path.normpath(yaml_conf['io']['model_path'])
    pro_data_path = os.path.normpath(yaml_conf['io']['pro_data_path'])

    patient_df_name = yaml_conf['io']['patient_df_name']
    image_df_name = yaml_conf['io']['image_df_name']
    array_name = yaml_conf['io']['array_name']
    patient_predict_name = yaml_conf['io']['patient_predict_name']
    plot_only = yaml_conf['io']['plot_only']

    # preprocessing and inference parameters
    input_channel = yaml_conf['processing']['input_channel']
    saved_model = yaml_conf['processing']['saved_model']
    random_state = yaml_conf['processing']['random_state']
    bootstrap = yaml_conf['processing']['bootstrap']

    # model config
    run_type = yaml_conf['processing']['run_type']
    run_model = yaml_conf['processing']['run_model']
    thr_img = yaml_conf['processing']['thr_img']
    thr_prob = yaml_conf['model']['thr_prob']
    thr_pos = yaml_conf['model']['thr_pos']
    epoch = yaml_conf['model']['epoch']
    activation = yaml_conf['model']['activation']
    batch_size = yaml_conf['processing']['batch_size']
    lr = yaml_conf['processing']['lr']
    epoch = yaml_conf['model']['epoch']
    activation = yaml_conf['model']['activation']
    batch_size = yaml_conf['processing']['batch_size']
    lr = yaml_conf['processing']['lr']

   
    #------------------------------------------
    # run the localization pipeline
    #------------------------------------------
    print '\n--- STEP 3 - MODEL VALIDATION ---\n'
    
    print(saved_model)
    run_step3_test()



    

    
