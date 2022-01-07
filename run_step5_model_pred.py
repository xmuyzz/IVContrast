
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
import globi
import yaml
import argparse
from time import gmtime, strftime
from datetime import datetime
import timeit
from get_data.pred_dataset import pred_pat_dataset
from get_data.pred_dataset import pred_img_dataset
from go_model.pred_model import pred_model



def run_step6_pred():

    pred_pat_dataset(
        crop_shape=crop_shape,
        new_spacing=new_spacing,
        fixed_img_dir=fixed_img_dir,
        slice_range=slice_range,
        pred_data_dir=pred_data_dir,
        fns_pat_df=fns_pat_df,
        data_dirs=data_dirs,
        reg_dirs=reg_dirs
        )

    pred_img_dataset(
        pred_data_dir=pred_data_dir,
        slice_range=slice_range,
        fns_pat_df=fns_pat_df, 
        fns_img_df=fns_img_df, 
        fns_arr=fns_arr
        )

    pred_model(
        model_dir=model_dir,
        pred_data_dir=pred_data_dir,
        saved_model=saved_model,
        thr_img=thr_img,
        thr_prob=thr_prob,
        fns_pat_pred=fns_pat_pred,
        fns_img_pred=fns_img_pred,
        fns_arr=fns_arr,
        fns_img_df=fns_img_df
        )


if __name__ == '__main__':

    base_conf_file_path = 'config/'
    conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']
    parser = argparse.ArgumentParser(description = 'Run pipeline step 3 - model prediction.')
    parser.add_argument(
        '--conf',
        required = False,
        help = 'Specify the YAML configuration file containing the run details.' \
                + 'Defaults to 'step6_pred.yaml'',
        choices = conf_file_list,
        default = 'step6_pred.yaml',
       )
    args = parser.parse_args()
    conf_file_path = os.path.join(base_conf_file_path, args.conf)

    with open(conf_file_path) as f:
      yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

    # input-output
    raw_data_path = os.path.normpath(yaml_conf['io']['raw_data_path'])
    reg_data_path = os.path.normpath(yaml_conf['io']['reg_data_path'])
    ahmed_data_path = os.path.normpath(yaml_conf['io']['ahmed_data_path'])
    
    patient_df_name = yaml_conf['io']['patient_df_name']
    image_df_name = yaml_conf['io']['image_df_name']
    array_name = yaml_conf['io']['array_name']
    patient_predict_name = yaml_conf['io']['patient_predict_name']

    # preprocessing and inference parameters
    downsample_size = yaml_conf['processing']['downsample_size']
    crop_shape = yaml_conf['processing']['crop_shape']
    new_spacing = yaml_conf['processing']['new_spacing']
    slice_range = yaml_conf['processing']['slice_range']
    thr_img = yaml_conf['processing']['thr_img']
    thr_prob = yaml_conf['processing']['thr_prob']

    # model config
    saved_model_name = yaml_conf['model']['saved_model_name']


    os.mkdir(pred_data_dir) if not os.path.isdir(pred_data_path) else None
    os.mkdir(raw_data_path) if not os.path.isdir(raw_data_path) else None
    os.mkdir(reg_data_path) if not os.path.isdir(reg_data_path) else None
    
    #------------------------------------------
    # run the localization pipeline
    #------------------------------------------
    print '\n--- STEP 6 - MODEL PREDICTION ---\n'

    run_step6_pred()
    
 






#    # raw data dirs
#    maastro_dir = '/mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTATION_DATA/maastro/0_image_raw'
#    ## registration data dirs
#    maastro_reg = '/mnt/aertslab/USERS/Zezhong/contrast_detection/ahmed_data/maastro'
#    ahmed_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/ahmed_data'
#    model_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/model'
#    saved_model = 'FineTuned_model_2021_07_27_16_44_40'
#    fixed_img_dir = os.path.join(ahmed_data_dir, 'NSCLC001.nrrd')
#    downsample_size = (96, 96, 36)
#    crop_shape = [192, 192, 140]
#    new_spacing = [1, 1, 3]
#    slice_range = range(50, 120)
#    thr_img = 0.5
#    thr_prob = 0.5
#
#    ## patient df
#    fns_pat_df = 'maastro_pat.csv'
#    ## image df
#    fns_img_df = 'maastro_img.csv'
#    ## image numpy array
#    fns_arr = 'maastro_arr.npy'
#    ## patient level prediction df
#    fns_pat_pred = 'maastro_pat_pred.csv'
#    ## patient level prediction df
#    fns_img_pred = 'maastro_img_pred.csv'
