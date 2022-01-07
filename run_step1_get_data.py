
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
import matplotlib
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
import argparse
from get_data.get_img_dataset import get_img_dataset
from get_data.get_pat_dataset import get_pat_dataset
from get_data.respacing_reg_crop import respacing_reg_crop



def run_step1_get_data():
    
   ## respacing and registration
    respacing_reg_crop(
        PMH_data_dir=PMH_data_dir,
        CHUM_data_dir=CHUM_data_dir,
        CHUS_data_dir=CHUS_data_dir,
        MDACC_data_dir=MDACC_data_dir,
        PMH_reg_dir=PMH_reg_dir,
        CHUM_reg_dir=CHUM_reg_dir,
        CHUS_reg_dir=CHUS_reg_dir,
        MDACC_reg_dir=MDACC_reg_dir,
        fixed_img_dir=fixed_img_dir,
        interp_type=interp_type,
        new_spacing=new_spacing,
        return_type=return_type1,
        data_exclude=data_exclude,
        crop_shape=crop_shape
        )

    data_tot, label_tot, ID_tot = get_pat_dataset(
        label_dir=label_dir,
        CHUM_label_csv=CHUM_label_csv,
        CHUS_label_csv=CHUS_label_csv,
        PMH_label_csv=PMH_label_csv, 
        MDACC_label_csv=MDACC_label_csv,
        CHUM_reg_dir=CHUM_reg_dir,
        CHUS_reg_dir=CHUS_reg_dir,
        PMH_reg_dir=PMH_reg_dir,
        MDACC_reg_dir=MDACC_reg_dir,
        MDACC_data_dir=MDACC_data_dir,
        pro_data_dir=pro_data_dir
        )

    get_img_dataset(
        data_tot=data_tot, 
        ID_tot=ID_tot, 
        label_tot=label_tot, 
        slice_range=slice_range,
        return_type1=return_type1, 
        return_type2=return_type2, 
        interp_type=interp_type, 
        input_channel=input_channel, 
        output_size=output_size, 
        norm_type=norm_type,
        pro_data_dir=pro_data_dir,
        run_type=None,
        fns_arr_3ch,
        fns_df
        )

#----------------------------------------------------------------------------------------
# main function
#----------------------------------------------------------------------------------------

if __name__ == '__main__':

    base_conf_file_path = 'config/'
    conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']
    parser = argparse.ArgumentParser(description = 'Run pipeline step 1 - get data.')
    parser.add_argument(
        '--conf',
        required = False,
        help = 'Specify the YAML configuration file containing the run details.' \
                + 'Defaults to 'step1_get_data.yaml'',
        choices = conf_file_list,
        default = 'step1_get_data.yaml',
       )
    args = parser.parse_args()
    conf_file_path = os.path.join(base_conf_file_path, args.conf)

    with open(conf_file_path) as f:
      yaml_conf = yaml.load(f, Loader=yaml.FullLoader)

    # input-output
    train_img_path = os.path.normpath(yaml_conf['io']['train_img_path'])
    val_img_path = os.path.normpath(yaml_conf['io']['val_img_path'])
    test_img_path = os.path.normpath(yaml_conf['io']['test_img_path'])
    CHUM_data_path = os.path.normpath(yaml_conf['io']['CHUM_data_path'])
    CHUS_data_path = os.path.normpath(yaml_conf['io']['CHUS_data_path'])
    PMH_data_path = os.path.normpath(yaml_conf['io']['PMH_data_path'])
    MDACC_data_path = os.path.normpath(yaml_conf['io']['MDACC_img_path'])
    CHUM_reg_path = os.path.normpath(yaml_conf['io']['CHUM_reg_path'])
    CHUS_reg_path = os.path.normpath(yaml_conf['io']['CHUS_reg_path'])
    PMH_reg_path = os.path.normpath(yaml_conf['io']['PMH_reg_path'])
    MDACC_reg_path = os.path.normpath(yaml_conf['io']['MDACC_reg_path'])
    label_path = os.path.normpath(yaml_conf['io']['label_path'])
    pro_data_path = os.path.normpath(yaml_conf['io']['pro_data_path'])
    data_pro_path = os.path.normpath(yaml_conf['io']['data_pro_path'])

    CHUM_label_csv = yaml_conf['io']['CHUM_label_csv']
    CHUS_label_csv = yaml_conf['io']['CHUS_label_csv']
    PMH_label_csv = yaml_conf['io']['PMH_label_csv']
    MDACC_label_csv = yaml_conf['io']['MDACC_label_csv']
    
    # preprocessing and inference parameters
    downsample_size = yaml_conf['processing']['downsample_size']
    crop_shape = yaml_conf['processing']['crop_shape']
    interp_type = yaml_conf['processing']['interp_type']
    new_spacing = yaml_conf['processing']['new_spacing']
    data_exclude = yaml_conf['processing']['data_exclude']
    slice_range = yaml_conf['processing']['slice_range']
    output_size = yaml_conf['processing']['output_size']
    input_channel = yaml_conf['processing']['input_channel']
    return_type1 = yaml_conf['processing']['return_type1']
    return_type2 = yaml_conf['processing']['return_type2']
    norm_type = yaml_conf['processing']['norm_type']
    crop_shape = yaml_conf['processing']['crop_shape']
    data_exclude = yaml_conf['processing']['data_exclude']
    fns_arr_1ch = yaml_conf['processing']['fns_arr_1ch']
    fns_arr_3ch = yaml_conf['processing']['fns_arr_3ch']
    fns_df = yaml_conf['processing']['fns_df']

    
    #------------------------------------------
    # run the localization pipeline
    #------------------------------------------
    print '\n--- STEP 1 - GET DATA ---\n'   

    run_step1_get_data()


#    train_img_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/train_img_dir'
#    val_img_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/val_img_dir'
#    test_img_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/test_img_dir'
#    CHUM_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_CHUM'
#    CHUS_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_CHUS'
#    PMH_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_PMH'
#    MDACC_data_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_MDACC'
#    CHUM_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/CHUM_data_reg'
#    CHUS_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/CHUS_data_reg'
#    PMH_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/PMH_data_reg'
#    MDACC_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/MDACC_data_reg'
#    label_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data_pro'
#    data_pro_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data_pro'
#    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
#    fixed_img_dir = os.path.join(data_pro_dir, 'PMH050.nrrd')
#    CHUM_label_csv = 'label_CHUM.csv'
#    CHUS_label_csv = 'label_CHUS.csv'
#    PMH_label_csv = 'label_PMH.csv'
#    MDACC_label_csv = 'label_MDACC.csv'
#    interp_type = 'linear'
#    new_spacing = (1, 1, 3)
#    data_exclude = None
#    slice_range = range(17, 83)
#    output_size = (96, 96, 36)
#    input_channel = 3
#    return_type1 = 'nrrd'
#    return_type2 = 'npy'
#    norm_type = 'np_clip'  #'np_interp'          # 'np_clip'
#    crop_shape = [192, 192, 100]
#    data_exclude = None
#    fns_arr_1ch = ['train_arr_1ch.npy', 'val_arr_1ch.npy', 'test_arr_1ch.npy']
#    fns_arr_3ch = ['train_arr_3ch.npy', 'val_arr_3ch.npy', 'test_arr_3ch.npy']
#    fns_df = ['train_img_df.csv', 'val_img_df.csv', 'test_img_df.csv']
    
