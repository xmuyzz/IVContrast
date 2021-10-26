
"""
  ----------------------------------------------
  DeepContrast - run DeepContrast pipeline step5
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
from go_model.test_model import test_model
from go_model.finetune_model import finetune_model
from utils.write_txt import write_txt
from utils.make_plots import make_plots
from go_model.exval_model import exval_model
from get_data.exval_dataset import exval_pat_dataset
from get_data.exval_dataset import exval_img_dataset


#-------------------------------------------------------------
# test model
#---------------------------------------------------------------
def run_step4_exval():
  
    exval_pat_dataset(
        NSCLC_pin_file=NSCLC_pin_file, 
        NSCLC_label_file=NSCLC_label_file, 
        NSCLC_data_dir=NSCLC_data_dir, 
        crop_shape=crop_shape,
        return_type1=return_type1, 
        return_type2=return_type2, 
        interp_type=interp_type, 
        input_channel=input_channel, 
        output_size=None,
        NSCLC_reg_dir=NSCLC_reg_dir, 
        norm_type=norm_type, 
        data_exclude=data_exclude,
        new_spacing=new_spacing,
        fixed_img_dir=fixed_img_dir,
        pro_data_dir=pro_data_dir
        )

    exval_img_dataset(
        slice_range=slice_range,
        interp_type=interp_type,
        input_channel=input_channel,
        norm_type=norm_type,
        pro_data_dir=pro_data_dir,
        split=split,
        return_type1=return_type1,
        return_type2=return_type2,
        fn_arr_1ch=fn_arr_1ch,
        fn_arr_3ch=fn_arr_3ch,
        fn_df=fn_df
        )

    if fine_tune == True:
        tuned_model, model_fn = finetune_model(
            input_channel=input_channel, 
            pro_data_dir=pro_data_dir, 
            model_dir=model_dir, 
            saved_model=saved_model,
            batch_size=batch_size, 
            epoch=epoch, 
            freeze_layer=freeze_layer,
            run_model=run_model            
            )
        for run_type in ['exval2']:
            loss, acc = exval_model(
                model_dir=model_dir,
                pro_data_dir=pro_data_dir,
                threshold=thr_img,
                run_type=run_type,
                save_dir=exval_dir,
                activation=activation,
                saved_model=None,
                tuned_model=tuned_model
                )
            make_plots(
                run_type=run_type,
                thr_img=thr_img,
                thr_prob=thr_prob,
                thr_pos=thr_pos,
                bootstrap=bootstrap,
                pro_data_dir=pro_data_dir,
                save_dir=None,
                loss=loss,
                acc=acc,
                run_model=run_model,
                saved_model=model_fn,
                epoch=epoch,
                batch_size=batch_size,
                lr=lr
                )

    elif fine_tune == False:
        for run_type in ['exval2']:
            loss, acc = exval_model(
                model_dir=model_dir,
                pro_data_dir=pro_data_dir,
                threshold=thr_img,
                run_type=run_type,
                activation=activation,
                saved_model=saved_model,
                tuned_model=None
                ) 
            make_plots(
                run_type=run_type,
                thr_img=thr_img,
                thr_prob=thr_prob,
                thr_pos=thr_pos,
                bootstrap=bootstrap,
                pro_data_dir=pro_data_dir,
                save_dir=None,
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
    exval_path = os.path.normpath(yaml_conf['io']['exval_path'])
    fixed_img_path = os.path.normpath(yaml_conf['io']['fixed_img_path'])
    model_path = os.path.normpath(yaml_conf['io']['model_path'])
    pro_data_path = os.path.normpath(yaml_conf['io']['pro_data_path'])
    NSCLC_data_path = os.path.normpath(yaml_conf['io']['model_path'])
    NSCLC_reg_path = os.path.normpath(yaml_conf['io']['pro_data_path'])

    NSCLC_label_name = yaml_conf['io']['NSCLC_label_name']
    NSCLC_pin_name = yaml_conf['io']['NSCLC_pin_name']

    # preprocessing and inference parameters
    interp_type = yaml_conf['processing']['input_channel']
    norm_type = yaml_conf['processing']['saved_model']
    input_channel = yaml_conf['processing']['random_state']
    return_type1 = yaml_conf['processing']['bootstrap']
    return_type2 = yaml_conf['processing']['input_channel']
    crop_shape = yaml_conf['processing']['saved_model']
    new_spacing = yaml_conf['processing']['random_state']
    data_exclude = yaml_conf['processing']['data_exclude']
    slice_range = yaml_conf['processing']['slice_range']
    bootstrap = yaml_conf['processing']['bootstrap']
    thr_img = yaml_conf['processing']['thr_img']
    thr_prob = yaml_conf['processing']['thr_prob']
    thr_pos = yaml_conf['processing']['thr_pos']


    #------------------------------------------
    # run the localization pipeline
    #------------------------------------------
    print '\n--- STEP 4 - MODEL EX VAL ---\n'

    print(saved_model)
    run_step4_exval()

    # model config
    lr = yaml_conf['processing']['lr']
    run_model = yaml_conf['processing']['run_model']
    freeze_layer = yaml_conf['processing']['freeze_layer']
    fine_tune = yaml_conf['model']['fine_tune']
    run_type = yaml_conf['model']['run_type']
    epoch = yaml_conf['model']['epoch']
    activation = yaml_conf['model']['activation']
    split = yaml_conf['model']['split']
    saved_model = yaml_conf['model']['saved_model']


#    model_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/model'
#    exval_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval'
#    pro_data_dir = '/home/bhkann/zezhong/git_repo/IV-Contrast-CNN-Project/pro_data'
#    NSCLC_data_dir = '/mnt/aertslab/DATA/Lung/TOPCODER/nrrd_data'
#    NSCLC_reg_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/data/NSCLC_data_reg'
#    fixed_img_dir = os.path.join(exval_dir, 'NSCLC001.nrrd')
#    NSCLC_label_file = 'label_NSCLC.csv'
#    NSCLC_pin_file = 'harvard_rt.csv'
#    interp_type = 'linear'
#    norm_type = 'np_clip'
#    input_channel = 3
#    return_type1 = 'nrrd'
#    return_type2 = 'nrrd'
#    crop_shape = [192, 192, 140]
#    new_spacing = [1, 1, 3]
#    data_exclude = None
#    slice_range = range(50, 120)
#    bootstrap = 1000
#    batch_size = 32
#    lr = 1e-5
#    input_channel = 3
#    thr_img = 0.5
#    thr_prob = 0.5
#    thr_pos = 0.5
#    activation = 'sigmoid'
#    split = True
#
#    freeze_layer = None
#    epoch = 10
#    fine_tune = False
#    plots_only = True
#    run_type = 'exval2'
#    run_models = ['EfficientNetB4']
#    saved_models = ['Tuned_EfficientNetB4_2021_08_27_20_26_55']
#   
#    start = timeit.default_timer()
#    for saved_model, run_model in zip(saved_models, run_models):
#        run_step5_exval_model()
#    stop = timeit.default_timer()
#    print('Run Time:', np.around((stop - start)/60, 0), 'mins')
#

    



    

    
