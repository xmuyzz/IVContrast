import glob
import shutil
import os
import pandas as pd
import nrrd
import re
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from time import gmtime, strftime
from datetime import datetime
import timeit
from utils.respacing import respacing
from utils.nrrd_reg import nrrd_reg_rigid_ref



#----------------------------------------------------------------------------------
# respacing and registration
#----------------------------------------------------------------------------------

def data_respacing_reg(pmh_data_dir, chum_data_dir, chus_data_dir, mdacc_data_dir,
                       pmh_reg_dir, chum_reg_dir, chus_reg_dir, madcc_reg_dir,
                       fixed_img_dir, interp_type, new_spacing, return_type):

    ## PMH data
    fns = [fn for fn in sorted(glob.glob(PMH_data_dir + '/*nrrd'))]
    #print(fns)
    IDs = []
    for fn in fns:
        ID = 'PMH' + fn.split('/')[-1].split('-')[1][2:5].strip()
        IDs.append(ID)
    df_PMH = pd.DataFrame({'ID': IDs, 'file': fns})
    pd.options.display.max_colwidth = 100
    #print(df_pmh)
    file = df_PMH['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('PMH data:', len(IDs))
    
    ## CHUM data
    fns = []
    for fn in sorted(glob.glob(CHUM_data_dir + '/*nrrd')):
        scan_ = fn.split('/')[-1].split('_')[2].strip()
        if scan_ == 'CT-SIM':
            fns.append(fn)
        else:
            continue
    #print('file:', len(fns))
    IDs = []
    for fn in fns:
        ID = 'CHUM' + fn.split('/')[-1].split('_')[1].split('-')[2].strip()
        IDs.append(ID)
    #print('ID:', len(IDs))
    df_CHUM = pd.DataFrame({'ID': IDs, 'file': fns})
    #print(df_chum)
    pd.options.display.max_colwidth = 100
    file = df_CHUM['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('CHUM data:', len(IDs))

    ## CHUS dataset
    fns = [fn for fn in sorted(glob.glob(CHUS_data_dir + '/*nrrd'))]
    IDs = []
    for fn in fns:
        ID = 'CHUS' + fn.split('/')[-1].split('_')[1].split('-')[2].strip()
        IDs.append(ID)
    df_CHUS = pd.DataFrame({'ID': IDs, 'file': fns})
    pd.options.display.max_colwidth = 100
    #print(df_chus)
    file = df_CHUS['file'][0]
    data, header = nrrd.read(file)
    print(data.shape)
    print('CHUS data:', len(IDs))
    print('CHUS dataset created.')

    ## MDACC dataset
    fns = [fn for fn in sorted(glob.glob(MDACC_data_dir + '/*nrrd'))]
    #print(fns)
    IDs = []
    for fn in fns:
        ID = 'MDACC' + fn.split('/')[-1].split('-')[2][1:4].strip()
        IDs.append(ID)
    df_MDACC = pd.DataFrame({'ID': IDs, 'file': fns})
    print('total test scan:', df.shape[0])


    ## combine dataset for train-val split
    df = pd.concat([df_PMH, df_CHUM, df_CHUS, df_MDACC], ignore_index=True)
    print(df.shape[0])
    #print(df[700:])
    ## exclude data with certain conditions
    if data_exclude != None:
        df_exclude = df[df['ID'].isin(data_exclude)]
        print(df_exclude)
        df.drop(df[df['ID'].isin(data_exclude)].index, inplace=True)
        print(df.shape[0])
        #print(df[700:])

    for fn, ID in zip(df['fns'], df['ID']):
        if ID[:-2] == 'PMH':
            save_dir = PMH_reg_dir
        elif ID[:-2] == 'CHUM':
            save_dir = CHUM_reg_dir
        elif ID[:-2] == 'CHUS':
            save_dir = CHUS_reg_dir
        elif ID[:-2] == 'MDACC':
            save_dir = MDACC_reg_dir

        img_nrrd = respacing(
            nrrd_dir=fn,
            interp_type=interp_type,
            new_spacing=new_spacing,
            patient_id=ID,
            return_type=return_type,
            save_dir=''
            )

        nrrd_reg_rigid_ref(
            img_nrrd=img_nrrd, 
            fixed_img_dir=fixed_img_dir, 
            patient_id=ID, 
            save_dir=save_dir
            )

        
#-----------------------------------------------------------------------
# main function
#----------------------------------------------------------------------
if __name__ == '__main__':

    CHUM_data_dir   = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_CHUM'
    CHUS_data_dir   = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_CHUS'
    PMH_data_dir    = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_PMH'
    MDACC_data_dir  = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_MDACC'
    CHUM_reg_dir    = '/media/bhkann/HN_RES1/HN_CONTRAST/CHUM_data_reg'
    CHUS_reg_dir    = '/media/bhkann/HN_RES1/HN_CONTRAST/CHUS_data_reg'
    PMH_reg_dir     = '/media/bhkann/HN_RES1/HN_CONTRAST/PMH_data_reg'
    MDACC_reg_dir   = '/media/bhkann/HN_RES1/HN_CONTRAST/MDACC_data_reg'
    fixed_img_dir   = os.path.join(PMH_data_dir, 'PMH050.nrrd')

    interp_type = 'linear'
    new_spacing = (1, 1, 3)
    data_exclude = None
    return_type = 'nrrd'

    os.mkdir(CHUM_reg_dir)  if not os.path.isdir(CHUM_reg_dir)  else None
    os.mkdir(CHUS_reg_dir)  if not os.path.isdir(CHUS_reg_dir)  else None
    os.mkdir(PMH_reg_dir)   if not os.path.isdir(PMH_reg_dir)   else None
    os.mkdir(MDACC_reg_dir) if not os.path.isdir(MDACC_reg_dir) else None
    
    start = timeit.default_timer()

    data_respacing_reg(
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
        return_type=return_type
        )

    stop = timeit.default_timer()
    print('Run Time:', np.around((stop - start)/60, 2), 'seconds')
