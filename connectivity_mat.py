"""
using connectivity for classification
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clean_questions import * 
from compare_control import *
from predict_digestive import *


def load_connectivity(task_name='paintype', add_questionnaire=False, add_idp=False):
    """load connectivity given classification task"""
    conmat_dir = '/vols/Data/pain/asdahl/uk_biobank/suyi_extend/ukbf/rfMRI/fullcorr_100/'
    eid_ls = []
    # collect eid based on task name
    dff = check_eid(task_name, add_questionnaire=add_questionnaire, add_idp=add_idp)
    for f in os.listdir(conmat_dir):
        if f.endswith('_0.txt'):
            # record eid
            eid = f.split('_')[0]
            eid_ls.append(eid)
    eids = np.array(eid_ls)
    # slice out those with conmat
    dff_slice = dff[dff['eid'].isin(eids)]
    # load data
    conmat_ls = []
    for n, r in dff_slice.iterrows():
        fname = str(int(r['eid']))+'_25751_2_0.txt'
        conmat = np.loadtxt(os.path.join(conmat_dir, fname))
        conmat_ls.append(conmat)
    # convert to df
    arr_conmat = np.array(conmat_ls, dtype=float)
    df = pd.DataFrame(arr_conmat)

    # adding idp or questionnaire if needed
    if add_questionnaire or add_idp:
        df_out = pd.concat([df, dff_slice.reset_index(drop=True)], axis=1)
        print(df_out)
    else:
        df_out = df
    df['eid'] = dff_slice['eid'].values
    df['label'] = dff_slice['label'].values    
    return df_out

def check_eid(task_name='paintype', add_questionnaire=False, add_idp=False):
    """check eid and make labels"""
    # load settings
    questionnaire = None
    idp = None
    if add_questionnaire==True and add_idp==False:
        questionnaire = 'all'
    elif add_idp==True and add_questionnaire==False:
        idp = 'all'
    elif add_idp==True and add_questionnaire==True:
        questionnaire = 'all'
        idp = 'all'
    else: # placeholder for eid
        idp = ['t2weighted']
    visits = [2]
    impute_flag = True
    if task_name=='paintype':
        dff_imputed = load_patient_grouped(questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag, patient_grouping='simplified')
    elif task_name=='digestive':
        dff_imputed = load_digestive_data(label_type='severe', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)
    elif task_name=='paincontrol':
        dff_imputed = load_patient_matched(questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)
    return dff_imputed

# running
if __name__=="__main__":
    res_ls = []

    for d in ['digestive', 'paincontrol', 'paintype']:
        print(d)
        # df = load_connectivity(task_name=d)
        df = load_connectivity(task_name=d, add_questionnaire=False, add_idp=False)
        print(df.shape)
        # cv classification
        df_res = cv_classify(df, classifier='rforest', cv_fold=4, scaler=True, balance=True)
        # save result
        df_res['dataset'] = d
        res_ls.append(df_res)
    # performance df
    df_perf = pd.concat(res_ls)
    # save to csv
    fname = f'./model_performance/output/all_connectivity.csv'
    print(fname)
    df_perf.to_csv(fname, index=None)
