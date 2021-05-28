"""
using connectivity for classification
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clean_questions import * 
from compare_control import *
from predict_digestive import *


def load_connectivity(task_name='paintype', dff=None, conn_type='fullcorr_100',add_questionnaire=False, add_idp=False, add_conn=True):
    """load connectivity given classification task"""
    # define connectivity path
    corr_dir = '/vols/Data/pain/asdahl/uk_biobank/suyi_extend/ukbf/rfMRI/'
    corr_ls = ['fullcorr_100', 'fullcorr_25', 'parcorr_100', 'parcorr_25', 'compamp_100', 'compamp_25']
    conn_codes = {'fullcorr_100':25751,
                'fullcorr_25':25750, 
                'parcorr_100':25753, 
                'parcorr_25':25752, 
                'compamp_100':25755, 
                'compamp_25':25754}
    if conn_type in corr_ls:
        conmat_dir = corr_dir + conn_type
        conn_code = conn_codes[conn_type]
    else:
        raise ValueError('connectivity type not exist')
    
    eid_ls = []
    # collect eid based on task name
    if task_name is not None:
        dff = check_eid(task_name, add_questionnaire=add_questionnaire, add_idp=add_idp)
    else:
        dff = dff
        
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
        fname = str(int(r['eid']))+f'_{conn_code}_2_0.txt'
        conmat = np.loadtxt(os.path.join(conmat_dir, fname))
        conmat_ls.append(conmat)
    # convert to df
    arr_conmat = np.array(conmat_ls, dtype=float)
    df = pd.DataFrame(arr_conmat)

    # adding idp or questionnaire if needed
    if (add_questionnaire or add_idp) and add_conn==True:
        df_out = pd.concat([df, dff_slice.reset_index(drop=True)], axis=1)
        # print(df_out)
    elif add_conn==False:
        df_out = dff_slice.reset_index(drop=True)
    else:
        df_out = df
    # print(f'total df size:{df.shape}')
    df['eid'] = dff_slice['eid'].values
    df['label'] = dff_slice['label'].values 
    return df_out

def check_eid(task_name='paintype', dff_imputed=None, add_questionnaire=False, add_idp=False):
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
    if task_name=='paintype_all':
        dff_imputed = load_patient_grouped(pain_status='all', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag, patient_grouping='simplified')
    elif task_name=='paintype_must':
        dff_imputed = load_patient_grouped(pain_status='must', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag, patient_grouping='simplified')
    elif task_name=='paintype_restricted':
        dff_imputed = load_patient_grouped(pain_status='restricted', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag, patient_grouping='simplified')
    elif task_name=='digestive':
        dff_imputed = load_digestive_data(label_type='severe', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)
    elif task_name=='paincontrol_all':
        dff_imputed = load_pain_matched(pain_status='all', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)
    elif task_name=='paincontrol_must':
        dff_imputed = load_pain_matched(pain_status='must', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)
    elif task_name=='paincontrol_restricted':
        dff_imputed = load_pain_matched(pain_status='restricted', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)
    elif task_name==None and dff_imputed is not None:
        dff_imputed = dff_imputed
    return dff_imputed

# running
if __name__=="__main__":

    clf = sys.argv[1]  # clf = ['rforest','lgb']
    conn_type = sys.argv[2] #['fullcorr_100', 'fullcorr_25', 'parcorr_100', 'parcorr_25', 'compamp_100', 'compamp_25']

    dir_name = f'./model_performance/output/{clf}/{conn_type}/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for conn_id in [True, False]:
        for qs_id in [True, False]:
            for idp_id in [True, False]:
                res_ls = []
                for d in ['digestive', 'paintype_all', 'paintype_must', 'paintype_restricted', 'paincontrol_all', 'paincontrol_must', 'paincontrol_restricted']:
                    print(f'conntype={conn_type}, conn={conn_id}, qs={qs_id}, idp={idp_id}, d={d}')
                    # df = load_connectivity(task_name=d)
                    df = load_connectivity(task_name=d, conn_type=conn_type, add_questionnaire=qs_id, add_idp=idp_id, add_conn=conn_id)
                    print(df.shape)
                    # cv classification
                    df_res = cv_classify(df, classifier=clf, cv_fold=4, scaler=True, balance=True)
                    # save result
                    df_res['dataset'] = d
                    res_ls.append(df_res)
                # performance df
                df_perf = pd.concat(res_ls)
                # save to csv
                qsidp_id = '_connectivity'*conn_id+'_qs'*qs_id+'_idp'*idp_id
                fname = f'./model_performance/output/{clf}/{conn_type}/all{qsidp_id}.csv'
                print(fname)
                df_perf.to_csv(fname, index=None)
