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


def load_connectivity(task_name='paintype', dff=None, conn_type='fullcorr_100',add_questionnaire=False, add_idp=False, add_conn=True, patient_grouping='simplified'):
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
        dff = check_eid(task_name, add_questionnaire=add_questionnaire, add_idp=add_idp, patient_grouping=patient_grouping)
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
    df_con = pd.DataFrame(arr_conmat)
    # add _c to distinguish conn cols
    df_conn = add_colappend(df_con, append='_c', is_conn=True)

    # adding idp or questionnaire if needed
    if (add_questionnaire or add_idp) and add_conn==True:
        df_out = pd.concat([df_conn, dff_slice.reset_index(drop=True)], axis=1)
        # print(df_out)
    elif add_conn==False:
        df_out = dff_slice.reset_index(drop=True)
    else:
        df_out = df_conn
    # print(f'total df size:{df.shape}')
    df_out['eid'] = dff_slice['eid'].values
    df_out['label'] = dff_slice['label'].values 
    return df_out

def add_colappend(df, append='_q', is_conn=False):
    """add column name appendix"""
    cols = df.columns
    if is_conn: # connectivity cols integers
        q_cols = [str(col)+append for col in cols]
    else:
        q_cols = [str(col)+append for col in cols if '-' in col]
    mapping = dict(zip(cols, q_cols))
    df_rename = df.rename(columns=mapping)
    return df_rename

def df_colappend(pain_status, add_questionnaire=True, add_idp=False, patient_grouping='grouped', visits=[2]):
    """add i/q to idp/qs cols"""
    if add_questionnaire==True and add_idp==False:
        dff_imputed = load_patient_grouped(pain_status=pain_status, questionnaire='all', idp=None, question_visits=visits, imputed=True, patient_grouping=patient_grouping)
        dff_imputed_rename = add_colappend(dff_imputed, append='_q')
    elif add_idp==True and add_questionnaire==False:
        dff_imputed = load_patient_grouped(pain_status=pain_status, questionnaire=None, idp='all', question_visits=visits, imputed=True, patient_grouping=patient_grouping)
        dff_imputed_rename = add_colappend(dff_imputed, append='_i')
    elif add_idp==True and add_questionnaire==True:
        dff_imputed = load_patient_grouped(pain_status=pain_status, questionnaire='all', idp=None, question_visits=visits, imputed=True, patient_grouping=patient_grouping)
        dff_imputed_q = add_colappend(dff_imputed, append='_q')
        dff_imputed = load_patient_grouped(pain_status=pain_status, questionnaire=None, idp='all', question_visits=visits, imputed=True, patient_grouping=patient_grouping)
        dff_imputed_i = add_colappend(dff_imputed, append='_i')
        dff_imputed_i.drop(['label','eid'], axis='columns', inplace=True)
        dff_imputed_rename = pd.concat([dff_imputed_i.reset_index(drop=True), dff_imputed_q.reset_index(drop=True)], axis=1)
    return dff_imputed_rename

def check_eid(task_name='paintype', dff_imputed=None, add_questionnaire=False, add_idp=False, patient_grouping='simplified'):
    """check eid and make labels"""
    # load settings
    visits = [2]
    impute_flag = True
    if task_name=='paintype_all':
        dff_imputed = df_colappend('all', add_questionnaire=add_questionnaire, add_idp=add_idp, patient_grouping=patient_grouping, visits=visits)
    elif task_name=='paintype_must':
        dff_imputed = df_colappend('must', add_questionnaire=add_questionnaire, add_idp=add_idp, patient_grouping=patient_grouping, visits=visits)
    elif task_name=='paintype_restricted':
        dff_imputed = df_colappend('restricted', add_questionnaire=add_questionnaire, add_idp=add_idp, patient_grouping=patient_grouping, visits=visits)
    elif task_name=='digestive':
        dff_imputed = load_digestive_data(label_type='severe', questionnaire=add_questionnaire, idp=add_idp, question_visits=visits, imputed=impute_flag)
    elif task_name=='paincontrol_all':
        dff_imputed = load_pain_matched(pain_status='all', questionnaire=add_questionnaire, idp=add_idp, question_visits=visits, imputed=impute_flag)
    elif task_name=='paincontrol_must':
        dff_imputed = load_pain_matched(pain_status='must', questionnaire=add_questionnaire, idp=add_idp, question_visits=visits, imputed=impute_flag)
    elif task_name=='paincontrol_restricted':
        dff_imputed = load_pain_matched(pain_status='restricted', questionnaire=add_questionnaire, idp=add_idp, question_visits=visits, imputed=impute_flag)
    elif task_name==None and dff_imputed is not None:
        dff_imputed = dff_imputed
    return dff_imputed

# running
if __name__=="__main__":
    # check_eid(task_name='paintype_all', dff_imputed=None, add_questionnaire=True, add_idp=True, patient_grouping='grouped')
    dff = load_connectivity(task_name='paintype_all', dff=None, conn_type='fullcorr_100',add_questionnaire=True, add_idp=True, add_conn=True, patient_grouping='grouped')
    print(dff.head())
# # running
# if __name__=="__main__":

#     clf = sys.argv[1]  # clf = ['rforest','lgb']
#     conn_type = sys.argv[2] #['fullcorr_100', 'fullcorr_25', 'parcorr_100', 'parcorr_25', 'compamp_100', 'compamp_25']

#     main_dir = f'./model_performance/output_patient/{clf}/'
#     if not os.path.exists(main_dir):
#         os.mkdir(main_dir)
#     dir_name = f'{main_dir}/{conn_type}/'
#     if not os.path.exists(dir_name):
#         os.mkdir(dir_name)

#     for conn_id in [True, False]:
#         for qs_id in [True, False]:
#             for idp_id in [True, False]:
#                 res_ls = []
#                 # for d in ['digestive', 'paintype_all', 'paintype_must', 'paintype_restricted', 'paincontrol_all', 'paincontrol_must', 'paincontrol_restricted']:
#                 for d in ['paintype_all', 'paintype_must', 'paintype_restricted']:
#                     print(f'conntype={conn_type}, conn={conn_id}, qs={qs_id}, idp={idp_id}, d={d}')
#                     # df = load_connectivity(task_name=d)
#                     df = load_connectivity(task_name=d, conn_type=conn_type, add_questionnaire=qs_id, add_idp=idp_id, add_conn=conn_id, patient_grouping='grouped')
#                     print(df.shape)
#                     # cv classification
#                     df_res = cv_classify(df, classifier=clf, cv_fold=4, scaler=True, balance=True)
#                     # save result
#                     df_res['dataset'] = d
#                     res_ls.append(df_res)
#                 # performance df
#                 df_perf = pd.concat(res_ls)
#                 # save to csv
#                 qsidp_id = '_connectivity'*conn_id+'_qs'*qs_id+'_idp'*idp_id
#                 # fname = f'./model_performance/output/{clf}/{conn_type}/all{qsidp_id}.csv'
#                 fname = dir_name + f'all{qsidp_id}.csv'
#                 print(fname)
#                 df_perf.to_csv(fname, index=None)
