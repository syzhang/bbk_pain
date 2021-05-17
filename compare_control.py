"""
compare control and patients
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clean_questions import * 

def extract_control(df_control, df_disease, save_csv=True, save_name='matched_control'):
    """extract age/sex matched control to disease group"""
    # use age/sex only
    dfp = df_disease[['31-0.0', '21003-2.0', 'eid']]
    dfc = df_control[['31-0.0', '21003-2.0', 'eid']]
    # loop through
    matched = []
    for ip, rp in dfp.iterrows():
        for ic, rc in dfc.iterrows():
            if rp['31-0.0']==rc['31-0.0'] and rp['21003-2.0']==rc['21003-2.0']:
                if rc['eid'] not in matched:
                    matched.append(rc['eid'])
                    break
    # save 
    df_matched = pd.Series(matched).astype(int)
    if save_csv:
        df_matched.to_csv(f'./data/{save_name}.csv', header=None, index=None)
    return matched

def load_pain_matched(pain_status='plus', questionnaire='all', idp='all', question_visits=[2], imputed=True):
    """prepare pain/matched control set for classify"""
    # load data
    if pain_status == 'plus': # pain plus/minus
        df_pain = pd.read_csv('./data/qsidp_pain_plus.csv')
        df_matched = pd.read_csv('./data/qsidp_pain_minus_matched.csv')
    elif pain_status == 'all': # patients/patients matched
        df_pain = pd.read_csv('./data/qsidp_patients.csv')
        df_matched = pd.read_csv('./data/qsidp_patients_matched.csv')
    elif pain_status == 'must': # patients with pain/matched
        df_pain = pd.read_csv('./data/qsidp_patients_pain.csv')
        df_matched = pd.read_csv('./data/qsidp_patients_pain_matched.csv')

    df_pain['label'] = 1
    df_matched['label'] = 0
    dfs = pd.concat([df_pain, df_matched])
    print(f'Patients={df_pain.shape[0]}, controls={df_matched.shape[0]}')
    # extract questions/idps
    qs = load_qscode(questionnaire=questionnaire, idp=idp)
    # extract questionnaire of interest
    dff = extract_qs(dfs, df_questionnaire=qs, visits=question_visits)
    dff['label'] = dfs['label']
    # impute
    if imputed==True:
        print(f'Questionnaires from visits {question_visits} shape={dff.shape}')
        dff_imputed = impute_qs(dff, freq_fill='median', nan_percent=0.9)
        print(f'After imputation shape={dff_imputed.shape}')
    elif imputed==False:
        dff_imputed = dff.dropna()
    else:
        dff_imputed = dff
    return dff_imputed


# running
if __name__=="__main__":

    # patient/matched control classify
    questionnaire = 'all'
    idp = 'all'
    dff_imputed = load_pain_matched(pain_status='all', questionnaire=questionnaire, idp=idp, question_visits=[2], imputed=True)
    # basic classification
    classifiers = ['rforest']#'dtree', 
    for c in classifiers:
        basic_classify(dff_imputed, classifier=c, random_state=0, test_size=0.25, save_plot=True, num_importance=20, questionnaire=questionnaire, idp=idp,
        save_name='paincontrol_qsidp')

    
