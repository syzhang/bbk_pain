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
    df_matched = pd.Series(matched)
    if save_csv:
        df_matched.to_csv('./data/{save_name}.csv', header=None, index=None)
    return matched

def load_patients(visits=[2], single_disease=True):
    """load patients removing multi diseases"""
    # from clean_questions import exclude_multidisease, disease_label
    dfp = pd.read_csv(os.path.join('..', 'funpack_cfg', 'qsidp_subjs_disease_visit2_extended.tsv'), sep='\t')  
    # load disease
    df_disease_label = disease_label(dfp, visits=visits, grouping='simplified')
    # exclude multi diseases subjects
    df_exclude, df_label_exclude = exclude_multidisease(dfp, df_disease_label)
    if single_disease==True:
        return df_exclude
    else:
        return df_qs

def load_patient_matched(questionnaire='all', idp='all', question_visits=[2], imputed=True):
    """prepare patient/matched control set for classify"""
    # load data
    df_disease = load_patients(visits=[2], single_disease=True)
    df_disease['label'] = 1
    df_matched = pd.read_csv('../funpack_cfg/qsidp_subjs_control_visit2_matched.tsv', sep='\t')
    df_matched['label'] = 0
    dfs = pd.concat([df_disease, df_matched])
    print(f'Patients={df_disease.shape[0]}, controls={df_matched.shape[0]}')
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
    # # generate list of matched control
    # df_control = pd.read_csv('../funpack_cfg/qsidp_subjs_control_visit2_extended.tsv', sep='\t')
    # df_disease = load_patients(visits=[2], single_disease=True)
    # match_ls = extract_control(df_control, df_disease, save_csv=True)

    # patient/matched control classify
    questionnaire = 'all'
    idp = 'all'
    dff_imputed = load_patient_matched(questionnaire=questionnaire, idp=idp, question_visits=[2], imputed=True)
    # basic classification
    classifiers = ['rforest']#'dtree', 
    for c in classifiers:
        basic_classify(dff_imputed, classifier=c, random_state=0, test_size=0.25, save_plot=True, num_importance=20, questionnaire=questionnaire, idp=idp,
        save_name='paincontrol_qs')

    
