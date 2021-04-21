"""
predict digestive pain from no pain at imaging
"""
import os
import sys
import numpy as np
import pandas as pd
from clean_questions import *

def load_digestive_data(label_type='severe', questionnaire='all', idp='all', question_visits=[2], imputed=True, nan_percent=0.9):
    """load digestive_after_imaging dataset"""
    df = pd.read_csv('../funpack_cfg/qsidp_subjs_digestive_imaging.tsv', sep='\t')
    print(df.shape)
    # create label
    dfq = pain_label(df, label_type=label_type)
    dfq.set_index('eid', inplace=True)
    # load question code
    qs = load_qscode(questionnaire=questionnaire, idp=idp)
    # extract questionnaire of interest
    df_qs = extract_qs(df, df_questionnaire=qs, visits=question_visits)
    df_qs.set_index('eid', inplace=True)
    # merge label
    dff = df_qs.merge(dfq['label'], left_index=True, right_index=True)
    dff.reset_index(inplace=True)
    # impute
    if imputed==True:
        print(f'Questionnaires from visits {question_visits} shape={dff.shape}')
        dff_imputed = impute_qs(dff, freq_fill='median', nan_percent=nan_percent)
        print(f'After imputation shape={dff_imputed.shape}')
    elif imputed==False:
        dff_imputed = dff.dropna()
    else:
        dff_imputed = dff
    return dff_imputed

def pain_label(df, label_type='severe'):
    """create pain label based on digestive qs"""
    # load digestive qs
    qs = load_qscode(questionnaire='digestive', idp=None)
    dfq = extract_qs(df, df_questionnaire=qs, visits=[2])
    # pain label
    if label_type == 'severe': # only include bothered a lot
        dfq['label'] = [
            (dfq['21027-0.0']==1) | # Abdominal discomfort/pain for 6 months or longer
            (dfq['21035-0.0']==1) | # Currently (in last 3 months) suffer from abdominal pain
            (dfq['21048-0.0']==-602) | # Degree bothered by back pain in the past 3 months
            (dfq['21052-0.0']==-602) | # Degree bothered by chest pain in the last 3 months
            (dfq['21051-0.0']==-602) | # Degree bothered by headaches in the last 3 months
            (dfq['21049-0.0']==-602) | # Degree bothered by pain in arms/legs/joints in the past 3 months
            (dfq['21057-0.0']==-602) # Degree bothered by pain/problems during intercourse in the last 3 months
        ][0].astype(int)
    elif label_type == 'mild': # include bothered a little
        dfq['label'] = [
            (dfq['21027-0.0']==1) | # Abdominal discomfort/pain for 6 months or longer
            (dfq['21035-0.0']==1) | # Currently (in last 3 months) suffer from abdominal pain
            (dfq['21048-0.0']!=-600) | # Degree bothered by back pain in the past 3 months
            (dfq['21052-0.0']!=-600) | # Degree bothered by chest pain in the last 3 months
            (dfq['21051-0.0']!=-600) | # Degree bothered by headaches in the last 3 months
            (dfq['21049-0.0']!=-600) | # Degree bothered by pain in arms/legs/joints in the past 3 months
            (dfq['21057-0.0']!=-600) # Degree bothered by pain/problems during intercourse in the last 3 months
        ][0].astype(int)
    elif label_type == 'severe_wide': # include bothered a little
        dfq['label'] = [
            (dfq['21027-0.0']==1) | # Abdominal discomfort/pain for 6 months or longer
            (dfq['21035-0.0']==1) | # Currently (in last 3 months) suffer from abdominal pain
            (dfq['21048-0.0']==-602) | # Degree bothered by back pain in the past 3 months
            (dfq['21052-0.0']==-602) | # Degree bothered by chest pain in the last 3 months
            (dfq['21051-0.0']==-602) | # Degree bothered by headaches in the last 3 months
            (dfq['21049-0.0']==-602) | # Degree bothered by pain in arms/legs/joints in the past 3 months
            (dfq['21057-0.0']==-602) | # Degree bothered by pain/problems during intercourse in the last 3 months
            (dfq['21048-0.0']==-601) | # Degree bothered by back pain in the past 3 months
            (dfq['21052-0.0']==-601) | # Degree bothered by chest pain in the last 3 months
            (dfq['21051-0.0']==-601) | # Degree bothered by headaches in the last 3 months
            (dfq['21049-0.0']==-601) | # Degree bothered by pain in arms/legs/joints in the past 3 months
            (dfq['21057-0.0']==-601) # Degree bothered by pain/problems during intercourse in the last 3 months
        ][0].astype(int)
    return dfq

def pain_wideness(df):
    """construct a measure of pain widespreadness"""
    

# running
if __name__=="__main__":
    # load questionnaire codes
    # question_visits = [0,1,2]
    question_visits = [2]
    questionnaire = 'all'
    idp = 'all'#['t1vols','taskfmri']#
    # load data
    dff_imputed = load_digestive_data(label_type='severe', questionnaire=questionnaire, idp=idp, question_visits=[2], imputed=True) 
    # dff_imputed = load_digestive_data(label_type='severe', questionnaire=questionnaire, idp=idp, question_visits=[2], imputed=False)

    # basic classification
    classifiers = ['rforest']#'dtree', 
    for c in classifiers:
        basic_classify(dff_imputed, classifier=c, random_state=0, test_size=0.25, save_plot=True, num_importance=20, questionnaire=questionnaire, idp=idp, save_name='digestive_qs')
        # dfr = cv_classify(dff_imputed, classifier=c, cv_fold=10, questionnaire=questionnaire, idp=idp)