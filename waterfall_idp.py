"""
waterfall test for IDP sets
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clean_questions import * 
from compare_control import *
from predict_digestive import *

# running
if __name__=="__main__":
    # which dataset to use (paintype or paincontrol classification)
    dataset = sys.argv[1] 
    # patient/matched control classify
    idp_ls = ['fast','subcorticalvol','t1vols','t2star','t2weighted','taskfmri','dmri','dmriweighted']
    questionnaire = None
    visits = [2]
    # impute_flag = False
    impute_flag = True
    # initialise
    res_ls = []
    # combinations
    idp_sets = []
    for i in range(1, len(idp_ls)+1):
        idp_sets.extend(itertools.combinations(idp_ls, i))
    print(len(idp_sets))

    # loop through combinations
    for idpx in idp_sets:
        idp = list(idpx)
        idpn = [t[:2] for t in idp]
        idp_name = '+'.join(idpn)
        print(idp_name)

        if dataset == 'paincontrol':
            dff_imputed = load_patient_matched(questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)
        elif dataset == 'paintype':
            dff_imputed = load_patient_grouped(questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)
        elif dataset == 'digestive':
            dff_imputed = load_digestive_data(label_type='severe', questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=impute_flag)

        # cv classification
        df_res = cv_classify(dff_imputed, classifier='rforest', cv_fold=10, questionnaire=questionnaire, idp=idp)
        # save result
        df_res['IDP'] = idp_name
        res_ls.append(df_res)

    # performance df
    df_perf = pd.concat(res_ls)
    # save to csv
    df_perf.to_csv(f'./model_performance/{dataset}_idp_waterfall.csv', index=None)
