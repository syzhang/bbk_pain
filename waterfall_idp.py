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
    idp_ls = ['fast','subcorticalvol','t1vols','t2star','t2weighted','taskfmri','dmri','wdmri']
    qs_ls = ['lifestyle','mental','cognitive','demographic']
    visits = [2]
    impute_flag = True
    # initialise
    res_ls = []
    # combinations
    idp_sets = []
    for i in range(1, len(idp_ls)+len(qs_ls)+1):
        idp_sets.extend(itertools.combinations(idp_ls+qs_ls, i))
    print(len(idp_sets))

    # loop through combinations
    for idpx in idp_sets:
        idp = list(idpx)
        idpn = [t[:3] for t in idp]
        idp_name = '+'.join(idpn)

        # separate into qs and idp
        qs_in = [f for f in idp if f in qs_ls]
        id_in = [f for f in idp if f in idp_ls]
        print(qs_in)
        print(id_in)
        # load data
        if dataset == 'paincontrol':
            dff_imputed = load_patient_matched(questionnaire=qs_in, idp=id_in, question_visits=visits, imputed=impute_flag)
        elif dataset == 'paintype':
            dff_imputed = load_patient_grouped(questionnaire=qs_in, idp=id_in, question_visits=visits, imputed=impute_flag)
        elif dataset == 'digestive':
            dff_imputed = load_digestive_data(label_type='severe', questionnaire=qs_in, idp=id_in, question_visits=visits, imputed=impute_flag)

        # cv classification
        df_res = cv_classify(dff_imputed, classifier='rforest', cv_fold=4, questionnaire=qs_in, idp=id_in, scaler=True, balance=True)
        # save result
        df_res['QS/IDP'] = idp_name
        res_ls.append(df_res)

    # performance df
    df_perf = pd.concat(res_ls)
    # save to csv
    df_perf.to_csv(f'./model_performance/{dataset}_waterfall.csv', index=None)
