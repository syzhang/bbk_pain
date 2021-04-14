"""
waterfall test for IDP sets
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clean_questions import * 
from compare_control import *

# running
if __name__=="__main__":
    # which dataset to use (paintype or paincontrol classification)
    dataset = sys.argv[1] 
    # patient/matched control classify
    idp_ls = ['fast','subcorticalvol','t1vols','t2star','t2weighted','taskfmri','dmri','dmriweighted']
    questionnaire = None
    visits = [2]
    # initialise
    name_ls, auc_ls, tr_ls, ts_ls = [], [], [], []
    # combinations
    idp_sets = []
    for L in range(2, len(idp_ls)+1):
        idp_sets.append(idp_ls[:L])
    # loop through combinations
    for idp in idp_ls+idp_sets:
        if type(idp) is list:
            idp_name = '+'.join(idp)
        else:
            idp_name = idp
        print(idp_name)

        if dataset == 'paincontrol':
            dff_imputed = load_patient_matched(questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=True)
        elif dataset == 'paintype':
            dff_imputed = load_patient_grouped(questionnaire=questionnaire, idp=idp, question_visits=visits, imputed=True)

        # basic classification
        auc, train_acc, test_acc = basic_classify(dff_imputed, classifier='rforest', random_state=0, test_size=0.25, plot_figs=False, save_plot=False, num_importance=20, questionnaire=questionnaire, idp=idp)
        # store idp classifier performance
        name_ls.append(idp_name)
        auc_ls.append(auc)
        tr_ls.append(train_acc)
        ts_ls.append(test_acc)

    # performance df
    df_perf = pd.DataFrame({'IDP': name_ls, 'AUC': auc_ls, 'Train accuracy': tr_ls, 'Test accuracy': ts_ls})
    # save to csv
    df_perf.to_csv(f'./model_performance/{dataset}_idp_waterfall.csv', index=None)
