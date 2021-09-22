"""
extract info for bmrc use
"""

import os
import numpy as np
import pandas as pd

from clean_questions import disease_label
from predict_digestive import pain_label

# running
if __name__=="__main__":
    # extract disease labels
    # fnames = ['patients_pain', 'patients', 'patients_pain_restricted']
    # for fname in fnames:
    #     qs_fname = os.path.join('./data', 'qsidp_'+fname+'.csv')
    #     df_qs = pd.read_csv(qs_fname)
    #     df_qs_label = disease_label(df_qs, visits=[2], grouping='grouped')
    #     print(df_qs.shape)
    #     print(df_qs_label.shape)
    #     save_path = os.path.join('./data', 'label_'+fname+'.csv')
    #     df_qs_label.to_csv(save_path)

    # digestive labels
    fname = 'digestive'
    qs_fname = os.path.join('./data', 'qsidp_'+fname+'.csv')
    df_qs = pd.read_csv(qs_fname)
    df_qs_label = pain_label(df_qs, label_type='severe')
    # print(df_qs.shape)
    # print(df_qs.head())
    # print(df_qs_label[['eid', 'label']])
    # print(df_qs_label.shape)
    # save
    df_label = df_qs_label[['eid', 'label']]
    df_label.to_csv(os.path.join('./data', 'label_'+fname+'.csv'), index=None)
    df_sj = df_qs_label['eid']
    df_sj.to_csv(os.path.join('./data', 'subjs_'+fname+'.csv'), header=None, index=None)
