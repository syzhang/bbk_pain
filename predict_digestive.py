"""
predict digestive pain from no pain at imaging
"""
import os
import sys
import numpy as np
import pandas as pd
from clean_questions import *

def load_data(label_type='severe'):
    """load digestive_after_imaging dataset"""
    df = pd.read_csv('../funpack_cfg/qsidp_subjs_digestive_imaging.tsv', sep='\t')
    print(df.shape)
    # create label
    dfq = pain_label(df, label_type=label_type)
    df_out = df.merge(dfq, how='outer', on='eid')
    df_out.drop_duplicates(inplace=True)
    print(df_out['label']==True)
    return df_out

def pain_label(df, label_type='severe'):
    """create pain label based on digestive qs"""
    # load digestive qs
    qs = load_qscode(questionnaire='digestive', idp=None)
    dfq = extract_qs(df, df_questionnaire=qs, visits=[2])
    # pain label
    if label_type == 'severe': # only include bothered a lot
        dfq['label'] = [(dfq['21027-0.0']==1) | (dfq['21035-0.0']==1)
               | (dfq['21048-0.0']==-602) | (dfq['21052-0.0']==-602)
               | (dfq['21051-0.0']==-602)  | (dfq['21049-0.0']==-602)
               | (dfq['21057-0.0']==-602)][0].astype(int)
    elif label_type == 'mild': # include bothered a little
        dfq['label'] = [(dfq['21027-0.0']==1) | (dfq['21035-0.0']==1)
               | (dfq['21048-0.0']!=-600) | (dfq['21052-0.0']!=-600)
               | (dfq['21051-0.0']!=-600)  | (dfq['21049-0.0']!=-600)
               | (dfq['21057-0.0']!=-600)][0].astype(int)
    return dfq

# running
if __name__=="__main__":
    df = load_data(label_type='severe')
    print(df.shape)
    print(sum(df['label']==1))