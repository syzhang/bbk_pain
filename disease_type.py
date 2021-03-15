"""
extract ids for disease classification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_disease(df, disease_code, visit=None):
    """extract disease type into df"""
    # slice reported disease
    df_d = df[[col for col in df.columns if str(20002) in col]]
    df_d = pd.concat([df['eid'], df_d], axis=1)
    # slice particular visit
    df_dvd = pd.DataFrame()
    df_dvd['eid'] = df_d['eid']
    if visit is not None:
        for vis in visit:
            p_cols = [col for col in df_d.columns if '-'+str(vis) in col]
            # extract visit df
            df_dv = pd.concat([df_d['eid'], df_d[p_cols]], axis=1)
            df_dv.set_index('eid', inplace=True)
            # find matching disease
            disease_tag = []
            # if type(disease_code) is list:
            if len(disease_code)> 10:
                dcode_float = [float(i) for i in list(disease_code[1:-2].split(','))]
            else:
                dcode_float = [float(disease_code)]
            for i, r in df_dv.iterrows():
                if any(item in dcode_float for item in r.values):
                    disease_tag.append(1)
                else:
                    disease_tag.append(0)
            df_dvd[str(dcode_float[0])+'-'+str(vis)] = disease_tag
        df_dvd.set_index('eid', inplace=True)

    return df_dvd

def plot_disease(df, df_disease_group, visits=[0,1,2,3], save_plot=True):
    """plot groups of disease"""
    # check data count
    f, axes = plt.subplots(2,int(np.ceil(df_disease_group.shape[0]/2)), figsize=(10, 6))#, sharey=True)
    c = 0
    df_out = pd.DataFrame()
    # loop disease groups
    for i, r in df_disease_group.iterrows():
        df_tmp = extract_disease(df, r['code'], visit=visits)
        df_tmp.replace(0.0, np.nan, inplace=True) # keep pain only
        # plot
        g = sns.countplot(data=df_tmp, ax=axes.flat[c])
        axes.flat[c].set_title(r['disease'])
        axes.flat[c].set_xticklabels([str(i) for i in visits])
        # store disease df
        if i == 0:
            df_out = df_tmp.dropna(how='all').reset_index()
        else:
            # print(df_out.columns)
            df_out = df_out.merge(df_tmp.dropna(how='all').reset_index(), on='eid', how='outer')
        c += 1
    # save plot
    if save_plot:
        save_name = './figs/disease_groups.png'
        plt.savefig(save_name, bbox_inches='tight')
    return df_out

def group_disease_id(df, df_disease_group, visits=[0,1,2,3], save=True):
    """save groups of disease"""
    c = 0
    df_out = pd.DataFrame()
    # loop disease groups
    for i, r in df_disease_group.iterrows():
        df_tmp = extract_disease(df, r['code'], visit=visits)
        df_tmp.replace(0.0, np.nan, inplace=True) # keep pain only
        # store disease df
        if i == 0:
            df_out = df_tmp.dropna(how='all').reset_index()
        else:
            # print(df_out.columns)
            df_out = df_out.merge(df_tmp.dropna(how='all').reset_index(), on='eid', how='outer')
        c += 1
    if save:
        df_out.to_csv('../output/disease_group_id.csv', header=None)
    return df_out

# running
if __name__=="__main__":
    # load data with conditions
    tsv_cond = os.path.join('..', 'funpack_cfg', 'subj_with_condition_with_clinical_variables.tsv')
    df_cond = pd.read_csv(tsv_cond, sep='\t')
    # load disease groups
    df_disease_group = pd.read_csv('./bbk_codes/disease_code_grouped.csv')
    # plot disease numbers
    plot_disease(df_cond, df_disease_group, visits=[0,1,2,3])
    # save disease group id
    df_out = group_disease_id(df_cond, df_disease_group, visits=[0,1,2,3])