"""
extract ids for disease classification
"""

import os
import sys
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
            # if len(disease_code)> 10:
            if isinstance(disease_code, str) and disease_code.startswith('['):
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

def plot_disease(df, df_disease_group, visits=[0,1,2,3], save_plot=True, save_name='disease_groups'):
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
        save_name = f'./figs/{save_name}.png'
        plt.savefig(save_name, bbox_inches='tight')
    return df_out

def group_disease_id(df, df_disease_group, visits=[0,1,2,3], save=True, save_name='disease_group_id'):
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
        df_out.to_csv(f'../output/{save_name}.csv')
    return df_out

def generate_clinical_vars():
    """generate clinical variables from bbk codes"""
    all_code = []
    for f in os.listdir('./bbk_codes/'):
        if not f.startswith('disease_code_'):
            print(f)
            df_tmp = pd.read_csv('./bbk_codes/'+f)
            all_code.append(df_tmp.code.astype(int))
    df_code = np.concatenate(all_code)
    dfc = pd.DataFrame(df_code)
    dfc.to_csv('../funpack_cfg/clinical_idp_variables.txt', header=None, index=None)

# running
if __name__=="__main__":
    # which visits to use
    visits = sys.argv[1] 
    # save string
    visits_ls = [int(x) for x in list(visits.split(','))]
    if len(visits_ls)==1:
        visits_str = 'visit'+str(visits_ls[0])
    else:
        visits_str = 'allvisits'
    # load data with conditions
    # tsv_cond = os.path.join('..', 'funpack_cfg', 'subj_with_condition_with_clinical_variables.tsv')
    # load extended subjects with conditions
    tsv_cond = os.path.join('..', 'funpack_cfg', 'subjs_with_condition_extended.tsv')
    df_cond = pd.read_csv(tsv_cond, sep='\t')
    # load disease groups
    df_disease_group = pd.read_csv('./bbk_codes/disease_code_grouped.csv')
    # plot disease numbers
    plot_disease(df_cond, df_disease_group, visits=visits_ls, save_name=f'disease_groups_{visits_str}_extended')
    # save disease group id
    df_out = group_disease_id(df_cond, df_disease_group, visits=visits_ls, save_name=f'disease_group_id_{visits_str}_extended')
    # extract eid for funpack
    df_sliced_sj = pd.read_csv(os.path.join('..', 'output', f'disease_group_id_{visits_str}_extended.csv'))
    df_sliced_sj['eid'].to_csv(os.path.join('..', 'output', f'subjs_disease_{visits_str}_extended.csv'), index=False, header=None)
    # exclude multiple conditinos
    # df_excluded = exclude_multidisease(df_out)
