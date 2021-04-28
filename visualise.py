"""
plot various outputs
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sort_compare(df_tmp, comp_var='IDP', criteria='accuracy', sort_top=2):
    """sort comparison results"""
    # limit number of results
    df_gb = df_tmp.groupby([comp_var]).mean()
    if df_gb.shape[0] >= sort_top:
        dfs = df_gb.sort_values(by=['test_'+criteria], ascending=False)
        df_top = dfs.iloc[:sort_top]
        # df = df_tmp[df_tmp['IDP'].isin(df_top.index)]
        df = pd.concat([df_tmp[df_tmp[comp_var]==i] for i in df_top.index])
    else:
        df = df_tmp
    return df

def load_outputs(task_name='paintype'):
    """load all model performance given group name"""
    output_dir = './model_performance/output'
    df_ls = []
    for f in os.listdir(output_dir):
        if f.startswith(task_name):
            df_tmp = pd.read_csv(os.path.join(output_dir, f))
            df_ls.append(df_tmp)
    df = pd.concat(df_ls)
    df.drop_duplicates(inplace=True)
    return df

def plot_compare(df, save_name='paintype', comp_var='IDP'):
    """plot model comparison"""
    # plot
    f, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    plot_ls = [c for c in df.columns if 'test_' in c]
    for i, c in enumerate(plot_ls):
        g = sns.barplot(x=df[comp_var], y=df[c], palette="rocket", ax=axes[i])
        plt.xticks(rotation=90)
        axes[i].axhline(np.mean(df[c]), color='k', clip_on=False)
        axes[i].set_ylabel(c)
        dfc_diff = df[c].max() - df[c].min()
        axes[i].set_ylim(df[c].mean()-0.5*dfc_diff, df[c].mean()+0.5*dfc_diff)
    # save
    sns.despine(bottom=True)
    plt.savefig(f'./model_performance/figs/{save_name}_compare.png', bbox_inches='tight')

# running
if __name__=="__main__":
    # what to plot (idp/clf)
    plot_type = sys.argv[1] 

    if plot_type=='idp':
        # plot waterfall
        comp_var = 'QS/IDP'
        for d in ['paincontrol', 'paintype', 'digestive']:
            print(d)
            df_tmp = load_outputs(d)
            # df = sort_compare(df_tmp, comp_var=comp_var, criteria='accuracy', sort_top=15)
            df = sort_compare(df_tmp, comp_var=comp_var, criteria='roc_auc', sort_top=15)
            plot_compare(df, save_name=d+'_QSIDP', comp_var=comp_var)
    elif plot_type=='clf':
        # plot all clf compare
        preproc = 'all_data'
        comp_var = 'classifier'
        df_tmp = pd.read_csv(f'./model_performance/{preproc}_classifiers.csv')#classifiers x datasets (3) x 10cv
        for d in ['paincontrol', 'paintype', 'digestive']:
            df_tmpd = df_tmp[df_tmp['dataset']==d]
            df = sort_compare(df_tmpd, comp_var=comp_var, criteria='accuracy', sort_top=len(np.unique(df_tmpd[comp_var])))
            plot_compare(df, save_name=d+'_'+preproc, comp_var=comp_var)