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

def plot_compare(df, save_name='paintype', comp_var='IDP', hue_var=None):
    """plot model comparison"""
    # plot
    plot_ls = [c for c in df.columns if 'test_' in c]
    f, axes = plt.subplots(len(plot_ls), 1, figsize=(9, 7), sharex=True)
    for i, c in enumerate(plot_ls):
        if hue_var:
            g = sns.barplot(x=df[comp_var], y=df[c], hue=df[hue_var], palette="Paired", ax=axes[i])
            g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            g = sns.barplot(x=df[comp_var], y=df[c], palette="Paired", ax=axes[i])
        for bar in g.patches:
            g.annotate(f'{bar.get_height()*100:.0f}',
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=9, xytext=(0, 5),
                   textcoords='offset points')
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
        # preproc = 'all_data'
        preproc = 'compare_classifiers_qsidp' #'all_data_connectivity'
        comp_var = 'classifier'
        df_tmp = pd.read_csv(f'./model_performance/output/{preproc}.csv')#classifiers x datasets (3) x 10cv
        # for d in ['paincontrol', 'paintype', 'digestive']:
        for d in np.unique(df_tmp['dataset']):
            df_tmpd = df_tmp[df_tmp['dataset']==d]
            df = sort_compare(df_tmpd, comp_var=comp_var, criteria='roc_auc', sort_top=len(np.unique(df_tmpd[comp_var])))
            plot_compare(df, save_name=d+'_'+preproc, comp_var=comp_var)
    elif plot_type=='dataset':
        comp_var = 'dataset'
        # plot single classfier with all datasets
        all_ls = []
        clf = 'lgb'
        # clf = 'rforest'
        # fpath = f'./model_performance/output/{clf}/'
        fpath = f'./model_performance/output_patient/{clf}/'
        conn_ls = os.listdir(fpath)

        for conn in conn_ls:
            fpath_conn = os.path.join(fpath, conn)
            for f in os.listdir(fpath_conn):
                if f.startswith('all_'):
                    fname = f
                    df_tmp = pd.read_csv(os.path.join(fpath_conn,fname))
                    # print(fname,df_tmp['test_roc_auc'].mean())
                    # plot_compare(df_tmp, save_name=fname, comp_var=comp_var)
                    # combine to single
                    df_tmp['features'] = ('_').join(fname.split('_')[1:])
                    all_ls.append(df_tmp)
            dff = pd.concat(all_ls)
            # plot_compare(dff, save_name=f'feature_{clf}_{conn}', comp_var=comp_var, hue_var='features')
            plot_compare(dff, save_name=f'feature_{clf}_{conn}_ptn', comp_var=comp_var, hue_var='features')
    elif plot_type=='feature':
        fpath = f'./model_performance/output_features/'
        f_ls = os.listdir(fpath)
        c = 'test_roc_auc_ovr'
        for f in f_ls:
            df_tmp = pd.read_csv(os.path.join(fpath,f))
            sns.barplot(x=df_tmp['dataset'], y=df_tmp[c], palette="Paired")
            plt.xticks(rotation=90)
            # axes[i].axhline(np.mean(df[c]), color='k', clip_on=False)
            # axes[i].set_ylabel(c)
            dfc_diff = df_tmp[c].max() - df_tmp[c].min()
            plt.ylim(df_tmp[c].mean()-0.5*dfc_diff, df_tmp[c].mean()+0.5*dfc_diff)
            save_name = f.split('.')[0]
            plt.savefig(f'./model_performance/figs/{save_name}_compare.png', bbox_inches='tight')

