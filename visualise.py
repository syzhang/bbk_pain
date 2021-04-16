"""
plot various outputs
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_waterfall(dataset='paintype', criteria='accuracy', sort_top=2):
    """load waterfall comparison results"""
    df_tmp = pd.read_csv(f'./model_performance/{dataset}_idp_waterfall.csv')
    # limit number of results
    df_gb = df_tmp.groupby(['IDP']).mean()
    if df_gb.shape[0] > sort_top:
        dfs = df_gb.sort_values(by=['test_'+criteria], ascending=False)
        df_top = dfs.iloc[:sort_top]
        # df = df_tmp[df_tmp['IDP'].isin(df_top.index)]
        df = pd.concat([df_tmp[df_tmp['IDP']==i] for i in df_top.index])
    else:
        df = df_tmp
    return df

def plot_waterfall(dataset='paintype', criteria='accuracy', sort_top=20):
    """plot waterfall model comparison"""
    # load data
    df = load_waterfall(dataset=dataset, sort_top=sort_top)
    # plot
    f, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    plot_ls = [c for c in df.columns if 'test_' in c]
    for i, c in enumerate(plot_ls):
        g = sns.barplot(x=df['IDP'], y=df[c], palette="rocket", ax=axes[i])
        plt.xticks(rotation=90)
        axes[i].axhline(np.mean(df[c]), color='k', clip_on=False)
        axes[i].set_ylabel(c)
        axes[i].set_ylim(df[c].mean()-0.02, df[c].mean()+0.02)
    # save
    sns.despine(bottom=True)
    plt.savefig(f'./model_performance/figs/{dataset}_idp_waterfall.png', bbox_inches='tight')

# running
if __name__=="__main__":
    plot_waterfall(dataset='paincontrol', sort_top=20)
    plot_waterfall(dataset='paintype', sort_top=20)
    plot_waterfall(dataset='digestive', sort_top=20)