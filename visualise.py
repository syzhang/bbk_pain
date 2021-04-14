"""
plot various outputs
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_waterfall(dataset='paintype'):
    """plot waterfall model comparison"""
    df = pd.read_csv(f'./model_performance/{dataset}_idp_waterfall.csv')
    # shorten labels
    short_labels = []
    for i,r in df.iterrows():
        if '+' in r['IDP']:
            tmp = r['IDP'].split('+')
            short_labels.append(str(len(tmp)))
        else:
            short_labels.append(r['IDP'])
    df['IDP name'] = short_labels
    # Set up the matplotlib figure
    f, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    plot_ls = ['AUC', 'Train accuracy', 'Test accuracy']
    for i, c in enumerate(plot_ls):
        g = sns.barplot(x=df['IDP name'], y=df[c], palette="rocket", ax=axes[i])
        plt.xticks(rotation=90)
        axes[i].axhline(np.mean(df[c]), color='k', clip_on=False)
        axes[i].set_ylabel(c)
        axes[i].set_ylim(df[c].min()-0.05, df[c].max()+0.05)
    # Finalize the plot
    sns.despine(bottom=True)
    # plt.setp(f.axes, yticks=[])
    # plt.tight_layout(h_pad=2)
    plt.savefig(f'./model_performance/figs/{dataset}_idp_waterfall.png', bbox_inches='tight')

# running
if __name__=="__main__":
    plot_waterfall(dataset='paincontrol')
    plot_waterfall(dataset='paintype')