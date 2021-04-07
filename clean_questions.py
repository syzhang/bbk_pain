"""
clean questionnaire data
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_qs(df_subjects, df_questionnaire, visits=[2]):
    """extract questionnaire set out of 5 possible"""
    # load questionnaire code of interest
    field_code = df_questionnaire['code'].to_list()
    # extract all fields with questionnaire code
    field_cols = []
    for code in field_code:
#         print(code)
#         print(field_cols)
        cols_ls = [col for col in df_subjects.columns if str(code)+'-' in col]
        if visits != None: # limit to visits only
            if len(cols_ls) > 1:
                cols_exclude = []
                for visit in visits:
                    for col in cols_ls:
                        if '-'+str(visit) in col:
                            cols_exclude.append(col)
                cols_ls = cols_exclude
        else:
            cols_ls = cols_ls
        field_cols += cols_ls
    # append eid
    field_cols += ['eid']
    # remove duplicate
    field_cols_rm = list(set(field_cols))
    df_qs = df_subjects[field_cols_rm]
    # remove duplicated columns
    df_qs_rm = df_qs.loc[:, ~df_qs.columns.duplicated()]
    return df_qs_rm

def load_qscode(questionnaire='all'):
    """load questionnaire code (lifestyle/mental/cognitive/digestive)"""
    base_dir = './bbk_codes/'
    questionnaire_ls = ['lifestyle','mental','cognitive','digestive']
    if (questionnaire!='all') and (questionnaire in questionnaire_ls):
        df_qs = pd.read_csv(base_dir+questionnaire+'_code.csv')
    elif questionnaire=='all':
        qs_ls = []
        for qs in questionnaire_ls:
            qs_ls.append(pd.read_csv(base_dir+qs+'_code.csv'))
        df_qs = pd.concat(qs_ls)
    else:
        raise ValueError('Questionnaire code does not exist.')
    return df_qs

def clean_dtype(df):
    """clean up dtype related"""
    df_tmp = df.dropna(how='all')
    print(df_tmp.shape)
    perc = df_tmp.shape[0]/df.shape[0]*100
    print(f'{perc:.1f}% have all questions')
    return df_tmp

def disease_label(df_subjects, visits=[2]):
    """create disease label df"""
    from disease_type import extract_disease
    df_disease_group = pd.read_csv('./bbk_codes/disease_code_grouped.csv')
    for i, r in df_disease_group.iterrows():
        # print(i,r['disease'])
        df_tmp = extract_disease(df_subjects, r['code'], visit=visits)
        df_tmp.replace(np.nan, 0.0, inplace=True)
        # rename column to disease
        df_tmp.rename(columns={df_tmp.columns[0]: r['disease']}, inplace=True)
        # merge
        if i == 0:
            df_disease_label = df_tmp
        else:
            df_disease_label = df_disease_label.join(df_tmp, on='eid')
    return df_disease_label

def exclude_multidisease(df, df_label):
    """exclude subjects with multiple diseases"""
    # subjects with multi diseases
    df_label_copy = df_label.copy()
    df_label_copy['sum'] = df_label_copy[df_label_copy.columns.to_list()].sum(axis=1)
    df_label_exclude = df_label_copy[df_label_copy['sum']<=1]
    df_label_exclude = df_label_exclude.drop('sum', axis=1)
    # multidisease eid
    df_copy = df.set_index('eid')
    df_copy['eid'] = df_copy.index
    df_exclude = df_copy.loc[df_label_copy['sum']<=1]
    return df_exclude, df_label_exclude

def impute_qs(df, nan_percent=0.9, freq_fill='median'):
    """impute questionnaire df"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if col!='label': # exclude label
            # remove time stamp cols
            if df_copy[col].dtype==object:
                df_copy.drop(col, axis=1, inplace=True)
            # replace nan with -818 (prefer not to say)
            elif np.any(df_copy[col]<-810):
#                 print(col)
                df_copy[col].replace({np.nan: -818.}, inplace=True)
    # fill freq nan with median
    df_copy = replace_freq(df_copy, use=freq_fill)
    # replace specific fields
    df_copy = replace_specific(df_copy)
    # drop columns with threshold percentage nan
    df_copy.dropna(axis=1, thresh=int(nan_percent*df_copy.shape[0]), inplace=True)
    return df_copy

def replace_freq(df, use='median'):
    """replace nan in freq with median"""
    df_copy = df.copy()
    for c in df_copy.columns:
        tmp = df_copy[c].value_counts()
        if tmp.shape[0]>7 and c!='label': # most likely frequency
            if use == 'median':
                df_copy[c].fillna(tmp.median(), inplace=True)
            elif use == 'mean':
                df_copy[c].fillna(tmp.mean(), inplace=True)
        elif tmp.shape[0]<=7 and c!='label': # other types of freq
            if np.any(df_copy[c]==-3.) or np.any(df_copy[c]==-1.): # prefer not to say
                df_copy[c].replace({np.nan: -3.}, inplace=True)
            elif np.any(df_copy[c]==-600.): # degree of bother, also has prefer not to say
                df_copy[c].replace({np.nan: -818.}, inplace=True)

    return df_copy

def replace_specific(df):
    """replace specific categories"""
    df_copy = df.copy()
    categories_zero = [
        '6160',#Leisure/social activities
        '6145',#Illness, injury, bereavement, stress in last 2 years
        '20123',#Single episode of probable major depression
        '20124',#Probable recurrent major depression (moderate)
        '20125', #Probable recurrent major depression (severe)
        '20481', #Self-harmed in past year
        '20484', #Attempted suicide in past year
        '20122', #Bipolar disorder status
        '20126', #Bipolar and major depression status
                 ]
    categories_nts = [
        '20414', #Frequency of drinking alcohol
    ]
    categories_to = [
        '20246', #Trail making completion status
        '20245', #Pairs matching completion status
        '20244', #Symbol digit completion status
    ]
    for c in df_copy.columns:
        for cat in categories_zero:
            if cat in c: 
                df_copy[c].replace(np.nan, 0., inplace=True)
        for cat in categories_nts:
            if cat in c:
                df_copy[c].replace(np.nan, -818., inplace=True) # treat as prefer not to say
        for cat in categories_to:
            if cat in c:
                df_copy[c].replace(np.nan, 1., inplace=True) # treat as abandoned
    return df_copy

def basic_dtree(df, test_size=0.5, save_plot=True):
    """basic decision tree classification"""
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    
    y = df['label']
    X = df.drop('label', axis=1)
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
    
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train)
    predicted = clf.predict(X_test)
    
    # creating a confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, predicted)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
    if save_plot:
        plt.xticks(rotation=90)
        plt.savefig('./figs/dtree_cm.png', bbox_inches='tight')

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    print(f"Classification report for classifier {clf}:\n"
        f"{classification_report(y_test, predicted)}\n"
        f"ROC AUC={auc}")

# running
if __name__=="__main__":
    # main questionnaire file
    qs_path = os.path.join('..', 'funpack_cfg', 'questions_subjs_disease_visit2_extended.tsv')
    # load subjects
    df_subjects = pd.read_csv(qs_path, sep='\t')
    # create disease label
    df_disease_label = disease_label(df_subjects, visits=[2])
    # load questionnaire codes
    # questionnaire_ls = ['lifestyle','mental','cognitive','digestive','all']
    questionnaire_ls = ['all']
    for q in questionnaire_ls:
        df_qs = load_qscode(q)
        # extract questionnaire of interest
        df_qs = extract_qs(df_subjects, df_questionnaire=df_qs)
        # exclude multi diseases subjects
        df_exclude, df_label_exclude = exclude_multidisease(df_qs, df_disease_label)

    # reverse one hot encoding
    label_exclude = df_label_exclude.idxmax(axis=1)
    dff = df_exclude.merge(label_exclude.rename('label'), left_index=True, right_index=True)
    # impute
    print(dff.shape)
    dff_imputed = impute_qs(dff, freq_fill='median', nan_percent=0.)
    print(dff_imputed.shape)
    dff_imputed = dff_imputed.dropna(how='all', axis=1)
    print(dff_imputed.shape)
    # basic classification
    basic_dtree(dff_imputed, test_size=0.5, save_plot=True)