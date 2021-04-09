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
        # cols_ls = [col for col in df_subjects.columns if str(code)+'-' in col]
        code_root = str(code)+'-'
        cols_ls = [col for col in df_subjects.columns if col[:len(code_root)]==code_root]
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

def load_qscode(questionnaire='all', idp=None):
    """load questionnaire and idp code"""
    base_dir = './bbk_codes/'
    # questionnaire data
    df_qs = pd.DataFrame()
    if questionnaire!=None:
        questionnaire_ls = ['lifestyle','mental','cognitive','digestive','cwp','demographic']
        if (questionnaire!='all') and (questionnaire in questionnaire_ls):
            df_qs = pd.read_csv(os.path.join(base_dir, questionnaire+'_code.csv'))
        elif questionnaire=='all':
            qs_ls = []
            for qs in questionnaire_ls:
                qs_ls.append(pd.read_csv(os.path.join(base_dir,qs+'_code.csv')))
            df_qs = pd.concat(qs_ls)
        else:
            raise ValueError('Questionnaire code does not exist.')
    # idp data
    df_idp = pd.DataFrame()
    if idp!=None:
        idp_ls = ['baseg','dktseg','subcorticalseg','subcorticalvol','subroiseg','whiteseg']
        if (idp!='all') and (idp in idp_ls):
            df_idp = pd.read_csv(os.path.join(base_dir, idp+'_code.csv'))
        elif idp=='all':
            idpc_ls = []
            for i in idp_ls:
                fname = 'idp_'+i+'_code.csv'
                fpath = os.path.join(base_dir, fname)
                idpc_ls.append(pd.read_csv(fpath))
            df_idp = pd.concat(idpc_ls)
        else:
            raise ValueError('IDP code does not exist.')
    # combine questionnaire with idp
    df_out = pd.concat([df_qs, df_idp])
    return df_out

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

def basic_classify(df, classifier='dtree', test_size=0.5, random_state=10, save_plot=True, num_importance=20, questionnaire='all', idp='None'):
    """basic classification"""
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    
    y = df['label']
    X = df.drop(['label','eid'], axis=1)
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    if classifier == 'dtree':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
    elif classifier == 'rforest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=5).fit(X_train, y_train)
    # predict test set
    y_test_predicted = clf.predict(X_test)
    
    # creating a confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_test_predicted)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
    if save_plot:
        plt.xticks(rotation=90)
        plt.savefig(f'./figs/{classifier}_cm.png', bbox_inches='tight')

    # plot permutation importantce
    from sklearn.inspection import permutation_importance
    plot_num = num_importance
    result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    # result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    question_labels = match_question(X_test.columns[sorted_idx[-plot_num:]],questionnaire=questionnaire, idp=idp)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.boxplot(result.importances[sorted_idx[-plot_num:]].T, vert=False, labels=question_labels)
    ax.set_title("Permutation Importances (test set)")
    # ax.set_title("Permutation Importances (train set)")
    if save_plot:
        plt.savefig(f'./figs/{classifier}_importance.png', bbox_inches='tight')

    # calculate accuracy / auc
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    print(f"Classification report for classifier {clf}:\n"
        f"{classification_report(y_test, y_test_predicted)}\n"
        f"ROC AUC={auc:.4f}, train accuracy={clf.score(X_train, y_train):.4f}, test accuracy={clf.score(X_test, y_test):.4f}")

def match_question(q_codes, questionnaire='all', idp=None):
    """backward search questions to match question code"""
    df_qs = load_qscode(questionnaire=questionnaire, idp=idp)
    question_ls = []
    for c in q_codes:
        code = int(c.split('-')[0])
        question = df_qs[df_qs['code']==code]['Field title'].values
        question_ls.append(question)
    return question_ls

# # running
# if __name__=="__main__":

# running
if __name__=="__main__":
    # main questionnaire file
    qs_path = os.path.join('..', 'funpack_cfg', 'questions_subjs_disease_visit2_extended.tsv')
    # load subjects
    df_subjects = pd.read_csv(qs_path, sep='\t')
    # create disease label
    df_disease_label = disease_label(df_subjects, visits=[2])
    # load questionnaire codes
    # questionnaire_ls = ['all']
    # question_visits = [0,1,2]
    question_visits = [2]
    
    # load data
    questionnaire = 'all'
    idp = 'all'
    df_qs = load_qscode(questionnaire=questionnaire, idp=idp)
    # extract questionnaire of interest
    df_qs = extract_qs(df_subjects, df_questionnaire=df_qs, visits=question_visits)
    # exclude multi diseases subjects
    df_exclude, df_label_exclude = exclude_multidisease(df_qs, df_disease_label)

    # reverse one hot encoding
    label_exclude = df_label_exclude.idxmax(axis=1)
    dff = df_exclude.merge(label_exclude.rename('label'), left_index=True, right_index=True)
    # impute
    print(f'Questionnaires from visits {question_visits} shape={dff.shape}')
    dff_imputed = impute_qs(dff, freq_fill='median', nan_percent=0.9)
    print(f'After imputation shape={dff_imputed.shape}')
    dff_imputed = dff_imputed.dropna(how='all', axis=1)
    print(f'Drop all nan cols shape={dff_imputed.shape}')
    # basic classification
    classifiers = ['dtree', 'rforest']
    for c in classifiers:
        basic_classify(dff_imputed, classifier=c, random_state=0, test_size=0.25, save_plot=True, num_importance=20, questionnaire=questionnaire, idp=idp)