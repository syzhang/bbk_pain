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
    if questionnaire!=None and len(questionnaire)!=0:
        questionnaire_ls = ['lifestyle','mental','cognitive','digestive','cwp','demographic']
        if (questionnaire!='all') and (questionnaire in questionnaire_ls):
            df_qs = pd.read_csv(os.path.join(base_dir, questionnaire+'_code.csv'))
        elif (questionnaire!='all') and (type(questionnaire) is list): # multiple qs sets
            qs_ls = []
            for i in questionnaire:
                fname = i+'_code.csv'
                fpath = os.path.join(base_dir, fname)
                qs_ls.append(pd.read_csv(fpath))
            df_qs = pd.concat(qs_ls)
        elif questionnaire=='all':
            questionnaire_ls = ['lifestyle','mental','cognitive','demographic']
            qs_ls = []
            for qs in questionnaire_ls:
                qs_ls.append(pd.read_csv(os.path.join(base_dir,qs+'_code.csv')))
            df_qs = pd.concat(qs_ls)
        else:
            raise ValueError('Questionnaire code does not exist.')
    # idp data
    df_idp = pd.DataFrame()
    if idp!=None and len(idp)!=0:
        idp_ls = ['dmri','wdmri','fast','subcorticalvol','t1vols','t2star','t2weighted','taskfmri']
        if (idp!='all') and (idp in idp_ls): # single idp set
            df_idp = pd.read_csv(os.path.join(base_dir, 'idp_'+idp+'_code.csv'))
        elif (idp!='all') and (type(idp) is list): # multiple idp sets
            idpc_ls = []
            for i in idp:
                fname = 'idp_'+i+'_code.csv'
                fpath = os.path.join(base_dir, fname)
                idpc_ls.append(pd.read_csv(fpath))
            df_idp = pd.concat(idpc_ls)
        elif idp=='all': # all idp sets
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

def disease_label(df, visits=[2], grouping='simplified'):
    """create disease label df"""
    from disease_type import extract_disease
    # different grouping
    if grouping == 'simplified':
        df_disease_group = pd.read_csv('./bbk_codes/disease_code_simplified.csv')
    elif grouping == 'detailed':
        df_disease_group = pd.read_csv('./bbk_codes/disease_code.csv')
    elif grouping == 'select':
        df_disease_group = pd.read_csv('./bbk_codes/disease_code_select.csv')
    else:
        # df_disease_group = pd.read_csv('./bbk_codes/disease_code_grouped.csv')
        df_disease_group = pd.read_csv('./bbk_codes/disease_code_grouped4.csv') # excluding fibro
    # drop duplicates to avoid merging issues
    if df.index.name == 'eid':
        df_dropd = df[~df.index.duplicated()]
    elif 'eid' in df:
        df_dropd = df[~df['eid'].duplicated()]
    else:
        raise ValueError('eid not present.')
    # iterate diseases
    df_disease_label = []
    for i, r in df_disease_group.iterrows():
        df_tmp = extract_disease(df_dropd, r['code'], visit=visits)
        df_tmp.replace(np.nan, 0.0, inplace=True)
        # rename column to disease
        df_tmp.rename(columns={df_tmp.columns[0]: r['disease']}, inplace=True)
        # merge
        df_disease_label.append(df_tmp)
    return pd.concat(df_disease_label,axis=1)

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

def impute_qs(df, nan_percent=0.9, freq_fill='median', 
              transform=False, transform_fn='sqrt'):
    """impute questionnaire df"""
    df_copy = df.copy()
    # replace prefer not to say and remove object
    df_copy = replace_noans(df_copy)
    # replace multiple choice fields
    df_copy = replace_multifield(df_copy)
    # replace specific fields
    df_copy = replace_specific(df_copy)
    # fill freq nan with median
    df_copy = replace_freq(df_copy, use=freq_fill)
    # transform freq cols
    if transform:
        df_copy = apply_transform(df_copy, use=transform_fn)
    # drop columns with threshold percentage nan
    df_copy.dropna(axis=1, thresh=int(nan_percent*df_copy.shape[0]), inplace=True)
    return df_copy

def replace_noans(df):
    """replace prefer not to say if avaialable and remove object cols"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if col!='label': # exclude label
            # remove time stamp cols
            if df_copy[col].dtype==object:
                df_copy.drop(col, axis=1, inplace=True)
            # replace nan with -818 (prefer not to say)
            elif np.any(df_copy[col]==-818):
                df_copy[col].replace({np.nan: -818.}, inplace=True)
    return df_copy

def replace_multifield(df):
    """replace multiple choice fields"""
    df_copy = df.copy()
    categories_multi = [
        '6160',#Leisure/social activities
        '6145',#Illness, injury, bereavement, stress in last 2 years
    ]
    for cat in categories_multi:
        p_cols = [col for col in df_copy.columns if col[:len(cat)+1]==str(cat)+'-']
        for c in p_cols: # replace with none of the above -7
            df_copy[c].replace(np.nan, -7., inplace=True)
    return df_copy

def replace_specific(df):
    """replace specific categories"""
    df_copy = df.copy()
    categories_zero = [
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

def replace_freq(df, use='median'):
    """replace nan in freq with median"""
    df_copy = df.copy()
    for c in df_copy.columns:
        tmp = df_copy[c].value_counts()
        if tmp.shape[0]>7 and c!='label': # most likely frequency/idp
            if use == 'median':
                df_copy[c].fillna(df_copy[c].median(), inplace=True)
            elif use == 'mean':
                df_copy[c].fillna(df_copy[c].mean(), inplace=True)
        elif tmp.shape[0]<=7 and c!='label': # other types of freq
            if np.any(df_copy[c]==-3.) or np.any(df_copy[c]==-1.): # prefer not to say
                df_copy[c].replace({np.nan: -3.}, inplace=True)
#             elif np.any(df_copy[c]==-600.): # degree of bother, also has prefer not to say
#                 df_copy[c].replace({np.nan: -818.}, inplace=True)
    return df_copy

def apply_transform(df, use='sqrt'):
    """adding additional freq cols with transforms"""
    df_copy = df.copy()
    trans_freq = [
        '22040', # Summed MET minutes per week for all activity
        '20156', # Duration to complete numeric path (trail #1)
        '20157', # Duration to complete alphanumeric path (trail #2)
    ]
    if use=='log':
        fn = np.log
    elif use=='sqrt':
        fn = np.sqrt
    for c in df_copy.columns:
        for cat in trans_freq:
            if cat in c:
                df_copy[c+'-1'] = df_copy[c].apply(fn)
    return df_copy

def cv_classify(df, classifier='dtree', cv_fold=10, scaler=True, balance=True):
    """n-fold cross validation classification"""
    from sklearn.model_selection import cross_validate
    # dummify labels
    y = df['label']
    # y_label = df['label']
    # y = pd.get_dummies(y_label).iloc[:,0]
    X = df.drop(['label','eid'], axis=1)
    # balance dataset
    if balance:
        from imblearn.under_sampling import RandomUnderSampler
        # define undersampling strategy
        under = RandomUnderSampler(random_state=0)
        # fit and apply the transform
        X, y = under.fit_resample(X, y)
    # apply scaler
    if scaler:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    # define classifier
    if classifier == 'dtree':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=5)
    elif classifier == 'rforest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=5)
    elif classifier == 'lgb':
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_jobs=-1)
    # cv result
    print(len(np.unique(y)))
    if len(np.unique(y)) <= 2: # binary
        cv_results = cross_validate(clf, X, y, cv=cv_fold, return_train_score=False, scoring=('accuracy', 'f1', 'roc_auc'))
        df_res = pd.DataFrame(cv_results)
        # print res
        print(f"{cv_fold}-fold CV classification with classifier {clf}:\n"
            f"test ROC AUC={df_res['test_roc_auc'].mean():.4f}, test accuracy={df_res['test_accuracy'].mean():.4f}, test f1={df_res['test_f1'].mean():.4f}")
    else:
        cv_results = cross_validate(clf, X, y, cv=cv_fold, return_train_score=False, scoring=('accuracy', 'f1_micro', 'roc_auc_ovo'))
        df_res = pd.DataFrame(cv_results)
        # print res
        print(f"{cv_fold}-fold CV classification with classifier {clf}:\n"
            f"test ROC AUC={df_res['test_roc_auc_ovo'].mean():.4f}, test accuracy={df_res['test_accuracy'].mean():.4f}, test f1={df_res['test_f1_micro'].mean():.4f}")

    return df_res

def basic_classify(df, classifier='dtree', test_size=0.5, random_state=10, plot_figs=True, save_plot=True, save_name='', num_importance=20, questionnaire='all', idp=None, scaler=True, balance=True):
    """basic classification"""
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    
    y = df['label']
    X = df.drop(['label','eid'], axis=1)
    # balance dataset
    if balance:
        from imblearn.under_sampling import RandomUnderSampler
        # define undersampling strategy
        under = RandomUnderSampler(random_state=0)
        # fit and apply the transform
        X, y = under.fit_resample(X, y)
    # apply scaler
    if scaler:
        from sklearn.preprocessing import StandardScaler
        X_scale = StandardScaler().fit_transform(X)
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=test_size, random_state=random_state, stratify=y)
    _, X_test_cols, _, _ = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    if classifier == 'dtree':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
    elif classifier == 'rforest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=5).fit(X_train, y_train)
    elif classifier == 'lgb':
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_jobs=-1)
    # predict test set
    y_test_predicted = clf.predict(X_test)
    
    # plotting confusion matrix and feature importance
    if plot_figs:
        # creating a confusion matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, y_test_predicted)
        cm_display = ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
        if save_plot:
            plt.xticks(rotation=90)
            plt.savefig(f'./figs/{save_name}_{classifier}_cm.png', bbox_inches='tight')

        # plot permutation importantce
        from sklearn.inspection import permutation_importance
        plot_num = num_importance
        result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        # result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()
        question_labels = match_question(X_test_cols.columns[sorted_idx[-plot_num:]],questionnaire=questionnaire, idp=idp)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.boxplot(result.importances[sorted_idx[-plot_num:]].T, vert=False, labels=question_labels)
        ax.set_title("Permutation Importances (test set)")
        # ax.set_title("Permutation Importances (train set)")
        if save_plot:
            plt.savefig(f'./figs/{save_name}_{classifier}_importance.png', bbox_inches='tight')

    # calculate accuracy / auc
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y))==2: # binary class
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    else: # multiclass
        auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    # calculate train/test accuracy
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Classification report for classifier {clf}:\n"
        f"{classification_report(y_test, y_test_predicted)}\n"
        f"ROC AUC={auc:.4f}, train accuracy={train_acc:.4f}, test accuracy={test_acc:.4f}")
    return auc, train_acc, test_acc

def match_question(q_codes, questionnaire='all', idp=None):
    """backward search questions to match question code"""
    df_qs = load_qscode(questionnaire=questionnaire, idp=idp)
    question_ls = []
    for c in q_codes:
        code = int(c.split('-')[0])
        question = df_qs[df_qs['code']==code]['Field title'].values
        question_ls.append(question)
    return question_ls

def load_patient_grouped(pain_status='all', questionnaire='all', idp='all', question_visits=[2], imputed=True, patient_grouping='simplified'):
    """load patient grouped and impute"""
    # load subjects
    if pain_status == 'all': # no necessarily have pain
        df_subjects = pd.read_csv('./data/qsidp_patients.csv')
    elif pain_status == 'must': # condition and pain
        df_subjects = pd.read_csv('./data/qsidp_patients_pain.csv')
    elif pain_status == 'restricted': # pain site restricted to condition
        df_subjects = pd.read_csv('./data/qsidp_patients_pain_restricted.csv')
    # create disease label
    df_disease_label = disease_label(df_subjects, visits=[2], grouping=patient_grouping)
    # load question code
    qs = load_qscode(questionnaire=questionnaire, idp=idp)
    # extract questionnaire of interest
    df_qs = extract_qs(df_subjects, df_questionnaire=qs, visits=question_visits)
    # exclude multi diseases subjects
    df_exclude, df_label_exclude = exclude_multidisease(df_qs, df_disease_label)

    # reverse one hot encoding
    label_exclude = df_label_exclude.idxmax(axis=1)
    # code label to binary
    if 'chronic central pain' in label_exclude: # binary case
        lab = label_exclude=='chronic central pain'
        label = lab.astype(int)
    else: # non-binary case
        label = label_exclude

    dff = df_exclude.merge(label.rename('label'), left_index=True, right_index=True)
    # impute
    if imputed==True:
        print(f'Questionnaires from visits {question_visits} shape={dff.shape}')
        dff_imputed = impute_qs(dff, freq_fill='median', nan_percent=0.9, transform=False)
        print(f'After imputation shape={dff_imputed.shape}')
    elif imputed==False:
        dff_imputed = dff.dropna()
    else:
        dff_imputed = dff
    return dff_imputed


# running
if __name__=="__main__":
    # load questionnaire codes
    question_visits = [2]
    questionnaire = 'all'
    idp = 'all'
    pain_status = 'restricted'# 'must', 'all'
    # load data
    # dff_imputed = load_patient_grouped(pain_status=pain_status, questionnaire=questionnaire, idp=idp, question_visits=question_visits, imputed=True, patient_grouping='simplified')
    dff_imputed = load_patient_grouped(pain_status=pain_status, questionnaire=questionnaire, idp=idp, question_visits=question_visits, imputed=True, patient_grouping='grouped')

    # basic classification
    classifiers = ['rforest']#'dtree', 
    for c in classifiers:
        # basic_classify(dff_imputed, classifier=c, random_state=0, test_size=0.25, save_plot=True, num_importance=20, questionnaire=questionnaire, idp=idp, save_name='paintype', scaler=True, balance=True)
        basic_classify(dff_imputed, classifier=c, random_state=0, test_size=0.25, save_plot=True, num_importance=20, questionnaire=questionnaire, idp=idp, save_name='paintype_restricted_2', scaler=True, balance=True)
        # dfr = cv_classify(dff_imputed, classifier=c, cv_fold=10, questionnaire=questionnaire, idp=idp)