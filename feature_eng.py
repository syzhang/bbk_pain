"""
feature engineering and transform
"""
import os
import sys
import numpy as np
from numpy.core.fromnumeric import transpose
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings; warnings.simplefilter('ignore')

from connectivity_mat import load_connectivity

def prep_colnames(df):
    """preparing column names"""
    QS = [col for col in df.columns if '_q' in str(col)]
    IDP = [col for col in df.columns if '_i' in str(col)]
    CONN = [col for col in df.columns if '_c' in str(col)]
    return QS, IDP, CONN

def parse_qic(df, qic_str):
    """parse qic string"""
    QS, IDP, CONN = prep_colnames(df)
    proc_cols = []
    if 'q' in qic_str:
        proc_cols += QS
    elif 'i' in qic_str:
        proc_cols += IDP
    elif 'c' in qic_str:
        proc_cols += CONN
    return proc_cols

def quantile_trans(df_train, df_test, qic='qic', is_train=True):
    """applying quantile transformer"""
    from sklearn.preprocessing import QuantileTransformer
    model_output_folder = './feats_out'
    os.makedirs(model_output_folder, exist_ok=True)
    # which cols to transform
    trans_cols = parse_qic(df_train, qic)
    # apply transform
    for col in trans_cols:
        vec_len = len(df_train[col].values)
        vec_len_test = len(df_test[col].values)
        raw_vec = df_train[col].values.reshape(vec_len, 1)
        if is_train:
            transformer = QuantileTransformer(n_quantiles=100,
                                            random_state=0,
                                            output_distribution="normal")
            transformer.fit(raw_vec)
            pd.to_pickle(transformer, f'{model_output_folder}/{col}_quantile_transformer.pkl')
        else:
            transformer = pd.read_pickle(f'{model_output_folder}/{col}_quantile_transformer.pkl')

        df_train[col] = transformer.transform(raw_vec)
        df_test[col] = transformer.transform(
            df_test[col].values.reshape(vec_len_test,1)).reshape(1, vec_len_test)[0]
        return df_train, df_test

def factor_analysis(df_train, df_test, n_comp=50, qic='qic', is_train=True):
    """factor analysis"""
    from sklearn.decomposition import PCA,FactorAnalysis
    model_output_folder = './feats_out'
    # which cols to transform
    QS, IDP, CONN = prep_colnames(df_train)
    trans_cols = parse_qic(df_train, qic)
    label = 'qic'
    data = pd.concat([pd.DataFrame(df_train[trans_cols]),
                        pd.DataFrame(df_test[trans_cols])])
    if is_train:
        fa = FactorAnalysis(n_components=n_comp,
                            random_state=1903).fit(data[trans_cols])
        pd.to_pickle(fa, f'{model_output_folder}/factor_analysis_{label}.pkl')
    else:
        fa = pd.read_pickle(f'{model_output_folder}/factor_analysis_{label}.pkl')
    data2 = fa.transform(data[trans_cols])
    train2 = data2[:df_train.shape[0]]
    test2 = data2[-df_test.shape[0]:]
    # rename cols
    train2 = pd.DataFrame(train2, columns=[f'{label}-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'{label}-{i}' for i in range(n_comp)])

    train = pd.concat((df_train.reset_index(drop=True), train2.reset_index(drop=True)), axis=1)
    test = pd.concat((df_test.reset_index(drop=True), test2.reset_index(drop=True)), axis=1)
    return train, test 

def umap_analysis(df_train, df_test, n_comp=50, qic='qic', is_train=True):
    """umap analysis"""
    from umap import UMAP
    model_output_folder = './feats_out'
    # which cols to transform
    trans_cols = parse_qic(df_train, qic)
    label = qic
    data = pd.concat([pd.DataFrame(df_train[trans_cols]),
                        pd.DataFrame(df_test[trans_cols])])
    if is_train:
        um = UMAP(n_components=n_comp, random_state=1903).fit(data[trans_cols])
        pd.to_pickle(um, f'{model_output_folder}/umap_analysis_{label}.pkl')
    else:
        um = pd.read_pickle(f'{model_output_folder}/umap_analysis_{label}.pkl')
    data2 = um.transform(data[trans_cols])
    train2 = data2[:df_train.shape[0]]
    test2 = data2[-df_test.shape[0]:]
    # rename cols
    train2 = pd.DataFrame(train2, columns=[f'{label}-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'{label}-{i}' for i in range(n_comp)])

    train = pd.concat((df_train.reset_index(drop=True), train2.reset_index(drop=True)), axis=1)
    test = pd.concat((df_test.reset_index(drop=True), test2.reset_index(drop=True)), axis=1)
    return train, test 

def cluster_feats(df_train, df_test, n_comp=50, qic='qic'):
    """kmeans cluster features"""    
    from sklearn.cluster import KMeans
    model_output_folder = './feats_out'
    # which cols to transform
    trans_cols = parse_qic(df_train, qic)
    label = 'kmeans'
    
    def create_cluster(train, test, features, kind, n_clusters = n_comp):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans = KMeans(n_clusters = n_clusters, random_state = 1903).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        # print(train.head)
        # print(test.head)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(df_train, df_test, trans_cols, kind=label, n_clusters = n_comp)

    return train, test

def tsplit(df, test_size, random_state, scale=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    from imblearn.under_sampling import RandomUnderSampler
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, 
                              stratify=df['label'])
    train.drop(columns='eid', inplace=True)
    test.drop(columns='eid', inplace=True)
    cols = train.columns.to_list()
    cols.remove('label')
    
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train[cols])
        train[cols] = scaler.transform(train[cols])
        test[cols] = scaler.transform(test[cols])

    return train, test

def rforest_train(train, test):
    """random forest quick train"""
    from sklearn.ensemble import RandomForestClassifier
    target_col = 'label'
    clf = RandomForestClassifier(max_depth=5,random_state=42)
    clf.fit(train.drop(columns=target_col), train[target_col].values.ravel())
    test_pred = clf.predict(test.drop(columns=target_col))
    test_proba = clf.predict_proba(test.drop(columns=target_col))
    test_targ = test[target_col]
    # one hot encode multiclass
    if len(np.unique(test_targ)) > 2:
        auc = roc_auc_score(test_targ, test_proba, multi_class='ovo')
        print(f'test roc auc={auc}')
    else:
        auc = roc_auc_score(test_targ, test_proba)
    df_res = pd.DataFrame({'test_roc_auc_ovr': [auc]})
    return df_res

# running
if __name__=="__main__":

    data_name = sys.argv[1]
    conn_type = 'fullcorr_100'
    # data_name = 'paintype_restricted'

    dir_name = f'./model_performance/output_features/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    df = load_connectivity(task_name=data_name, conn_type=conn_type,add_questionnaire=True, add_idp=True, add_conn=True, patient_grouping='grouped')
    # df = load_connectivity(task_name='paintype_restricted', conn_type='fullcorr_100',add_questionnaire=qs_id, add_idp=idp_id, add_conn=conn_id, patient_grouping='simplified')
    print(df.shape)
    print(df['label'].value_counts())
    # encode label
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(df['label']))
    df['label'] = le.transform(df['label'])
    print(df['label'].value_counts())

    test_size=0.25
    quantile_label = ''
    factor_label = ''
    results = []    
    opts = [False, True]
    qic_opts = ['q', 'i', 'c', 'qi', 'qc', 'ic', 'qic']

    save_fname = dir_name + f'feature_eng_umap_{data_name}.csv'
    ### testing best umap setting
    for qic_id in qic_opts:
        for n in range(2, 20, 3): # number of components
            for random_state in range(4): # cv
                # train test split
                train, test = tsplit(df, test_size, random_state, scale=False)
                # umap analysis
                train, test = umap_analysis(train, test, n_comp=n, qic=qic_id, is_train=True)
                # rforest train
                res = rforest_train(train, test)
                res['dataset'] = qic_id+'_'+str(n)
                results.append(res)

    # save_fname = dir_name + f'feature_eng_{data_name}.csv'
    ### testing various combinations of feats
    # for quantile_id in opts:
    #     for umap_id in opts:
    #         for factor_id in opts:
    #             for kmeans_id in [False]:
    #                 for qic_id in qic_opts:
    #                     for random_state in range(4): # cv
    #                         # train test split
    #                         train, test = tsplit(df, test_size, random_state, scale=False)
    #                         # label
    #                         quantile_label = 'quantile-'+qic_id
    #                         kmeans_label = 'kmeans-'+qic_id
    #                         factor_label = 'fa-'+qic_id
    #                         umap_label = 'umap-'+qic_id
    #                         all_labels = [quantile_label*quantile_id, kmeans_label*kmeans_id, factor_label*factor_id, umap_label*umap_id]
    #                         all_label = '_'.join(all_labels)
    #                         check_steps = 0
    #                         # transform
    #                         if quantile_id:
    #                             train, test = quantile_trans(train, test, is_train=True, qic=qic_id)
    #                             check_steps += 1
    #                         # kmeans cluster
    #                         if kmeans_id:
    #                             train, test = cluster_feats(train, test, qic=qic_id, n_comp=20)
    #                             check_steps += 1
    #                         # factor analysis
    #                         if factor_id:
    #                             train, test = factor_analysis(train, test, n_comp=20, qic=qic_id, is_train=True)
    #                             check_steps += 1
    #                         # umap analysis
    #                         if umap_id:
    #                             train, test = umap_analysis(train, test, n_comp=20, qic=qic_id, is_train=True)
    #                             check_steps += 1
    #                         # rforest train
    #                         if check_steps>0:
    #                             res = rforest_train(train, test)
    #                             print(all_label)
    #                             res['dataset'] = all_label
    #                             results.append(res)
    df_res = pd.concat(results)
    print(df_res)
    # save
    print(save_fname)
    df_res.to_csv(save_fname, index=None)