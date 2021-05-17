"""
compare classifer
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_validate
import xgboost as xgb
import lightgbm as lgb

from imblearn.under_sampling import RandomUnderSampler

from compare_control import *
from clean_questions import *
from predict_digestive import *
from connectivity_mat import load_connectivity

names = ["LGB", "Nearest Neighbors", "Linear SVM", 
         "Decision Tree", "Random Forest", 
         "AdaBoost", "QDA", "XGB"]

classifiers = [
    lgb.LGBMClassifier(n_jobs=-1),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis(),
    xgb.XGBClassifier()
    ]

rng = np.random.RandomState(2)

# classifier flags
questionnaire = 'all'
idp = 'all'
question_visits = [2]
impute_flag = True # fillna w median
# impute_flag = False # dropna
# data_used = 'qsidp' # 'connectivity' 
data_used = 'connectivity' 

# load all datasets
if data_used == 'qsidp':
    datasets = [
                load_patient_grouped(pain_status='all', questionnaire=questionnaire, idp=idp, question_visits=question_visits, imputed=impute_flag, patient_grouping='simplified'), # pain type (all)
                load_patient_grouped(pain_status='must', questionnaire=questionnaire, idp=idp, question_visits=question_visits, imputed=impute_flag, patient_grouping='simplified'), # pain type (must have pain)
                load_digestive_data(label_type='severe', questionnaire=questionnaire, idp=idp, question_visits=question_visits, imputed=impute_flag), # digestive
                # load_patient_matched(questionnaire=questionnaire, idp=idp, question_visits=question_visits, imputed=impute_flag) # patient control
                ]
elif data_used == 'connectivity':
    datasets = [
                load_connectivity(task_name='paintype_all'), # pain type
                load_connectivity(task_name='paintype_must'), # pain type
                load_connectivity(task_name='digestive'), # digestive
                # load_connectivity(task_name='paincontrol') # patient control
                ]
dataset_names = ['paintype_all', 'paintype_must', 'digestive']#, 'paincontrol']

res_ls = []
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    # X, y = ds
    y = ds['label']
    if y.dtype==object:
        y = pd.get_dummies(y).iloc[:,0]
    X = ds.drop(['label','eid'], axis=1)

    # balance data
    # define undersampling strategy
    under = RandomUnderSampler(random_state=0)
    # fit and apply the transform
    X, y = under.fit_resample(X, y)

    # standardise X
    X = StandardScaler().fit_transform(X)
    
    # use pca (so that can plot decision boundary)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # X = pca.fit_transform(X)

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        # cv result
        cv_fold = 4
        cv_results = cross_validate(clf, X, y, cv=cv_fold, return_train_score=False, scoring=('accuracy', 'f1', 'roc_auc'))
        df_res = pd.DataFrame(cv_results)
        df_res['classifier'] = str(clf)
        df_res['dataset'] = dataset_names[ds_cnt]
        score = df_res['test_accuracy'].mean()
        res_ls.append(df_res)
        # print res
        print(f"{cv_fold}-fold CV classification with classifier {clf}:\n"
            f"test ROC AUC={df_res['test_roc_auc'].mean():.4f}, test accuracy={df_res['test_accuracy'].mean():.4f}, test f1={df_res['test_f1'].mean():.4f}")

# performance df
df_perf = pd.concat(res_ls)
# save to csv
df_perf.to_csv(f'./model_performance/output/compare_classifiers_{data_used}.csv', index=None)