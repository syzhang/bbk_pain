"""
selecting subjects of interest
"""
import os
import numpy as np
import pandas as pd
from clean_questions import *
from disease_type import extract_disease
from compare_control import extract_control

def select_patients(save=True):
    """select patients with conditions (pain or no pain)"""
    # load funpack output
    df_path = os.path.join('..', 'funpack_cfg', 'qsidp_subjs_disease_allvisits_extended.tsv')
    df_subjects = pd.read_csv(df_path, sep='\t')
    # select disease of interest
    disease_status = disease_label(df_subjects, visits=[2], grouping='simplified')
    # slice out disease subjects
    disease_eid = disease_status[disease_status.sum(axis=1)>0].index
    df_out = df_subjects[df_subjects['eid'].isin(disease_eid)]
    # save
    if save:
        df_out.to_csv('./data/qsidp_patients.csv', index=None)
    return df_out

def select_patients_pain(save=True):
    """select patients with conditions and pain"""
    # read patients
    df_patients = pd.read_csv('./data/qsidp_patients.csv')
    # restrict to those with pain
    df_out = cwp_positive(df_patients)
    # save
    if save:
        df_out.to_csv('./data/qsidp_patients_pain.csv', index=None)
    return df_out

def select_patients_pain_restricted(save=True):
    """select patients with conditions and pain restricted to body sites"""
    # load patients with pain
    df_pp = pd.read_csv('./data/qsidp_patients_pain.csv')
    # hard coded condition-pain pairs
    df_rp0 = restrict_pain_to_disease(df_pp, disease_code=1265, cwp_code=[3799, 4067]) # migraine, 3+m headache or facial pain
    df_rp1 = restrict_pain_to_disease(df_pp, disease_code=1465, cwp_code=[3733, 3414, 3404]) # OA, 3+m knee/hip/neck or shoulder pain
    df_rp2 = restrict_pain_to_disease(df_pp, disease_code=1464, cwp_code=[3733, 3414, 3404]) # RA, 3+m knee/hip/neck or shoulder pain
    df_rp3 = restrict_pain_to_disease(df_pp, disease_code=1477, cwp_code=[3733, 3414, 3404]) # psoriatic arthropathy, 3+m knee/hip/neck or shoulder pain
    df_rp4 = restrict_pain_to_disease(df_pp, disease_code=1538, cwp_code=[3733, 3414, 3404]) # psoriatic arthropathy, 3+m knee/hip/neck or shoulder pain
    df_rp5 = restrict_pain_to_disease(df_pp, disease_code=1154, cwp_code=3741) # IBS, 3+m stomach pain
    df_rp6 = restrict_pain_to_disease(df_pp, disease_code=1294, cwp_code=3571) # back problem, 3+m back pain
    df_rp7 = restrict_pain_to_disease(df_pp, disease_code=1478, cwp_code=3571) # cervical spondylosis, 3+m back pain
    df_rp8 = restrict_pain_to_disease(df_pp, disease_code=1311, cwp_code=3571) # spine arthritis/spondylitis, 3+m back pain
    df_rp9 = restrict_pain_to_disease(df_pp, disease_code=1312, cwp_code=3571) # prolapsed disc/slipped disc, 3+m back pain
    df_rp10 = restrict_pain_to_disease(df_pp, disease_code=1532, cwp_code=3571) # disc problem, 3+m back pain
    df_rp11 = restrict_pain_to_disease(df_pp, disease_code=1533, cwp_code=3571) # disc degeneration, 3+m back pain
    df_rp12 = restrict_pain_to_disease(df_pp, disease_code=1534, cwp_code=3571) # back pain, 3+m back pain
    df_rp13 = restrict_pain_to_disease(df_pp, disease_code=1542, cwp_code=2956) # fibromyalgia, 3+m general pain
    # concat all
    df_out = pd.concat([df_rp0, df_rp1,df_rp2, df_rp3, df_rp4, df_rp5, df_rp6, df_rp7, df_rp8, df_rp9, df_rp10, df_rp11, df_rp12, df_rp13])
    df_out.drop_duplicates(subset='eid', inplace=True)
    # save
    if save:
        df_out.to_csv('./data/qsidp_patients_pain_restricted.csv', index=None)
    return df_out

def restrict_pain_to_disease(df, disease_code, cwp_code):
    """returns pain site restricted to disease"""
    dfd_tmp = extract_disease(df, int(disease_code), visit=[2]) # disease
    dfd_eid = pd.Series(dfd_tmp[dfd_tmp.values==1].index)
    # loop through cwps
    if isinstance(cwp_code, list):
        cols = []
        for cwp in cwp_code:
            cols_tmp = check_field(df, int(cwp), visit=2) # cwp
            cols += cols_tmp
    else:
        cols = check_field(df, int(cwp_code), visit=2) # cwp
    # extract patients from df
    dfp_tmp = check_count(df, cols, 1)
    dfp_tmp.drop_duplicates(subset='eid', inplace=True)
    dfp_eid = dfp_tmp['eid']
    eids = dfd_eid[dfd_eid.isin(dfp_eid)]
    df_out = df[df['eid'].isin(eids)]
    return df_out

def cwp_positive(df, positive=True):
    """return subset of subjects with pain"""
    # restrict to those with Pain type(s) experienced in last month
    cols_ls = check_field(df, 6159, visit=2)
    df_np = check_count(df, cols_ls, -7)
    if positive:
        df_plm = df[~df['eid'].isin(df_np['eid'])]
    else:
        df_plm = df_np
    # restrict to those with one of the 3+ month cwp
    fields = pd.read_csv('./bbk_codes/cwp_code.csv')
    ff = fields[fields['code']!=6159]['code'].to_list()
    cols = []
    for f in ff:
        cols += check_field(df_plm, f, visit=2)
    if positive:
        df_ppp = check_count(df_plm, cols, 1)
    else:
        df_ppp = df_plm
    df_out = df_ppp.drop_duplicates('eid')
    return df_out
    
def select_pain_plus(save=True):
    """select those with pain"""
    # read patients
    df_patients = pd.read_csv(os.path.join('..', 'funpack_cfg', 'qsidp_subjs_disease_allvisits_extended.tsv'), sep='\t')
    # read controls 
    df_controls = pd.read_csv(os.path.join('..', 'funpack_cfg', 'qsidp_subjs_control_allvisits_extended.tsv'), sep='\t')
    # restrict to those with pain
    dfp_out = cwp_positive(df_patients)
    dfc_out = cwp_positive(df_controls)
    # combine
    df_out = pd.concat([dfp_out, dfc_out])
    # save
    if save:
        df_out.to_csv('./data/qsidp_pain_plus.csv', index=None)
    return df_out

def select_pain_minus(save=True):
    """select those without pain"""
    # read patients
    df_patients = pd.read_csv(os.path.join('..', 'funpack_cfg', 'qsidp_subjs_disease_allvisits_extended.tsv'), sep='\t')
    # read controls 
    df_controls = pd.read_csv(os.path.join('..', 'funpack_cfg', 'qsidp_subjs_control_allvisits_extended.tsv'), sep='\t')
    # restrict to those with pain
    dfp_out = cwp_positive(df_patients, positive=False)
    dfc_out = cwp_positive(df_controls, positive=False)
    # combine
    df_out = pd.concat([dfp_out, dfc_out])
    # save
    if save:
        df_out.to_csv('./data/qsidp_pain_minus.csv', index=None)
    return df_out

def pain_minus_matched(save=True):
    """age/gender match pain minus to pain plus"""
    df_disease = pd.read_csv('./data/qsidp_pain_plus.csv')
    df_control = pd.read_csv('./data/qsidp_pain_minus.csv')
    df_subjs = extract_control(df_control, df_disease, save_csv=True, save_name='subjs_pain_minus_matched')
    # extract subjs
    df_out = df_control[df_control['eid'].isin(df_subjs)]
    if save:
        df_out.to_csv('./data/qsidp_pain_minus_matched.csv', index=None)
    return df_out

def patients_matched(save=True):
    """age/gender match controls to patients"""
    df_disease = pd.read_csv('./data/qsidp_patients.csv')
    df_control = pd.read_csv('./data/qsidp_controls.csv')
    df_subjs = extract_control(df_control, df_disease, save_csv=True, save_name='subjs_patients_matched')
    # extract subjs
    df_out = df_control[df_control['eid'].isin(df_subjs)]
    if save:
        df_out.to_csv('./data/qsidp_patients_matched.csv', index=None)
    return df_out

def patients_pain_matched(save=True):
    """age/gender match controls to patients_pain"""
    df_disease = pd.read_csv('./data/qsidp_patients_pain.csv')
    df_control = pd.read_csv('./data/qsidp_controls.csv')
    df_subjs = extract_control(df_control, df_disease, save_csv=True, save_name='subjs_patients_pain_matched')
    # extract subjs
    df_out = df_control[df_control['eid'].isin(df_subjs)]
    if save:
        df_out.to_csv('./data/qsidp_patients_pain_matched.csv', index=None)
    return df_out

def patients_pain_restricted_matched(save=True):
    """age/gender match controls to patients_pain_restricted"""
    df_disease = pd.read_csv('./data/qsidp_patients_pain_restricted.csv')
    df_control = pd.read_csv('./data/qsidp_controls.csv')
    df_subjs = extract_control(df_control, df_disease, save_csv=True, save_name='subjs_patients_pain_restricted_matched')
    # extract subjs
    df_out = df_control[df_control['eid'].isin(df_subjs)]
    if save:
        df_out.to_csv('./data/qsidp_patients_pain_restricted_matched.csv', index=None)
    return df_out

def select_controls(save=True):
    """select controls (pain and conditions free)"""
    # read controls 
    df_controls = pd.read_csv(os.path.join('..', 'funpack_cfg', 'qsidp_subjs_control_allvisits_extended.tsv'), sep='\t')
    # restrict to those with no pain
    dfc_out = cwp_positive(df_controls, positive=False)
    # exclude those with conditions
    disease_status = disease_label(dfc_out, visits=[2], grouping='detailed')
    # slice out disease subjects
    nd_eid = disease_status[disease_status.sum(axis=1)==0].index
    df_out = dfc_out[dfc_out['eid'].isin(nd_eid)]
    # save
    if save:
        df_out.to_csv('./data/qsidp_controls.csv', index=None)
    return df_out

def select_digestive(save=True):
    """select digestive prediction set"""
    # load controls
    df_controls = pd.read_csv('./data/qsidp_controls.csv')
    # extract dates
    qs = load_qscode(questionnaire=['digestive', 'demographic'], idp=None)
    df_qs = extract_qs(df_controls, df_questionnaire=qs, visits=[2])
    # rename dates
    df_qs.rename(columns={'53-2.0':'imaging_date', '21023-0.0':'digest_date'}, inplace=True)
    # slice out digestive after imaging
    df_dt = pd.to_datetime(df_qs['digest_date'])
    df_it = pd.to_datetime(df_qs['imaging_date'])
    # calcualte diff
    diff = (df_dt-df_it).values
    dff = diff.astype('int64')
    # those finished digestive after imaging
    df_dig = df_qs[dff>0]
    # output
    df_out = df_controls[df_controls['eid'].isin(df_dig['eid'])]
    # save
    if save:
        df_out.to_csv('./data/qsidp_digestive.csv', index=None)
    return df_out

def check_field(df, field_code, visit=2):
    """check field in visit x"""
    code = field_code # 6159/2956
    df_copy = df.copy()
    if visit is not None:
        code_root = f'{code}-{visit}'
    else:
        code_root = f'{code}-'
    cols_ls = [col for col in df_copy.columns if col[:len(code_root)]==code_root]
    # print(cols_ls)
    return cols_ls

def check_count(df, field_ls, field_status):
    """check status count given field list"""
    df_copy = df.copy()
    rec_ls = []
    for c in field_ls:
        df_tmp = df_copy[df_copy[c]==field_status]
        # print(c, df_tmp.shape)
        rec_ls.append(df_tmp)
    # make df from rec list
    df_out = pd.concat(rec_ls)
    return df_out

# running
if __name__=="__main__":
    # df = select_digestive()
    # df = pain_minus_matched()
    # df = patients_matched()
    # df = patients_pain_matched()
    # df = select_patients_pain_restricted()
    df = patients_pain_restricted_matched()
    print(df.shape)