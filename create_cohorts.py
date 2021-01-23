import pandas as pd 
import numpy as np
import os 
import pickle as pk 
import random
import argparse
from tqdm import tqdm

from options import args


def main():
    """
        Prepare two DRG cohorts from MIMIC-III: MS, APR
        And extract notes for each stay prior to threshold hour
    """

    for cohort in ['ms', 'apr']:
    # for cohort in ['apr']:
        print('Construct cohort for %s drg' % cohort)
        df = getDF(cohort, args.mimic_dir)
        notes_df = loadNOTES(df[['SUBJECT_ID', 'HADM_ID', 'INTIME']])

        construct_cohort(df, notes_df, cohort)
        print('\n\n\n')



def construct_cohort(drg_df, notes_df, drg_type='ms'):
    drg_path = '%s/%s' % (args.data_dir, drg_type)
    text_dir = '%s/text_raw/' % drg_path
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    pairs = []
    for _, row in tqdm(drg_df.iterrows(), total=len(drg_df)):
        sub = row['SUBJECT_ID']
        hadm = row['HADM_ID']
        stay = row['stay']
        adm_diag = row['DIAGNOSIS']

        hours = extract_note_append_adm_diag(sub, hadm, adm_diag, notes_df, text_dir)

        if hours:
            pairs.append((stay, *hours))

    df_tmp = pd.DataFrame.from_records(pairs, columns=['stay','hour0','hour12','hour24','hour36'])
    drg_df_h = pd.merge(drg_df, df_tmp, on=['stay'])
    print("at least one note before the 48h threshold at icu")

    pt, st = len(drg_df_h.SUBJECT_ID.unique()), len(drg_df_h)
    print("..there are {} pt, {} stays w/ {} drg in total".format(pt, st, drg_type))

    split_cohort(drg_df_h, drg_path)

def getDF(drg_type, mimic_dir):
    if drg_type == 'ms':
        drg_df = getMS(mimic_dir)
    elif drg_type == 'apr':
        drg_df = getAPR(mimic_dir)

    return filterCohort(drg_df, mimic_dir)

def getMS(mimic_dir):
    drg = pd.read_csv(os.path.join(mimic_dir, 'DRGCODES.csv'), usecols=['SUBJECT_ID', 'HADM_ID','DRG_TYPE', 'DRG_CODE', 'DESCRIPTION'])
    ms_df = drg[drg.DRG_TYPE == 'MS'] # using only MS codes
    ms_df = ms_df.dropna() # dropping one invalid code that does not have description nor mapping to the official drg document: 9 and 170; 12 cases in total
    drg_df = ms_df[['SUBJECT_ID', 'HADM_ID','DRG_CODE']] # 21551 pts and 27267 events
    print("raw cases w/ MS drg: pt - {}, stays - {}".format( len(drg_df.SUBJECT_ID.unique()), len(drg_df)))

    return drg_df

def getAPR(mimic_dir):
    drg = pd.read_csv(os.path.join(mimic_dir, 'DRGCODES.csv'), usecols=['SUBJECT_ID', 'HADM_ID','DRG_TYPE', 'DRG_CODE', 'DESCRIPTION'])
    apr_df = drg[drg.DRG_TYPE == 'APR '] # using only APR codes
    apr_df = apr_df.drop_duplicates(subset=['HADM_ID','DRG_CODE', 'DESCRIPTION'])
    raw = len(apr_df)

    # drop duplicates
    # first drop same desc but different severities -- keeping only the highest one
    apr_df = apr_df.sort_values(by=['HADM_ID', 'DESCRIPTION'], ascending=False).drop_duplicates(subset=['HADM_ID', 'DESCRIPTION'])
    print("{} raw apr codes in MIMIC, {} after first dropping duplicated severity".format(raw, len(apr_df)))

    # second drop hadm that are assigned with two codes 
    dup_mask = apr_df[apr_df.duplicated(subset=['HADM_ID'])].HADM_ID
    apr_df = apr_df[~apr_df.HADM_ID.isin(dup_mask)]
    drg_df = apr_df[['SUBJECT_ID', 'HADM_ID','DRG_CODE']]
    print("raw cases w/ APR drg: pt - {}, stays - {}".format( len(drg_df.SUBJECT_ID.unique()), len(drg_df)))

    return drg_df

def filterCohort(drg_df, mimic_dir):
    """
        1. filter adult
        2. filter single ICU stay - only one icu stay & no transfer

        return: df of stay information
    """

    icu = pd.read_csv(os.path.join(mimic_dir, 'ICUSTAYS.csv'))
    icu_count = icu.groupby(['HADM_ID']).size()
    icu_once = icu.HADM_ID.isin(icu_count[icu_count==1].index)
    icu_no_transfer = (icu.FIRST_WARDID == icu.LAST_WARDID) & (icu.FIRST_CAREUNIT == icu.LAST_CAREUNIT)
    icu_single = icu[icu_once & icu_no_transfer][['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID','INTIME', 'DBSOURCE','LOS']]
    adm = pd.read_csv(os.path.join(mimic_dir, 'ADMISSIONS.csv'), usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME','DISCHTIME','DIAGNOSIS'])
    pt = pd.read_csv(os.path.join(mimic_dir, 'PATIENTS.csv'), usecols=['SUBJECT_ID','DOB'])

    # map to adm and pt to get age
    drg_adm = pd.merge(drg_df, adm, on=['SUBJECT_ID', 'HADM_ID'])
    drg_dob = pd.merge(drg_adm, pt, on=['SUBJECT_ID'])
    admtime = pd.to_datetime(drg_dob.ADMITTIME).dt.date.apply(np.datetime64)
    dobtime = pd.to_datetime(drg_dob.DOB).dt.date.apply(np.datetime64)
    age = (admtime.subtract(dobtime)).astype('timedelta64[Y]')
    drg_dob['AGE'] = age.apply(lambda x: 90 if x < 0 else x)

    drg_adult = drg_dob[drg_dob.AGE >= 18]
    print("age over 18 w/ drg: pt - {}, visits - {}".format(len(drg_adult.SUBJECT_ID.unique()), len(drg_adult) ))

    drg_final = pd.merge(drg_adult, icu_single, on=['SUBJECT_ID', 'HADM_ID'])
    print("plus criterion on single icu: pt - {}, visits - {}".format(len(drg_final.SUBJECT_ID.unique()), len(drg_final) ))

    drg_final['stay'] = drg_final.apply(lambda row: "{}_{}".format(row['SUBJECT_ID'], row['HADM_ID']), axis=1)

    drg_final.drop(['DOB', 'ICUSTAY_ID', 'DISCHTIME'], axis=1)

    return drg_final

def loadNOTES(event_df):
    """
        load noteevents.csv and map to event_df (sub, hadm) and calculate relative note time
    """
    # map notes 
    cols = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'CATEGORY', 'TEXT']
    print("Loading MIMIC notes...209iter")
    iter_notes = pd.read_csv(os.path.join(args.mimic_dir, 'NOTEEVENTS.csv'), iterator=True, usecols=cols, chunksize=10000)
    dfList = []
    for d in tqdm(iter_notes, total=209):
        dfList.append(event_df.merge(d, on=['SUBJECT_ID', 'HADM_ID']))
    notes_df = pd.concat(dfList)
    print("%d note events in total..." % len(notes_df))
    # notes_df, remove those without charttime, like discharge summary
    notes_df = notes_df.dropna(subset=['CHARTTIME'])
    print("%d note events w/ charttime" % len(notes_df))
    # get diff time for each note based on icu intime
    intime = notes_df.INTIME.apply(np.datetime64)
    charttime = notes_df.CHARTTIME.apply(np.datetime64)
    diff = (charttime-intime).astype('timedelta64[m]').astype(float)
    notes_df['DIFFTIME'] = diff / 60. # in hours

    return notes_df

def extract_note_append_adm_diag(sub, hadm, adm_diag, notes_df, output_dir):
    """
        extract and save notes for each visit
        get info about note availability
    """
    note_slice = notes_df[(notes_df.SUBJECT_ID == sub) & (notes_df.HADM_ID == hadm)].sort_values('CHARTTIME')
    stay = "{}_{}".format(sub, hadm)
    output_file = stay+'.pk' 

    if len(note_slice) == 0:
        return None

    note_slice = note_slice[['CATEGORY', 'DIFFTIME', 'TEXT']]

    earlies = note_slice.DIFFTIME.iloc[0]
    hour0 = 1 if earlies < 0 else 0
    hour12 = 1 if earlies<12 else 0
    hour24 = 1 if earlies<24 else 0
    hour36 = 1 if earlies<36 else 0
    # hour48 = 1 if earlies<48 else 0
    hours = (hour0, hour12, hour24, hour36)

    valid_mask = note_slice.DIFFTIME <= args.threshold
    valid_df = note_slice[valid_mask]

    if len(valid_df) == 0:
        return None
    else:
        if adm_diag != adm_diag: # check for nan
            adm_diag = ''
        adm_diag_df = pd.DataFrame([{'CATEGORY': 'admission_diag', 'DIFFTIME': -1e+5, 'TEXT': adm_diag}])
        valid_df = pd.concat([adm_diag_df, valid_df], ignore_index=True)
        with open(os.path.join(output_dir, output_file), 'wb') as f:
            pk.dump(valid_df, f, pk.HIGHEST_PROTOCOL)
            # pk.dump(note_slice, f, pk.HIGHEST_PROTOCOL)
        return hours

def split_cohort(drg_df, output_dir):
    """
        create train, val, and test split 
        make sure stays of same pt in the single split 
    """

    def split_patients(df, frac=0.1):
        pt_count = df.groupby(['SUBJECT_ID']).size()
        pt_multi = pd.Series(pt_count[pt_count >1].index)
        pt_single= pd.Series(pt_count[pt_count==1].index)

        assert len(pt_single) + len(pt_multi) == len(df.SUBJECT_ID.unique())

        test_multi = pt_multi.sample(frac=frac, random_state=1443)
        test_single = pt_single.sample(frac=frac, random_state=1443)
        test_pt = test_single.append(test_multi)
        test_mask = df.SUBJECT_ID.isin(test_pt)

        test_df = df[test_mask]
        train_df = df[~test_mask]

        return train_df, test_df

    # create test split
    train_val, test = split_patients(drg_df)

    assert len(train_val)+len(test) == len(drg_df)

    # save all info
    drg_df.to_csv('%s/drg_cohort.csv' % output_dir, index=False)
    train_val.to_csv('%s/train_val.csv' % output_dir, index=False)
    test.to_csv('%s/test.csv' % output_dir, index=False)

    tr_pt, tr_st = len(train_val.SUBJECT_ID.unique()), len(train_val)
    te_pt, te_st = len(test.SUBJECT_ID.unique()), len(test)

    print("..split into train ({} pt, {} st), and test ({} pt, {} st).".format(tr_pt, tr_st, te_pt, te_st))


if __name__ == "__main__":
    main()
