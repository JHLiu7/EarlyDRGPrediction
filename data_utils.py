import os
import pandas as pd
import numpy as np
import pickle as pk

import torch
from torch.utils.data import Dataset


class DrgTextDataset(Dataset):
    def __init__(self, args, df, rule_path):
        self.df = df
        self.max_seq_length = args.max_seq_length
        self.data_dir = '%s/%s' % (args.data_dir, args.cohort)
        self.text_dir = '%s/text_processed' % self.data_dir
        self.token2id_dir = '%s/token2id.dict' % self.data_dir
        self.token2id = pd.read_pickle(self.token2id_dir)

        _, self.d2i, _, _, self.d2w = load_rule(rule_path)

        self.unique_pt_df = self.df.sort_values(by=['SUBJECT_ID', 'hour0', 'hour12'], ascending=False).drop_duplicates(subset=['SUBJECT_ID']).reset_index(drop=True)

        self.load_data(48)

        self.Y = len(self.d2i)

    def load_data(self, hour):
        self.size = len(self.df)
        self.data = self.read_df(self.df, hour)
        print('dataset loaded with', self.size, 'stays')

    def load_pop_for_hpa(self, hour):
        self.size = len(self.unique_pt_df)
        self.data = self.read_df(self.unique_pt_df, hour)
        print('examine unique %s pt at %sth hour' % (self.size, hour))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def read_df(self, df, hour):
        data = {}

        df = df.reset_index()

        for n, row in df.iterrows():
            stay = row['stay']
            drg = self.d2i[row['DRG_CODE']]
            rw = self.d2w[row['DRG_CODE']] # relative weight

            X_text = self.read_text(stay, hour)
            sample = {
                'entry': stay,
                'text': torch.tensor(X_text).long(),
                'drg': torch.tensor(drg).long(),
                'rw': torch.tensor(rw).float()
            }
            data[n] = sample
        return data

    def read_text(self, stay, hour):
        path = '%s/%s.dict' % (self.text_dir, stay)
        with open(path, 'rb') as f:
            text_dict = pk.load(f)

        tmp = []
        for time in text_dict:
            if time <= hour:
                text = text_dict[time]
                tmp.extend([self.token2id[w] if w in self.token2id else self.token2id['<unk>'] for w in text])
        tokens = np.array(tmp)

        X_text = np.zeros(self.max_seq_length)
        length = min(self.max_seq_length, len(tokens))

        X_text[:length] = tokens[:length]

        return X_text

def load_rule(path):
    rule_df = pd.read_csv(path)
    
    # remove MDC 15 - neonate and couple other codes related to postcare
    if 'MS' in path:
        msk = (rule_df['MDC']!='15') & (~rule_df['MS-DRG'].isin([945, 946, 949, 950, 998, 999])) 
        space = sorted(rule_df[msk]['DRG_CODE'].unique())
    elif 'APR' in path:
        msk = (rule_df['MDC']!='15') & (~rule_df['APR-DRG'].isin([860, 863])) 
        space = sorted(rule_df[msk]['DRG_CODE'].unique())
        
    drg2idx = {}
    for d in space:
        drg2idx[d] = len(drg2idx)
    i2d = {v:k for k,v in drg2idx.items()}

    d2mdc, d2w = {}, {}
    for _, r in rule_df.iterrows():
        drg = r['DRG_CODE']
        mdc = r['MDC']
        w = r['WEIGHT']
        d2mdc[drg] = mdc
        d2w[drg] = w
        
    return rule_df, drg2idx, i2d, d2mdc, d2w