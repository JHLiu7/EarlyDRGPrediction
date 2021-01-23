import pandas as pd
import numpy as np
import pickle as pk

import os
import time
import math
import copy 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace


from eval_utils import full_metrics, result2str, reg_metrics
from data_utils import load_rule

def eval_test(model, dataset, target, score_func, rule_path=None, print_results=False, batch_size=16):
    loss, score, (y_pred, y, stays) = test(model, dataset, target, batch_size, score_func)

    inf = {'y_pred':y_pred,'y':y,'stay':stays}

    if rule_path == None:
        return score, rule_path
    else:
        if target == 'drg':
            rule_df, d2i, i2d, d2mdc, d2w = load_rule(rule_path)
            result_dict = full_metrics(y_pred, y, rule_df, d2i)
        elif target == 'rw':
            result_dict = reg_metrics(y_pred, y)

    if print_results:
        print(result2str(result_dict))

    return result_dict, inf  

def train_with_early_stopping(model, train_dataset, dev_dataset, epochs, patience, target, batch_size, optimizer, score_func, small_base=True):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0 if small_base else 1e+5
    dev_history, count = [], 0

    for epoch in range(epochs):
        print("training on epoch", epoch+1)
        since = time.time()

        tr_loss = train(model, train_dataset, target, batch_size, optimizer)

        dev_loss, dev_score, _ = test(model, dev_dataset, target, batch_size, score_func)

        better = 'No..'
        cond1 = small_base and (dev_score > best_score)
        cond2 = (not small_base) and (dev_score < best_score)
        if cond1 or cond2:
            best_score = dev_score
            best_model_wts = copy.deepcopy(model.state_dict())
            better = 'Yes!'
            count = epoch+1

        time_elapsed = time.time() - since
        to_print = (time_elapsed // 60, time_elapsed % 60, tr_loss, dev_loss, dev_score, better)
        print("finish in {:.0f}m{:.0f}s tr_loss: {:.3f}, dev_loss: {:.3f}, dev_score: {:.3f}...better? -> {}".format(*to_print))

        # early stopping
        dev_history.append(dev_score)
        if count <= len(dev_history) - patience:
            print('enough patience of', patience, 'and stops at %dth' % count, 'epoch with best dev score %.3f' % best_score)
            break

    return best_model_wts


def train(model, dataset, target, batch_size, optimizer):
    tr_loss, nb_tr_steps = 0., 0.

    model.cuda()
    model.train()
    dloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in dloader:
        x = batch['text'].cuda()
        label = batch[target].cuda()

        optimizer.zero_grad()
        _, loss = model(x, label)

        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_steps += 1

    tr_loss /= nb_tr_steps
    return tr_loss 

def test(model, dataset, target, batch_size, score_func):
    te_loss, nb_te_steps = 0., 0.
    y_pred, y, stays = [], [], []

    model.cuda()
    model.eval()
    dloader =  DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in dloader:
        x = batch['text'].cuda()
        label = batch[target].cuda()
        entry  = batch['entry']

        with torch.no_grad():
            logit, loss = model(x, label)

        te_loss += loss.item()
        nb_te_steps += 1

        logit = logit.detach().cpu().numpy()
        label = label.cpu().numpy()

        y_pred.append(logit)
        y.append(label)
        stays.append(entry)

    y_pred = np.concatenate(y_pred)
    y = np.concatenate(y)
    stays = np.concatenate(stays)

    te_loss /= nb_te_steps
    te_score = score_func(y_pred, y)

    return te_loss, te_score, (y_pred, y, stays)




# func to split train into (train, dev) or folds of (train, dev)
def split_df_by_pt(df, frac=None, k=None):
    pt_count = df.groupby(['SUBJECT_ID']).size()
    pt_multi = pd.Series(pt_count[pt_count >1].index)
    pt_single= pd.Series(pt_count[pt_count==1].index)

    
    assert len(pt_single) + len(pt_multi) == len(df.SUBJECT_ID.unique())

    if frac:
        test_multi = pt_multi.sample(frac=frac, random_state=1443)
        test_single = pt_single.sample(frac=frac, random_state=1443)

        test_pt = test_single.append(test_multi)
        test_mask = df.SUBJECT_ID.isin(test_pt)

        test_df = df[test_mask]
        train_df = df[~test_mask]

        return train_df, test_df

    elif k:
        np.random.RandomState(seed=1443).shuffle(pt_multi)
        np.random.RandomState(seed=1443).shuffle(pt_single)

        tick1 = int( len(pt_multi) / k )
        tick2 = int( len(pt_single)/ k )

        pt_multi10 = int( len(pt_multi) * .1)
        pt_single10= int( len(pt_single)* .1)

        splits = []    
        i = 0
        while i < k-1:
            subj1 = pt_multi[i*tick1 : (i+1)*tick1]
            subj2 = pt_single[i*tick2 : (i+1)*tick2]
            test_subj = pd.concat([subj1, subj2])

            train_df = df[~df.SUBJECT_ID.isin(test_subj)]
            test_df = df[df.SUBJECT_ID.isin(test_subj)]

            test_subj10=pd.concat([subj1[:pt_multi10], subj2[:pt_single10]])
            train_df90 = df[~df.SUBJECT_ID.isin(test_subj10)]
            test_df10 = df[df.SUBJECT_ID.isin(test_subj10)]

            splits.append((train_df, test_df, train_df90, test_df10))
            i+=1

        subj1 = pt_multi[i*tick1 : ]
        subj2 = pt_single[i*tick2 : ]
        test_subj = pd.concat([subj1, subj2])

        train_df = df[~df.SUBJECT_ID.isin(test_subj)]
        test_df = df[df.SUBJECT_ID.isin(test_subj)]

        test_subj10=pd.concat([subj1[:pt_multi10], subj2[:pt_single10]])
        train_df90 = df[~df.SUBJECT_ID.isin(test_subj10)]
        test_df10 = df[df.SUBJECT_ID.isin(test_subj10)]

        splits.append((train_df, test_df, train_df90, test_df10))
        
        assert len(splits) == k
        return splits

def update_args(args, params):
    # update hyperparams in args
    argcopy = vars(args)
    for k,v in params.items():
        argcopy[k] = v
    args = Namespace(**argcopy)
    return args

# utils to save results
def dump_outputs(result_dir, test_infs, hpa_results=None, hpa_infs=None, checkpoint=None, hyperparam=None, dev_infs=None):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open('%s/test_infs.pk' % result_dir, 'wb') as outf1:
        pk.dump(test_infs, outf1)

    if hpa_infs:
        with open('%s/hpa_infs.pk' % result_dir, 'wb') as outf2:
            pk.dump(hpa_infs, outf2)

    if hpa_results:
        with open('%s/hpa_results.pk' % result_dir, 'wb') as outf3:
            pk.dump(hpa_results, outf3)

    if checkpoint:
        # with open('%s/checkpoint.pk' % result_dir, 'wb') as outf4:
        #     pk.dump(checkpoint, outf4)
        torch.save(checkpoint, '%s/checkpoint.bin' % result_dir)

    if hyperparam:
        with open('%s/hyperparam.pk' % result_dir, 'wb') as outf5:
            pk.dump(hyperparam, outf5)

    if dev_infs:
        with open('%s/dev_infs.pk' % result_dir, 'wb') as outf6:
            pk.dump(dev_infs, outf6)

    print('dumped everything to {}'.format(result_dir))
