import pandas as pd
import numpy as np
import pickle as pk

import os
import time
import torch
import torch.nn as nn

from train_utils import train_with_early_stopping, split_df_by_pt, dump_outputs, update_args, eval_test
from eval_utils import full_metrics, result2str, reg_metrics, score_f1, score_mae
from data_utils import DrgTextDataset, load_rule

from options import args
from models import pick_model

os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# def main():

  

if __name__ == '__main__':
    # main()
    RULE_PATH = '%s/%sDRG_RULE%s.csv' % (args.rule_dir, args.cohort.upper(), args.rule)

    data_dir = '%s/%s' % (args.data_dir, args.cohort)
    TEXT_DIR = '%s/text_embed' % data_dir
    embedding = np.load('%s/embedding.npy' % data_dir) 

    run_time = time.strftime('%b_%d_%H', time.localtime())
    result_dir = 'results/%s' % '_'.join([args.cohort, args.model, 'text', run_time])


    train_val_df = pd.read_csv('%s/train_val.csv' % data_dir)
    test_df = pd.read_csv('%s/test.csv' % data_dir)

    train_df, dev_df = split_df_by_pt(train_val_df, frac=0.1)

    train_dataset = DrgTextDataset(args, train_df, RULE_PATH)
    dev_dataset = DrgTextDataset(args, dev_df, RULE_PATH)
    test_dataset = DrgTextDataset(args, test_df, RULE_PATH)
    args.Y = train_dataset.Y

    if not args.eval_model:
        model = pick_model(args, embedding)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if args.target == 'drg':
            score_func = score_f1
            small_base = True
        else:
            score_func = score_mae
            small_base = False
        model_wts = train_with_early_stopping(model, train_dataset, dev_dataset, args.epochs, args.patience, args.target, args.batch_size, optimizer, score_func, small_base)
    else:
        print("load checkpoint from", args.eval_model)
        hyperparam_path = '%s/hyperparam.pk' % args.eval_model
        if os.path.exists(hyperparam_path):
            hyperparam = pd.read_pickle(hyperparam_path)
            args = update_args(args, hyperparam)
        model = pick_model(args, embedding)
        model_wts = torch.load('%s/checkpoint.bin' % args.eval_model)

    # eval 
    model.load_state_dict(model_wts)
    text_infs = {}
    for hour in [24, 48]:
        print('Test Hour', str(hour), 'Evaluation Results')
        test_dataset.load_data(hour)
        te_score, te_inf = eval_test(model, test_dataset, args.target, score_func, RULE_PATH, True, args.batch_size)
        text_infs['inf%s' % hour] = te_inf

    if args.save_model:
        dump_outputs(result_dir, text_infs, checkpoint=model_wts, hyperparam=vars(args))

