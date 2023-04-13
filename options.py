import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='data_n', type=str, help="Directory for data folder which contains root cohort, as well as subfolders for text files")
parser.add_argument("--mimic_dir", default='~/mimic/', type=str, help="Dir for MIMIC-III")

parser.add_argument("--threshold", default=48, type=int, help="threshold hour to decide what data to include, eg, 48h after icu")
parser.add_argument("--cohort", default='ms', type=str, choices=['ms', 'apr'])
parser.add_argument("--rule_dir", default='rules', type=str, help="path for officials `rules' for DRG grouping and weights")
parser.add_argument("--rule", default='13', type=str, choices=['12', '13', '14'], help="Fiscal Year for DRG rules. Valid for both MS and APR")

parser.add_argument("--target", default='drg', type=str, choices=['drg', 'rw'], help="define the target and corresponding objective, choices include drg code and relative weight")

parser.add_argument("--pretrained_embed_dir", default='../', type=str, help="dir to store pretrained embedding weights")
parser.add_argument("--word_min_freq", default=3, type=int)
parser.add_argument("--max_seq_length", default=2000, type=int)

parser.add_argument("--model", default='CAML', type=str) #, choices=['CAML', 'MultiCNN', 'MultiResCNN'])
parser.add_argument("--multi_kernel_sizes", default='3,4,5', type=str, help='kernels for multi-cnn')
parser.add_argument("--single_kernel_size", default=5, type=int, help="kernel size for caml")
parser.add_argument("--cnn_filter_maps", default=256, type=int)
parser.add_argument("--hidden_size", default=256, type=int, help="rnn hidden size")

parser.add_argument("--dropout_rate", default=0.3, type=float)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--wd", default=1e-5, type=float)
parser.add_argument("--epochs", default=2, type=int)
parser.add_argument("--patience", default=8, type=int)
parser.add_argument("--batch_size", default=32, type=int)

parser.add_argument("--device", default='0', type=str)

parser.add_argument("--eval_model", type=str, required=False, help="if specified, will evaluate the checkpoint saved in the path")
parser.add_argument("--save_model", action="store_const", const=True, default=False, help="train and evaluate a model, saving checkpoint and results")

# hmm
parser.add_argument("--cohort_sim", action="store_const", const=True, default=False, help="examine a model with cohort simulation to predict CMI thru time")
parser.add_argument("--random_seed", default=0, type=int)


args = parser.parse_args()





