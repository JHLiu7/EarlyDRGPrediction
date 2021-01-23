import pandas as pd
import numpy as np
import pickle as pk
import os 
import re

from tqdm import tqdm
from collections import Counter, OrderedDict
from nltk import sent_tokenize, word_tokenize

from options import args

from gensim.models import KeyedVectors

def main():
    """
        tokenize raw text and save token2id dict, embeddings, and processed text
    """
    data_dir = '%s/%s' % (args.data_dir, args.cohort)
    text_dir = '%s/text_raw' % data_dir
    output_dir = '%s/text_processed' % data_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [f for f in os.listdir(text_dir) if f.endswith('pk')]

    # get vocab and embeddings
    words = get_common(files, text_dir, output_dir, args.threshold)
    token2id, embedding = get_embeddings(words, args.pretrained_embed_dir)

    # save vocab and embeddings
    t2i_path = os.path.join(data_dir, 'token2id.dict')
    with open(t2i_path, 'wb') as f:
        pk.dump(token2id, f)

    embed_path = os.path.join(data_dir, 'embedding.npy')
    np.save(embed_path, embedding)

    # save processed text to new subfolder
    for file in tqdm(files):
        save2id(file, token2id, text_dir, output_dir, args.threshold)



# the following re patterns and cleaning processes are adapted from the biowordvec repo
# ==============================================================================
SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text):
    """
    Clean text
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text

def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    tokens = []
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            tokens.extend([w.lower() for w in word_tokenize(sent)])
    return tokens
# ==============================================================================

def get_stay_tokens(file, text_dir, hour, keeptime=False):
    """
        input: path
        output: tokens in order for all texts representing the stay
    """
    stay_df = pd.read_pickle(os.path.join(text_dir, file))
    note_dict = OrderedDict()
    for _, row in stay_df.iterrows():
        diff = row['DIFFTIME']
        if diff < hour:
            text = preprocess_mimic(row['TEXT'])
            note_dict[diff] = text

    if keeptime:
        return note_dict
    else:
        tokens = [t for note in note_dict.values() for t in note]
        return tokens

def get_common(files, text_dir, output_dir, threshold_hour):
    all_tokens=[]
    for file in tqdm(files):
        all_tokens.extend(get_stay_tokens(file, text_dir, threshold_hour))
    token_count = Counter(all_tokens)

    common = [w for (w,c) in token_count.most_common() if c >= args.word_min_freq]  
    print("{} tokens in text, {} unique, and {} of them appeared at least three times".format(len(all_tokens), len(token_count),len(common)))
    # with open(os.path.join(output_dir, 'unique_common_words.txt'), 'w') as f:
    #     for w in common:
    #         f.write(w+'\n')
    return common

def get_embeddings(words, embed_dir):
    print("loading biovec...")
    model = KeyedVectors.load_word2vec_format(os.path.join(embed_dir, 'BioWordVec_PubMed_MIMICIII_d200.vec.bin'), binary=True)
    print("loaded, start to get embed for tokens")

    model_vocab = set(model.index2word)

    valid_words = []
    oov = []
    for w in words:
        if w in model_vocab:
            valid_words.append(w)
        else:
            oov.append(w)
    print("oov", oov)

    valid_words = sorted(valid_words)

    # vocab dicts
    token2id = {}
    token2id['<pad>'] = 0
    for word in valid_words:
        token2id[word] = len(token2id)
    token2id['<unk>'] = len(token2id)

    # get embeddings; pad initiliazed as zero, unk as random
    dim = model.vectors.shape[1]
    embedding = np.zeros( (len(valid_words)+2, dim), dtype=np.float32)
    embedding[0] = np.zeros(dim,)
    embedding[-1] = np.random.randn(dim,)
    print("embed shape", embedding.shape)
    for i, w in enumerate(valid_words):
        embedding[i+1] = model[w]

    return token2id, embedding

def save2id(file, token2id, text_dir, output_dir, threshold_hour):
    output_path = os.path.join(output_dir, file.replace('pk','dict'))
    note_dict = get_stay_tokens(file, text_dir, threshold_hour, keeptime=True)

    out_dict = OrderedDict()
    for key, tokens in note_dict.items():
        out_dict[key] = tokens

    with open(output_path, 'wb') as f:
        pk.dump(out_dict, f)

if __name__ == '__main__':
    main()
