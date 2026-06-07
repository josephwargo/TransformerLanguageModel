import os
import re
import zipfile
import numpy as np
import cupy as cp
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def parse_corpus(dataset, start_token, end_token, num_samples):
    files = dataset["train"]["text"][:num_samples]
    return [[start_token] + [re.sub(r'[^\w]', '', w.lower()) for w in f.split(" ")] + [end_token] for f in files]

def word_to_ind(corpus, pad_val):
    corpus_words = [y for x in corpus for y in x]
    corpus_words = list(set(corpus_words))
    word2ind={}
    for i in range(len(corpus_words)+1):
        word2ind[corpus_words[i-1]] = i
    word2ind[pad_val] = 0

    return word2ind

def ind_to_word(corpus, pad_val):
    corpus_words = [y for x in corpus for y in x]
    corpus_words = list(set(corpus_words))

    corpus_words.insert(0, pad_val)
    return corpus_words