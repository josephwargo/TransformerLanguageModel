import os
import re
import zipfile
import numpy as np
import cupy as cp
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def parse_corpus(dataset, start_token, end_token, num_samples=None):
    if num_samples is not None:
        files = dataset["train"]["text"][:num_samples]
    else:
        files = dataset["train"]["text"]
    corpus = [[start_token] + [re.sub(r'[^\w]', '', w.lower()) for w in f.split(" ")] + [end_token] for f in files]
    return corpus

def word_to_ind(corpus, pad_val):
    corpus_words = [y for x in corpus for y in x]
    corpus_words = list(set(corpus_words))

    word2ind={word: i for i, word in enumerate(corpus_words, start=1)}
    word2ind[pad_val] = 0

    return word2ind

def ind_to_word(corpus, pad_val):
    corpus_words = [y for x in corpus for y in x]
    corpus_words = list(set(corpus_words))

    corpus_words.insert(0, pad_val)
    return corpus_words

def chunk_corpus(corpus, max_text_len):
    corpus_copy = corpus.copy()
    for text, index in enumerate(corpus_copy):
        if len(text) > max_text_len:
            text = text[:max_text_len]
            corpus_copy[index] = text
            remaining_text = text[max_text_len:]
            corpus_copy.append(remaining_text)
    return corpus_copy