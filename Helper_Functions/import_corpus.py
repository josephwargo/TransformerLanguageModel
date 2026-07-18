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
    for index, text in enumerate(corpus_copy):
        if len(text) > max_text_len:
            text = text[:max_text_len]
            corpus_copy[index] = text
            remaining_text = text[max_text_len:]
            corpus_copy.append(remaining_text)
    return corpus_copy


def batch_corpus(corpus, word2ind, batch_size):

    x_batches = []
    Y_batches = []

    numerical_corpus = []
    for text in corpus:
        numerical_corpus.append([word2ind.get(word, 0) for word in text])

    for text_num in range(0, len(corpus), batch_size):

        batch = numerical_corpus[text_num : text_num+batch_size]
        
        # finding max length in batch
        max_text_len = max(len(text) for text in batch)
        
        # padding
        padded_batch = np.zeros((len(batch), max_text_len), dtype=np.int32)
        for i, text in enumerate(batch):
            padded_batch[i, :len(text)] = text
        

        # transferring to GPU
        gpu_batch_array_ind = cp.asarray(padded_batch)

        # cutting off first token for x, and last token for Y
        x_batch_ind = gpu_batch_array_ind[:, :-1]
        Y_batch_ind = gpu_batch_array_ind[:, 1:]


        # appending indices to later be turned into embeddings (saves memory)
        x_batches.append(x_batch_ind)
        
        # only need the indices for Y
        Y_batches.append(Y_batch_ind)

    return x_batches, Y_batches