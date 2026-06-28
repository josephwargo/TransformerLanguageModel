import numpy as np
import cupy as cp

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