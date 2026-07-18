import os
import re
import zipfile
import numpy as np
import cupy as cp
from datasets import load_dataset
from huggingface_hub import hf_hub_download

def download_word_embeddings(repo_id, filename_zipped, filename_unzipped):
    print("Downloading GloVe embeddings from Hugging Face...")
    glove_zip_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename_zipped
    )

    extraction_dir = os.path.dirname(glove_zip_path)
    embeddings_filepath = os.path.join(extraction_dir, filename_unzipped)


    if not os.path.exists(embeddings_filepath):
        print("Extracting text file...")
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            zip_ref.extract(filename_unzipped, path=extraction_dir)
    return embeddings_filepath

def get_embeddings_for_corpus(filepath, words, dimensions):
    vocab_size = len(words)
    
    # numpy first before converting to cupy
    cpu_embeddings = np.zeros((vocab_size, dimensions), dtype=np.float32)

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()

            if word in words: 
                index = words[word]
                cpu_embeddings[index] = np.array(vector, dtype=np.float32)[:dimensions]
                
    # transfer to cupy
    gpu_embeddings = cp.asarray(cpu_embeddings)
    return gpu_embeddings
