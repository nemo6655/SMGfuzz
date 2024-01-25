from gensim.models import Word2Vec
import os
import codecs
import numpy as np

def read_seed_file(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:
        seed_text = f.read()
    # 删除空格和换行符
    seed_text = seed_text.replace(' ', '').replace('\n', '')
    return seed_text

def train_word2vec_model(seed_folder):
    seed_files = os.listdir(seed_folder)
    seed_sentences = []
    for seed_file in seed_files:
        if seed_file.startswith("id"):
            file_path = os.path.join(seed_folder, seed_file)
            seed_text = read_seed_file(file_path)
            seed_sentences.append(seed_text)

    model = Word2Vec([seed_sentences], min_count=1, vector_size=256,window=1)
    seed_vectors = {seed_file: model.wv[seed_text] for seed_file, seed_text in zip(seed_files, seed_sentences)}
    return seed_vectors


seed_folder = 'Decode_Data/seed/'
seed_vectors = train_word2vec_model(seed_folder)

seed_vec_names=seed_vectors.keys()
seed_vec_values=seed_vectors.values()
for seed_vec_name,seed_vec_value in zip(seed_vec_names,seed_vec_values):
    np.savetxt('Seed_Vec/'+seed_vec_name,seed_vec_value)


