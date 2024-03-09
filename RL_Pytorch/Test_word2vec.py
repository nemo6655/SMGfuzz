from gensim.models import Word2Vec
import os
import codecs
import numpy as np
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

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

# 调用函数清空文件夹
if os.path.exists('Decode_Data/seed_vec/'):
    clear_folder('Decode_Data/seed_vec/')
else:
    os.mkdir('Decode_Data/seed_vec/')

seed_vectors = train_word2vec_model(seed_folder)

seed_vec_names=seed_vectors.keys()
seed_vec_values=seed_vectors.values()
for seed_vec_name,seed_vec_value in zip(seed_vec_names,seed_vec_values):
    np.savetxt('Decode_Data/seed_vec/'+seed_vec_name,seed_vec_value)
    print('seed:'+ seed_vec_name +'已转换为词向量')


