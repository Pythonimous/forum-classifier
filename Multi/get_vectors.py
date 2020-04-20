import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

base = pd.read_csv('data/base.csv')
classes_list = list(set(base['class']))

with open('data/top_300_ppmi.pickle', 'rb') as top_file:
    top_tokens = pickle.load(top_file)
top_file.close()

top_words = top_tokens['top_words']
top_trigrams = top_tokens['top_trigrams']

word_vectors = []
trigram_vectors = []

for _, row in tqdm(base.iterrows(), total=len(base)):
    text = row['text']

    words = text.split()
    trigrams = [text[i:i + 3] for i in range(len(text) - 3)]

    word_vector, trigram_vector = [0] * 10, [0] * 10

    for i in range(len(classes_list)):
        word_lookup_set = set(top_words[classes_list[i]])
        trigram_lookup_set = set(top_trigrams[classes_list[i]])

        for word in words:
            if word in word_lookup_set:
                word_vector[i] += 1
        for trigram in trigrams:
            if trigram in trigram_lookup_set:
                trigram_vector[i] += 1

        word_vector[i] /= len(word_lookup_set)
        trigram_vector[i] /= len(trigram_lookup_set)

    word_vectors.append(word_vector)
    trigram_vectors.append(trigram_vector)

decimal_vectors = {'word_vectors': np.array(word_vectors),
                   'trigram_vectors': np.array(trigram_vectors)}

with open('data/top_token_vectors.pickle', 'wb') as vector_file:
    pickle.dump(decimal_vectors, vector_file)
vector_file.close()
