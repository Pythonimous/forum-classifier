import pickle
import pandas as pd

from Multi import utils

base = pd.read_csv('data/base.csv')
with open('data/counts.pickle', 'rb') as counts_file:
    counts_dict = pickle.load(counts_file)
counts_file.close()

classes_list = list(set(base['class']))

words_ppmi = utils.build_matrix(counts_dict['word_counts_absolute'])
trigrams_ppmi = utils.build_matrix(counts_dict['trigram_counts_absolute'])
ppmi_dict = {'words_ppmi': words_ppmi, 'trigrams_ppmi': trigrams_ppmi}
with open('data/ppmi_matrices.pickle', 'wb') as ppmi_file:
    pickle.dump(ppmi_dict, ppmi_file)
ppmi_file.close()

words_top300 = {c: {} for c in classes_list}
trigrams_top300 = {c: {} for c in classes_list}

''' For each class add words / trigrams and their respective class ppmi values'''
for word, class_ppmis in words_ppmi.items():
    for word_class in classes_list:
        words_top300[word_class][word] = words_ppmi[word][word_class]
for trigram, class_ppmis in trigrams_ppmi.items():
    for trigram_class in classes_list:
        trigrams_top300[trigram_class][trigram] = trigrams_ppmi[trigram][trigram_class]

words_top300 = {k: sorted(v, key=v.get, reverse=True)[:300] for k, v in words_top300.items()}
trigrams_top300 = {k: sorted(v, key=v.get, reverse=True)[:300] for k, v in trigrams_top300.items()}
top_tokens = {'top_words': words_top300,
              'top_trigrams': trigrams_top300}

with open('data/top_300_ppmi.pickle', 'wb') as top_file:
    pickle.dump(top_tokens, top_file)
top_file.close()
