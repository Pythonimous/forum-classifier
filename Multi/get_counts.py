import os
import pickle
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('data/base.csv')
classes = list(set(data['class']))


def get_counts_from_base(base, classes_list):
    word_counts_absolute = {}
    trigram_counts_absolute = {}

    for _, row in tqdm(base.iterrows(), total=len(base)):
        text_class = row['class']
        text = row['text']
        trigrams = set([text[i:i + 3] for i in range(len(text) - 2)])

        for word in set(text.split()):
            if word not in word_counts_absolute:
                word_counts_absolute[word] = {i: 2 for i in classes_list}  # add-2 smoothing
            word_counts_absolute[word][text_class] += 1

        for trigram in trigrams:
            if trigram not in trigram_counts_absolute:
                trigram_counts_absolute[trigram] = {i: 2 for i in classes_list}
            trigram_counts_absolute[trigram][text_class] += 1

    category_length = {i: base['class'].value_counts()[i] for i in classes_list}

    word_counts_relative = {}
    trigram_counts_relative = {}
    for word, abs_counts in word_counts_absolute.items():
        word_counts_relative[word] = {category: count / category_length[category]
                                      for category, count in abs_counts.items()}
    for trigram, abs_counts in trigram_counts_absolute.items():
        trigram_counts_relative[trigram] = {category: count / category_length[category]
                                            for category, count in abs_counts.items()}

    for word, rel_counts in list(word_counts_relative.items()):
        if max(list(rel_counts.values())) < 0.01:  # if word occurs in less than 1% of its most prevalent class
            del word_counts_relative[word]  # remove it from both dictionaries
            del word_counts_absolute[word]

    for trigram, rel_counts in list(trigram_counts_relative.items()):
        if max(list(rel_counts.values())) < 0.01:  # if trigram occurs in less than 1% of its most prevalent class
            del trigram_counts_relative[trigram]  # remove it from both dictionaries
            del trigram_counts_absolute[trigram]

    counts_dictionary = {'word_counts_absolute': word_counts_absolute,
                         'word_counts_relative': word_counts_relative,
                         'trigram_counts_absolute': trigram_counts_absolute,
                         'trigram_counts_relative': trigram_counts_relative}

    if not os.path.isdir('data'):
        os.mkdir('data')

    with open('data/counts.pickle', 'wb') as output_file:
        pickle.dump(counts_dictionary, output_file)
    output_file.close()

    return counts_dictionary


output_dict = get_counts_from_base(data, classes)
