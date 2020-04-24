import math
import pickle
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split


def p_wc(word, context_word, counts, total_counts):
    return counts[word][context_word] / total_counts


def p_w(word, counts, total_counts):
    return sum(counts[word].values()) / total_counts


def p_c(context_word, counts, total_counts):
    return sum(freqs.get(context_word, 0) for freqs in counts.values()) / total_counts


def pmi(word, context_word, counts, total_counts):
    ans = (p_wc(word, context_word, counts, total_counts) /
           (p_w(word, counts, total_counts) * p_c(context_word, counts, total_counts)))
    if ans:
        return math.log2(ans)
    else:
        return 0


def ppmi(word, context_word, counts, total_counts):
    ans = pmi(word, context_word, counts, total_counts)
    return ans if ans > 0 else 0


def get_total_counts(counts):
    """counts is a dict of dicts"""
    return sum([sum(vals.values()) for vals in counts.values()])


def build_matrix(counts):
    """
    Builds PPMI matrix from absolute counts
    :param counts: dict of dict where k = word, v = num of texts of each class where word occurs
    :return: worked out PPMI matrix
    """
    total_counts = get_total_counts(counts)
    for w, contexts in counts.items():
        for c in contexts:
            counts[w][c] = ppmi(w, c, counts, total_counts)
    return counts


def load_data(folder):
    data = pd.read_csv(f'{folder}/base.csv')
    with open(f'{folder}/top_300_ppmi.pickle', 'rb') as top_file:
        top_300 = pickle.load(top_file)
    top_file.close()
    return data, top_300


def train_dev_test(frame, state):
    frame = frame.sample(frac=1, random_state=state)  # shuffling dataset
    train, dev = train_test_split(frame, test_size=0.1, random_state=state)
    dev, test = train_test_split(dev, test_size=0.5, random_state=state)
    return train, dev, test


def make_map_from_nested(to_unnest):
    unnested = list(set(list(itertools.chain.from_iterable(to_unnest))))
    return {unnested[i]: i for i in range(len(unnested))}


def word_vectors(sentences, word_map):
    vectors = []
    for sentence in sentences:
        vector = []
        words = sentence.split()
        for word in words:
            if word in word_map:
                vector.append(word_map[word])
        vectors.append(vector)
    return vectors


def trigram_vectors(sentences, trigram_map):
    vectors = []
    for sentence in sentences:
        vector = []
        trigrams = [sentence[i:i + 3] for i in range(len(sentence) - 2)]
        for trigram in trigrams:
            if trigram in trigram_map:
                vector.append(trigram_map[trigram])
        vectors.append(vector)
    return vectors


def all_vectors(sentences, all_map):
    vectors = []
    for sentence in sentences:
        vector = []
        words = sentence.split()
        for word in words:
            if word in all_map:
                vector.append(all_map[word])
        vectors.append(vector)
    return vectors
