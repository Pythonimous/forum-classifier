#! python3


import re


def load_lines(file_name, encoding='utf-8'):
    with open(file_name, 'r', encoding=encoding) as f:
        lines = f.read().split('\n')
    return lines

    
def save_lines(file_name, lines, encoding='utf-8'):
    with open(file_name, 'w', encoding=encoding) as f:
        f.write('\n'.join(lines))
    

def get_word_unigram_vector(text, word_features):
    """Return a word unigram feature vector for text."""
    tokens = re.findall('\w+', text.lower())
    vec = []
    for f in word_features:
        c = tokens.count(f)
        vec.append(c)
    return vec
    
    
def get_char_ngram_vector(text, ngrams):
    vec = []
    for f in ngrams:
        vec.append(text.count(f))
    return vec
    
    
def get_char_trigrams(message):
    ngrams = {}
    length = len(message)
    for i in range(length-2):
        ngram = message[i:i+3]
        if ngram not in ngrams:
            ngrams[ngram] = 0
        ngrams[ngram] += 1
    return ngrams


def learn_char_trigrams(messages, file_name):
    print('extracting char ngrams for {}'.format(file_name))
    trigrams = {}
    for m in messages:
        for tri, f in get_char_trigrams(m).items():
            if tri not in trigrams:
                trigrams[tri] = 0
            trigrams[tri] += f
    _trigrams = [(f, tri) for tri, f in trigrams.items()]
    _trigrams.sort(reverse=True)
    _trigrams = '\n'.join(['{} {!r}'.format(f, tri) for f, tri in _trigrams])
    print(_trigrams, file=open('{}_char_tri.txt'.format(file_name), 'w', encoding='utf-8'))
    return trigrams
    
    
def get_informative(f_dict1, f_dict2, k):
    all_features = set(list(f_dict1) + list(f_dict2))
    scores = {}
    freqs1 = sum(f for f in f_dict1.values())
    freqs2 = sum(f for f in f_dict2.values())
    for f in all_features:
        f1 = f_dict1.get(f, 1) / freqs1
        f2 = f_dict2.get(f, 1) / freqs2
        scores[f] = f1 / f2
    scores = [(score, feature) for feature, score in scores.items()]
    scores.sort()
    features = [feature for score, feature in scores]
    # print(features[:50], '\n\n', features[-50:])
    selected = features[:k] + features[-k:]
    n = len(selected)
    with open('informative_char_ngrams_{}.txt'.format(n), 'w', encoding='utf-8') as f:
        f.write('\n'.join('{!r}'.format(ngram) for ngram in selected))
    return selected
    
'''
if __name__ == '__main__':
    sen1 = load_lines('intol_sent_train.txt')
    tri1 = learn_char_trigrams(sen1, 'intol')
    sen0 = load_lines('non-intol_sent_train.txt')
    tri0 = learn_char_trigrams(sen0, 'non-intol')
    
    best = get_informative(tri1, tri0, 500)
    ord_best = sorted(best)
    
    sen_test1 = load_lines('intol_sent_test.txt')
    sen_test0 = load_lines('non-intol_sent_test.txt')
    
    vecs = []
    vecs.append('\t'.join([str(i+1) for i, t in enumerate(ord_best)] + ['class']))
    vecs.append('discrete\t'*len(ord_best) + 'discrete')
    vecs.append('\t'*len(ord_best) + 'class')
    for sens, lab in ((sen1, '1'), (sen0, '0'), (sen_test1, '1'), (sen_test0, '0')):
        print(lab)
        for s in sens:
            vecs.append('\t'.join([str(val) for val in get_char_ngram_vector(s, ord_best)] + [lab]))
    save_lines('data.tab', vecs)
    
    input('Press Enter to exit')
'''