import math


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
