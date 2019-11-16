# -*- coding:utf-8 -*-
# @Author : QinYue <qinyuestatistics@163.com>

import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from Corpus import Corpus

class WordNetwork(object):
    """
    Building words' co-occurrence network by inputting text.

    Parameters
    ----------
    text : List-like
        The list of strings. Each string is an unit of counting words' co-occurrence.
    """
    def __init__(self, text, keywords=None, remove_stopword=True, with_segs=False, weight_type='count'):
        self.text = text
        self.corpus = Corpus(text, keywords=keywords, remove_stopword=remove_stopword, with_segs=with_segs)
        self.network = nx.Graph()
        self._network(weight_type)

    def similarity(self, vec1, vec2):
        dot_val = 0.
        a_norm = 0.
        b_norm = 0.
        for a, b in zip(vec1, vec2):
            dot_val += a * b
            a_norm += a ** 2
            b_norm += b ** 2
        if a_norm * b_norm == 0:
            cos = -1
        else:
            cos = dot_val / ((a_norm * b_norm) ** 0.5)
        return cos

    def count(self, vec1, vec2):
        if len(vec1) != len(vec2):
            raise ValueError('Length of two vectors must be equal.')
        res = 0
        for i, j in zip(vec1, vec2):
            if i and j:
                res += 1
        return res

    def _network(self, weight_type='sim'):
        V = self.corpus.words
        countarray = self.corpus.countarray.T
        E = []
        for i in range(len(V)):
            vec1 = countarray[i]
            for j in range(i, len(V)):
                vec2 = countarray[j]
                if weight_type == 'sim':
                    w = self.similarity(vec1, vec2)
                elif weight_type == 'count':
                    w = self.count(vec1, vec2)
                else:
                    raise ValueError('Parameter "weight_type" must be "sim" or "count".')
                if w:
                    E.append((V[i], V[j], w))
        self.network.add_nodes_from(V)
        self.network.add_weighted_edges_from(E)

class OneGramNetwork(object):
    """
    Building 1-gram network by inputting text.

    Parameters
    ----------
    text : list-like
        The list of strings. Each string is an unit of counting words' co-occurrence.
    """
    def __init__(self, text, keywords=None, remove_stopword=True, with_segs=False):
        self.text = text
        self.corpus = Corpus(text, keywords=keywords, remove_stopword=remove_stopword, with_segs=with_segs)
        self.network = nx.Graph()
        self.build_network()

    def build_network(self):
        # self.network.add_nodes_from(self.corpus.words)
        E = []
        length = len(self.corpus.words)
        indexes = {j:i for i, j in enumerate(self.corpus.words)}
        adjacent = [[0 for i in range(length)] for j in range(length)]
        for line in self.corpus.corpus:
            line = line.split()
            len_line = len(line)
            for i in range(1, len_line):
                word_head = line[i-1].lower()  # Note: CountVectorizer will turn English words to lower case.
                word_end = line[i].lower()
                adjacent[indexes[word_head]][indexes[word_end]] += 1
        for i in range(length):
            word_i = self.corpus.words[i]
            for j in range(i, length):
                word_j = self.corpus.words[j]
                weight = adjacent[indexes[word_i]][indexes[word_j]] + adjacent[indexes[word_j]][indexes[word_i]]
                if weight:
                    E.append((word_i, word_j, weight))
        self.network.add_weighted_edges_from(E)
