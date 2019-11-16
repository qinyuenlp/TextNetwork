# -*- coding:utf-8 -*-
# @Date   : 2019/10/31
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

class KeywordNetwork_sim(object):
    def __init__(self, corpus, keywords):
        self.corpus = corpus
        self.keywords = keywords
        self.network = nx.Graph()
        self._build_graph()

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

    def _get_vectors(self):
        indexes = {word: pos for pos, word in enumerate(self.corpus.words) if word in self.keywords}
        vectors = {}
        error_list = []  # temporary solution to the bug that i not exist in indexes' keys
        for i in self.keywords:
            try:
                index_i = indexes[i]
                vector = self.corpus.countarray[:, index_i]
                vectors[i] = vector
            except:
                error_list.append(i)
        for i in error_list:
            self.keywords.remove(i)
        return {i:j for i, j in vectors.items() if i not in error_list}

    def _build_graph(self):
        self.network.add_nodes_from(self.keywords)
        vectors = self._get_vectors() # dict : {node : vector}
        E = []
        for i in range(len(self.keywords)):
            word_i = self.keywords[i]
            vec_i = vectors[word_i]
            for j in range(i, len(self.keywords)):
                if j == i:
                    continue
                word_j = self.keywords[j]
                vec_j = vectors[word_j]
                weight = self.similarity(vec_i, vec_j)
                if weight:
                    E.append((word_i, word_j, weight))
        self.network.add_weighted_edges_from(E)

class KeywordNetwork_co(object):
    """Building a co-occurrence network of keywords, weights of edges in the network are calculated as co-occurrence times.

    Parameter
    --------
    keywords_input : keywords list of an abstract
        List of keywords set by the author of the corresponding paper.
    """
    def __init__(self, keywords_input):
        self.corpus = keywords_input
        self.keywords = []
        self.network = nx.Graph()
        self._build_network()

    def _build_network(self):
        tmp_corpus = [' '.join(i) for i in self.corpus]
        vectorizer = CountVectorizer()
        count_array = vectorizer.fit_transform(tmp_corpus).toarray().T  # each line is the vector of the word
        self.keywords = vectorizer.get_feature_names()
        edges_list = self._get_edges(self.keywords, count_array)
        self.network.add_nodes_from(self.keywords)
        self.network.add_weighted_edges_from(edges_list)

    def _get_edges(self, words, mat):
        n = len(words)
        words_dict = {i:j for i, j in enumerate(words)}
        mat_dict = {i:j for i, j in enumerate(mat)}
        res = []
        for i in range(n):
            word_i = words_dict[i]
            vec_i = mat_dict[i]
            for j in range(i, n):
                if i == j:
                    continue
                word_j = words_dict[j]
                vec_j = mat_dict[j]
                weight = self.count_num(vec_i, vec_j)
                if weight:
                    res.append((word_i, word_j, weight))
        return res

    def count_num(self, vec1, vec2):
        if len(vec1) != len(vec2):
            raise ValueError("Lengths of two vector must be equal.")
        num = 0
        for i, j in zip(vec1, vec2):
            if i and j:
                num += 1
        return num