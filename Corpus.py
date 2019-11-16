# -*- coding:utf-8 -*-
# @Date   : 2019/10/31
# @Author : QinYue <qinyuestatistics@163.com>

import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer

class Corpus(object):
    def __init__(self, text, keywords=None, stopword='./Data/stopword.txt', keywords_seg='n', remove_stopword=True, with_segs=False):
        self.corpus = []
        self.keywords = keywords
        self.stopword = []
        self.segs = {}
        self.words = None
        self.countarray = None
        self.tfidfarray = None
        self._countmat = None
        self.remove_stopword = remove_stopword
        if keywords:
            for i in keywords:
                jieba.add_word(i)
        self.preprocess(text, stopword, with_segs)
        if keywords:
            self.set_keywords_seg(keywords, keywords_seg)


    def preprocess(self, text, stopword, with_segs):
        self._cut(text, stopword, with_segs)
        self._get_matrix()

    def _get_stopword(self, stopword):
        if self.remove_stopword:
            with open(stopword, encoding='utf-8') as f:
                for i in f:
                    self.stopword.append(i[:-1])
                f.close()

    def _cut(self, text, stopword, with_segs):
        self._get_stopword(stopword)
        for line in text:
            line_cut = [i for i in pseg.cut(line) if i.word not in self.stopword]
            if with_segs:
                for i in line_cut:
                    i_seg = i.flag
                    i_word = i.word.lower()
                    if i_word in self.segs and i_seg not in self.segs[i_word]:
                        self.segs[i_word].append(i_seg)
                    elif i_word not in self.segs:
                        self.segs[i_word] = [i_seg]
            self.corpus.append([i.word for i in line_cut])

    def _get_matrix(self):
        self.corpus = [' '.join(line) for line in self.corpus]
        vectorizer = CountVectorizer(token_pattern='\S+')
        self._countmat = vectorizer.fit_transform(self.corpus)
        self.countarray = self._countmat.toarray()
        self.words = vectorizer.get_feature_names()

    def set_keywords_seg(self, keywords, seg="n"):
        for i in keywords:
            if i in self.segs and seg not in self.segs[i]:
                self.segs[i].append(seg)
            elif i not in self.segs:
                self.segs[i] = seg