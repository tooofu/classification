# -*- coding: utf-8 -*-
"""
Dec 2018 by Haeryong Jeong
1024.ding@gmail.com
https://www.github.com/tooofu/classification
"""

from __future__ import print_function

import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import LabelEncoder


class Dataset(object):

    def __init__(self, file='bbc-text.csv', max_len=20, embed_size=10, n_class=5):
        self.max_len = max_len
        self.embed_size = embed_size
        self.n_class = n_class
        self.file = file
        self.data = pd.read_csv(file, encoding='utf8')
        self.wv = None
        self.X = None
        self.Y = None

    def get_vocab_embed(self):
        vec_file = '{}.{}.vec'.format(os.path.splitext(self.file)[0], self.embed_size)
        if not self.wv:
            if os.path.exists(vec_file):
                self.wv = KeyedVectors.load(vec_file)
            else:
                self.wv = Word2Vec(self.data['text'].str.split(' '), size=self.embed_size, min_count=1).wv
                self.wv.save(vec_file)
        return self.wv.vectors

    def load_train_data(self, is_expand=False):
        if not self.wv:
            self.get_vocab_embed()

        self.encoder = LabelEncoder().fit(self.data['category'])
        self.X = [[self.wv.vocab[word].index for word in sentence.split(' ')[:self.max_len]] for sentence in self.data['text']]
        self.Y = self.encoder.transform(self.data['category'])

    def get_batch_data(self, batch_size, is_expand=False):
        if not self.X:
            self.load_train_data(is_expand)

            if is_expand:
                Y = np.zeros((len(self.Y), self.n_class), dtype=float)
                Y[np.arange(Y.shape[0]), self.Y] = 1
                self.Y = Y

        num_batch = len(self.X) // batch_size

        for i in range(num_batch):
            start, end = i * batch_size, (i + 1) * batch_size
            yield self.X[start: end], self.Y[start: end]

    def idx2w(self, idxs):
        return [self.wv.index2word[i] for i in idxs]

    def label2w(self, y, is_expand=False):
        if not is_expand:
            return y
        return self.encoder.inverse_transform(y.argmax(axis=1))
