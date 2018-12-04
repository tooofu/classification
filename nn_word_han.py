# -*- coding: utf-8 -*-
"""
Dec 2018 by Haeryong Jeong
1024.ding@gmail.com
https://www.github.com/tooofu/classification
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import BasicLSTMCell


class WordHAN(object):
    """
    """
    def __init__(self, FLAGS, vocab_embed):
        self.n_class = FLAGS.n_class
        self.max_len = FLAGS.max_len
        self.embed_size = FLAGS.embed_size

        self.vocab_embed = tf.convert_to_tensor(vocab_embed, name='vocab_embed')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.int32, [None, self.max_len], name='X')
            self.y = tf.placeholder(tf.float32, [None, self.n_class], name='y')

        with tf.name_scope('embed'):
            embed_words = tf.nn.embedding_lookup(self.vocab_embed, self.X, name='embed_words')

        with tf.name_scope('encode-word'):
            # case 1
            # encode_words = embed_words
            # case 2
            (fw_outputs, bw_outputs), _ = bidirectional_dynamic_rnn(
                BasicLSTMCell(self.max_len), BasicLSTMCell(self.max_len), inputs=embed_words, dtype=tf.float32)
            encode_words = fw_outputs + bw_outputs

        with tf.name_scope('word-attn'):
            v = self.attention(encode_words)

        with tf.name_scope('output'):
            self.logits = self.output(v)

        # mean loss
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.mean_loss = tf.reduce_mean(self.loss, name='mean_loss')
        tf.summary.scalar('mean_loss', self.mean_loss)

        # accuracy
        self.target = tf.argmax(self.y, 1, name='target')
        self.prediction = tf.argmax(self.logits, 1, name='prediction')
        correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')
        tf.summary.scalar('accuracy', self.accuracy)

    def attention(self, encode_words):
        """
            http://www.aclweb.org/anthology/N16-1174

            1. u = tanh(xw + b)
            2. a = softmax(u * vec)
            3. v = sum(a * x)
        """
        #  h = tf.stack(h, axis=1)  # shape: [batch_size, max_len, hidden_size * 2]
        #  x = tf.reshape(h, shape=[-1, self.hidden_size * 2])  # shape: [batch_size * max_len, hidden_size * 2]
        #  w = tf.get_variable('word_attn_w', shape=[self.hidden_size * 2, self.hidden_size * 2], initializer=self.initializer)
        #  b = tf.get_variable('word_attn_b', shape=[self.hidden_size * 2])
        #  u = tf.nn.tanh(tf.matmul(x, w) + b)  # shape: [batch_size * max_len, hidden_size * 2]
        #  u = tf.reshape(u, shape=[-1, self.max_len, self.hidden_size * 2])  # shape: [batch_size, max_len, hidden_size * 2]

        #  u = encode_words
        u = layers.fully_connected(encode_words, self.max_len, activation_fn=tf.nn.tanh)
        self.vec = tf.Variable(tf.truncated_normal([self.max_len]), name='word_attn_vec')
        a = tf.nn.softmax(tf.reduce_sum(tf.multiply(u, self.vec), axis=2, keep_dims=True), dim=1)
        v = tf.reduce_sum(tf.multiply(encode_words, a), axis=1)
        return v

    def output(self, v):
        W = tf.Variable(tf.truncated_normal([self.max_len, self.n_class], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[self.n_class]), name='b')
        logits = tf.nn.xw_plus_b(v, W, b, name='logits')
        return logits
