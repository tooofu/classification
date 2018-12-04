# -*- coding: utf-8 -*-
"""
Dec 2018 by Haeryong Jeong
1024.ding@gmail.com
https://www.github.com/tooofu/classification
"""

from __future__ import print_function

import tensorflow as tf


class TextCNN(object):

    def __init__(self, FLAGS, vocab_embed):
        self.n_class = FLAGS.n_class
        self.max_len = FLAGS.max_len
        self.embed_size = FLAGS.embed_size
        self.kernels = eval(FLAGS.kernels)
        self.feature_maps = eval(FLAGS.feature_maps)
        self.vocab_embed = tf.convert_to_tensor(vocab_embed, name='vocab_embed')
        self.dropout_prob = tf.convert_to_tensor(FLAGS.dropout_prob)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.total_feature_map = sum(self.feature_maps)

        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.int32, [None, self.max_len], name='X')
            self.y = tf.placeholder(tf.float32, [None, self.n_class], name='y')

        with tf.name_scope('embed'):
            embed_words = tf.nn.embedding_lookup(self.vocab_embed, self.X, name='embed_words')  # shape (N, M)
            embed_words_expanded = tf.expand_dims(embed_words, -1)  # shape (N, M, 1)

        with tf.name_scope('cnn'):
            h_pool_flat = self.cnn(embed_words_expanded)

        with tf.name_scope('dropout'):
            h_drop = self.dropout(h_pool_flat)

        with tf.name_scope('output'):
            self.logits = self.output(h_drop)

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

    def cnn(self, embed_words_expanded):
        pooled_outputs = []
        for i, (kernel, feature_map) in enumerate(zip(self.kernels, self.feature_maps)):
            with tf.name_scope('conv-pool'):
                # conv
                W = tf.Variable(tf.truncated_normal([kernel, self.embed_size, 1, feature_map], stddev=0.1), name='W')
                conv = tf.nn.conv2d(embed_words_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                # bias
                b = tf.Variable(tf.constant(0.1, shape=[feature_map]), name='b')
                h = tf.nn.bias_add(conv, b)
                # activation
                relu_h = tf.nn.relu(h, name='relu')
                # max-pool
                pooled = tf.nn.max_pool(relu_h, ksize=[1, self.max_len - kernel + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')

                pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.total_feature_map])
        return h_pool_flat

    def dropout(self, h_pool_flat, seed=1):
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_prob, seed=seed)
        return h_drop

    def output(self, h_drop):
        W = tf.Variable(tf.truncated_normal([self.total_feature_map, self.n_class], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[self.n_class]), name='b')
        logits = tf.nn.xw_plus_b(h_drop, W, b, name='logits')
        return logits
