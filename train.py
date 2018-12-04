# -*- coding: utf-8 -*-
"""
Dec 2018 by Haeryong Jeong
1024.ding@gmail.com
https://www.github.com/tooofu/classification
"""

from __future__ import print_function

import tensorflow as tf

from nn_text_cnn import TextCNN as NNTextCNN
from nn_word_han import WordHAN as NNWordHAN
from dataset import Dataset
from utils import mkdir


flags = tf.app.flags
flags.DEFINE_integer('epoch', 50, 'Epoch to train [50]')
flags.DEFINE_integer('n_class', 5, '')
flags.DEFINE_integer('embed_size', 650, 'The dimension of word embedding matix [650]')
flags.DEFINE_integer('max_len', 65, 'The maximum length of word [65]')
flags.DEFINE_integer('batch_size', 100, 'The size of batch sentence [100]')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate [1e-3]')
flags.DEFINE_float('dropout_prob', 0.5, 'Probability of dropout layer [0.5]')
flags.DEFINE_string('model', 'NNTextCNN', 'The type of model to train and test [NNTextCNN, ...]')
flags.DEFINE_string('ckpt_dir', 'ckpt', 'Director name to save the checkpoint [ckpt]')
flags.DEFINE_string('logs_dir', 'logs', 'Director name to save the log [logs]')
flags.DEFINE_boolean('is_training', True, '')
# with define CNN
flags.DEFINE_string('feature_maps', '[64, 128, 128]', 'The feature maps in CNN [64, 128, 128]')
flags.DEFINE_string('kernels', '[2, 3, 4]', 'The width of CNN kernels [2, 3, 4]')
FLAGS = flags.FLAGS


model_dict = {
    'NNTextCNN': NNTextCNN,
    'NNWordHAN': NNWordHAN,
}


def train(dataset):
    vocab_embed = dataset.get_vocab_embed()
    with tf.Graph().as_default(), tf.Session() as sess:
        model = model_dict[FLAGS.model](FLAGS, vocab_embed)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(model.mean_loss, global_step=model.global_step)

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)

        sess.run(tf.global_variables_initializer())

        for epoch in range(0, FLAGS.epoch):
            for X, y in dataset.get_batch_data(FLAGS.batch_size, is_expand=True):
                _, summary, mean_loss, accuracy = sess.run([train_op, summary_op, model.mean_loss, model.accuracy],
                                                           feed_dict={model.X: X, model.y: y})
            print('epoch:', epoch, mean_loss, accuracy)
            writer.add_summary(summary, global_step=sess.run(model.global_step))
        saver.save(sess, FLAGS.ckpt_dir + '/model', global_step=model.global_step)
    print("Done")


def test(dataset):
    with tf.Graph().as_default(), tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt))
        saver.restore(sess, ckpt)

        X = sess.graph.get_operation_by_name('input/X').outputs[0]
        y = sess.graph.get_operation_by_name('input/y').outputs[0]
        mean_loss = sess.graph.get_operation_by_name('mean_loss').outputs[0]
        accuracy = sess.graph.get_operation_by_name('accuracy').outputs[0]

        # nn_word_han
        vec = sess.graph.get_operation_by_name('word-attn/word_attn_vec').outputs[0]

        for X_data, y_data in dataset.get_batch_data(10, is_expand=True):
            loss, acc, vec = sess.run([mean_loss, accuracy, vec], feed_dict={X: X_data, y: y_data})
            print('loss: %f, acc: %.2f' % (loss, acc))
            break

        # nn_word_han
        print(vec)
        for X, label in zip(X_data, dataset.label2w(y_data, is_expand=True)):
            print(label, sorted(zip(vec, dataset.idx2w(X)), key=lambda x: x[0]))


def main(_):
    FLAGS.ckpt_dir += '-{}-{}'.format(FLAGS.max_len, FLAGS.embed_size)
    FLAGS.logs_dir += '-{}-{}'.format(FLAGS.max_len, FLAGS.embed_size)

    dataset = Dataset(max_len=FLAGS.max_len, embed_size=FLAGS.embed_size, n_class=FLAGS.n_class)

    if FLAGS.is_training:
        mkdir([FLAGS.ckpt_dir, FLAGS.logs_dir])
        train(dataset)
    else:
        test(dataset)

if __name__ == '__main__':
    tf.app.run()
