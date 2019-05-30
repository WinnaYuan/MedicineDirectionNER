#!/usr/bin/python
# -*- coding: UTF-8 -*-

import time, sys, os
import hyperparams as hp
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from dataPreprocess import getBatchData, padSequences
from util import get_logger
from eval import conlleval


class BiLSTM_CRF(object):
    def __init__(self, embeddings, word2id, config, paths, use_normalize):
        self.embeddings = embeddings
        self.word2id = word2id
        self.config = config

        self.tag2label = hp.tag2label
        self.label2tag = hp.label2tag
        self.num_tags = len(self.tag2label)
        self.batch_size = hp.BATCH_SIZE
        self.lr = hp.LR
        self.hidden_size = hp.HIDDEN_SIZE
        self.keep_pro = hp.KEEP_PRO
        self.shuffle = hp.SHUFFLE

        self.model_path = paths['model_path']
        self.results_path = paths['result_path']
        self.summary_path = paths['summary_path']
        self.log_path = paths['log_path']
        self.logger = get_logger(self.log_path)
        self.use_normalize = use_normalize

    def normalize(self, inputs, epsilon=1e-8):#( batch_size, seq_len, 2*hidden_size)
        with tf.variable_scope('ln'):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    # start build graph
    def buildGraph(self):
        self.addPlaceHolders()
        self.lookupLayerOp()
        self.biLSTMLayerOp()
        self.softmax_pred_op()
        self.lossOp()
        self.trainStepOp()

    def addPlaceHolders(self):
        self.inputs = tf.placeholder(name='word2id', shape=[None, None], dtype=tf.int32)
        self.labels = tf.placeholder(name='labels', shape=[None, None], dtype=tf.int32)
        self.sequence_lengths = tf.placeholder(name='sequence_lengths', shape=[None], dtype=tf.int32)

        self.keep_pro_pl = tf.placeholder(name='dropout', shape=[], dtype=tf.float32)
        self.lr_pl = tf.placeholder(name='lr', shape=[], dtype=tf.float32)

    def lookupLayerOp(self):
        with tf.variable_scope("words"):
            word_embeddings = tf.Variable(name='word_embeddings',
                                          initial_value=self.embeddings,
                                          trainable=hp.UPDATE_EMBEDDINGS,
                                          dtype=tf.float32)
            inputs_embeddings = tf.nn.embedding_lookup(name='inputs_embeddings',
                                                       params=word_embeddings,
                                                       ids=self.inputs)
        # (batch_size, seq_len, embedding_dim)
        self.inputs_embeddings = tf.nn.dropout(inputs_embeddings, keep_prob=self.keep_pro_pl)

    def biLSTMLayerOp(self):
        with tf.variable_scope("BiLSTM"):
            cell_fw = LSTMCell(hp.HIDDEN_SIZE)
            cell_bw = LSTMCell(hp.HIDDEN_SIZE)
            # (batch_size, seq_len, hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                        cell_bw=cell_bw,
                                                                        inputs=self.inputs_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            # (batch_size, seq_len, 2*hidden_size)
            output = tf.concat((output_fw, output_bw), axis=-1)
            output = tf.nn.dropout(output, keep_prob=self.keep_pro_pl)

        with tf.variable_scope("projection"):
            W = tf.get_variable(name='W',
                                shape=[2*hp.HIDDEN_SIZE, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            if self.use_normalize:
                output = self.normalize(output)
            output = tf.reshape(output, [-1, 2*hp.HIDDEN_SIZE]) #( batch_size*seq_len, 2*hidden_size)
            self.logits = tf.reshape((tf.matmul(output, W) + b), [-1, s[1], self.num_tags]) #( batch_size, seq_len, num_tags)

    def softmax_pred_op(self):
        if not hp.USE_CRF:
            self.label_softmax = tf.cast(tf.argmax(self.logits, axis=-1), dtype=tf.int32)

    def lossOp(self):
        if not hp.USE_CRF:
            # (batch_size, max_seq)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        else:
            log_likelihood, self.transition_params  = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                                        tag_indices=self.labels,
                                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        tf.summary.scalar("loss", self.loss)

    def trainStepOp(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -hp.CLIP_GRAD, hp.CLIP_GRAD), v] for (g, v) in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    # end build graph


    # start training
    def train(self, train_data, dev_data):
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            self.merged = tf.summary.merge_all()
            self.file_writer = tf.summary.FileWriter(logdir=self.summary_path, graph=sess.graph)

            for epoch in range(hp.EPOCH):
                num_batches = (len(train_data) + self.batch_size - 1) // self.batch_size
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                batchs = getBatchData(train_data, self.batch_size, self.word2id, self.tag2label, shuffle=self.shuffle)

                for step, (seqs, labels) in enumerate(batchs):
                    step_num = epoch * num_batches + step +1
                    sys.stdout.write("\r processing: {} batch / {} batchs ".format(step+1, num_batches))
                    feed_dict, _ = self.getFeeddict(seqs, labels, self.lr, self.keep_pro)
                    _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                                 feed_dict=feed_dict)

                    if step+1 == 1 or step % 10 == 0 or num_batches-1 == step:
                            self.logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1, loss_train, step_num))

                    self.file_writer.add_summary(summary, step_num)

                    if step+1 == num_batches:
                        saver.save(sess, self.model_path, global_step=step_num)

                self.logger.info("============= validation test =============")
                labels_predict_test, seq_len_list_test = self.predictData(sess, dev_data)
                self.evaluate(labels_predict_test, dev_data, epoch)

    def getFeeddict(self, seqs, labels=None, lr=None, keep_pro=None):
        seq_list, seq_length_list = padSequences(seqs, pad_mark=0)
        feed_dict = {self.inputs: seq_list,
                     self.sequence_lengths: seq_length_list}
        if labels is not None:
            labels_, _ = padSequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if keep_pro is not None:
            feed_dict[self.keep_pro_pl] = keep_pro

        return feed_dict, seq_length_list

    # end training


    # start testing
    def test(self, test_data):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info("\n============ testing ============")
            saver.restore(sess, self.model_path)
            labels, _ = self.predictData(sess, test_data)
            self.evaluate(labels, test_data)

    # end testing


    def predictData(self, sess, dev_data):
        labels_predict, seq_len_list = [], []
        for seqs, labels in getBatchData(dev_data, self.batch_size, self.word2id, self.tag2label, shuffle=self.shuffle):
            labels_predict_ = []
            feed_dict, seq_len_list_ = self.getFeeddict(seqs, keep_pro=1.0)

            if hp.USE_CRF:
                logits, transition_params = sess.run([self.logits, self.transition_params],
                                                     feed_dict=feed_dict)
                for logit, seq_len in zip(logits, seq_len_list_):
                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
                    labels_predict_.append(viterbi_seq)
            else:
                labels_predict_ = sess.run(self.label_softmax, feed_dict=feed_dict)

            labels_predict.extend(labels_predict_)
            seq_len_list.extend(seq_len_list_)

        return labels_predict, seq_len_list

    def evaluate(self, labels_predict, data, epoch=None):
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label!= 0 else label

        model_predict = []
        for label_pred, (sent, tag) in zip(labels_predict, data):
            tag_pred = [label2tag[label] for label in label_pred]
            sent_info = []
            if len(label_pred) != len(sent):
                print(len(sent))
                print("sent",sent)
                print(len(label_pred))
                print("label_pred", label_pred)
                print(len(tag))
                print("tag", tag)
            else:
                for i in range(len(sent)):
                    sent_info.append([sent[i], tag[i], tag_pred[i]])
            model_predict.append(sent_info)

            epoch_num = epoch+1 if epoch != None else 'test'
            label_path = os.path.join(self.results_path, "label_" + str(epoch_num))
            metric_path = os.path.join(self.results_path, 'result_metric_' + str(epoch_num))
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)

    def demoOne(self, sess, demo_data):
        labels_predict, seq_len_list = [], []
        for seqs, label in getBatchData(demo_data, self.batch_size, self.word2id, self.tag2label, shuffle=False):
            labels_predict_ = []
            feed_dict, seq_len_list_ = self.getFeeddict(seqs, keep_pro=1.0)
            if hp.USE_CRF:
                logits, transition_params = sess.run([self.logits, self.transition_params],
                                                     feed_dict=feed_dict)
                for logit, seq_len in zip(logits, seq_len_list_):
                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
                    labels_predict_.append(viterbi_seq)
            else:
                labels_predict_ = sess.run(self.label_softmax, feed_dict=feed_dict)
            labels_predict.extend(labels_predict_)
        tags_predict = [self.label2tag[l] for l in labels_predict[0]]
        return tags_predict








