#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import os, time, sys, gc
import hyperparams as hp
import tensorflow as tf
import numpy as np
from model import BiLSTM_CRF
from dataPreprocess import readVocab, randomEmbedding, getTrainTestData
# from pytorch_pretrained_bert import BertModel

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only show "warnings" and "errors"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2


# get char embeddings
word2id = readVocab(hp.vocab_path_pickle)
if hp.pretrain_embedding == 'random':
    embeddings = randomEmbedding(word2id, hp.WORD_EMBEDDING_DIM)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')

# parser = argparse.ArgumentParser()
# parser.add_argument("--mode", type=str, default='',
#                     help="3 modes: train test and demo")
# mode = parser.parse_args().mode
mode = 'test'


# path setting
paths = {}
timestep = str(int(time.time()))+'_epoch'+str(hp.EPOCH) if mode == 'train' else '1559023444_epoch50'

output_path = os.path.join(hp.save_data, timestep)
if not os.path.exists(output_path): os.mkdir(output_path)
summary_path = os.path.join(output_path, 'summaries')
if not os.path.exists(summary_path): os.mkdir(summary_path)
paths['summary_path'] = summary_path
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.mkdir(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, 'results')
if not os.path.exists(result_path): os.mkdir(result_path)
paths['result_path'] = result_path
log_path = os.path.join(output_path, 'log.txt')
paths['log_path'] = log_path


if __name__ == '__main__':

    # training model
    if mode == 'train':
        train_data, valid_data, test_data = getTrainTestData(reload=True)
        train_size = len(train_data)
        valid_size = len(valid_data)
        model = BiLSTM_CRF(embeddings, word2id, config, paths, use_normalize=hp.USE_NORMALIZE)
        model.buildGraph()
        print("train data: {}".format(train_size))
        print("valid data: {}".format(valid_size))
        model.train(train_data=train_data, dev_data=valid_data)

    elif mode == 'test':
        _, _, test_data = getTrainTestData(reload=True)
        test_size = len(test_data)
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(embeddings, word2id, config, paths, use_normalize=hp.USE_NORMALIZE)
        model.buildGraph()
        print("test data: {}".format(test_size))
        model.test(test_data=test_data)

    elif mode == 'demo':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(embeddings, word2id, config, paths, use_normalize=hp.USE_NORMALIZE)
        model.buildGraph()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            print("============= demo ==============")
            saver.restore(sess, paths['model_path'])
            while(1):
                print("please input : ")
                sents = input()
                if sents == 'clear':
                    if os.path.exists('result'):
                        os.remove('result')
                        gc.collect()
                elif sents == '' or sents.isspace():
                    print("invalid input!")
                    break
                else:
                    sents = list(sents.strip())
                    data = [(sents, ['O']*len(sents))]
                    tags = model.demoOne(sess, data)
                    fw = open('result', 'a+', encoding='utf-8')
                    for sent, tag in zip(sents, tags):
                        fw.write(str(sent) + ' ' + str(tag) + '\n')
                    fw.write('\n')
                    fw.close()