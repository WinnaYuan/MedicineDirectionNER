#!/usr/bin/python
# -*- coding: UTF-8 -*-

import codecs, pickle, os, random
import numpy as np
from dataLoader import xml2NER
import hyperparams as hp


def getData(reload):
    """
    get sentences and tags from function xml2NER
    :return: sentences and tags
    """
    sentences, tags = xml2NER(hp.source_dir, hp.ner_file, reload)
    return sentences, tags

def vocabBuild(vocab_path, reload):
    """
    :param vocab_path: save vocab file
    :param reload: if reload is True, the save file will be constructed newly

    :return: dictionary word to id
    """
    vocab, word2id = {}, {}
    sentences, tags = getData(reload)
    for sent in sentences:
        for chara in sent:
            if chara not in word2id:
                if chara.isdigit():
                    chara = '<NUM>'
                elif ('\u0041' <= chara <= '\u005a') or ('\u0061' <= chara <= '\u007a'):
                    chara = '<ENG>'
                if chara not in vocab:
                    vocab[chara] = 1  # [id, count]
                else:
                    vocab[chara] += 1

    word2id_sort = sorted(vocab.items(), key=lambda item:item[1], reverse=True)  # list
    i = 1
    word2id['<PAD>'] = 0
    for item in word2id_sort:
        word2id[item[0]] = i
        i += 1
    word2id['<UNK>'] = i

    if reload == True:
        if os.path.exists(hp.vocab_path_json): os.remove(hp.vocab_path_json)
        if os.path.exists(hp.vocab_path_pickle): os.remove(hp.vocab_path_pickle)
        with codecs.open(hp.vocab_path_json, 'w', 'utf-8') as fw:
            for item in word2id:
                fw.write(item+' '+str(word2id[item])+'\n')
        with open(vocab_path, 'wb') as fw:
            pickle.dump(word2id, fw)

    print("word2id长度为：{}".format(len(word2id)))
    return word2id

def readVocab(vocab_path):
    """

    :param vocab_path: the path store vocab
    :return word2id: dictionary, the vocab
    """
    if os.path.exists(vocab_path):
        with open(hp.vocab_path_pickle, 'rb') as fr:
            word2id = pickle.load(fr)
    else:
        word2id = vocabBuild(vocab_path, True)
    return word2id

def sentence2id(sent, word2id):
    """

    :param sent: list, sentences
    :param word2id: dictionary, the vocab
    :return sentence_id: list, the correspond ids of sentences
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def randomEmbedding(vocab, embedding_dim):
    """

    :param vocab: dictionary, vocab
    :param embedding_dim: int, the dimension of embedding
    :return embedding_mat: numpy metric, embedding result
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def padSequences(sequences, pad_mark=0):
    """
    :param sequences: list, ids sequences
    :param pad_mark: use 0 to pad
    :return seq_list: list, pad result
    :return seq_len_list: list, the length of every sequences
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def getTrainTestData(reload=False):
    """
    get train and test data
    :return:
    """
    sentences, tags = getData(reload)
    data = []
    for sent, tag in zip(sentences, tags):
        data.append((sent, tag))
    length = len(data)
    train_data = data[:length//10*9]
    valid_data = train_data[:(len(train_data)//10)]
    test_data = data[length//10*9:]
    if reload == True:
        if os.path.exists('data/save/train'):  os.remove('data/save/train')
        if os.path.exists('data/save/valid'): os.remove('data/save/valid')
        if os.path.exists('data/save/test'): os.remove('data/save/test')
    f_train = open('data/save/train', 'a+', encoding='utf-8')
    f_valid = open('data/save/valid', 'a+', encoding='utf-8')
    f_test = open('data/save/test', 'a+', encoding='utf-8')
    for items in train_data:
        for s, t in zip(items[0], items[1]):
            f_train.write(str(s) + ' ' + str(t) +'\n')
        f_train.write('\n')
    for items in valid_data:
        for s, t in zip(items[0], items[1]):
            f_valid.write(str(s) + ' ' + str(t) +'\n')
        f_valid.write('\n')
    for items in test_data:
        for s, t in zip(items[0], items[1]):
            f_test.write(str(s) + ' ' + str(t) +'\n')
        f_test.write('\n')
    f_train.close()
    f_valid.close()
    f_test.close()

    return train_data, valid_data, test_data

def getBatchData(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """

    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

# getTrainTestData(True)
# vocabBuild(hp.vocab_path_pickle, reload=True)