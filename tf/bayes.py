# -*- coding: utf-8 -*-
import jieba
import json
import math
import numpy as np
import nltk
import pickle
import re
import time
from zhon.hanzi import punctuation  # 中文标点符号集合


def stop_words():
    stopwords_list = [line.strip() for line in open('./../framework/stop_words_jieba.utf8.txt', encoding='utf-8')]
    stopwords_list = stopwords_list + [' ', '\n', '\t', '，']
    return stopwords_list

def stop_words_gbk():
    stop_words_file = open('stop_words_ch.txt', 'r',encoding='gbk')
    stopwords_list = []
    for line in stop_words_file.readlines():
        stopwords_list.append(line[:-1])
    return stopwords_list


def jieba_fenci(raw, stopwords_list):
    word_list = list(jieba.cut(raw, cut_all=False))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    if '\n' in word_list:
        word_list.remove('\n')
    return word_list


def process_data(train_path, test_path, label_num, stop_word_list):
    """
    数据预处理
    :param train_path:  训练数据文件
    :param test_path:   测试数据文件
    :param label_num:   标签个数
    :param stop_word_list: 停用词列表
    :return:
    """
    word_bag_ = []
    word_bag = []
    samples = []
    labels = np.zeros((11), dtype=np.int)
    train_set = []
    test_set = []

    k = 0

    begin = int(time.time())

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            k += 1
            print("process line {0}".format(k))
            line = line.strip()
            tmp = line.split(',')
            sample = {}
            label = int(tmp[0])
            labels[label - 1] += 1
            content = re.sub(r"[a-zA-Z0-9]+", '', "".join(tmp[1:]))
            word_list = jieba_fenci(content, stop_word_list)
            train_set.append((word_list, label))
            word_bag_ = word_bag_ + (word_list)
            sample['label'] = label
            sample['word_list'] = word_list
            samples.append(sample)

        for w in word_bag_:
            if w not in word_bag:
                word_bag.append(w)

        end = int(time.time())
        print("load sample used: {}s".format(end - begin))

        begin = int(time.time())
        A = np.zeros((len(word_bag), label_num), dtype=np.int)

        i = 0
        word_bag_index = {}
        for w in word_bag:
            word_bag_index[w] = i
            i += 1

        print("word bag size: {}".format(i))

        for sample in samples:
            j = sample['label'] - 1

            words = set(sample['word_list'])

            for w in words:
                i = word_bag_index[w]
                A[i][j] += 1

        end = int(time.time())
        print("build A matrix used {}s".format(end - begin))
        print(A)

        w = open('./word_bag.json', 'w', encoding='utf-8')
        json.dump(word_bag, fp=w)
        w.close()

        l = open('./label.pickle', 'wb')
        pickle.dump(labels, file=l)
        l.close()

        f = open('./A.pickle', 'wb')
        pickle.dump(A, file=f)
        f.close()
        print("train_set finished")

    with open(test_path, 'r', encoding='utf-8') as g:
        for line in g:
            label = int(line.split(',')[0]) - 1
            content = ""
            for aa in line.split(',')[1:]:
                content += aa
            word_list = jieba_fenci(content, stop_word_list)
            test_set.append((word_list, label))

    return A, train_set, test_set, word_bag, labels


def cal_b_from_a(A):
    K = A.shape[1]
    dig_k = np.eye(K, K, dtype=np.int)
    m_k = np.ones((K, K), dtype=np.int)
    B = np.matmul(A, (m_k - dig_k))

    # fix B if A[i][j] == 0, B[i][j] should be 0.
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] == 0:
                B[i][j] = 0
    return B


def cal_chi_from_a_b(A, B, labels):
    K = labels.shape[0]
    N = np.matmul(labels, np.ones((K)))  # 总样本数
    print("total samples: {0}".format(N))

    D = np.zeros(A.shape, dtype=np.int)
    C = np.zeros(A.shape, dtype=np.int)
    CHI = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] == 0:
                D[i][j] = 0
                C[i][j] = 0
            else:
                D[i][j] = N - labels[j] - B[i][j]
                C[i][j] = labels[j] - A[i][j]

    print("C matrix: ")
    print(C)
    print("D matrix: ")
    print(D)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] == 0:
                CHI[i][j] = 0
            else:
                tmp = (np.square((A[i][j]*D[i][j] - B[i][j]*C[i][j]))) / ((A[i][j] + B[i][j]) * (C[i][j] + D[i][j]))
                #print("chi[{0}][{1}] = math.log({2}/{3}) * {4}".format(i, j, N, A[i][j] + B[i][j], tmp))
                CHI[i][j] = math.log(N / (A[i][j] + B[i][j])) * tmp

    return CHI


def feature_select(chi, word_bag):
    word_dict = []
    for j in range(chi.shape[1]):
        a = chi[:, j]
        y = enumerate(a)
        a = sorted(y, key=lambda x: x[1], reverse=True)[:100]
        b = []
        for aa in a:
            b.append(aa[0])
        word_dict.extend(b)

    words = []
    for w in word_dict:
        if word_bag[w] not in words:
            words.append(word_bag[w])
    return word_dict, words


def document_features(data, word_bag):
    """
    labeled_featuresets: A list of classified featuresets,
    i.e., a list of tuples ``(featureset, label)``.
    """
    feature = {}
    tokens = set(data)
    for w in word_bag:
        if w in tokens:
            feature[w] = 1
        else:
            feature[w] = 0
    return feature


sample_num = 11

A, train_set, test_set, word_bag, labels = process_data("./data/training.csv", './data/testing.csv', sample_num, stop_word_list=stop_words_gbk())

fp = open("./train_set.json", "w", encoding='utf-8')
json.dump(train_set, fp)
fp.close()

fp = open("./test_set.json", "w", encoding='utf-8')
json.dump(train_set, fp)
fp.close()

print("word bag ")
print(word_bag)
print("A matrix ")
print(A)
B = cal_b_from_a(A)
print("B matrix ")
print(B)
CHI = cal_chi_from_a_b(A, B, labels)
print("chi matrix ")
print(CHI)
chi_fp = open("chi.pickle", "wb")
pickle.dump(CHI, file=chi_fp)

word_dict, words = feature_select(CHI, word_bag)
print(words)


documents = [(document_features(data[0], words), data[1]) for data in train_set]
test_documents_feature = [(document_features(data[0], words), data[1]) for data in test_set]

json.dump(documents, open('./documents_feature.json', 'w', encoding='utf-8'))
json.dump(test_documents_feature, open('./test_documents_feature.json', 'w', encoding='utf-8'))

classifier = nltk.NaiveBayesClassifier.train(documents[:4000])
test_error = nltk.classify.accuracy(classifier, documents[4000:4773])
print("test_error:{}".format(test_error))
classifier.show_most_informative_features(20)

