# -*- coding: utf-8 -*-
import jieba
import json
import math
import numpy as np
import nltk
import os
import pickle
import re
import sys
import time


def stop_words():
    stopwords_list = [line.strip() for line in open('./../../framework/stop_words_jieba.utf8.txt', encoding='utf-8')]
    stopwords_list = stopwords_list + [' ', '\n', '\t', '，']
    return stopwords_list


def stop_words_local():
    stop_words_file = open(os.path.join(os.path.dirname(__file__), './stop_words_utf8.txt'), 'r', encoding='utf-8')
    stopwords_list = []
    for line in stop_words_file.readlines():
        stopwords_list.append(line[:-1])

    if os.path.exists("./unrelative_words.json"):
        words = json.load(open("./unrelative_words.json", 'r', encoding="utf-8"))
        stopwords_list.extend(words)
    stopwords_list.extend(["满意","唆使"])
    return list(set(stopwords_list))


def jieba_fenci(raw, stopwords_list):
    word_list = list(jieba.cut(raw))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    word_list = [word for word in word_list if word not in stopwords_list]
    for word in word_list:
        if word in ['•', '轮轮', '定义']:
            word_list.remove(word)
    """if '\n' in word_list:
        word_list.remove('\n')"""
    return word_list


def process_all_terms(term_file, stop_word_list):
    tokens = []
    for line in open(term_file, "r", encoding="utf-8"):
        t = jieba_fenci("".join(line.strip().split(",")[1:]), stop_word_list)
        tokens.extend(t)
    tokens = list(set(tokens))
    return tokens


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
    labels = np.zeros((label_num), dtype=np.int)
    train_set = []
    test_set = []

    k = 0

    begin = int(time.time())

    if not os.path.exists(train_path):
        print("FATAL couldn't find file {0}".format(train_path))
        exit(0)

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            k += 1
            line = line.strip()
            tmp = line.split(',')
            sample = {}
            label = int(tmp[0])
            labels[label - 1] += 1
            content = re.sub(r"[a-zA-Z0-9_\u0020]+", '', "".join(tmp[1:]))
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

        w = open('./.word_bag.json', 'w', encoding='utf-8')
        json.dump(word_bag, fp=w, ensure_ascii=False)
        w.close()

        l = open('./.label.pickle', 'wb')
        pickle.dump(labels, file=l)
        l.close()

        f = open('./.A.pickle', 'wb')
        pickle.dump(A, file=f)
        f.close()
        print("train_set finished")

    if os.path.exists(test_path):
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

    fp = open("./.label_keywords.txt", "w", encoding="utf-8")

    for j in range(chi.shape[1]):
        a = chi[:, j]
        y = enumerate(a)
        a = sorted(y, key=lambda x: x[1], reverse=True)[:5]
        b = []
        for aa in a:
            b.append(aa[0])
        c = [word_bag[x] for x in b]
        c = list(set(c))
        fp.write("{0},{1}".format(j + 1, " ".join(c) + "\n"))
        word_dict.extend(b)

    fp.close()

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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python3 bayes.py {training file} {testing file} {the total of class}")
        print("")
        print("examples: python3 bayes.py training.txt testing.txt 10")
        print("")
        exit(0)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    sample_num = int(sys.argv[3])

    term_word_bag = process_all_terms("./../../corpus/samples/ts/regular_terms.txt", stop_word_list=stop_words_local())

    print(term_word_bag)
    A, train_set, test_set, word_bag, labels = process_data(train_file, test_file, sample_num, stop_word_list=stop_words_local())

    fp = open("./.train_set.json", "w", encoding='utf-8')
    json.dump(train_set, fp, ensure_ascii=False)
    fp.close()

    if len(test_set):
        fp = open("./.test_set.json", "w", encoding='utf-8')
        json.dump(test_set, fp, ensure_ascii=False)
        fp.close()

    B = cal_b_from_a(A)

    CHI = cal_chi_from_a_b(A, B, labels)

    chi_fp = open(".chi.pickle", "wb")
    pickle.dump(CHI, file=chi_fp)

    word_dict, words = feature_select(CHI, word_bag)
    #words = word_bag

    # 加上term中的词汇，看看效果
    words.extend(term_word_bag)

    # 写入词袋文件
    json.dump(words, open("./.word_bag.json", "w", encoding="utf-8"), ensure_ascii=False)

    f = open("chi_selected_words.txt", "w", encoding="utf-8")
    f.write("\n".join(words))
    f.close()

    documents = [(document_features(data[0], words), data[1]) for data in train_set]

    json.dump(documents, open('./.documents_feature.json', 'w', encoding='utf-8'), ensure_ascii=False)

    classifier = nltk.NaiveBayesClassifier.train(documents[:550])
    test_error = nltk.classify.accuracy(classifier, documents[550:605])
    print("test_error:{}".format(test_error))
    if len(test_set):
        test_documents_feature = [(document_features(data[0], words), data[1]) for data in test_set]
        json.dump(test_documents_feature, open('./.test_documents_feature.json', 'w', encoding='utf-8'), ensure_ascii=False)
        test_error = nltk.classify.accuracy(classifier, test_documents_feature)
        print("test_error:{}".format(test_error))
    else:
        print("NO TestSet!")

    classifier.show_most_informative_features(20)

    pickle.dump(classifier, open("./.bayes_classifier.pickle", 'wb'))

    input_set = jieba_fenci("交割先决条件", stopwords_list=stop_words_local())
    input_feature = document_features(input_set, words)
    print(input_feature)
    result = classifier.prob_classify(input_feature)
    print("测试结果：")
    print(result.max())
