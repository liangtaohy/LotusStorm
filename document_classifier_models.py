# -*- coding: utf-8 -*-
from framework import MLog
import os
import jieba
import jieba.analyse
import nltk
import random
import pickle

corpus_data_dir = './sample'

word_features = []

categories_arr = []


def document_features(document):
    """
    feature extractor
    :param document:
    :return:
    """
    global word_features
    document_words = set(document)
    features = {}
    for word, freq in word_features:
        if word in document_words:
            features[word] = freq
        else:
            features[word] = 0
    return features


def save_to_pickle(obj, filename):
    """
    save obj to pickle
    :param obj:
    :param filename:
    :return:
    """
    save_f = open(filename, "wb")
    pickle.dump(obj, save_f)


def load_documents(filename):
    """
    load corpus from file
    :param filename:
    :return: dictionary {'category':[words...]}
    """
    global word_features
    corpus_file = os.path.join(corpus_data_dir, filename)
    if not os.path.exists(corpus_file):
        #MLog.logger.fatal('corpus file not existed: ' + corpus_file)
        exit()

    file_ = open(corpus_file)
    try:
        text = file_.readline()

        documents = []

        while text:
            list_t = text.split(' ', 1)
            category = list_t[0]
            raw = list_t[1]
            #MLog.logger.debug('raw text: ' + text)
            #seg_list = jieba.cut(raw)

            # feature extractor (特征词抽取。对于短文本而言，这个就够用了)
            seg_list = jieba.analyse.extract_tags(raw, topK=20, withWeight=False)
            documents.append((list(seg_list), category))
            text = file_.readline()

        random.shuffle(documents)

        save_to_pickle(documents, "documents.pickle")

        # bag of words 词袋
        all_words = nltk.FreqDist(w for (words, _) in documents
                                  for w in words)

        # feature selection 特征选择（TF-词频）
        N = all_words.N()
        most_common = all_words.most_common(2000)
        for (w, freq) in most_common:
            word_features.append((w, freq/N))

        save_to_pickle(word_features, "word_features.pickle")

        featuresets = [(document_features(d), c) for (d, c) in documents]

        #total = len(featuresets)
        train_set, test_set = featuresets[100:], featuresets[:100]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print(nltk.classify.accuracy(classifier, test_set))
        classifier.show_most_informative_features(5)

        """
        Test Case
        """
        labels = classifier.labels()
        for label in labels:
            p = classifier.prob_classify(document_features(jieba.analyse.extract_tags("全国人民代表大会常务委员会关于特赦确实改恶从善的罪犯的决定［失效］"))).prob(label)
            print("全国人民代表大会常务委员会关于特赦确实改恶从善的罪犯的决定［失效］:%s-%f" % (label, p))

        for label in labels:
            p = classifier.prob_classify(document_features(jieba.analyse.extract_tags("我找你啊"))).prob(label)
            print("我找你啊:%s-%f" % (label, p))

        save_to_pickle(classifier, "NaiveBayesClassifier.pickle")

        """
        Too Slow!!!
        """
        if False:
            classifier = nltk.DecisionTreeClassifier.train(train_set)
            print(nltk.classify.accuracy(classifier, test_set))

        save_to_pickle(classifier, "DecisionTreeClassifier.pickle")

    finally:
        file_.close()


if __name__ == "__main__":
    file = 'corpus_findlaw_list.txt'
    load_documents(file)